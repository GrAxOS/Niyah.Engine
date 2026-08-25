#!/usr/bin/env python3
"""
Training Loop for NiyahMini - Original Model Training from Scratch

REAL, working implementation (not a prototype):
  - Forward: RoPE + GQA attention + RMSNorm (pre-norm) + SwiGLU FFN, causal
    masking, tied embeddings.
  - Full manual backpropagation. Gradients verified by finite differences
    (run: python train.py --grad-check).
  - SGD-with-momentum and AdamW optimizers that actually update parameters.
  - Checkpointing with provenance, LR scheduling, gradient clipping, determinism.

NO borrowed code from any existing model implementation.
All algorithms implemented from first principles and numerically verified.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _rmsnorm_fwd(x, w, eps):
    ss = np.sum(x * x, axis=-1, keepdims=True) / x.shape[-1]
    r = 1.0 / np.sqrt(ss + eps)
    return x * r * w, r


def _rmsnorm_bwd(do, x, w, r):
    D = x.shape[-1]
    S = np.sum(do * x * w, axis=-1, keepdims=True)
    coeff = -(r * r * r) / D
    dx = do * w * r + coeff * x * S
    dw = do * x * r
    return dx, dw


def _rope_cos_sin(T, head_dim, theta):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(T, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1).astype(np.float32)
    return cos, sin


def _rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rope_fwd(x, cos, sin):
    # x: (B, T, H, head_dim); cos/sin: (T, head_dim)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return x * c + _rotate_half(x) * s


def _rope_bwd(dy, cos, sin):
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return dy * c - _rotate_half(dy) * s


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NiyahMiniModel:
    """Original transformer: pre-norm, RoPE, GQA, SwiGLU, tied LM head."""

    LAYER_KEYS = ["attn_norm", "wq", "wk", "wv", "wo",
                  "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"]

    def __init__(self, config, seed=42, dtype=np.float32):
        self.config = dict(config)
        self.seed = seed
        self.dtype = dtype
        self._derive_dims()
        self.rng = np.random.RandomState(seed)
        self.weights = self._init_weights()
        self.grads = self._zero_grads()
        self.cache = {}
        cos, sin = _rope_cos_sin(self.n_ctx, self.head_dim, self.rope_theta)
        self._cos = cos
        self._sin = sin

    def _derive_dims(self):
        c = self.config
        self.n_layers = int(c.get("n_layers", 12))
        self.n_dim = int(c.get("n_dim", 512))
        self.n_heads = int(c.get("n_heads", 8))
        self.n_kv_heads = int(c.get("n_kv_heads", max(1, self.n_heads // 2)))
        self.n_ff = int(c.get("n_ff", 4 * self.n_dim))
        self.n_vocab = int(c.get("n_vocab", 32768))
        self.n_ctx = int(c.get("n_ctx", 2048))
        self.rope_theta = float(c.get("rope_theta", 10000.0))
        self.norm_eps = float(c.get("norm_eps", 1e-5))
        self.tie = bool(c.get("tie_word_embeddings", True))
        assert self.n_dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = self.n_dim // self.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.rep = self.n_heads // self.n_kv_heads

    @staticmethod
    def _xavier(shape, rng):
        n_in, n_out = shape[1], shape[0]
        scale = math.sqrt(2.0 / (n_in + n_out))
        return rng.uniform(-1, 1, shape) * scale

    def _init_weights(self):
        rng = self.rng
        D, V, FF, KVD = self.n_dim, self.n_vocab, self.n_ff, self.kv_dim
        dt = self.dtype
        w = {}
        w["embedding"] = (rng.uniform(-1, 1, (V, D)) * math.sqrt(1.0 / D)).astype(dt)
        for l in range(self.n_layers):
            ly = {}
            ly["attn_norm"] = np.ones(D, dtype=dt)
            ly["wq"] = self._xavier((D, D), rng).astype(dt)
            ly["wk"] = self._xavier((KVD, D), rng).astype(dt)
            ly["wv"] = self._xavier((KVD, D), rng).astype(dt)
            ly["wo"] = self._xavier((D, D), rng).astype(dt)
            ly["ffn_norm"] = np.ones(D, dtype=dt)
            ly["ffn_gate"] = self._xavier((FF, D), rng).astype(dt)
            ly["ffn_up"] = self._xavier((FF, D), rng).astype(dt)
            ly["ffn_down"] = self._xavier((D, FF), rng).astype(dt)
            w[f"layer_{l}"] = ly
        w["final_norm"] = np.ones(D, dtype=dt)
        return w

    def _zero_grads(self):
        g = {"embedding": np.zeros_like(self.weights["embedding"])}
        for l in range(self.n_layers):
            g[f"layer_{l}"] = {k: np.zeros_like(v) for k, v in self.weights[f"layer_{l}"].items()}
        g["final_norm"] = np.zeros_like(self.weights["final_norm"])
        return g

    def zero_grad(self):
        self.grads = self._zero_grads()

    def num_params(self):
        return int(sum(p.size for _, p in self._named_params()))

    def _named_params(self):
        yield "embedding", self.weights["embedding"]
        yield "final_norm", self.weights["final_norm"]
        for l in range(self.n_layers):
            for k in self.LAYER_KEYS:
                yield f"layer_{l}.{k}", self.weights[f"layer_{l}"][k]

    def _named_grads(self):
        yield "embedding", self.grads["embedding"]
        yield "final_norm", self.grads["final_norm"]
        for l in range(self.n_layers):
            for k in self.LAYER_KEYS:
                yield f"layer_{l}.{k}", self.grads[f"layer_{l}"][k]

    def grads_for(self, name):
        if name in ("embedding", "final_norm"):
            return self.grads[name]
        l, k = name.split(".")
        return self.grads[l][k]

    # -- forward ---------------------------------------------------------
    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.n_ctx
        D, H, HD, FF, V = self.n_dim, self.n_heads, self.head_dim, self.n_ff, self.n_vocab
        cos, sin = self._cos[:T], self._sin[:T]
        emb = self.weights["embedding"]
        x = emb[input_ids].astype(self.dtype)
        cache = {"layers": [], "input_ids": input_ids, "T": T}
        mask = np.triu(np.ones((T, T), dtype=self.dtype), k=1) * -1e30
        for l in range(self.n_layers):
            ly = self.weights[f"layer_{l}"]
            h, r_an = _rmsnorm_fwd(x, ly["attn_norm"], self.norm_eps)
            q = (h @ ly["wq"].T).reshape(B, T, H, HD)
            k = (h @ ly["wk"].T).reshape(B, T, self.n_kv_heads, HD)
            v = (h @ ly["wv"].T).reshape(B, T, self.n_kv_heads, HD)
            q = _rope_fwd(q, cos, sin)                                  # (B,T,H,HD)
            k_rope = _rope_fwd(k, cos, sin)                             # (B,T,KV,HD)
            k_rep = np.repeat(k_rope, self.rep, axis=2)                 # (B,T,H,HD)
            v_rep = np.repeat(v, self.rep, axis=2)                      # (B,T,H,HD)
            scores = np.einsum("bthd,bshd->bhts", q, k_rep) / math.sqrt(HD)
            scores = scores + mask                                      # (B,H,T,T)
            probs = _softmax(scores, axis=-1)                          # (B,H,T,T)
            out = np.einsum("bhts,bshd->bhtd", probs, v_rep)            # (B,H,T,HD)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, D)           # (B,T,D)
            ao = out @ ly["wo"].T
            res = x + ao
            h2, r_fn = _rmsnorm_fwd(res, ly["ffn_norm"], self.norm_eps)
            gate = h2 @ ly["ffn_gate"].T
            up = h2 @ ly["ffn_up"].T
            ff = _silu(gate) * up
            fo = ff @ ly["ffn_down"].T
            cache["layers"].append({
                "x": x, "h": h, "r_an": r_an, "q_rope": q, "k_rope": k_rope,
                "v": v, "probs": probs, "out_heads": out, "ao": ao, "res": res,
                "h2": h2, "r_fn": r_fn, "gate": gate, "up": up, "ff": ff, "fo": fo,
            })
            x = res + fo
        final_pre = x
        x, r_fnorm = _rmsnorm_fwd(x, self.weights["final_norm"], self.norm_eps)
        logits = x @ emb.T
        cache["final_pre"] = final_pre
        cache["final_in"] = x
        cache["r_fnorm"] = r_fnorm
        self.cache = cache
        return logits

    def loss_and_dlogits(self, logits, targets):
        B, T, V = logits.shape
        probs = _softmax(logits, axis=-1)
        log_probs = np.log(probs + 1e-30)
        n = targets.size
        loss = -float(np.sum(np.take_along_axis(log_probs, targets[..., None], axis=-1)) / n)
        onehot = np.zeros_like(probs)
        np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
        dlogits = (probs - onehot) / n
        return loss, dlogits.astype(logits.dtype)

    # -- backward --------------------------------------------------------
    def backward(self, dlogits):
        c = self.cache
        layers = c["layers"]
        T = c["T"]
        B = dlogits.shape[0]
        D, H, HD, KVD, FF = self.n_dim, self.n_heads, self.head_dim, self.kv_dim, self.n_ff
        cos, sin = self._cos[:T], self._sin[:T]
        emb = self.weights["embedding"]
        h_final = c["final_in"]
        glm = np.einsum("btv,btd->vd", dlogits, h_final)
        self.grads["embedding"] += glm
        dx = dlogits @ emb
        d_pre, d_fnorm = _rmsnorm_bwd(dx, c["final_pre"], self.weights["final_norm"], c["r_fnorm"])
        self.grads["final_norm"] += np.sum(d_fnorm, axis=(0, 1))
        dx = d_pre
        for l in range(self.n_layers - 1, -1, -1):
            lc = layers[l]
            ly = self.weights[f"layer_{l}"]
            gl = self.grads[f"layer_{l}"]
            d_res = dx.copy()
            d_fo = dx.copy()
            gl["ffn_down"] += np.einsum("btd,btf->df", d_fo, lc["ff"])
            d_ff = d_fo @ ly["ffn_down"]
            gate = lc["gate"]; up = lc["up"]
            sig = 1.0 / (1.0 + np.exp(-gate))
            d_up = d_ff * (sig * gate)
            d_gate = d_ff * up * sig * (1.0 + gate * (1.0 - sig))
            gl["ffn_gate"] += np.einsum("btf,btd->fd", d_gate, lc["h2"])
            gl["ffn_up"] += np.einsum("btf,btd->fd", d_up, lc["h2"])
            d_h2 = d_gate @ ly["ffn_gate"] + d_up @ ly["ffn_up"]
            d_res2, d_fn = _rmsnorm_bwd(d_h2, lc["res"], ly["ffn_norm"], lc["r_fn"])
            gl["ffn_norm"] += np.sum(d_fn, axis=(0, 1))
            d_res_total = d_res + d_res2
            d_ao = d_res_total
            gl["wo"] += np.einsum("btd,btD->dD", d_ao, lc["out_heads"])
            d_out = (d_ao @ ly["wo"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)  # (B,H,T,HD)
            probs = lc["probs"]
            v = lc["v"]
            k_rope = lc["k_rope"]
            q_rope = lc["q_rope"]
            v_rep = np.repeat(v, self.rep, axis=2)
            k_rep = np.repeat(k_rope, self.rep, axis=2)
            d_v_rep = np.einsum("bhts,bhtd->bshd", probs, d_out)        # (B,T,H,HD)
            g_probs = np.einsum("bhtd,bshd->bhts", d_out, v_rep)        # (B,H,T,T)
            sum_pg = np.sum(probs * g_probs, axis=-1, keepdims=True)
            d_scores = probs * (g_probs - sum_pg)                      # (B,H,T,T)
            inv_s = 1.0 / math.sqrt(HD)
            d_q_rope = np.einsum("bhts,bshd->bthd", d_scores, k_rep) * inv_s   # (B,T,H,HD)
            d_k_rep = np.einsum("bhts,bthd->bshd", d_scores, q_rope) * inv_s   # (B,T,H,HD)
            d_v = d_v_rep.reshape(B, T, self.n_kv_heads, self.rep, HD).sum(axis=3)
            d_k = d_k_rep.reshape(B, T, self.n_kv_heads, self.rep, HD).sum(axis=3)
            d_q = _rope_bwd(d_q_rope, cos, sin).reshape(B, T, D)
            d_k = _rope_bwd(d_k, cos, sin).reshape(B, T, KVD)
            d_v = d_v.reshape(B, T, KVD)
            gl["wq"] += np.einsum("btd,btD->dD", d_q, lc["h"])
            gl["wk"] += np.einsum("btd,btD->dD", d_k, lc["h"])
            gl["wv"] += np.einsum("btd,btD->dD", d_v, lc["h"])
            d_h = d_q @ ly["wq"] + d_k @ ly["wk"] + d_v @ ly["wv"]
            d_x_from_attn, d_an = _rmsnorm_bwd(d_h, lc["x"], ly["attn_norm"], lc["r_an"])
            gl["attn_norm"] += np.sum(d_an, axis=(0, 1))
            dx = d_res_total + d_x_from_attn
        np.add.at(self.grads["embedding"], c["input_ids"], dx)

    # -- optimizers ------------------------------------------------------
    def clip_grads(self, max_norm):
        total = math.sqrt(sum(float(np.sum(g * g)) for _, g in self._named_grads()))
        if max_norm and total > max_norm and total > 0:
            scale = max_norm / total
            for _, g in self._named_grads():
                g *= scale
        return total

    def step_sgd(self, lr, state, momentum=0.9, weight_decay=0.0):
        for name, p in self._named_params():
            g = self.grads_for(name)
            m = state["m"][name]
            m *= momentum
            m += g
            p -= lr * (m + weight_decay * p)

    def step_adamw(self, lr, state, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        state["t"] += 1
        t = state["t"]
        for name, p in self._named_params():
            g = self.grads_for(name)
            m = state["m"][name]
            v = state["v"][name]
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * (g * g)
            mhat = m / (1 - beta1 ** t)
            vhat = v / (1 - beta2 ** t)
            p -= lr * (mhat / (np.sqrt(vhat) + eps) + weight_decay * p)

    def init_optim_state(self):
        m, v = {}, {}
        for name, p in self._named_params():
            m[name] = np.zeros_like(p)
            v[name] = np.zeros_like(p)
        return {"m": m, "v": v, "t": 0}

    # -- serialization ---------------------------------------------------
    def state_dict(self):
        sd = {"config": self.config, "seed": self.seed}
        for name, p in self._named_params():
            sd[name] = p.copy()
        return sd

    def load_state_dict(self, sd):
        self.config = dict(sd["config"])
        self._derive_dims()
        for name, p in self._named_params():
            p[...] = np.asarray(sd[name], dtype=self.dtype)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "weights.npz", **self.state_dict())

    def load(self, path):
        z = np.load(Path(path) / "weights.npz", allow_pickle=True)
        self.load_state_dict({k: z[k] for k in z.files})


# ---------------------------------------------------------------------------
# Trainer (uses the real model)
# ---------------------------------------------------------------------------

class NiyahMiniTrainer:
    """Original trainer for NiyahMini: wires the real model to the training loop."""

    def __init__(self, config, output_dir, seed=42):
        self.config = config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.step = 0
        self.best_loss = float("inf")
        self.start_time = None
        self.rng = np.random.RandomState(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_cfg = config.get("model", config)
        self.model = NiyahMiniModel(model_cfg, seed=seed)
        self.optimizer = None
        self.opt_state = None
        self.history = []

    def initialize_optimizer(self):
        oc = self.config.get("optimizer", {})
        self.optimizer = {
            "type": oc.get("type", "adamw"),
            "base_lr": oc.get("lr", 1e-3),
            "lr": oc.get("lr", 1e-3),
            "beta1": oc.get("beta1", 0.9),
            "beta2": oc.get("beta2", 0.999),
            "eps": oc.get("eps", 1e-8),
            "momentum": oc.get("momentum", 0.9),
            "weight_decay": oc.get("weight_decay", 0.01),
        }
        self.opt_state = self.model.init_optim_state()

    def get_learning_rate(self):
        lr_cfg = self.config.get("lr_scheduler", {})
        base_lr = self.optimizer["base_lr"]
        if lr_cfg.get("type") == "cosine":
            warmup = lr_cfg.get("warmup_steps", 1000)
            total = lr_cfg.get("total_steps", 100000)
            if self.step < warmup:
                return base_lr * (self.step + 1) / warmup
            progress = (self.step - warmup) / max(1, (total - warmup))
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        if lr_cfg.get("type") == "linear":
            warmup = lr_cfg.get("warmup_steps", 1000)
            total = lr_cfg.get("total_steps", 100000)
            if self.step < warmup:
                return base_lr * (self.step + 1) / warmup
            progress = (self.step - warmup) / max(1, (total - warmup))
            return base_lr * (1.0 - progress)
        return base_lr

    def train_step(self, batch):
        lr = self.get_learning_rate()
        self.optimizer["lr"] = lr
        self.model.zero_grad()
        logits = self.model.forward(batch["input_ids"])
        loss, dlogits = self.model.loss_and_dlogits(logits, batch["targets"])
        self.model.backward(dlogits)
        grad_norm = self.model.clip_grads(self.config.get("gradient_clip", 1.0))
        if self.optimizer["type"] == "sgd":
            self.model.step_sgd(lr, self.opt_state,
                                self.optimizer["momentum"], self.optimizer["weight_decay"])
        else:
            self.model.step_adamw(lr, self.opt_state,
                                  self.optimizer["beta1"], self.optimizer["beta2"],
                                  self.optimizer["eps"], self.optimizer["weight_decay"])
        self.step += 1
        self.history.append({"step": self.step, "loss": loss, "lr": lr,
                             "grad_norm": float(grad_norm)})
        return {"loss": loss, "lr": lr, "grad_norm": float(grad_norm), "step": self.step}

    def evaluate(self, data, batch_size=8):
        total = 0.0
        nb = max(1, len(data) // batch_size)
        for i in range(0, len(data), batch_size):
            items = data[i:i + batch_size]
            if not items:
                continue
            ml = max(len(it["input_ids"]) for it in items)
            ids = np.zeros((len(items), ml), dtype=np.int32)
            tg = np.zeros((len(items), ml), dtype=np.int32)
            for j, it in enumerate(items):
                ids[j, :len(it["input_ids"])] = it["input_ids"]
                tg[j, :len(it["targets"])] = it["targets"]
            logits = self.model.forward(ids)
            loss, _ = self.model.loss_and_dlogits(logits, tg)
            total += loss
        return total / nb

    def save_checkpoint(self, checkpoint_dir=None):
        checkpoint_dir = Path(checkpoint_dir or (self.output_dir / f"checkpoint_{self.step}"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(checkpoint_dir)
        with (checkpoint_dir / "config.json").open("w") as f:
            json.dump(self.config, f, indent=2)
        with (checkpoint_dir / "optimizer.json").open("w") as f:
            json.dump({k: v for k, v in self.optimizer.items() if k != "lr"} |
                      {"lr": self.optimizer["lr"], "t": self.opt_state["t"]}, f, indent=2)
        with (checkpoint_dir / "state.json").open("w") as f:
            json.dump({"step": self.step, "best_loss": self.best_loss,
                       "history": self.history[-100:]}, f, indent=2)
        self._save_manifest(checkpoint_dir)
        return checkpoint_dir

    def _save_manifest(self, checkpoint_dir):
        hashes = {}
        for fn in ["weights.npz", "config.json", "optimizer.json", "state.json"]:
            fp = checkpoint_dir / fn
            if fp.exists():
                with fp.open("rb") as f:
                    hashes[fn] = hashlib.sha256(f.read()).hexdigest()
        manifest = {
            "checkpoint_id": f"step_{self.step}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": self.step,
            "loss": self.history[-1]["loss"] if self.history else None,
            "model_config": self.config.get("model", self.config),
            "files": hashes,
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        }
        with (checkpoint_dir / "checkpoint_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        with (checkpoint_dir / "config.json").open("r") as f:
            self.config = json.load(f)
        self.model = NiyahMiniModel(self.config.get("model", self.config), seed=self.seed)
        self.model.load(checkpoint_dir)
        self.initialize_optimizer()
        with (checkpoint_dir / "state.json").open("r") as f:
            st = json.load(f)
        self.step = st["step"]
        self.best_loss = st["best_loss"]

    def train(self, train_data, val_data=None, num_epochs=1, batch_size=8,
              checkpoint_every=1000, validate_every=100):
        self.start_time = time.time()
        nb = max(1, len(train_data) // batch_size)
        for epoch in range(num_epochs):
            self.rng.shuffle(train_data)
            epoch_loss = 0.0
            t0 = time.time()
            for bi in range(nb):
                items = train_data[bi * batch_size:(bi + 1) * batch_size]
                ml = max(len(it["input_ids"]) for it in items)
                ids = np.zeros((len(items), ml), dtype=np.int32)
                tg = np.zeros((len(items), ml), dtype=np.int32)
                for j, it in enumerate(items):
                    ids[j, :len(it["input_ids"])] = it["input_ids"]
                    tg[j, :len(it["targets"])] = it["targets"]
                res = self.train_step({"input_ids": ids, "targets": tg})
                epoch_loss += res["loss"]
                if self.step % checkpoint_every == 0:
                    self.save_checkpoint()
                if val_data and self.step % validate_every == 0:
                    vl = self.evaluate(val_data, batch_size)
                    if vl < self.best_loss:
                        self.best_loss = vl
                        self.save_checkpoint(self.output_dir / "best")
                if (bi + 1) % 100 == 0:
                    print(f"Epoch {epoch+1} Batch {bi+1}/{nb} loss={epoch_loss/(bi+1):.4f} "
                          f"lr={res['lr']:.2e} gn={res['grad_norm']:.2f} "
                          f"t={time.time()-t0:.1f}s")
            print(f"Epoch {epoch+1} done avg_loss={epoch_loss/nb:.4f}")
        self.save_checkpoint()
        print(f"Training complete. steps={self.step}")


# ---------------------------------------------------------------------------
# Gradient check (self-validating via finite differences)
# ---------------------------------------------------------------------------

def gradient_check(config=None, seed=0, tol=1e-4, n_per_tensor=4):
    cfg = config or {
        "n_layers": 2, "n_dim": 16, "n_heads": 4, "n_kv_heads": 2, "n_ff": 32,
        "n_vocab": 32, "n_ctx": 16, "rope_theta": 10000.0, "norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    model = NiyahMiniModel(cfg, seed=seed, dtype=np.float64)
    T, B = 6, 1
    rng = np.random.RandomState(seed + 1)
    ids = rng.randint(0, cfg["n_vocab"], size=(B, T)).astype(np.int32)
    targets = ids.copy()

    logits = model.forward(ids)
    loss0, dlogits = model.loss_and_dlogits(logits, targets)
    model.zero_grad()
    model.backward(dlogits)

    def loss_only():
        lg = model.forward(ids)
        lo, _ = model.loss_and_dlogits(lg, targets)
        return lo

    eps = 1e-6
    max_rel = 0.0
    checked = 0
    worst = []

    def check(name, ref, grad):
        nonlocal max_rel, checked
        o = ref.copy()
        ref += eps
        lp = loss_only()
        ref -= 2 * eps
        lm = loss_only()
        ref[...] = o
        num = (lp - lm) / (2 * eps)
        rel = abs(num - grad) / (abs(num) + abs(grad) + 1e-12)
        if rel > max_rel:
            max_rel = rel
        checked += 1
        worst.append((name, rel, float(num), float(grad)))

    def pick2(name, M, G):
        for _ in range(n_per_tensor):
            i = rng.randint(M.shape[0])
            j = rng.randint(M.shape[1])
            check(f"{name}[{i},{j}]", M[i:i+1, j], float(G[i, j]))

    def pick1(name, arr, garr):
        for _ in range(n_per_tensor):
            i = rng.randint(arr.shape[0])
            check(f"{name}[{i}]", arr[i:i+1], float(garr[i]))

    NORM_KEYS = {"attn_norm", "ffn_norm"}

    pick2("embedding", model.weights["embedding"], model.grads["embedding"])
    for l in range(model.n_layers):
        for k in model.LAYER_KEYS:
            w = model.weights[f"layer_{l}"][k]
            g = model.grads[f"layer_{l}"][k]
            if k in NORM_KEYS:
                pick1(f"L{l}.{k}", w, g)
            else:
                pick2(f"L{l}.{k}", w, g)
    pick1("final_norm", model.weights["final_norm"], model.grads["final_norm"])

    worst.sort(key=lambda x: -x[1])
    return max_rel, checked, worst[:6]


# ---------------------------------------------------------------------------
# Dataset helper (placeholder tokenizer; real BPE is in niyah_mini_vocab.c)
# ---------------------------------------------------------------------------

def create_dataset_from_corpus(corpus_path, tokenizer=None, max_len=128):
    corpus_path = Path(corpus_path)
    with corpus_path.open("r") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    train_data, val_data = [], []
    for i, rec in enumerate(corpus):
        text = rec.get("text", "")
        # NOTE: placeholder char-level ids. Replace with the real BPE tokenizer
        # (native/niyah_mini/niyah_mini_vocab.c) once it trains real merges.
        token_ids = [min(ord(c), 255) for c in text[:max_len]]
        if len(token_ids) > 4:
            split = len(token_ids) // 2
            ex = {
                "input_ids": token_ids[:split],
                "targets": token_ids[split:split + (len(token_ids) - split)],
                "document_id": rec.get("document_id", ""),
                "source_url": rec.get("source_url", ""),
                "license": rec.get("license", ""),
                "language": rec.get("language", "unknown"),
                "domain": rec.get("domain", "general"),
            }
            (val_data if i % 10 == 0 else train_data).append(ex)
    return train_data, val_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NiyahMini from scratch")
    parser.add_argument("--config", type=Path, help="Training config JSON")
    parser.add_argument("--corpus", type=Path, help="Training corpus JSONL")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-check", action="store_true",
                        help="Run finite-difference gradient check and exit")
    args = parser.parse_args()

    if args.grad_check:
        max_rel, checked, worst = gradient_check()
        print(f"Gradient check: {checked} params sampled, max rel error = {max_rel:.3e}")
        for name, rel, num, an in worst:
            print(f"  {name:28s} rel={rel:.2e} num={num:+.4e} analytic={an:+.4e}")
        ok = max_rel < 1e-4
        print("RESULT: PASS" if ok else "RESULT: FAIL")
        sys.exit(0 if ok else 1)

    if not (args.config and args.corpus and args.output_dir):
        parser.error("--config, --corpus and --output-dir are required (or use --grad-check)")

    with args.config.open("r") as f:
        config = json.load(f)
    trainer = NiyahMiniTrainer(config, args.output_dir, args.seed)
    trainer.initialize_optimizer()
    print(f"Model params: {trainer.model.num_params():,}")

    train_data, val_data = create_dataset_from_corpus(args.corpus)
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")
    trainer.train(train_data, val_data, num_epochs=args.epochs, batch_size=args.batch_size)
    trainer.save_checkpoint(trainer.output_dir / "final")
    print(f"Final checkpoint: {trainer.output_dir / 'final'}")


if __name__ == "__main__":
    main()
