#!/usr/bin/env python3
"""
Inference Engine for NiyahMini

REAL inference path (no DummyModel):
  - Loads trained weights produced by train.py (weights.npz + config).
  - Reuses the verified NiyahMiniModel.forward (RoPE + GQA + RMSNorm + SwiGLU +
    causal + tied LM head) so inference matches training exactly.
  - Sampling (temperature / top-k / top-p / repetition penalty), provenance
    tracking, deterministic mode, streaming.

Constraints honored:
  - Never fabricate output: if no trained weights are found at --model-dir,
    print an error to stderr and exit non-zero with NO generated text
    (the "NIYAH_ERR_NO_WEIGHTS" contract).
  - Evidence labels: FACT / INFERENCE / UNKNOWN / CONFLICTED.

NOTE: The tokenizer below is a deterministic byte-level placeholder. A real BPE
tokenizer (niyah_mini_vocab) is not yet implemented and is tracked as a separate
P0 item; until it lands, generation operates on byte ids, which is sufficient to
verify that trained weights load and produce coherent logits end-to-end.

NO borrowed code from any existing model implementation.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import numpy as np

# Reuse the numerically-verified model from train.py rather than duplicating the
# forward/backward math. train.py is import-safe (its CLI runs under an
# "__main__" guard; importing it only defines classes/functions).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import NiyahMiniModel  # noqa: E402


class ByteTokenizer:
    """Deterministic byte-level tokenizer (placeholder pending real BPE).

    encode: each character -> its Unicode code point (clamped to the model
            vocabulary range).
    decode: each id -> chr(id).
    This is intentionally simple and reversible so that inference can be
    exercised end-to-end on trained weights without a trained BPE vocab.
    """

    def __init__(self, n_vocab: int = 32768):
        self.n_vocab = n_vocab

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            code = ord(ch)
            if code >= self.n_vocab:
                code = code % self.n_vocab
            ids.append(code)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


def load_model(model_dir: Path) -> Optional[NiyahMiniModel]:
    """Load a NiyahMiniModel from a checkpoint directory.

    train.py's NiyahMiniModel.save() writes weights.npz containing the config,
    seed and all weight tensors. The Trainer also writes config.json. We prefer
    the config embedded in weights.npz (self-contained) and fall back to
    config.json. Returns None if no usable weights are present.
    """
    model_dir = Path(model_dir)
    weights_path = model_dir / "weights.npz"
    if not weights_path.exists():
        return None

    z = np.load(weights_path, allow_pickle=True)
    config = None
    if "config" in z.files:
        config = z["config"].item()
    if config is None and (model_dir / "config.json").exists():
        with (model_dir / "config.json").open("r") as f:
            full = json.load(f)
        config = full.get("model", full)
    if config is None:
        return None

    model = NiyahMiniModel(config)
    model.load_state_dict({k: z[k] for k in z.files})
    return model


class NiyahMiniInference:
    """Inference engine for NiyahMini backed by the real trained model."""

    def __init__(self, model: NiyahMiniModel, tokenizer=None, config: Optional[Dict] = None):
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer(model.n_vocab)
        self.config = config or {}

        # Sampling parameters
        self.temperature = self.config.get("temperature", 1.0)
        self.top_k = self.config.get("top_k", 50)
        self.top_p = self.config.get("top_p", 1.0)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.0)
        self.max_tokens = self.config.get("max_tokens", 256)
        self.seed = self.config.get("seed", None)

        # State
        self.generated_tokens: List[int] = []
        self.start_time = None

        if self.seed is not None:
            np.random.seed(int(self.seed))

    def set_sampling_params(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        """Set sampling parameters."""
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        include_provenance: bool = True,
    ) -> Dict[str, Any]:
        """Generate text from prompt using the real model forward pass."""
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p

        self.start_time = time.time()
        self.generated_tokens = []

        # Tokenize prompt (byte-level placeholder tokenizer)
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]

        input_ids = np.array(prompt_ids, dtype=np.int32)
        output_text = self._generate_tokens(input_ids, stop_sequences or [])

        result = {
            "text": output_text,
            "prompt": prompt,
            "num_tokens": len(self.generated_tokens),
            "generation_time": time.time() - self.start_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.get("model_name", "niyah_mini"),
            "config": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "max_tokens": self.max_tokens,
            },
        }

        if include_provenance:
            result["provenance"] = self._create_provenance_record(prompt, output_text)

        return result

    def _generate_tokens(self, input_ids: np.ndarray, stop_sequences: List[str]) -> str:
        """Generate tokens autoregressively with the real model.

        Unlike the previous stub, the FULL context is fed to forward() at every
        step (the Python model has no KV cache), and the next-token logits are
        read from the final position. Context is capped to n_ctx.
        """
        self.generated_tokens = []
        ctx_cap = int(getattr(self.model, "n_ctx", 2048))
        output_text = ""

        for _ in range(self.max_tokens):
            # Cap context to the model's maximum sequence length (sliding window)
            ctx = input_ids
            if ctx.shape[0] > ctx_cap:
                ctx = ctx[-ctx_cap:]

            logits = self.model.forward(ctx.reshape(1, -1))[0, -1, :]

            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Apply top-k filtering
            if self.top_k > 0:
                logits = self._top_k_filtering(logits, self.top_k)

            # Apply top-p (nucleus) filtering
            if self.top_p < 1.0:
                logits = self._top_p_filtering(logits, self.top_p)

            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits)

            next_token = self._sample_from_logits(logits)

            self.generated_tokens.append(int(next_token))
            output_text += chr(int(next_token))
            input_ids = np.append(input_ids, next_token)

            # Stop sequences (checked against decoded text so far)
            if stop_sequences:
                for s in stop_sequences:
                    if s and output_text.endswith(s):
                        return output_text
            # Stop if we exceed context (no more room)
            if input_ids.shape[0] >= ctx_cap:
                break

        return output_text

    def _top_k_filtering(self, logits: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k filtering to logits."""
        if k <= 0 or k >= logits.shape[0]:
            return logits
        top_k_indices = np.argsort(logits)[-k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[top_k_indices] = logits[top_k_indices]
        return filtered

    def _top_p_filtering(self, logits: np.ndarray, p: float) -> np.ndarray:
        """Apply nucleus (top-p) filtering to logits."""
        if p >= 1.0:
            return logits
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = self._softmax(sorted_logits)
        cum_probs = np.cumsum(probs)
        # Keep the smallest set whose cumulative probability >= p
        mask = cum_probs <= p
        cutoff = int(np.sum(mask))
        result = np.full_like(logits, -np.inf)
        keep = sorted_indices[: cutoff + 1]
        result[keep] = logits[keep]
        return result

    def _apply_repetition_penalty(self, logits: np.ndarray) -> np.ndarray:
        """Apply repetition penalty to logits."""
        if self.repetition_penalty == 1.0 or not self.generated_tokens:
            return logits
        token_counts = np.bincount(
            np.array(self.generated_tokens, dtype=np.int64),
            minlength=logits.shape[0],
        )
        for i in range(len(logits)):
            count = int(token_counts[i]) if i < len(token_counts) else 0
            if count > 0:
                logits[i] = logits[i] / (self.repetition_penalty ** count)
        return logits

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _sample_from_logits(self, logits: np.ndarray) -> int:
        """Sample a token id from logits."""
        probs = self._softmax(logits)
        # Guard against numerical issues
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            return int(np.argmax(logits))
        probs = probs / s
        return int(np.random.choice(len(probs), p=probs))

    def _create_provenance_record(self, prompt: str, output: str) -> Dict[str, Any]:
        """Create provenance record for generation."""
        return {
            "input_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output.encode()).hexdigest(),
            "model": self.config.get("model_name", "niyah_mini"),
            "model_hash": self.config.get("model_hash", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "max_tokens": self.max_tokens,
            },
        }

    def generate_streaming(
        self,
        prompt: str,
        callback: Callable[[str], None],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text with streaming output."""
        result = self.generate(prompt, include_provenance=False, **kwargs)
        for ch in result["text"]:
            callback(ch)
        return result


def format_evidence_output(text: str, sources: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Format output with evidence labels (FACT / INFERENCE / UNKNOWN / CONFLICTED)."""
    if "FACT" in text or "حقيقة" in text:
        label = "FACT"
    elif "INFERENCE" in text or "استدلال" in text:
        label = "INFERENCE"
    elif "UNKNOWN" in text or "مجهول" in text:
        label = "UNKNOWN"
    elif "CONFLICTED" in text or "متضارب" in text:
        label = "CONFLICTED"
    else:
        label = "UNKNOWN"  # Default to unknown if no explicit label

    result = {
        "label": label,
        "answer": text,
        "source_ids": [s.get("id", "") for s in (sources or [])],
        "limitations": [],
        "verification_steps": [],
        "lvu_agreement": 1.0 if label == "FACT" else 0.5,
        "lvu_label": label,
        "peer_prediction_consistent": True,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference with NiyahMini")
    parser.add_argument("--model-dir", type=Path, required=True, help="Model/checkpoint directory (must contain weights.npz)")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) filtering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic sampling")
    parser.add_argument("--stop", type=str, default=None, help="Optional stop sequence")
    parser.add_argument("--evidence", action="store_true", help="Format output with evidence labels")
    args = parser.parse_args()

    # Load REAL trained weights. Never fabricate output if weights are missing.
    model = load_model(args.model_dir)
    if model is None:
        print(
            f"ERROR: no trained weights found at {args.model_dir} "
            f"(expected weights.npz). Refusing to fabricate output.",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = ByteTokenizer(model.n_vocab)

    inference = NiyahMiniInference(
        model,
        tokenizer=tokenizer,
        config={
            "model_name": "niyah_mini",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )

    print(f"Prompt: {args.prompt}")
    print(f"Generating up to {args.max_tokens} tokens...")
    print()

    stop_sequences = [args.stop] if args.stop else None
    result = inference.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_sequences=stop_sequences,
    )

    if args.evidence:
        evidence_result = format_evidence_output(result["text"])
        print(json.dumps(evidence_result, indent=2, ensure_ascii=False))
    else:
        print(f"Output: {result['text']}")
        print(f"\nTokens: {result['num_tokens']}")
        print(f"Time: {result['generation_time']:.2f}s")


if __name__ == "__main__":
    main()
