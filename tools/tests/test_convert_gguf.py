#!/usr/bin/env python3
"""End-to-end validation of tools/convert_gguf_to_niyah.py.

Run directly:

    python3 tools/tests/test_convert_gguf.py

What this actually proves
-------------------------
A real GGUF container is written by tools/tests/gguf_fixture.py, converted,
and the resulting blob is compared byte position by byte position against a
reference decoder implemented HERE, in llama.cpp's own algebraic form:

    Q4_0:  y[j] = (q - 8) * d
    Q4_1:  y[j] = q * d + m

The converter uses m + d*q with m = -8*d for Q4_0. The two forms are
mathematically equal but written differently on purpose, so a shared
misunderstanding of the format cannot pass both.

Not covered
-----------
Big-endian hosts. The byteswap paths in _stream_* and emit() only execute
when sys.byteorder != "little", and faking that flag on a little-endian host
would exercise the swap against data of the wrong endianness, which proves
nothing. Those paths remain unverified.
"""
import gc
import json
import math
import os
import struct
import sys
import tempfile
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # tools/
sys.path.insert(0, _HERE)                   # tools/tests/

import convert_gguf_to_niyah as conv  # noqa: E402
import gguf_fixture as fx             # noqa: E402

FAILURES = []
CHECKS = [0]


def check(condition, label):
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(label)
        print("  FAIL  %s" % label)
    return bool(condition)


def check_eq(got, want, label):
    return check(got == want, "%s (got %r, want %r)" % (label, got, want))


def close(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------- reference dequantiser


def ref_decode(payload, count, type_code):
    """Independent dequantiser. Returns a list of Python floats."""
    if type_code == fx.GGML_F32:
        return list(struct.unpack("<%df" % count, payload[:count * 4]))
    if type_code == fx.GGML_F16:
        return [float(x) for x in struct.unpack("<%de" % count,
                                                payload[:count * 2])]
    if type_code in (fx.GGML_Q4_0, fx.GGML_Q4_1):
        block_bytes = 18 if type_code == fx.GGML_Q4_0 else 20
        out = [0.0] * count
        for i in range(count // 32):
            off = i * block_bytes
            d = float(struct.unpack_from("<e", payload, off)[0])
            if type_code == fx.GGML_Q4_0:
                qs = off + 2
                for j in range(16):
                    b = payload[qs + j]
                    out[i * 32 + j] = ((b & 0x0F) - 8) * d
                    out[i * 32 + 16 + j] = ((b >> 4) - 8) * d
            else:
                m = float(struct.unpack_from("<e", payload, off + 2)[0])
                qs = off + 4
                for j in range(16):
                    b = payload[qs + j]
                    out[i * 32 + j] = (b & 0x0F) * d + m
                    out[i * 32 + 16 + j] = (b >> 4) * d + m
        return out
    raise ValueError("reference decoder has no path for type %d" % type_code)


def expected_floats(writer, order):
    values = []
    for name in order:
        t = writer.tensor(name)
        values.extend(ref_decode(t["payload"], t["count"], t["type"]))
    return values


def read_blob(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 4:
        raise AssertionError("blob length %d is not a multiple of 4" % len(raw))
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


# ------------------------------------------------------------------ the tests


def test_layout_and_values(tmp):
    print("test_layout_and_values")
    writer, order = fx.build_model()
    gguf = writer.write(os.path.join(tmp, "tiny.gguf"))
    blob = os.path.join(tmp, "tiny.bin")
    cfg_path = os.path.join(tmp, "tiny.json")

    config, written = conv.convert(gguf, blob, cfg_path)

    want = expected_floats(writer, order)
    check_eq(written, len(want), "written float count")
    got = read_blob(blob)
    check_eq(len(got), len(want), "blob float count")

    if len(got) == len(want):
        bad = [i for i in range(len(want)) if not close(got[i], want[i])]
        check(not bad, "all %d values match reference (%d mismatches, first "
                       "at %s)" % (len(want), len(bad), bad[:3]))

    # Slot 5 of each layer must be ffn_norm, not a second attn_norm.
    dim = fx.TINY["n_embd"]
    cursor = fx.TINY["n_vocab"] * dim
    per_layer = [dim, dim * dim, 16 * dim, 16 * dim, dim * dim,
                 dim, fx.TINY["n_ff"] * dim, fx.TINY["n_ff"] * dim,
                 dim * fx.TINY["n_ff"]]
    for layer in range(fx.TINY["n_layers"]):
        at = cursor + sum(per_layer[:5])
        ffn = writer.tensor("blk.%d.ffn_norm.weight" % layer)
        attn = writer.tensor("blk.%d.attn_norm.weight" % layer)
        want_ffn = ref_decode(ffn["payload"], ffn["count"], ffn["type"])
        want_attn = ref_decode(attn["payload"], attn["count"], attn["type"])
        slot = got[at:at + dim]
        check(all(close(slot[i], want_ffn[i]) for i in range(dim)),
              "layer %d slot 5 is ffn_norm" % layer)
        check(slot != want_attn, "layer %d slot 5 is not attn_norm" % layer)
        cursor += sum(per_layer)

    # Q4_0 nibble halves must land in the right halves of each block.
    emb = writer.tensor("token_embd.weight")
    if emb["type"] == fx.GGML_Q4_0:
        d = float(struct.unpack_from("<e", emb["payload"], 0)[0])
        first_byte = emb["payload"][2]
        check(close(got[0], ((first_byte & 0x0F) - 8) * d),
              "Q4_0 element 0 is the low nibble of qs[0]")
        check(close(got[16], ((first_byte >> 4) - 8) * d),
              "Q4_0 element 16 is the high nibble of qs[0]")

    check_eq(config["n_vocab"], fx.TINY["n_vocab"], "config n_vocab")
    check_eq(config["n_dim"], fx.TINY["n_embd"], "config n_dim")
    check_eq(config["n_heads"], fx.TINY["n_heads"], "config n_heads")
    check_eq(config["n_kv_heads"], fx.TINY["n_kv_heads"], "config n_kv_heads")
    check_eq(config["n_layers"], fx.TINY["n_layers"], "config n_layers")
    check_eq(config["n_ff"], fx.TINY["n_ff"], "config n_ff")
    check_eq(config["n_ctx"], fx.TINY["n_ctx"], "config n_ctx")
    check_eq(config["eos_token"], fx.TINY["eos_token"], "config eos_token")
    check_eq(config["tie_word_embeddings"], False, "config tie")
    check(close(config["rope_theta"], 10000.0), "config rope_theta")
    check(close(config["norm_eps"], 1e-5, tol=1e-3), "config norm_eps")
    return cfg_path


def test_config_json(cfg_path):
    print("test_config_json")
    with open(cf