#!/usr/bin/env python3
"""End-to-end validation of tools/convert_gguf_to_niyah.py.

    python3 tools/tests/test_convert_gguf.py

A real GGUF container is written by tools/tests/gguf_fixture.py, converted,
and the blob is compared element by element against a reference decoder
implemented here in llama.cpp's own algebraic form:

    Q4_0:  y = (q - 8) * d
    Q4_1:  y = q * d + m

The converter computes m + d*q with m = -8*d for Q4_0. Equal in value,
deliberately different in expression, so one shared misunderstanding of the
format cannot satisfy both.

Not covered: big-endian hosts. The byteswap branches in _stream_* and emit()
run only when sys.byteorder != "little", and forcing that flag on a
little-endian host would byteswap correctly-ordered data and prove nothing.
"""
import gc
import json
import os
import struct
import sys
import tempfile
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import convert_gguf_to_niyah as conv  # noqa: E402
import gguf_fixture as fx             # noqa: E402

FAILURES = []
TOTAL = [0]


def check(cond, label):
    TOTAL[0] += 1
    if not cond:
        FAILURES.append(label)
        print("    FAIL  " + label)
    return bool(cond)


def check_eq(got, want, label):
    return check(got == want, "%s (got %r, want %r)" % (label, got, want))


def close(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------- reference dequantiser


def ref_decode(payload, count, type_code):
    if type_code == fx.GGML_F32:
        return list(struct.unpack("<%df" % count, payload[:count * 4]))
    if type_code == fx.GGML_F16:
        return [float(x) for x in
                struct.unpack("<%de" % count, payload[:count * 2])]
    if type_code in (fx.GGML_Q4_0, fx.GGML_Q4_1):
        stride = 18 if type_code == fx.GGML_Q4_0 else 20
        out = [0.0] * count
        for i in range(count // 32):
            off = i * stride
            d = float(struct.unpack_from("<e", payload, off)[0])
            if type_code == fx.GGML_Q4_0:
                m = None
                qs = off + 2
            else:
                m = float(struct.unpack_from("<e", payload, off + 2)[0])
                qs = off + 4
            for j in range(16):
                b = payload[qs + j]
                lo = b & 0x0F
                hi = b >> 4
                if m is None:
                    out[i * 32 + j] = (lo - 8) * d
                    out[i * 32 + 16 + j] = (hi - 8) * d
                else:
                    out[i * 32 + j] = lo * d + m
                    out[i * 32 + 16 + j] = hi * d + m
        return out
    raise ValueError("no reference path for ggml type %d" % type_code)


def expected_floats(writer, order):
    values = []
    for name in order:
        t = writer.tensor(name)
        values.extend(ref_decode(t["payload"], t["count"], t["type"]))
    return values


def read_blob(path):
    with open(path, "rb") as f:
        raw = f.read()
    check_eq(len(raw) % 4, 0, "blob length is a multiple of 4")
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


# --------------------------------------------------------------------- checks


def test_layout_and_values(tmp):
    print("  layout_and_values")
    writer, order = fx.build_model()
    gguf = writer.write(os.path.join(tmp, "tiny.gguf"))
    blob = os.path.join(tmp, "tiny.bin")
    cfg_path = os.path.join(tmp, "tiny.json")

    config, written = conv.convert(gguf, blob, cfg_path)
    want = expected_floats(writer, order)
    got = read_blob(blob)

    check_eq(written, len(want), "reported float count")
    check_eq(len(got), len(want), "blob float count")
    if len(got) == len(want):
        bad = [i for i in range(len(want)) if not close(got[i], want[i])]
        check(not bad, "all %d values match reference (%d bad, first %s)"
                       % (len(want), len(bad), bad[:3]))

    check_eq(config["n_vocab"], fx.TINY["n_vocab"], "n_vocab")
    check_eq(config["n_dim"], fx.TINY["n_embd"], "n_dim")
    check_eq(config["n_heads"], fx.TINY["n_heads"], "n_heads")
    check_eq(config["n_kv_heads"], fx.TINY["n_kv_heads"], "n_kv_heads")
    check_eq(config["n_layers"], fx.TINY["n_layers"], "n_layers")
    check_eq(config["n_ff"], fx.TINY["n_ff"], "n_ff")
    check_eq(config["n_ctx"], fx.TINY["n_ctx"], "n_ctx")
    check_eq(config["eos_token"], fx.TINY["eos_token"], "eos_token")
    check_eq(config["tie_word_embeddings"], False, "tie flag")
    check(close(config["rope_theta"], 10000.0), "rope_theta")
    check(close(config["norm_eps"], 1e-5, tol=1e-3), "norm_eps")
    return writer, order, got, cfg_path


def layer_slot_sizes():
    dim = fx.TINY["n_embd"]
    ff = fx.TINY["n_ff"]
    kv_dim = fx.TINY["n_kv_heads"] * (dim // fx.TINY["n_heads"])
    return [dim, dim * dim, kv_dim * dim, kv_dim * dim, dim * dim,
            dim, ff * dim, ff * dim, dim * ff]


def test_ffn_norm_slot(writer, got):
    print("  ffn_norm_slot")
    dim = fx.TINY["n_embd"]
    sizes = layer_slot_sizes()
    cursor = fx.TINY["n_vocab"] * dim
    for layer in range(fx.TINY["n_layers"]):
        at = cursor + sum(sizes[:5])
        slot = got[at:at + dim]
        ffn = writer.tensor("blk.%d.ffn_norm.weight" % layer)
        attn = writer.tensor("blk.%d.attn_norm.weight" % layer)
        want_ffn = ref_decode(ffn["payload"], ffn["count"], ffn["type"])
        want_attn = ref_decode(attn["payload"], attn["count"], attn["type"])
        check(len(slot) == dim and all(close(slot[i], want_ffn[i])
                                      for i in range(dim)),
              "layer %d slot 5 holds ffn_norm" % layer)
        check(not all(close(slot[i], want_attn[i]) for i in range(dim)),
              "layer %d slot 5 is not a second attn_norm" % layer)
        cursor += sum(sizes)


def test_q4_0_halves(writer, got):
    print("  q4_0_halves")
    emb = writer.tensor("token_embd.weight")
    if emb["type"] != fx.GGML_Q4_0:
        check(False, "fixture embedding is Q4_0")
        return
    d = float(struct.unpack_from("<e", emb["payload"], 0)[0])
    b0 = emb["payload"][2]
    check(close(got[0], ((b0 & 0x0F) - 8) * d),
          "element 0 is the low nibble of qs[0]")
    check(close(got[16], ((b0 >> 4) - 8) * d),
          "element 16 is the high nibble of qs[0]")
    check(not close(got[1], ((b0 >> 4) - 8) * d) or (b0 & 0x0F) == (b0 >> 4),
          "element 1 is not the high nibble of qs[0]")


def c_json_find_int(text, key):
    """Reimplementation of json_find_int's quoted-key strstr scan."""
    i = text.find('"' + key + '"')
    if i < 0:
        return None
    j = i + len(key) + 2
    while j < len(text) and text[j] in " \t:":
        j += 1
    k = j
    if k < len(text) and text[k] in "+-":
        k += 1
    while k < len(text) and text[k].isdigit():
        k += 1
    if k == j:
        return None
    return int(text[j:k])


def test_config_json(cfg_path):
    print("  config_json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        check(False, "emitted config is valid JSON (%s)" % exc)
        return
    check(True, "emitted config is valid JSON")

    for out_key, _src in conv.CONFIG_KEYS:
        check(out_key in parsed, "config contains %s" % out_key)

    # The dual-schema safety argument, checked against the real text.
    wanted = {
        "vocab_size": fx.TINY["n_vocab"],
        "dim": fx.TINY["n_embd"],
        "heads": fx.TINY["n_heads"],
        "kv_heads": fx.TINY["n_kv_heads"],
        "layer_count": fx.TINY["n_layers"],
        "context_size": fx.TINY["n_ctx"],
        "hidden_dim": fx.TINY["n_ff"],
        "eos_token": fx.TINY["eos_token"],
    }
    for key, value in wanted.items():
        check_eq(c_json_find_int(text, key), value,
                 'json_find_int("%s") resolves without collision' % key)


def test_chunk_windows(tmp, reference):
    print("  chunk_windows")
    writer, order = fx.build_model()
    gguf = writer.write(os.path.join(tmp, "chunk.gguf"))
    saved = conv.OUT_CHUNK_FLOATS
    try:
        for window in (1, 17, 31, 32, 33, 64, 100, 4096, 1 << 18):
            conv.OUT_CHUNK_FLOATS = window
            blob = os.path.join(tmp, "chunk_%d.bin" % window)
            conv.convert(gguf, blob, None)
            got = read_blob(blob)
            same = (len(got) == len(reference)
                    and all(got[i] == reference[i] for i in range(len(got))))
            check(same, "OUT_CHUNK_FLOATS=%d gives an identical blob" % window)
            os.remove(blob)
    finally:
        conv.OUT_CHUNK_FLOATS = saved


def test_tied_embeddings(tmp):
    print("  tied_embeddings")
    writer, order = fx.build_model(tie=True)
    gguf = writer.write(os.path.join(tmp, "tied.gguf"))
    blob = os.path.join(tmp, "tied.bin")
    config, written = conv.convert(gguf, blob, None)
    check_eq(config["tie_word_embeddings"], True, "tie flag read from metadata")
    check_eq(written, len(expected_floats(writer, order)), "tied float count")
    check("output.weight" not in order, "lm_head omitted from tied order")

    untied_writer, untied_order = fx.build_model(tie=False)
    delta = (len(expected_floats(untied_writer, untied_order))
             - len(expected_floats(writer, order)))
    check_eq(delta, fx.TINY["n_vocab"] * fx.TINY["n_embd"],
             "tied blob is exactly one embedding shorter")


def test_tie_inferred(tmp):
    print("  tie_inferred")
    writer, order = fx.build_model(tie=True, declare_tie=False)
    gguf = writer.write(os.path.join(tmp, "inferred.gguf"))
    blob = os.path.join(tmp, "inferred.bin")
    try:
        config, written = conv.convert(gguf, blob, None)
    except RuntimeError as exc:
        check(False, "tie inferred from missing output.weight (%s)" % exc)
        return
    check_eq(config["tie_word_embeddings"], True,
             "tie inferred from missing output.weight")
    check_eq(written, len(expected_floats(writer, order)),
             "inferred-tie float count")


def test_metadata_types(tmp):
    print("  metadata_types")
    writer, _order = fx.build_model()
    gguf = writer.write(os.path.join(tmp, "meta.gguf"))
    metadata, infos = conv.parse_gguf(gguf)

    check_eq(metadata.get("general.architecture"), "qwen2", "STRING value")
    check_eq(metadata.get("general.name"), "niyah-tiny-fixture", "STRING value 2")
    check_eq(metadata.get("general.alignment"), 32, "UINT32 value")
    check_eq(metadata.get("test.uint8"), 200, "UINT8 value")
    check_eq(metadata.get("test.int8"), -5, "INT8 value")
    check_eq(metadata.get("test.uint16"), 40000, "UINT16 value")
    check_eq(metadata.get("test.int16"), -300, "INT16 value")
    check_eq(metadata.get("test.int32"), -70000, "INT32 value")
    check_eq(metadata.get("test.uint64"), 2 ** 40 + 7, "UINT64 value")
    check_eq(metadata.get("test.int64"), -(2 ** 40) - 7, "INT64 value")
    check(close(metadata.get("test.float64", 0.0), 0.1), "FLOAT64 value")
    check_eq(metadata.get("test.bool_false"), False, "BOOL false")
    check_eq(metadata.get("qwen2.tie_word_embeddings"), False, "BOOL value")
    check(close(metadata.get("qwen2.rope.freq_base", 0.0), 10000.0),
          "FLOAT32 value")
    check_eq(metadata.get("test.small_array"), [-1, 0, 1],
             "short ARRAY materialised")

    tokens = metadata.get("tokenizer.ggml.tokens")
    check(isinstance(tokens, conv.SkippedArray) and tokens.count == 200,
          "long string ARRAY skipped, not materialised")
    scores = metadata.get("tokenizer.ggml.scores")
    check(isinstance(scores, conv.SkippedArray) and scores.count == 200,
          "long float ARRAY skipped, not materialised")

    # If any skip had gone wrong the tensor table would be unreadable.
    names = set(x["name"] for x in infos)
    check("token_embd.weight" in names and "output.weight" in names,
          "tensor table still parses after array skipping")
    check_eq(len(infos), 2 + 9 * fx.TINY["n_layers"], "tensor count")


def test_unsupported_type(tmp):
    print("  unsupported_type")
    types = dict(fx.DEFAULT_TYPES)
    types["token_embd"] = fx.GGML_Q4_K
    writer, _order = fx.build_model(types=types)
    gguf = writer.write(os.path.join(tmp, "kquant.gguf"))
    blob = os.path.join(tmp, "kquant.bin")

    message = None
    try:
        conv.convert(gguf, blob, None)
    except RuntimeError as exc:
        message = str(exc)
    check(message is not None, "Q4_K checkpoint is rejected")
    if message:
        check("Q4_K" in message, "error names Q4_K (%s)" % message)
        check("llama-quantize" in message, "error suggests llama-quantize")
        check("outside GGUF data region" not in message,
              "rejection is the clean type error, not an offset error")
    check(not os.path.exists(blob), "no output file left behind")


BIG = {
    "n_vocab": 1024,
    "n_embd": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 1,
    "n_ff": 128,
    "n_ctx": 128,
    "eos_token": 2,
    "rope_theta": 10000.0,
    "norm_eps": 1.0000000474974513e-05,
}


def test_bounded_memory(tmp):
    print("  bounded_memory")
    writer, order = fx.build_model(cfg=BIG, big_vocab_array=False)
    gguf = writer.write(os.path.join(tmp, "big.gguf"))
    total = sum(writer.tensor(n)["count"] for n in order)
    del writer, order
    gc.collect()

    saved = conv.OUT_CHUNK_FLOATS
    conv.OUT_CHUNK_FLOATS = 4096
    tracemalloc.start()
    try:
        _config, written = conv.convert(gguf, os.path.join(tmp, "big.bin"), None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        conv.OUT_CHUNK_FLOATS = saved

    check_eq(written, total, "big model float count")
    limit = 2 * 1024 * 1024
    check(peak < limit,
          "peak allocation %.1f KiB is under %d KiB for a %d-float model"
          % (peak / 1024.0, limit // 1024, total))
    print("      peak=%.1f KiB, floats=%d, blob=%.1f KiB"
          % (peak / 1024.0, total, total * 4 / 1024.0))


def main():
    print("convert_gguf_to_niyah end-to-end validation")
    print("  python %s, byteorder=%s"
          % (sys.version.split()[0], sys.byteorder))
    if sys.byteorder != "little":
        print("  NOTE: big-endian host, the swap paths are exercised but the"
              " reference decoder assumes little-endian payloads")

    with tempfile.TemporaryDirectory() as tmp:
        writer, order, blob_values, cfg_path = test_layout_and_values(tmp)
        test_ffn_norm_slot(writer, blob_values)
        test_q4_0_halves(writer, blob_values)
        test_config_json(cfg_path)
        test_chunk_windows(tmp, blob_values)
        test_tied_embeddings(tmp)
        test_tie_inferred(tmp)
        test_metadata_types(tmp)
        test_unsupported_type(tmp)
        test_bounded_memory(tmp)

    print("")
    if FAILURES:
        print("FAILED %d of %d checks:" % (len(FAILURES), TOTAL[0]))
        for label in FAILURES:
            print("  - " + label)
        return 1
    print("ok: %d checks passed" % TOTAL[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
