#!/usr/bin/env python3
"""Validate the converter's Q4_K and Q6_K decoders.

    python3 tools/tests/test_kquants.py

The reference decoders below are written as per-element index arithmetic:

    Q4_K   element i -> (outer = i//64, half = (i%64)//32, lane = i%32)
    Q6_K   element i -> (n = i//128, quarter = (i%128)//32, lane = i%32)

llama.cpp -- and the converter, which follows it -- instead walks grouped
pointers, advancing qs by 32 and the scale index by 2 per 64 outputs for
Q4_K, and ql/qh/scales by 64/32/8 per 128 outputs for Q6_K. Two different
formulations reaching the same numbers is meaningful evidence; one
transcription is unlikely to mirror a slip in the other.

What this does NOT prove: that the numbers are the ones a real checkpoint
holds. Fixture scales are synthetic. The remaining risk is retired only by
converting a real Qwen2.5 q4_k_m file and getting coherent generation out.
"""
import gc
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


def f16(buf, off):
    return float(struct.unpack_from("<e", buf, off)[0])


def i8(value):
    return value - 256 if value > 127 else value


# ------------------------------------------------------ reference decoders


def scale_min_k4(scales, j):
    """Unpack the j-th 6-bit (scale, min) pair from block_q4_K.scales[12].

    j in 0..7. The first four pairs are plain 6-bit fields; the last four
    borrow their two high bits from the top of earlier bytes.
    """
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    low_d = scales[j + 4] & 0x0F
    low_m = scales[j + 4] >> 4
    return (low_d | ((scales[j - 4] >> 6) << 4),
            low_m | ((scales[j] >> 6) << 4))


def ref_q4_k(payload, count):
    out = [0.0] * count
    for block in range(count // 256):
        off = block * 144
        d = f16(payload, off)
        dmin = f16(payload, off + 2)
        scales = payload[off + 4:off + 16]
        qs = off + 16
        for i in range(256):
            outer = i // 64
            half = (i % 64) // 32
            lane = i % 32
            sc, m = scale_min_k4(scales, outer * 2 + half)
            byte = payload[qs + outer * 32 + lane]
            nib = (byte & 0x0F) if half == 0 else (byte >> 4)
            out[block * 256 + i] = d * sc * nib - dmin * m
    return out


def ref_q6_k(payload, count):
    out = [0.0] * count
    for block in range(count // 256):
        off = block * 210
        ql = off
        qh = off + 128
        sc = off + 192
        d = f16(payload, off + 208)
        for i in range(256):
            n = i // 128
            quarter = (i % 128) // 32
            lane = i % 32
            high = payload[qh + n * 32 + lane]
            shift = quarter * 2
            top = ((high >> shift) & 3) << 4
            if quarter == 0:
                low = payload[ql + n * 64 + lane] & 0x0F
            elif quarter == 1:
                low = payload[ql + n * 64 + lane + 32] & 0x0F
            elif quarter == 2:
                low = payload[ql + n * 64 + lane] >> 4
            else:
                low = payload[ql + n * 64 + lane + 32] >> 4
            scale = i8(payload[sc + n * 8 + (lane // 16) + 2 * quarter])
            out[block * 256 + i] = d * scale * ((low | top) - 32)
    return out


def ref_decode(payload, count, type_code):
    if type_code == fx.GGML_F32:
        return list(struct.unpack("<%df" % count, payload[:count * 4]))
    if type_code == fx.GGML_Q4_K:
        return ref_q4_k(payload, count)
    if type_code == fx.GGML_Q6_K:
        return ref_q6_k(payload, count)
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
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


# ---------------------------------------------------------------- the checks


def test_kquant_model(tmp):
    print("  kquant_model")
    writer, order = fx.build_model(types=fx.KQUANT_TYPES)
    gguf = writer.write(os.path.join(tmp, "kq.gguf"))
    blob = os.path.join(tmp, "kq.bin")

    try:
        _config, written = conv.convert(gguf, blob, None)
    except RuntimeError as exc:
        check(False, "Q4_K/Q6_K model converts (%s)" % exc)
        return None

    want = expected_floats(writer, order)
    got = read_blob(blob)
    check_eq(written, len(want), "reported float count")
    check_eq(len(got), len(want), "blob float count")
    if len(got) == len(want):
        bad = [i for i in range(len(want)) if not close(got[i], want[i])]
        check(not bad, "all %d values match reference (%d bad, first %s)"
                       % (len(want), len(bad), bad[:3]))
    return got


def test_q4_k_group_scales(tmp):
    """Each 32-element group must use its own 6-bit scale pair.

    Sixty-four consecutive outputs sharing one scale would still look
    plausible, so this compares two groups' implied scale ratios against the
    packed bytes directly.
    """
    print("  q4_k_group_scales")
    name = "token_embd.weight"
    writer, _order = fx.build_model(types=fx.KQUANT_TYPES)
    t = writer.tensor(name)
    payload = t["payload"]
    scales = payload[4:16]

    pairs = [scale_min_k4(scales, j) for j in range(8)]
    distinct = len(set(pairs))
    check(distinct >= 4,
          "fixture block 0 has %d distinct scale pairs to distinguish" % distinct)

    gguf = writer.write(os.path.join(tmp, "groups.gguf"))
    blob = os.path.join(tmp, "groups.bin")
    conv.convert(gguf, blob, None)
    got = read_blob(blob)

    d = f16(payload, 0)
    dmin = f16(payload, 2)
    # Element 0 is lane 0 of group 0; element 32 is lane 0 of group 1. They
    # read the SAME qs byte -- low nibble both times -- and differ only by
    # which scale pair applies.
    byte0 = payload[16]
    nib = byte0 & 0x0F
    sc0, m0 = pairs[0]
    sc1, m1 = pairs[1]
    check(close(got[0], d * sc0 * nib - dmin * m0),
          "element 0 uses scale pair 0")
    check(close(got[32], d * sc1 * nib - dmin * m1),
          "element 32 uses scale pair 1, not pair 0")
    if sc0 != sc1 or m0 != m1:
        check(not close(got[32], d * sc0 * nib - dmin * m0),
              "element 32 is not a repeat of pair 0")


def test_q6_k_quarters(tmp):
    """Q6_K interleaves four quarters 32 elements apart out of one qh byte."""
    print("  q6_k_quarters")
    writer, _order = fx.build_model(types=fx.KQUANT_TYPES)
    name = "blk.0.attn_k.weight"
    t = writer.tensor(name)
    check_eq(t["type"], fx.GGML_Q6_K, "fixture attn_k is Q6_K")
    payload = t["payload"]

    gguf = writer.write(os.path.join(tmp, "q6k.gguf"))
    blob = os.path.join(tmp, "q6k.bin")
    conv.convert(gguf, blob, None)
    got = read_blob(blob)

    # Locate this tensor in the blob.
    _w2, order = fx.build_model(types=fx.KQUANT_TYPES)
    base = 0
    for other in order:
        if other == name:
            break
        base += writer.tensor(other)["count"]

    d = f16(payload, 208)
    high = payload[128]           # qh[0]
    for quarter in range(4):
        top = ((high >> (quarter * 2)) & 3) << 4
        if quarter == 0:
            low = payload[0] & 0x0F
        elif quarter == 1:
            low = payload[32] & 0x0F
        elif quarter == 2:
            low = payload[0] >> 4
        else:
            low = payload[32] >> 4
        scale = i8(payload[192 + 2 * quarter])
        want = d * scale * ((low | top) - 32)
        check(close(got[base + quarter * 32], want),
              "quarter %d lands at element %d" % (quarter, quarter * 32))


def test_kquant_chunk_windows(tmp, reference):
    """A K-quant superblock is 256 floats; windows below that must still work."""
    print("  kquant_chunk_windows")
    writer, _order = fx.build_model(types=fx.KQUANT_TYPES)
    gguf = writer.write(os.path.join(tmp, "kqchunk.gguf"))
    saved = conv.OUT_CHUNK_FLOATS
    try:
        for window in (1, 255, 256, 257, 512, 1000, 1 << 18):
            conv.OUT_CHUNK_FLOATS = window
            blob = os.path.join(tmp, "kqchunk_%d.bin" % window)
            conv.convert(gguf, blob, None)
            got = read_blob(blob)
            same = (len(got) == len(reference)
                    and all(got[i] == reference[i] for i in range(len(got))))
            check(same, "OUT_CHUNK_FLOATS=%d gives an identical blob" % window)
            os.remove(blob)
    finally:
        conv.OUT_CHUNK_FLOATS = saved


def test_q5_k_still_rejected(tmp):
    """Supporting Q4_K and Q6_K must not silently accept every K-quant."""
    print("  q5_k_still_rejected")
    types = dict(fx.KQUANT_TYPES)
    types["token_embd"] = fx.GGML_Q5_K
    writer, _order = fx.build_model(types=types)
    gguf = writer.write(os.path.join(tmp, "q5k.gguf"))
    blob = os.path.join(tmp, "q5k.bin")

    message = None
    try:
        conv.convert(gguf, blob, None)
    except RuntimeError as exc:
        message = str(exc)
    check(message is not None, "Q5_K is still rejected")
    if message:
        check("Q5_K" in message, "error names Q5_K (%s)" % message)
        check("outside GGUF data region" not in message,
              "rejection is the type error, not an offset error")
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


def test_kquant_bounded_memory(tmp):
    print("  kquant_bounded_memory")
    writer, order = fx.build_model(cfg=BIG, types=fx.KQUANT_TYPES,
                                   big_vocab_array=False)
    gguf = writer.write(os.path.join(tmp, "kqbig.gguf"))
    total = sum(writer.tensor(n)["count"] for n in order)
    del writer, order
    gc.collect()

    saved = conv.OUT_CHUNK_FLOATS
    conv.OUT_CHUNK_FLOATS = 4096
    tracemalloc.start()
    try:
        _config, written = conv.convert(gguf, os.path.join(tmp, "kqbig.bin"), None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        conv.OUT_CHUNK_FLOATS = saved

    check_eq(written, total, "big K-quant float count")
    limit = 2 * 1024 * 1024
    check(peak < limit,
          "peak allocation %.1f KiB is under %d KiB for a %d-float model"
          % (peak / 1024.0, limit // 1024, total))
    print("      peak=%.1f KiB, floats=%d" % (peak / 1024.0, total))


def main():
    print("Q4_K / Q6_K decoder validation")
    print("  python %s, byteorder=%s"
          % (sys.version.split()[0], sys.byteorder))

    with tempfile.TemporaryDirectory() as tmp:
        blob_values = test_kquant_model(tmp)
        if blob_values is None:
            print("")
            print("FAILED: conversion did not produce a blob; "
                  "remaining checks skipped")
            return 1
        test_q4_k_group_scales(tmp)
        test_q6_k_quarters(tmp)
        test_kquant_chunk_windows(tmp, blob_values)
        test_q5_k_still_rejected(tmp)
        test_kquant_bounded_memory(tmp)

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
