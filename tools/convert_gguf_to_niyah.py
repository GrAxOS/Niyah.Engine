#!/usr/bin/env python3
"""Convert a GGUF checkpoint to the flat float32 blob Niyah/NiyahMini expects.

GGUF is the GGML Universal File format used by llama.cpp -- it is not tied to
any single model family.

Stdlib only. No NumPy.

Memory
------
Tensors are decoded in bounded windows and flushed straight to disk, so peak
resident memory is a few MiB no matter how large the checkpoint is. Nothing
ever holds a whole tensor.

Output layout (matches the comment block in native/niyah.h)
-----------------------------------------------------------
    embedding
    for each layer:
        attn_norm, wq, wk, wv, wo, ffn_norm, ffn_gate, ffn_up, ffn_down
    final_norm
    lm_head            (omitted when tie_word_embeddings is true)

All projection matrices stay row-major [out_features][in_features].

Config schema
-------------
Two different C readers consume this JSON and they expect different key
names, so both sets are emitted:

    native/niyah_model.c          vocab_size, dim, heads, layer_count,
                                  context_size, kv_heads, hidden_dim,
                                  eos_token
    native/niyah_mini/..._model.c n_vocab, n_dim, n_heads, n_layers,
                                  n_ctx, n_kv_heads, n_ff, rope_theta,
                                  norm_eps, tie_word_embeddings

This is safe with the substring scanners both readers use, because the
search pattern includes the opening quote: "dim" cannot match inside
"n_dim" or "hidden_dim".
"""
import argparse
import array
import math
import struct
import sys

MAGIC = 0x46554747

GGML_TYPES = {
    0: ("F32", 4),
    1: ("F16", 2),
    2: ("Q4_0", 18),
    3: ("Q4_1", 20),
    6: ("Q5_0", 22),
    7: ("Q5_1", 24),
    8: ("Q8_0", 34),
    9: ("Q8_1", 36),
    10: ("Q2_K", 84),
    11: ("Q3_K", 110),
    12: ("Q4_K", 144),
    13: ("Q5_K", 176),
    14: ("Q6_K", 210),
    15: ("Q8_K", 292),
}

# Types this converter can actually decode. The rest of GGML_TYPES is known
# well enough to compute byte sizes (so offsets can be validated and a precise
# error produced) but cannot be dequantised yet.
SUPPORTED_TYPES = (0, 1, 2, 3)

QK = 32  # GGML block size for the Q4_* families

# Decode window. 262144 floats is 1 MiB of array('f'), and the largest raw
# read backing one window is also about 1 MiB, so peak overhead stays low.
OUT_CHUNK_FLOATS = 1 << 18

_LITTLE = sys.byteorder == "little"


def fail(message):
    raise RuntimeError(message)


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        fail("unexpected end of GGUF file")
    return data


# ---------------------------------------------------------------- primitives


def u8(f):
    return read_exact(f, 1)[0]


def i8(f):
    return struct.unpack("<b", read_exact(f, 1))[0]


def u16(f):
    return struct.unpack("<H", read_exact(f, 2))[0]


def i16(f):
    return struct.unpack("<h", read_exact(f, 2))[0]


def u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def i32(f):
    return struct.unpack("<i", read_exact(f, 4))[0]


def u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]


def i64(f):
    return struct.unpack("<q", read_exact(f, 8))[0]


def f32(f):
    return struct.unpack("<f", read_exact(f, 4))[0]


def f64(f):
    return struct.unpack("<d", read_exact(f, 8))[0]


def string(f):
    n = u64(f)
    if n > sys.maxsize:
        fail("GGUF string is too large")
    raw = read_exact(f, n)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def scalar_value(f, value_type):
    """Read one GGUF metadata value.

    INT8 and INT16 use struct formats 'b' and 'h'. They previously used
    "<i8" and "<i16", which are not valid struct formats at all -- struct
    reads a leading integer as a repeat count, so '8' and '16' were counts
    with no type character after them, raising struct.error on any GGUF that
    carried an int8 or int16 metadata value.
    """
    if value_type == 0:
        return u8(f)
    if value_type == 1:
        return i8(f)
    if value_type == 2:
        return u16(f)
    if value_type == 3:
        return i16(f)
    if value_type == 4:
        return u32(f)
    if value_type == 5:
        return i32(f)
    if value_type == 6:
        return u64(f)
    if value_type == 7:
        return i64(f)
    if value_type == 8:
        return f32(f)
    if value_type == 9:
        return f64(f)
    if value_type == 10:
        return u8(f) != 0
    if value_type == 11:
        return f"blob:{u64(f)}"
    if value_type == 12:
        return string(f)
    if value_type == 13:
        elem_type = u32(f)
        count = u64(f)
        if count > 1000000:
            fail("GGUF array is unreasonably large")
        return [scalar_value(f, elem_type) for _ in range(count)]
    fail(f"unsupported GGUF metadata type {value_type}")


def align_up(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def ggml_row_count(dims):
    total = 1
    for d in dims:
        total *= d
    return total


# ------------------------------------------------------------------ streaming


def _stream_f32(fh, count, name):
    """Yield array('f') windows for an F32 tensor."""
    remaining = count
    while remaining:
        n = min(remaining, OUT_CHUNK_FLOATS)
        chunk = array.array("f")
        chunk.frombytes(read_exact(fh, n * 4))
        if not _LITTLE:
            chunk.byteswap()
        if not all(map(math.isfinite, chunk)):
            fail(f"non-finite value found in tensor {name}")
        yield chunk
        remaining -= n


def _stream_f16(fh, count, name):
    """Yield array('f') windows for an F16 tensor."""
    remaining = count
    while remaining:
        n = min(remaining, OUT_CHUNK_FLOATS)
        values = struct.unpack(f"<{n}e", read_exact(fh, n * 2))
        if not all(map(math.isfinite, values)):
            fail(f"non-finite value found in tensor {name}")
        yield array.array("f", values)
        remaining -= n


def _stream_q4(fh, count, name, with_min):
    """Yield array('f') windows for a Q4_0 (with_min=False) or Q4_1 tensor.

    Element ordering follows llama.cpp's dequantize_row_q4_0/q4_1:

        y[i*qk + j]        <- low  nibble of qs[j]
        y[i*qk + j + qk/2] <- high nibble of qs[j]

    so the 16 low nibbles fill the FIRST half of each 32-element block and
    the 16 high nibbles fill the SECOND half. Writing (low, high) pairs
    consecutively instead -- as an earlier revision did -- yields a
    permutation of every block with identical byte count and shape, which
    nothing downstream can detect.

    Only the per-block scales are checked for finiteness. Output is
    m + d*q with q in [0, 15], so a finite scale guarantees finite output;
    that is count/32 checks rather than count.
    """
    block_bytes = 20 if with_min else 18
    if count % QK:
        fail(f"tensor {name} element count {count} is not divisible by {QK}")
    n_blocks = count // QK
    blocks_per_chunk = max(1, OUT_CHUNK_FLOATS // QK)
    half = QK // 2

    done = 0
    while done < n_blocks:
        nb = min(blocks_per_chunk, n_blocks - done)
        raw = read_exact(fh, nb * block_bytes)
        out = array.array("f", bytes(nb * QK * 4))

        off = 0
        base = 0
        for _ in range(nb):
            d = float(struct.unpack_from("<e", raw, off)[0])
            if with_min:
                m = float(struct.unpack_from("<e", raw, off + 2)[0])
                off += 4
            else:
                m = -8.0 * d
                off += 2
            if not (math.isfinite(d) and math.isfinite(m)):
                fail(f"non-finite scale found in tensor {name}")
            for j in range(half):
                byte = raw[off + j]
                out[base + j] = m + d * (byte & 0x0F)
                out[base + half + j] = m + d * (byte >> 4)
            off += half
            base += QK

        yield out
        done += nb


def stream_tensor(fh, info):
    """Yield array('f') windows for one tensor. fh must already be seeked."""
    type_code = info["type"]
    count = info["count"]
    name = info["name"]
    if type_code == 0:
        return _stream_f32(fh, count, name)
    if type_code == 1:
        return _stream_f16(fh, count, name)
    if type_code == 2:
        return _stream_q4(fh, count, name, with_min=False)
    if type_code == 3:
        return _stream_q4(fh, count, name, with_min=True)
    known = GGML_TYPES.get(type_code, (f"type{type_code}", 0))[0]
    fail(f"tensor type {known} is not implemented")


def emit(out, chunk):
    """Write one window as little-endian float32."""
    if not _LITTLE:
        chunk.byteswap()
    chunk.tofile(out)


# --------------------------------------------------------------------- parse


def parse_gguf(path):
    """Return (metadata, tensor_infos).

    Does not return the file handle: an earlier revision returned the `f`
    created by this function's own `with` block, which was already closed by
    the time the caller received it.
    """
    with open(path, "rb") as f:
        magic = u32(f)
        if magic != MAGIC:
            fail("not a GGUF file")
        version = u32(f)
        if version not in (2, 3):
            fail(f"unsupported GGUF version {version}")
        tensor_count = u64(f)
        metadata_count = u64(f)

        metadata = {}
        for _ in range(metadata_count):
            key = string(f)
            value_type = u32(f)
            metadata[key] = scalar_value(f, value_type)

        tensor_infos = []
        for _ in range(tensor_count):
            name = string(f)
            n_dims = u32(f)
            dims = [u64(f) for _ in range(n_dims)]
            type_code = u32(f)
            offset = u64(f)
            if type_code not in GGML_TYPES:
                fail(f"unsupported GGUF tensor type {type_code} for {name}")
            tensor_infos.append((name, dims, type_code, offset))

        alignment = metadata.get("general.alignment", 32)
        if (not isinstance(alignment, int) or alignment <= 0
                or alignment > (1 << 20) or alignment & (alignment - 1)):
            alignment = 32
        data_start = align_up(f.tell(), alignment)

        file_size = f.seek(0, 2)

        infos = []
        for index, (name, dims, type_code, rel_offset) in enumerate(tensor_infos):
            count = ggml_row_count(dims)
            info_name, block_bytes = GGML_TYPES[type_code]
            if type_code in (0, 1):
                byte_size = count * block_bytes
            else:
                if count % QK:
                    fail(f"tensor {name} element count {count} is invalid for {info_name}")
                byte_size = (count // QK) * block_bytes
            absolute = data_start + rel_offset
            if absolute < data_start or absolute + byte_size > file_size:
                fail(f"tensor {name} points outside GGUF data region")
            infos.append({
                "index": index,
                "name": name,
                "dims": dims,
                "count": count,
                "type": type_code,
                "type_name": info_name,
                "offset": absolute,
                "size": byte_size,
            })
        return metadata, infos


def first_metadata(metadata, keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def infer_config(metadata, infos):
    n_vocab = first_metadata(metadata, [
        "tokenizer.ggml.vocab_size",
        "qwen2.vocab_size",
        "llama.vocab_size",
    ])
    n_embd = first_metadata(metadata, [
        "qwen2.embedding_length",
        "llama.embedding_length",
    ])
    n_heads = first_metadata(metadata, [
        "qwen2.attention.head_count",
        "llama.attention.head_count",
    ])
    n_kv_heads = first_metadata(metadata, [
        "qwen2.attention.head_count_kv",
        "llama.attention.head_count_kv",
    ], n_heads)
    n_layers = first_metadata(metadata, [
        "qwen2.block_count",
        "llama.block_count",
    ])
    n_ctx = first_metadata(metadata, [
        "qwen2.context_length",
        "llama.context_length",
        "general.context_length",
    ], 2048)
    n_ff = first_metadata(metadata, [
        "qwen2.feed_forward_length",
        "llama.feed_forward_length",
    ])
    rope_theta = first_metadata(metadata, [
        "qwen2.rope.freq_base",
        "llama.rope.freq_base",
    ], 10000.0)
    norm_eps = first_metadata(metadata, [
        "qwen2.attention.layer_norm_rms_epsilon",
        "llama.attention.layer_norm_rms_epsilon",
    ], 1e-5)
    tie = first_metadata(metadata, [
        "qwen2.tie_word_embeddings",
        "llama.tie_word_embeddings",
    ], False)
    eos_token = first_metadata(metadata, [
        "tokenizer.ggml.eos_token_id",
    ], 0)

    shapes = {x["name"]: x["dims"] for x in infos}
    embedding_shape = (shapes.get("token_embd.weight")
                       or shapes.get("model.embed_tokens.weight"))
    if n_vocab is None and embedding_shape:
        n_vocab = embedding_shape[0]
    if n_embd is None and embedding_shape:
        n_embd = embedding_shape[1] if len(embedding_shape) > 1 else embedding_shape[0]
    if n_layers is None:
        layer_ids = []
        for info in infos:
            parts = info["name"].split(".")
            for marker in ("blk", "layers"):
                if marker in parts:
                    try:
                        layer_ids.append(int(parts[parts.index(marker) + 1]))
                    except (ValueError, IndexError):
                        pass
                    break
        if layer_ids:
            n_layers = max(layer_ids) + 1

    if any(x is None for x in (n_vocab, n_embd, n_heads, n_layers, n_ff)):
        fail("could not infer complete NiyahMini configuration from GGUF metadata")

    n_heads = int(n_heads)
    n_embd = int(n_embd)
    if n_heads <= 0:
        fail("head_count must be positive")
    if n_embd % n_heads:
        fail(f"embedding_length {n_embd} is not divisible by head_count {n_heads}")

    return {
        "n_layers": int(n_layers),
        "n_dim": n_embd,
        "n_heads": n_heads,
        "n_kv_heads": int(n_kv_heads),
        "n_ff": int(n_ff),
        "n_vocab": int(n_vocab),
        "n_ctx": int(n_ctx),
        "rope_theta": float(rope_theta),
        "norm_eps": float(norm_eps),
        "tie_word_embeddings": bool(tie),
        "eos_token": int(eos_token),
    }


def resolve_tensor(infos, candidates):
    by_name = {x["name"]: x for x in infos}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    fail("missing tensor: " + " | ".join(candidates))


def layer_tensor(infos, layer, candidates):
    """Resolve a per-layer tensor.

    Candidate order matters: resolve_tensor returns the FIRST match, so the
    canonical GGUF name must always be listed first.
    """
    names = []
    for prefix in (f"blk.{layer}.", f"layers.{layer}.", f"model.layers.{layer}."):
        for candidate in candidates:
            names.append(prefix + candidate)
    return resolve_tensor(infos, names)


def build_order(infos, config):
    """Produce the tensor emission order documented in native/niyah.h.

    The sixth entry of each layer is ffn_norm. It previously listed
    "attn_norm.weight" as its first candidate, and because
    blk.N.attn_norm.weight exists in every GGUF file, resolve_tensor matched
    attn_norm a second time: ffn_norm was never emitted and attn_norm was
    emitted twice. validate_shapes could not catch it because both norms
    have identical shape (dim,).
    """
    emb = resolve_tensor(infos, [
        "token_embd.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    ])
    final_norm = resolve_tensor(infos, [
        "output_norm.weight",
        "model.norm.weight",
        "transformer.ln_f.weight",
    ])
    lm_head = resolve_tensor(infos, [
        "output.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ])

    order = [emb]
    for layer in range(config["n_layers"]):
        order.extend([
            layer_tensor(infos, layer, [
                "attn_norm.weight", "input_layernorm.weight",
            ]),
            layer_tensor(infos, layer, ["attn_q.weight", "self_attn.q_proj.weight"]),
            layer_tensor(infos, layer, ["attn_k.weight", "self_attn.k_proj.weight"]),
            layer_tensor(infos, layer, ["attn_v.weight", "self_attn.v_proj.weight"]),
            layer_tensor(infos, layer, ["attn_output.weight", "self_attn.o_proj.weight"]),
            # ffn_norm FIRST -- see docstring.
            layer_tensor(infos, layer, [
                "ffn_norm.weight", "post_attention_layernorm.weight",
            ]),
            layer_tensor(infos, layer, ["ffn_gate.weight", "mlp.gate_proj.weight"]),
            layer_tensor(infos, layer, ["ffn_up.weight", "mlp.up_proj.weight"]),
            layer_tensor(infos, layer, ["ffn_down.weight", "mlp.down_proj.weight"]),
        ])
    order.append(final_norm)
    if not config["tie_word_embeddings"]:
        order.append(lm_head)
    return order


def validate_shapes(order, config):
    dim = config["n_dim"]
    vocab = config["n_vocab"]
    layers = config["n_layers"]
    heads = config["n_heads"]
    kv_heads = config["n_kv_heads"]
    ff = config["n_ff"]
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim

    expected = [(vocab, dim)]
    for _ in range(layers):
        expected.extend([
            (dim,), (dim, dim), (kv_dim, dim), (kv_dim, dim), (dim, dim),
            (dim,), (ff, dim), (ff, dim), (dim, ff),
        ])
    expected.append((dim,))
    if not config["tie_word_embeddings"]:
        expected.append((vocab, dim))

    if len(order) != len(expected):
        fail("internal tensor order mismatch")

    for index, (info, want) in enumerate(zip(order, expected)):
        dims = tuple(info["dims"])
        reversed_dims = tuple(reversed(info["dims"])) if len(dims) > 1 else dims
        if reversed_dims != want and dims != want:
            fail(
                f"tensor {index} {info['name']} shape {list(dims)} "
                f"incompatible with expected {list(want)}"
            )


def check_convertible(order):
    """Fail before the output file is opened if a tensor cannot be decoded."""
    bad = sorted({
        info["type_name"] for info in order if info["type"] not in SUPPORTED_TYPES
    })
    if bad:
        fail(
            "this checkpoint contains tensor types this converter cannot "
            "decode: " + ", ".join(bad) + ". Supported: F32, F16, Q4_0, Q4_1. "
            "Requantise first, e.g. "
            "llama-quantize model.gguf model-q4_0.gguf Q4_0"
        )


# Emitted in one pass; see module docstring for why both key sets exist.
CONFIG_KEYS = [
    # NiyahMini reader (native/niyah_mini/niyah_mini_model.c)
    ("n_layers", "n_layers"),
    ("n_dim", "n_dim"),
    ("n_heads", "n_heads"),
    ("n_kv_heads", "n_kv_heads"),
    ("n_ff", "n_ff"),
    ("n_vocab", "n_vocab"),
    ("n_ctx", "n_ctx"),
    ("rope_theta", "rope_theta"),
    ("norm_eps", "norm_eps"),
    ("tie_word_embeddings", "tie_word_embeddings"),
    # Niyah reader (native/niyah_model.c)
    ("layer_count", "n_layers"),
    ("dim", "n_dim"),
    ("heads", "n_heads"),
    ("kv_heads", "n_kv_heads"),
    ("hidden_dim", "n_ff"),
    ("vocab_size", "n_vocab"),
    ("context_size", "n_ctx"),
    ("eos_token", "eos_token"),
]


def write_config(config_path, config):
    with open(config_path, "w", encoding="utf-8", newline="\n") as cfg:
        cfg.write("{\n")
        for index, (out_key, src_key) in enumerate(CONFIG_KEYS):
            value = config[src_key]
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, float):
                text = f"{value:.9g}"
            else:
                text = str(value)
            comma = "," if index + 1 < len(CONFIG_KEYS) else ""
            cfg.write(f"  \"{out_key}\": {text}{comma}\n")
        cfg.write("}\n")


def convert(input_path, output_path, config_path, progress=False):
    metadata, infos = parse_gguf(input_path)
    config = infer_config(metadata, infos)
    order = build_order(infos, config)
    validate_shapes(order, config)
    check_convertible(order)

    expected_floats = sum(info["count"] for info in order)
    written = 0

    with open(input_path, "rb") as src, open(output_path, "wb") as out:
        for position, info in enumerate(order):
            src.seek(info["offset"])
            for chunk in stream_tensor(src, info):
                written += len(chunk)
                emit(out, chunk)
            if progress:
                print(
                    f"  [{position + 1}/{len(order)}] {info['name']} "
                    f"({info['type_name']}, {info['count']} elems)",
                    file=sys.stderr,
                )

    if written != expected_floats:
        fail(f"wrote {written} floats but expected {expected_floats}")

    if config_path:
        write_config(config_path, config)

    return config, written


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GGUF checkpoint to Niyah's flat float32 blob."
    )
    parser.add_argument("input", help="input .gguf")
    parser.add_argument("output", help="output flat float32 .bin")
    parser.add_argument("--config", help="optional Niyah JSON config path")
    parser.add_argument(
        "--progress", action="store_true", help="print per-tensor progress"
    )
    args = parser.parse_args()
    try:
        config, count = convert(
            args.input, args.output, args.config, progress=args.progress
        )
    except (OSError, RuntimeError, struct.error, OverflowError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"converted_f32={count}")
    print(f"bytes={count * 4}")
    for key in ("n_layers", "n_dim", "n_heads", "n_kv_heads", "n_ff", "n_vocab"):
        print(f"{key}={config[key]}")
    print(f"tie_word_embeddings={str(config['tie_word_embeddings']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
