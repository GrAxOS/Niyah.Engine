#!/usr/bin/env python3
import argparse
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


def fail(message):
    raise RuntimeError(message)


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        fail("unexpected end of GGUF file")
    return data


def u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


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
    if value_type == 0:
        return u8(f)
    if value_type == 1:
        return struct.unpack("<i8", read_exact(f, 1))[0]
    if value_type == 2:
        return u16(f)
    if value_type == 3:
        return struct.unpack("<i16", read_exact(f, 2))[0]
    if value_type == 4:
        return u32(f)
    if value_type == 5:
        return struct.unpack("<i", read_exact(f, 4))[0]
    if value_type == 6:
        return u64(f)
    if value_type == 7:
        return i64(f)
    if value_type == 8:
        return f32(f)
    if value_type == 9:
        return f64(f)
    if value_type == 10:
        return False if u8(f) == 0 else True
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


def u8(f):
    return read_exact(f, 1)[0]


def u16(f):
    return struct.unpack("<H", read_exact(f, 2))[0]


def align_up(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def ggml_row_count(dims):
    total = 1
    for d in dims:
        total *= d
    return total


def f16_to_f32(raw):
    if len(raw) % 2:
        fail("unaligned FP16 tensor")
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def q4_0_to_f32(raw, count):
    block_size = 32
    block_bytes = 18
    if count % block_size:
        fail("Q4_0 tensor size is not divisible by 32")
    if len(raw) != (count // block_size) * block_bytes:
        fail("Q4_0 tensor byte size mismatch")
    out = []
    off = 0
    for _ in range(count // block_size):
        d = struct.unpack_from("<e", raw, off)[0]
        off += 2
        qs = raw[off:off + 16]
        off += 16
        for byte in qs:
            q0 = byte & 0x0F
            q1 = byte >> 4
            out.append(float(d) * (q0 - 8))
            out.append(float(d) * (q1 - 8))
    return out


def q4_1_to_f32(raw, count):
    block_size = 32
    block_bytes = 20
    if count % block_size:
        fail("Q4_1 tensor size is not divisible by 32")
    if len(raw) != (count // block_size) * block_bytes:
        fail("Q4_1 tensor byte size mismatch")
    out = []
    off = 0
    for _ in range(count // block_size):
        d = struct.unpack_from("<e", raw, off)[0]
        m = struct.unpack_from("<e", raw, off + 2)[0]
        off += 4
        qs = raw[off:off + 16]
        off += 16
        for byte in qs:
            q0 = byte & 0x0F
            q1 = byte >> 4
            out.append(float(m) + float(d) * q0)
            out.append(float(m) + float(d) * q1)
    return out


def dequantize_tensor(raw, type_code, count):
    if type_code == 0:
        expected = count * 4
        if len(raw) != expected:
            fail(f"F32 tensor byte size mismatch: {len(raw)} != {expected}")
        return list(struct.unpack(f"<{count}f", raw))
    if type_code == 1:
        return f16_to_f32(raw)
    if type_code == 2:
        return q4_0_to_f32(raw, count)
    if type_code == 3:
        return q4_1_to_f32(raw, count)
    fail(f"unsupported GGML tensor type {type_code}; supported: F32, F16, Q4_0, Q4_1")


def parse_gguf(path):
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
        if not isinstance(alignment, int) or alignment <= 0 or alignment > (1 << 20) or alignment & (alignment - 1):
            alignment = 32
        data_start = align_up(f.tell(), alignment)

        infos = []
        file_size = f.seek(0, 2)
        for index, (name, dims, type_code, rel_offset) in enumerate(tensor_infos):
            count = ggml_row_count(dims)
            info_name, block_bytes = GGML_TYPES[type_code]
            if type_code in (0, 1):
                byte_size = count * block_bytes
            else:
                block_size = 32
                if count % block_size:
                    fail(f"tensor {name} element count {count} is invalid for {info_name}")
                byte_size = (count // block_size) * block_bytes
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
        return metadata, infos, f


def tensor_bytes(path, info):
    with open(path, "rb") as f:
        f.seek(info["offset"])
        return read_exact(f, info["size"])


def first_metadata(metadata, keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def detect_n_parameters(infos):
    return sum(x["count"] for x in infos)


def infer_config(metadata, infos):
    n_vocab = first_metadata(metadata, [
        "tokenizer.ggml.vocab_size",
        "qwen2.vocab_size",
        "llama.vocab_size",
    ])
    n_embd = first_metadata(metadata, [
        "qwen2.embedding_length",
        "llama.embedding_length",
        "llama.block_count_embedding_length",
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
    ], False)

    shapes = {x["name"]: x["dims"] for x in infos}
    embedding_shape = shapes.get("token_embd.weight") or shapes.get("model.embed_tokens.weight")
    if n_vocab is None and embedding_shape:
        n_vocab = embedding_shape[0]
    if n_embd is None and embedding_shape:
        n_embd = embedding_shape[1] if len(embedding_shape) > 1 else embedding_shape[0]
    if n_layers is None:
        layer_ids = []
        for info in infos:
            parts = info["name"].split(".")
            if "blk" in parts:
                try:
                    layer_ids.append(int(parts[parts.index("blk") + 1]))
                except (ValueError, IndexError):
                    pass
            elif "layers" in parts:
                try:
                    layer_ids.append(int(parts[parts.index("layers") + 1]))
                except (ValueError, IndexError):
                    pass
        if layer_ids:
            n_layers = max(layer_ids) + 1

    if any(x is None for x in (n_vocab, n_embd, n_heads, n_layers, n_ff)):
        fail("could not infer complete NiyahMini configuration from GGUF metadata")
    return {
        "n_layers": int(n_layers),
        "n_dim": int(n_embd),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "n_ff": int(n_ff),
        "n_vocab": int(n_vocab),
        "n_ctx": int(n_ctx),
        "rope_theta": float(rope_theta),
        "norm_eps": float(norm_eps),
        "tie_word_embeddings": bool(tie),
    }


def resolve_tensor(infos, candidates):
    by_name = {x["name"]: x for x in infos}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    fail("missing tensor: " + " | ".join(candidates))


def layer_tensor(infos, layer, candidates):
    names = []
    for prefix in (f"blk.{layer}.", f"layers.{layer}.", f"model.layers.{layer}."):
        for candidate in candidates:
            names.append(prefix + candidate)
    return resolve_tensor(infos, names)


def tensor_floats(path, info):
    raw = tensor_bytes(path, info)
    values = dequantize_tensor(raw, info["type"], info["count"])
    for value in values:
        if not math.isfinite(float(value)):
            fail(f"non-finite value found in tensor {info['name']}")
    return values


def write_f32(out, values):
    chunk = bytearray()
    pack = struct.Struct("<f").pack
    for value in values:
        chunk.extend(pack(float(value)))
        if len(chunk) >= 1024 * 1024:
            out.write(chunk)
            chunk.clear()
    if chunk:
        out.write(chunk)


def build_order(infos, config):
    emb = resolve_tensor(infos, ["token_embd.weight", "model.embed_tokens.weight", "transformer.wte.weight"])
    final_norm = resolve_tensor(infos, ["output_norm.weight", "model.norm.weight", "transformer.ln_f.weight"])
    lm_head = resolve_tensor(infos, ["output.weight", "lm_head.weight", "model.embed_tokens.weight"])
    order = [emb]
    for layer in range(config["n_layers"]):
        order.extend([
            layer_tensor(infos, layer, ["attn_norm.weight", "input_layernorm.weight", "self_attn.rms_norm.weight"]),
            layer_tensor(infos, layer, ["attn_q.weight", "self_attn.q_proj.weight"]),
            layer_tensor(infos, layer, ["attn_k.weight", "self_attn.k_proj.weight"]),
            layer_tensor(infos, layer, ["attn_v.weight", "self_attn.v_proj.weight"]),
            layer_tensor(infos, layer, ["attn_output.weight", "self_attn.o_proj.weight"]),
            layer_tensor(infos, layer, ["attn_norm.weight", "post_attention_layernorm.weight", "post_attn_norm.weight"]),
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
        shape = tuple(reversed(info["dims"])) if len(info["dims"]) > 1 else tuple(info["dims"])
        if shape != want and tuple(info["dims"]) != want:
            fail(f"tensor {index} {info['name']} shape {info['dims']} incompatible with expected {want}")


def convert(input_path, output_path, config_path):
    metadata, infos, _ = parse_gguf(input_path)
    config = infer_config(metadata, infos)
    order = build_order(infos, config)
    validate_shapes(order, config)

    with open(output_path, "wb") as out:
        for info in order:
            write_f32(out, tensor_floats(input_path, info))

    if config_path:
        with open(config_path, "w", encoding="utf-8", newline="\n") as cfg:
            cfg.write("{\n")
            keys = [
                "n_layers", "n_dim", "n_heads", "n_kv_heads", "n_ff",
                "n_vocab", "n_ctx", "rope_theta", "norm_eps", "tie_word_embeddings",
            ]
            for index, key in enumerate(keys):
                value = config[key]
                if isinstance(value, bool):
                    text = "true" if value else "false"
                elif isinstance(value, float):
                    text = f"{value:.9g}"
                else:
                    text = str(value)
                comma = "," if index + 1 < len(keys) else ""
                cfg.write(f"  \"{key}\": {text}{comma}\n")
            cfg.write("}\n")

    expected_floats = 0
    for info in order:
        expected_floats += info["count"]
    return config, expected_floats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input .gguf")
    parser.add_argument("output", help="output flat float32 .bin")
    parser.add_argument("--config", help="optional NiyahMini JSON config path")
    args = parser.parse_args()
    try:
        config, count = convert(args.input, args.output, args.config)
    except (OSError, RuntimeError, struct.error, OverflowError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"converted_f32={count}")
    print(f"n_layers={config['n_layers']}")
    print(f"n_dim={config['n_dim']}")
    print(f"n_heads={config['n_heads']}")
    print(f"n_kv_heads={config['n_kv_heads']}")
    print(f"n_ff={config['n_ff']}")
    print(f"n_vocab={config['n_vocab']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
