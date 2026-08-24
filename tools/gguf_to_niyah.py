#!/usr/bin/env python3
"""Convert a GGUF checkpoint to the Niyah native float32 layout.

Usage:
    python gguf_to_niyah.py \\
        --input  model.gguf \\
        --output model-f32.bin \\
        --config model.json

Outputs:
    <output>  raw little-endian float32 weights in NiyahModelWeights order
    <config>  JSON consumed by niyah_model_load_config_json()

Correctness notes (2026-08):

  * The architecture prefix is read from ``general.architecture`` instead of
    being hardcoded to ``llama``. A Qwen2 GGUF stores ``qwen2.embedding_length``,
    not ``llama.embedding_length``, so the previous converter silently emitted a
    config with no dim, heads or layer_count at all.

  * ``rope_theta`` and ``norm_eps`` are emitted. They were never written, and
    the C loader could not parse floats anyway, so it substituted 10000.0 and
    1e-5 for every model. 10000.0 is the Llama-2 rope base; Llama-3 uses
    500000.0 and Qwen2 uses 1000000.0.

  * ``context_size`` comes from the checkpoint instead of being pinned to 2048.

  * Tied embeddings are detected instead of crashing. Qwen2.5-0.5B ships no
    ``output.weight`` tensor, and ``extract_tensor`` used to raise ValueError.

  * Tensors are written with ``numpy.ndarray.tobytes()``. The previous code did
    ``struct.pack(f"{n}f", *values)``, expanding each tensor into that many
    Python call arguments; a 0.5B model needs tens of gigabytes that way.

  * Missing required metadata is a hard error. Nothing is defaulted silently.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not found. Install with: pip install numpy")
    sys.exit(1)

try:
    import gguf
except ImportError:
    print("ERROR: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


class ConversionError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def _as_str(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, memoryview):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "u", "i") and value.ndim == 1:
            try:
                return bytes(value.tolist()).decode("utf-8", "replace")
            except (TypeError, ValueError):
                pass
    return str(value)


def _field_value(field):
    """Best-effort scalar extraction across gguf-py versions."""
    contents = getattr(field, "contents", None)
    if callable(contents):
        try:
            value = field.contents()
            if value is not None:
                return value
        except Exception:
            pass

    data = getattr(field, "data", None)
    parts = getattr(field, "parts", None)
    if data is not None and parts is not None and len(data) > 0:
        part = parts[data[-1]]
        try:
            return part[0] if len(part) == 1 else part
        except TypeError:
            return part
    if parts:
        return parts[-1]
    return None


def _meta(reader, key, cast, required=True, default=None):
    field = reader.fields.get(key)
    if field is None:
        if required:
            raise ConversionError(
                f"required GGUF metadata key is missing: {key}"
            )
        return default

    raw = _field_value(field)
    if raw is None:
        if required:
            raise ConversionError(
                f"GGUF metadata key {key} has no readable value"
            )
        return default

    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"cannot read GGUF key {key}: {exc}") from exc


def get_config(reader) -> dict:
    """Extract the Niyah config from GGUF metadata. No silent defaults."""
    arch = _meta(reader, "general.architecture", _as_str)

    def key(suffix):
        return f"{arch}.{suffix}"

    dim = _meta(reader, key("embedding_length"), int)
    heads = _meta(reader, key("attention.head_count"), int)

    if heads <= 0 or dim <= 0:
        raise ConversionError(f"invalid dim={dim} heads={heads}")
    if dim % heads != 0:
        raise ConversionError(
            f"embedding_length {dim} is not divisible by head_count {heads}"
        )

    kv_heads = _meta(reader, key("attention.head_count_kv"), int,
                     required=False, default=heads)
    if kv_heads <= 0 or heads % kv_heads != 0:
        raise ConversionError(
            f"head_count {heads} is not divisible by head_count_kv {kv_heads}"
        )

    # The C loader rejects a config without these; see niyah_model.c.
    rope_theta = _meta(reader, key("rope.freq_base"), float)
    norm_eps = _meta(reader, key("attention.layer_norm_rms_epsilon"), float,
                     required=False, default=None)
    if norm_eps is None:
        norm_eps = _meta(reader, key("attention.layer_norm_epsilon"), float)

    if not rope_theta > 0.0:
        raise ConversionError(f"non-positive rope.freq_base: {rope_theta}")
    if not norm_eps > 0.0:
        raise ConversionError(f"non-positive rms epsilon: {norm_eps}")

    tensor_names = {t.name for t in reader.tensors}
    tied = "output.weight" not in tensor_names

    config = {
        "architecture": arch,
        "vocab_size": _meta(reader, "tokenizer.ggml.tokens", len,
                            required=False,
                            default=None)
        or _meta(reader, "tokenizer.vocab_size", int),
        "dim": dim,
        "heads": heads,
        "kv_heads": kv_heads,
        "layer_count": _meta(reader, key("block_count"), int),
        "hidden_dim": _meta(reader, key("feed_forward_length"), int),
        "context_size": _meta(reader, key("context_length"), int),
        "rope_theta": rope_theta,
        "norm_eps": norm_eps,
        "tie_word_embeddings": tied,
    }

    bos = _meta(reader, "tokenizer.ggml.bos_token_id", int,
                required=False, default=None)
    eos = _meta(reader, "tokenizer.ggml.eos_token_id", int,
                required=False, default=None)
    if bos is not None:
        config["bos_token"] = bos
    if eos is not None:
        config["eos_token"] = eos

    return config


# --------------------------------------------------------------------------
# Tensors
# --------------------------------------------------------------------------
def tensor_map(reader) -> dict:
    return {t.name: t for t in reader.tensors}


def write_tensor(handle, tensors: dict, name: str, expected: int) -> int:
    tensor = tensors.get(name)
    if tensor is None:
        raise ConversionError(f"tensor not found: {name}")

    # '<f4' pins little-endian float32 regardless of host byte order; the C
    # loader reads a raw float array and does no byte swapping.
    data = np.ascontiguousarray(tensor.data).astype("<f4", copy=False)
    count = int(data.size)

    if expected and count != expected:
        raise ConversionError(
            f"tensor {name} has {count} elements, expected {expected}"
        )

    handle.write(data.tobytes())
    return count


def write_niyah_weights(reader, output: Path, config: dict) -> int:
    """Write tensors in NiyahModelWeights order.

    embedding
    per layer: attn_norm, wq, wk, wv, wo, ffn_norm, ffn_gate, ffn_up, ffn_down
    final_norm
    lm_head   (omitted when embeddings are tied)
    """
    tensors = tensor_map(reader)

    vocab = config["vocab_size"]
    dim = config["dim"]
    heads = config["heads"]
    kv_heads = config["kv_heads"]
    layers = config["layer_count"]
    ff = config["hidden_dim"]
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim

    total = 0
    with open(output, "wb") as f:
        total += write_tensor(f, tensors, "token_embd.weight", vocab * dim)
        print(f"  embedding: {vocab * dim} floats")

        for i in range(layers):
            p = f"blk.{i}"
            total += write_tensor(f, tensors, f"{p}.attn_norm.weight", dim)
            total += write_tensor(f, tensors, f"{p}.attn_q.weight", dim * dim)
            total += write_tensor(f, tensors, f"{p}.attn_k.weight", kv_dim * dim)
            total += write_tensor(f, tensors, f"{p}.attn_v.weight", kv_dim * dim)
            total += write_tensor(f, tensors, f"{p}.attn_output.weight", dim * dim)
            total += write_tensor(f, tensors, f"{p}.ffn_norm.weight", dim)
            total += write_tensor(f, tensors, f"{p}.ffn_gate.weight", ff * dim)
            total += write_tensor(f, tensors, f"{p}.ffn_up.weight", ff * dim)
            total += write_tensor(f, tensors, f"{p}.ffn_down.weight", dim * ff)
            print(f"  layer {i}: done")

        total += write_tensor(f, tensors, "output_norm.weight", dim)

        if config["tie_word_embeddings"]:
            print("  lm_head: tied to embedding, not written")
        else:
            total += write_tensor(f, tensors, "output.weight", vocab * dim)

    return total


def expected_floats(config: dict) -> int:
    """Mirror of niyah_model_expected_floats() in native/niyah_model.c."""
    dim = config["dim"]
    head_dim = dim // config["heads"]
    kv_dim = config["kv_heads"] * head_dim
    ff = config["hidden_dim"]
    vocab = config["vocab_size"]

    per_layer = (
        dim
        + dim * dim
        + kv_dim * dim
        + kv_dim * dim
        + dim * dim
        + dim
        + ff * dim
        + ff * dim
        + dim * ff
    )

    total = vocab * dim + config["layer_count"] * per_layer + dim
    if not config["tie_word_embeddings"]:
        total += vocab * dim
    return total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    if not args.input.is_file():
        print(f"ERROR: no such file: {args.input}")
        return 2

    print(f"Loading {args.input}...")
    reader = gguf.GGUFReader(args.input, "r")

    try:
        print("Extracting config...")
        config = get_config(reader)

        args.config.write_text(json.dumps(config, indent=2) + "\n",
                               encoding="utf-8")
        print(f"Config saved to {args.config}")
        print(json.dumps(config, indent=2))

        want = expected_floats(config)
        print(f"Writing weights to {args.output} ({want} floats expected)...")
        got = write_niyah_weights(reader, args.output, config)

        if got != want:
            raise ConversionError(
                f"wrote {got} floats but the layout requires {want}; "
                "the C loader will reject this blob"
            )

        size_mb = got * 4 / 1e6
        print(f"Total: {got} floats ({size_mb:.1f} MB)")
    except ConversionError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
