#!/usr/bin/env python3
"""Write real GGUF files, for testing tools/convert_gguf_to_niyah.py.

Stdlib only. The point of this module is to be an INDEPENDENT implementation
of the GGUF container: it encodes metadata using the value-type codes from
enum gguf_type in llama.cpp, so if the converter's table drifts again the
tests fail immediately.

Container layout produced here:

    "GGUF"                      4 bytes
    version                     u32 (3)
    tensor_count                u64
    metadata_kv_count           u64
    metadata_kv_count x (key:str, value_type:u32, value)
    tensor_count x (name:str, n_dims:u32, dims:u64[n_dims],
                    type:u32, offset:u64)
    padding to `alignment`
    tensor data, each tensor starting at an `alignment` boundary

Tensor offsets are relative to the start of the data section.
"""
import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# enum gguf_type
UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12

# enum ggml_type, only what the fixtures need
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14

QK = 32
QK_K = 256


def align_up(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


# ------------------------------------------------------------------- encoding


def enc_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def enc_scalar(type_code, value):
    if type_code == UINT8:
        return struct.pack("<B", value)
    if type_code == INT8:
        return struct.pack("<b", value)
    if type_code == UINT16:
        return struct.pack("<H", value)
    if type_code == INT16:
        return struct.pack("<h", value)
    if type_code == UINT32:
        return struct.pack("<I", value)
    if type_code == INT32:
        return struct.pack("<i", value)
    if type_code == FLOAT32:
        return struct.pack("<f", value)
    if type_code == BOOL:
        return struct.pack("<B", 1 if value else 0)
    if type_code == STRING:
        return enc_string(value)
    if type_code == UINT64:
        return struct.pack("<Q", value)
    if type_code == INT64:
        return struct.pack("<q", value)
    if type_code == FLOAT64:
        return struct.pack("<d", value)
    raise ValueError("cannot encode metadata type %r" % (type_code,))


# --------------------------------------------------------- deterministic data


class Rng(object):
    """xorshift32. Deterministic across Python versions and platforms."""

    def __init__(self, seed):
        self.state = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def u32(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def byte(self):
        return self.u32() & 0xFF

    def unit(self):
        """Uniform in [0, 1)."""
        return self.u32() / 4294967296.0

    def signed(self, scale):
        """Uniform in (-scale, scale)."""
        return (self.unit() * 2.0 - 1.0) * scale

    def scale(self, magnitude):
        """Non-zero signed value, safely representable in f16."""
        value = self.signed(magnitude)
        if value == 0.0:
            return magnitude
        return value


def name_seed(name):
    seed = 0x811C9DC5
    for ch in name.encode("utf-8"):
        seed = ((seed ^ ch) * 16777619) & 0xFFFFFFFF
    return seed or 1


def payload_f32(name, count):
    rng = Rng(name_seed(name))
    return struct.pack("<%df" % count, *[rng.signed(2.0) for _ in range(count)])


def payload_f16(name, count):
    rng = Rng(name_seed(name))
    return struct.pack("<%de" % count, *[rng.signed(2.0) for _ in range(count)])


def payload_q4_0(name, count):
    """18 bytes per 32 elements: f16 scale, then 16 packed nibble pairs."""
    assert count % QK == 0
    rng = Rng(name_seed(name))
    out = bytearray()
    for block in range(count // QK):
        # Include a negative scale so a sign error cannot pass unnoticed.
        scale = rng.signed(0.25) if block % 3 else -abs(rng.signed(0.25)) - 0.01
        out += struct.pack("<e", scale)
        for _ in range(QK // 2):
            out.append(rng.byte())
    return bytes(out)


def payload_q4_1(name, count):
    """20 bytes per 32 elements: f16 scale, f16 min, 16 packed nibble pairs."""
    assert count % QK == 0
    rng = Rng(name_seed(name))
    out = bytearray()
    for _ in range(count // QK):
        out += struct.pack("<e", rng.signed(0.25) or 0.03125)
        out += struct.pack("<e", rng.signed(1.0))
        for _ in range(QK // 2):
            out.append(rng.byte())
    return bytes(out)


def payload_q4_k(name, count):
    """Structurally valid block_q4_K, 144 bytes per 256 elements.

        f16   d
        f16   dmin
        u8    scales[12]   six-bit scale/min pairs, packed
        u8    qs[128]      two 4-bit quants per byte

    The scales bytes are left fully random: every one of the 2^96 patterns is
    a legal encoding, so there is nothing to constrain. d and dmin are kept
    small and non-zero so products stay finite in f16.
    """
    assert count % QK_K == 0
    rng = Rng(name_seed(name))
    out = bytearray()
    for _ in range(count // QK_K):
        out += struct.pack("<e", rng.scale(0.05))
        out += struct.pack("<e", rng.scale(0.05))
        for _ in range(12):
            out.append(rng.byte())
        for _ in range(QK_K // 2):
            out.append(rng.byte())
    return bytes(out)


def payload_q6_k(name, count):
    """Structurally valid block_q6_K, 210 bytes per 256 elements.

        u8    ql[128]      low 4 bits
        u8    qh[64]       high 2 bits, four per byte
        i8    scales[16]   signed, one per 16 elements
        f16   d

    Note the order: d comes LAST in Q6_K, unlike Q4_K where it comes first.
    Every byte pattern is legal, including negative scales.
    """
    assert count % QK_K == 0
    rng = Rng(name_seed(name))
    out = bytearray()
    for _ in range(count // QK_K):
        for _ in range(QK_K // 2):
            out.append(rng.byte())
        for _ in range(QK_K // 4):
            out.append(rng.byte())
        for _ in range(QK_K // 16):
            out.append(rng.byte())
        out += struct.pack("<e", rng.scale(0.05))
    return bytes(out)


def payload_superblock(name, count, block_bytes):
    """Opaque but correctly sized payload for a K-quant tensor.

    Used for types the converter is expected to REJECT; the contents are
    never decoded, only the byte count matters.
    """
    assert count % QK_K == 0
    rng = Rng(name_seed(name))
    n = (count // QK_K) * block_bytes
    return bytes(bytearray(rng.byte() for _ in range(n)))


def make_payload(name, count, type_code):
    if type_code == GGML_F32:
        return payload_f32(name, count)
    if type_code == GGML_F16:
        return payload_f16(name, count)
    if type_code == GGML_Q4_0:
        return payload_q4_0(name, count)
    if type_code == GGML_Q4_1:
        return payload_q4_1(name, count)
    if type_code == GGML_Q4_K:
        return payload_q4_k(name, count)
    if type_code == GGML_Q6_K:
        return payload_q6_k(name, count)
    if type_code == GGML_Q5_K:
        return payload_superblock(name, count, 176)
    raise ValueError("no payload builder for ggml type %d" % type_code)


# --------------------------------------------------------------------- writer


class GgufWriter(object):
    def __init__(self, alignment=32):
        self.alignment = alignment
        self.metadata = []   # (key, type_code, encoded_bytes)
        self.tensors = []    # dict(name, dims, type, payload)

    def add_scalar(self, key, type_code, value):
        self.metadata.append((key, type_code, enc_scalar(type_code, value)))
        return self

    def add_array(self, key, elem_type, values):
        blob = struct.pack("<I", elem_type) + struct.pack("<Q", len(values))
        for value in values:
            blob += enc_scalar(elem_type, value)
        self.metadata.append((key, ARRAY, blob))
        return self

    def add_tensor(self, name, dims, type_code, payload=None):
        count = 1
        for d in dims:
            count *= d
        if payload is None:
            payload = make_payload(name, count, type_code)
        self.tensors.append({
            "name": name,
            "dims": list(dims),
            "type": type_code,
            "count": count,
            "payload": payload,
        })
        return self

    def tensor(self, name):
        for t in self.tensors:
            if t["name"] == name:
                return t
        raise KeyError(name)

    def write(self, path):
        # Assign aligned, data-section-relative offsets.
        cursor = 0
        for t in self.tensors:
            t["offset"] = cursor
            cursor = align_up(cursor + len(t["payload"]), self.alignment)
        data_size = cursor

        head = bytearray()
        head += GGUF_MAGIC
        head += struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors))
        head += struct.pack("<Q", len(self.metadata))
        for key, type_code, blob in self.metadata:
            head += enc_string(key)
            head += struct.pack("<I", type_code)
            head += blob
        for t in self.tensors:
            head += enc_string(t["name"])
            head += struct.pack("<I", len(t["dims"]))
            for d in t["dims"]:
                head += struct.pack("<Q", d)
            head += struct.pack("<I", t["type"])
            head += struct.pack("<Q", t["offset"])

        pad = align_up(len(head), self.alignment) - len(head)
        head += b"\x00" * pad

        body = bytearray(b"\x00" * data_size)
        for t in self.tensors:
            start = t["offset"]
            body[start:start + len(t["payload"])] = t["payload"]

        with open(path, "wb") as f:
            f.write(bytes(head))
            f.write(bytes(body))
        return path


# ------------------------------------------------------------- model fixtures

TINY = {
    "n_vocab": 64,
    "n_embd": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "n_ff": 64,
    "n_ctx": 128,
    "eos_token": 2,
    "rope_theta": 10000.0,
    "norm_eps": 1.0000000474974513e-05,  # exactly float32(1e-5)
}

# Mixed on purpose: every supported decode path is exercised in one file.
DEFAULT_TYPES = {
    "token_embd": GGML_Q4_0,
    "attn_norm": GGML_F32,
    "attn_q": GGML_F16,
    "attn_k": GGML_Q4_1,
    "attn_v": GGML_Q4_0,
    "attn_output": GGML_F32,
    "ffn_norm": GGML_F32,
    "ffn_gate": GGML_Q4_0,
    "ffn_up": GGML_F16,
    "ffn_down": GGML_Q4_1,
    "output_norm": GGML_F32,
    "output": GGML_Q4_0,
}

# K-quant coverage. The norms stay F32 because they are 32 elements wide in
# TINY and 256 does not divide 32 -- which mirrors real checkpoints, where
# llama-quantize always leaves norms in F32.
KQUANT_TYPES = {
    "token_embd": GGML_Q4_K,
    "attn_norm": GGML_F32,
    "attn_q": GGML_Q4_K,
    "attn_k": GGML_Q6_K,
    "attn_v": GGML_Q4_K,
    "attn_output": GGML_Q6_K,
    "ffn_norm": GGML_F32,
    "ffn_gate": GGML_Q4_K,
    "ffn_up": GGML_Q6_K,
    "ffn_down": GGML_Q4_K,
    "output_norm": GGML_F32,
    "output": GGML_Q6_K,
}


def build_model(cfg=None, tie=False, declare_tie=True, types=None,
                arch="qwen2", extra_metadata=True, big_vocab_array=True):
    """Return (writer, expected_order).

    expected_order is the list of tensor names in the blob order documented
    in native/niyah.h, derived here by hand so it is independent of the
    converter's build_order().
    """
    cfg = dict(cfg or TINY)
    types = dict(types or DEFAULT_TYPES)

    dim = cfg["n_embd"]
    vocab = cfg["n_vocab"]
    ff = cfg["n_ff"]
    layers = cfg["n_layers"]
    head_dim = dim // cfg["n_heads"]
    kv_dim = cfg["n_kv_heads"] * head_dim

    w = GgufWriter(alignment=32)

    w.add_scalar("general.architecture", STRING, arch)
    w.add_scalar("general.name", STRING, "niyah-tiny-fixture")
    w.add_scalar("general.alignment", UINT32, 32)
    if big_vocab_array:
        # Long arrays must be skipped without desynchronising the stream.
        w.add_array("tokenizer.ggml.tokens", STRING,
                    ["tok%d" % i for i in range(200)])
        w.add_array("tokenizer.ggml.scores", FLOAT32,
                    [i * 0.5 for i in range(200)])
    w.add_scalar("%s.embedding_length" % arch, UINT32, dim)
    w.add_scalar("%s.attention.head_count" % arch, UINT32, cfg["n_heads"])
    w.add_scalar("%s.attention.head_count_kv" % arch, UINT32, cfg["n_kv_heads"])
    w.add_scalar("%s.block_count" % arch, UINT32, layers)
    w.add_scalar("%s.context_length" % arch, UINT32, cfg["n_ctx"])
    w.add_scalar("%s.feed_forward_length" % arch, UINT32, ff)
    w.add_scalar("%s.rope.freq_base" % arch, FLOAT32, cfg["rope_theta"])
    w.add_scalar("%s.attention.layer_norm_rms_epsilon" % arch, FLOAT32,
                 cfg["norm_eps"])
    if declare_tie:
        w.add_scalar("%s.tie_word_embeddings" % arch, BOOL, tie)
    w.add_scalar("tokenizer.ggml.vocab_size", UINT32, vocab)
    w.add_scalar("tokenizer.ggml.eos_token_id", UINT32, cfg["eos_token"])
    if extra_metadata:
        # One value of every remaining spec type, to pin the type table.
        w.add_scalar("test.uint8", UINT8, 200)
        w.add_scalar("test.int8", INT8, -5)
        w.add_scalar("test.uint16", UINT16, 40000)
        w.add_scalar("test.int16", INT16, -300)
        w.add_scalar("test.int32", INT32, -70000)
        w.add_scalar("test.uint64", UINT64, 2 ** 40 + 7)
        w.add_scalar("test.int64", INT64, -(2 ** 40) - 7)
        w.add_scalar("test.float64", FLOAT64, 0.1)
        w.add_scalar("test.bool_false", BOOL, False)
        w.add_array("test.small_array", INT32, [-1, 0, 1])

    # GGUF stores dimensions fastest-varying first, so an [out][in] matrix is
    # written as [in, out].
    w.add_tensor("token_embd.weight", [dim, vocab], types["token_embd"])
    for layer in range(layers):
        p = "blk.%d." % layer
        w.add_tensor(p + "attn_norm.weight", [dim], types["attn_norm"])
        w.add_tensor(p + "attn_q.weight", [dim, dim], types["attn_q"])
        w.add_tensor(p + "attn_k.weight", [dim, kv_dim], types["attn_k"])
        w.add_tensor(p + "attn_v.weight", [dim, kv_dim], types["attn_v"])
        w.add_tensor(p + "attn_output.weight", [dim, dim], types["attn_output"])
        w.add_tensor(p + "ffn_norm.weight", [dim], types["ffn_norm"])
        w.add_tensor(p + "ffn_gate.weight", [dim, ff], types["ffn_gate"])
        w.add_tensor(p + "ffn_up.weight", [dim, ff], types["ffn_up"])
        w.add_tensor(p + "ffn_down.weight", [ff, dim], types["ffn_down"])
    w.add_tensor("output_norm.weight", [dim], types["output_norm"])
    if not tie:
        w.add_tensor("output.weight", [dim, vocab], types["output"])

    order = ["token_embd.weight"]
    for layer in range(layers):
        p = "blk.%d." % layer
        order.extend([
            p + "attn_norm.weight",
            p + "attn_q.weight",
            p + "attn_k.weight",
            p + "attn_v.weight",
            p + "attn_output.weight",
            p + "ffn_norm.weight",
            p + "ffn_gate.weight",
            p + "ffn_up.weight",
            p + "ffn_down.weight",
        ])
    order.append("output_norm.weight")
    if not tie:
        order.append("output.weight")

    return w, order
