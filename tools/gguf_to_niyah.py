#!/usr/bin/env python3
"""Convert Qwen2.5 GGUF to Niyah native float32 buffer.

Usage:
    python gguf_to_niyah.py \
        --input qwen2.5-0.5b-instruct-q4_k_m.gguf \
        --output qwen2.5-0.5b-f32.bin \
        --config config.json

Output:
    - qwen2.5-0.5b-f32.bin: Raw float32 weights (NiyahLlmModelWeights layout)
    - config.json: NiyahLlmConfig JSON
"""
import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import gguf
except ImportError:
    print("ERROR: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

def load_gguf(path: Path) -> gguf.GGUFReader:
    return gguf.GGUFReader(path, 'r')

def extract_tensor(reader: gguf.GGUFReader, name: str) -> list:
    """Extract tensor by name and convert to float32 list."""
    for tensor in reader.tensors:
        if tensor.name == name:
            data = tensor.data.astype('float32')
            return data.flatten().tolist()
    raise ValueError(f"Tensor not found: {name}")

def get_config(reader: gguf.GGUFReader) -> dict:
    """Extract model config from GGUF metadata."""
    config = {}
    
    # Vocab size
    for key, value in reader.fields.items():
        if key == 'tokenizer.vocab_size':
            config['vocab_size'] = int(value.parts[0])
        elif key == 'llama.embedding_length':
            config['dim'] = int(value.parts[0])
        elif key == 'llama.block_count':
            config['layer_count'] = int(value.parts[0])
        elif key == 'llama.feed_forward_length':
            config['hidden_dim'] = int(value.parts[0])
        elif key == 'llama.attention.head_count':
            config['heads'] = int(value.parts[0])
        elif key == 'llama.attention.head_count_kv':
            config['kv_heads'] = int(value.parts[0])
        elif key == 'tokenizer.ggml.eos_token_id':
            config['eos_token'] = int(value.parts[0])
    
    # Defaults
    config['context_size'] = 2048
    
    return config

def write_niyah_weights(reader: gguf.GGUFReader, output: Path):
    """Write tensors in NiyahLlmModelWeights layout.
    
    Layout:
    1. embedding (vocab_size * dim)
    2. For each layer:
       - attn_norm (dim)
       - q (heads * head_dim * dim)
       - k (kv_heads * head_dim * dim)
       - v (kv_heads * head_dim * dim)
       - o (dim * heads * head_dim)
       - ffn_norm (dim)
       - ffn_gate (hidden_dim * dim)
       - ffn_up (hidden_dim * dim)
       - ffn_down (dim * hidden_dim)
    3. final_norm (dim)
    4. lm_head (vocab_size * dim)
    """
    config = get_config(reader)
    vocab_size = config['vocab_size']
    dim = config['dim']
    layer_count = config['layer_count']
    hidden_dim = config['hidden_dim']
    heads = config['heads']
    kv_heads = config['kv_heads']
    head_dim = dim // heads
    
    with open(output, 'wb') as f:
        # 1. Embedding
        emb = extract_tensor(reader, 'token_embd.weight')
        f.write(struct.pack(f'{len(emb)}f', *emb))
        print(f"  embedding: {len(emb)} floats")
        
        # 2. Layers
        for layer_idx in range(layer_count):
            prefix = f'blk.{layer_idx}'
            
            # attn_norm
            norm = extract_tensor(reader, f'{prefix}.attn_norm.weight')
            f.write(struct.pack(f'{len(norm)}f', *norm))
            
            # q
            q = extract_tensor(reader, f'{prefix}.attn_q.weight')
            f.write(struct.pack(f'{len(q)}f', *q))
            
            # k
            k = extract_tensor(reader, f'{prefix}.attn_k.weight')
            f.write(struct.pack(f'{len(k)}f', *k))
            
            # v
            v = extract_tensor(reader, f'{prefix}.attn_v.weight')
            f.write(struct.pack(f'{len(v)}f', *v))
            
            # o
            o = extract_tensor(reader, f'{prefix}.attn_output.weight')
            f.write(struct.pack(f'{len(o)}f', *o))
            
            # ffn_norm
            ffn_norm = extract_tensor(reader, f'{prefix}.ffn_norm.weight')
            f.write(struct.pack(f'{len(ffn_norm)}f', *ffn_norm))
            
            # ffn_gate
            ffn_gate = extract_tensor(reader, f'{prefix}.ffn_gate.weight')
            f.write(struct.pack(f'{len(ffn_gate)}f', *ffn_gate))
            
            # ffn_up
            ffn_up = extract_tensor(reader, f'{prefix}.ffn_up.weight')
            f.write(struct.pack(f'{len(ffn_up)}f', *ffn_up))
            
            # ffn_down
            ffn_down = extract_tensor(reader, f'{prefix}.ffn_down.weight')
            f.write(struct.pack(f'{len(ffn_down)}f', *ffn_down))
            
            print(f"  layer {layer_idx}: done")
        
        # 3. final_norm
        final_norm = extract_tensor(reader, 'output_norm.weight')
        f.write(struct.pack(f'{len(final_norm)}f', *final_norm))
        
        # 4. lm_head (usually same as embedding for tied weights)
        lm_head = extract_tensor(reader, 'output.weight')
        f.write(struct.pack(f'{len(lm_head)}f', *lm_head))
        
        total_floats = f.tell() // 4
        print(f"Total: {total_floats} floats ({total_floats * 4 / 1e6:.1f} MB)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    
    print(f"Loading {args.input}...")
    reader = load_gguf(args.input)
    
    print("Extracting config...")
    config = get_config(reader)
    
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {args.config}")
    print(json.dumps(config, indent=2))
    
    print(f"Writing weights to {args.output}...")
    write_niyah_weights(reader, args.output)
    
    print("Done!")

if __name__ == '__main__':
    main()
