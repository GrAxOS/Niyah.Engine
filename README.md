# Niyah.Engine

**Local-first search + reasoning engine** with evidence-first philosophy.

## Quick Start

### 1. Build native library

```bash
cd native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 2. Download and convert model weights

```bash
# Download Qwen2.5-0.5B-Instruct GGUF and convert to native format
bash tools/download_and_convert.sh
```

This will:
- Download `qwen2.5-0.5b-instruct-q4_k_m.gguf` (~500MB)
- Convert to `weights/qwen2.5-0.5b-f32.bin` (~2GB float32)
- Generate `weights/config.json`

### 3. Run generation test

```bash
cd native/build
ctest -R test_niyah_generation --verbose
```

Expected output:
```
=== Test: Generation with real weights ===
Config validated
Loaded 2147483648 bytes of weights
Weights loaded
Tokenized prompt: "Hello, my name is" -> 5 tokens
Generation initialized
Generating...
  [0] token=12345, prob=0.8234, text=" Sulaiman"
  [1] token=67890, prob=0.7654, text=","
  ...
=== Test passed ===
```

## Architecture

```
native/          # C11 core (LLM, search, bridge)
neutral/         # Python training (QLoRA, LVU, MMR)
ui/Niyah.App/    # C# WPF desktop
search/          # C++ search engine
tools/           # GGUF converter, build scripts
```

## Features

- **BM25 search** with title/URL boosts
- **Transformer primitives** (attention, RoPE, SwiGLU, RMSNorm)
- **QLoRA fine-tuning** (neutral pipeline)
- **LVU consistency** + **Peer prediction** (reduce hallucination)
- **MMR audit log** (append-only cryptographic audit)
- **Production code training** (Linux, PostgreSQL, Nginx)

## License

Apache 2.0

## Acknowledgments

- Qwen Team (base model)
- Linux kernel, PostgreSQL, Nginx (production code)
- Sulaiman Alshammari (philosophy + MMR audit design)
