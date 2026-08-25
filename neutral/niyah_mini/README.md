# NiyahMini - Original Lightweight Transformer

**NiyahMini** is a from-scratch transformer model designed specifically for Niyah.Engine. It is:

- **Original**: No borrowed code, weights, or architectures from Llama, Qwen, Mistral, or any other model
- **Lightweight**: Small parameter counts for local training
- **Arabic-centric**: Optimized for Arabic, English, code, and mixed content
- **Evidence-aware**: Designed to work with Niyah's epistemic framework
- **Deterministic**: Reproducible behavior
- **Memory-efficient**: C11 implementation with careful memory management

## Architecture

### Model Design

NiyahMini uses a standard transformer architecture with the following features:

- **Multi-head self-attention** with Rotary Position Embedding (RoPE)
- **Grouped-Query Attention (GQA)** for efficiency
- **Pre-norm architecture** (RMSNorm before each sub-layer)
- **SwiGLU activation** in feed-forward networks
- **Residual connections**
- **Tied word embeddings** (optional)

### Variants

| Variant | Layers | Dim | Heads | FF Dim | Vocab | Params | Context |
|---------|--------|-----|-------|--------|-------|--------|---------|
| TINY | 4 | 128 | 4 | 512 | 8K | ~1M | 512 |
| SMALL | 8 | 256 | 8 | 1024 | 16K | ~4M | 1024 |
| BASE | 12 | 512 | 8 | 2048 | 32K | ~12M | 2048 |
| MEDIUM | 16 | 768 | 12 | 3072 | 64K | ~36M | 4096 |

## File Structure

```
native/niyah_mini/
├── niyah_mini_config.h    # Configuration and presets
├── niyah_mini_config.c    # Configuration validation
├── niyah_mini_vocab.h     # Vocabulary and BPE tokenizer
├── niyah_mini_vocab.c     # Vocabulary implementation
├── niyah_mini_model.h     # Model architecture
├── niyah_mini_model.c     # Model implementation
├── niyah_mini_bridge.h    # Integration with Niyah.Engine
├── niyah_mini_bridge.c    # Bridge implementation
├── Makefile               # Build configuration
└── test_*.c              # Unit tests

tools/niyah_mini/
├── build_arabic_vocab.py  # Vocabulary builder
└── gguf_to_niyah_mini.py  # Weight converter (future)

neutral/niyah_mini/
├── corpus_cleaner.py      # Provenance-preserving corpus cleaning
├── dataset_builder.py    # Training data preparation
├── dataset_validator.py  # Dataset validation
├── train.py              # Training loop
├── evaluate.py           # Evaluation framework
└── inference.py          # Inference engine
```

## Building

### Build NiyahMini Library

```bash
# Using CMake
cmake -S native/niyah_mini -B build/niyah_mini -DCMAKE_BUILD_TYPE=Release
cmake --build build/niyah_mini

# Using Make
cd native/niyah_mini
make lib

# Run tests
make test
```

### Build Vocabulary

```bash
# Build Arabic-centric vocabulary
python tools/niyah_mini/build_arabic_vocab.py \
    --output-dir data/vocab \
    --vocab-size 32768 \
    --n-merges 10000
```

## Training Data Preparation

### 1. Clean Corpus

```bash
python neutral/niyah_mini/corpus_cleaner.py \
    --input raw_data/ \
    --output data/corpus.jsonl \
    --manifest data/corpus_manifest.json \
    --source-name "My Data Source" \
    --source-url "https://example.com" \
    --license "CC-BY"
```

### 2. Build Dataset

```bash
python neutral/niyah_mini/dataset_builder.py \
    --corpus data/corpus.jsonl \
    --output-dir data/dataset
```

### 3. Validate Dataset

```bash
python neutral/niyah_mini/dataset_validator.py data/dataset
```

## Training

```bash
python neutral/niyah_mini/train.py \
    --config configs/niyah_mini_base.json \
    --corpus data/corpus.jsonl \
    --output-dir models/niyah_mini_base \
    --epochs 10 \
    --batch-size 16
```

## Inference

```bash
python neutral/niyah_mini/inference.py \
    --model-dir models/niyah_mini_base \
    --prompt "What is the capital of France?" \
    --temperature 0.7 \
    --evidence
```

## Evaluation

```bash
python neutral/niyah_mini/evaluate.py \
    --model-dir models/niyah_mini_base \
    --data-dir data/dataset
```

## Integration with Niyah.Engine

NiyahMini can be used with the existing Niyah.Engine infrastructure:

```c
#include "niyah.h"
#include "niyah_mini/niyah_mini_bridge.h"

int main(void) {
    NiyahMiniWrappedModel wrapped;
    NiyahMiniConfig config;
    
    niyah_mini_config_init(&config, NIYAH_MINI_BASE);
    niyah_mini_wrapped_init(&wrapped, &config);
    
    /* Load weights */
    niyah_mini_wrapped_load(&wrapped, "config.json", "weights.bin");
    
    /* Create tokenizer */
    NiyahTokenizer tokenizer;
    niyah_tokenizer_load(&tokenizer, "vocab.txt");
    
    /* Generate */
    NiyahLLMOutput output = niyah_mini_wrapped_generate(
        &wrapped, &tokenizer, "Hello", 50);
    
    if (output.status == NIYAH_OK) {
        printf("%s\n", output.text);
        niyah_llm_output_free(&output);
    } else if (output.status == NIYAH_ERR_NO_WEIGHTS) {
        /* Critical: no output text was fabricated */
        assert(output.text == NULL);
    }
    
    /* Cleanup */
    niyah_mini_wrapped_free(&wrapped);
    niyah_tokenizer_free(&tokenizer);
    
    return 0;
}
```

## Critical Contracts Preserved

NiyahMini **strictly preserves** all critical Niyah.Engine contracts:

1. **No Fabrication**: When weights are not loaded, `niyah_mini_wrapped_generate` returns:
   - `status = NIYAH_ERR_NO_WEIGHTS`
   - `text = NULL`

2. **Evidence Labels**: Output can be formatted with FACT/INFERENCE/UNKNOWN/CONFLICTED labels

3. **Provenance**: All training data includes source_url, license, and SHA-256 hashes

4. **Deterministic**: With fixed seeds, training and inference are reproducible

## Arabic Support

NiyahMini has special support for Arabic text:

- **Arabic characters**: Full Unicode support for Arabic script
- **Arabic tokenization**: BPE tokenizer trained on Arabic text
- **Mixed content**: Handles Arabic-English code-switching
- **RTL support**: Proper handling of right-to-left text

## Provenance Tracking

All data and models maintain full provenance:

- **Corpus**: Each document has source_url, license, SHA-256, retrieved_at_utc
- **Dataset**: Each example tracks its source document
- **Checkpoints**: Each checkpoint includes manifest with SHA-256 of all files
- **Inference**: Each generation includes input/output hashes and parameters

## License

All code in NiyahMini is original and follows the same license as Niyah.Engine.

## Known Limitations

- GPU support: Not yet implemented (CPU-only)
- Quantization: Not yet implemented (float32 only)
- Batched inference: Single-sample only
- Training: Python prototype (C11 training loop in development)

## Future Work

- C11 training loop implementation
- GPU backend (CUDA/HIP/Metal)
- Quantization support (INT8, INT4)
- Batched inference
- Distributed training
- More efficient attention kernels
