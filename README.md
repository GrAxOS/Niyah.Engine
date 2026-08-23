# Niyah Engine

A lightweight neural inference engine with knowledge graph integration.

## 📁 Project Structure

```
Niyah.Engine/
├── native/                 # C implementation
│   ├── niyah.h            # Main header (unified definitions)
│   ├── niyah_bridge.h     # Bridge API
│   ├── niyah_llm.h        # LLM generation
│   ├── niyah_runtime.h    # Runtime management
│   ├── niyah_tokenizer.h  # Tokenization
│   ├── niyah_telemetry.h  # Performance metrics
│   ├── niyah_attention.h  # Attention mechanisms
│   ├── niyah_transformer_layer.h  # Transformer layers
│   ├── *.c                # Implementation files
│   ├── CMakeLists.txt     # CMake build configuration
│   └── Makefile           # Make build configuration
├── tests/                  # Test files
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- GCC or Clang
- CMake 3.10+ (optional, for CMake build)
- Make (optional, for Make build)

### Build with CMake

```bash
cd native
mkdir build && cd build
cmake ..
make
```

### Build with Make

```bash
cd native
make clean
make
```

### Run Tests

```bash
cd native

# Run all tests
make test

# Or run individual tests
./test_niyah_core
./niyah_attention_test
./niyah_llm_test
# ... etc
```

## 📖 Usage

### Basic Example

```c
#include "niyah.h"

int main() {
    // Initialize model
    NiyahModelConfig config = {
        .n_vocab = 50257,
        .n_embd = 768,
        .n_head = 12,
        .n_layer = 12,
        .n_ctx = 1024
    };
    
    NiyahModel model = {
        .config = config,
        .weights = NULL,
        .weights_size = 0
    };
    
    // Initialize tokenizer
    NiyahTokenizer tokenizer = {0};
    
    // Initialize runtime
    NiyahRuntimeConfig runtime_config = {
        .memory_pool = NULL,
        .memory_size = 1024 * 1024 * 512,
        .device_id = 0,
        .use_gpu = false
    };
    
    NiyahRuntime runtime = {
        .config = runtime_config,
        .context = NULL
    };
    
    // Initialize sampler
    NiyahSamplerConfig sampler = {
        .strategy = NIYAH_SAMPLE_TOP_K,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f
    };
    
    // Create LLM
    NiyahLLM llm = {
        .model = model,
        .tokenizer = tokenizer,
        .runtime = runtime,
        .sampler = sampler
    };
    
    // Generate text
    const char* prompt = "Once upon a time";
    NiyahLLMOutput output = niyah_llm_generate(&llm, prompt, 100);
    
    printf("%s\n", output.text);
    
    // Cleanup
    free(output.text);
    
    return 0;
}
```

### Using the Bridge

```c
#include "niyah.h"
#include "niyah_bridge.h"

int main() {
    // Initialize LLM (see example above)
    NiyahLLM llm = {0};
    
    // Create bridge context
    NiyahBridgeContext* ctx = niyah_bridge_create(&llm);
    
    // Use bridge for knowledge-aware generation
    // ...
    
    // Cleanup
    niyah_bridge_destroy(ctx);
    
    return 0;
}
```

## 🔧 API Reference

### Core Types

- `NiyahModel` - Model configuration and weights
- `NiyahTokenizer` - Tokenization/detokenization
- `NiyahRuntime` - Runtime configuration
- `NiyahLLM` - Complete LLM instance
- `NiyahLLMOutput` - Generation output

### Key Functions

#### Generation
```c
NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm, const char* prompt, int32_t max_tokens);
```

#### Tokenization
```c
int32_t niyah_tokenize(NiyahTokenizer* tokenizer, const char* text, int32_t* tokens, int32_t max_tokens);
char* niyah_detokenize(NiyahTokenizer* tokenizer, const int32_t* tokens, int32_t n_tokens);
```

#### Bridge
```c
NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm);
void niyah_bridge_destroy(NiyahBridgeContext* ctx);
```

## 📊 Features

- ✅ Lightweight C implementation
- ✅ Knowledge graph integration
- ✅ Multiple sampling strategies (greedy, top-k, top-p, temperature)
- ✅ Transformer architecture
- ✅ Multi-head attention
- ✅ RoPE positional encoding
- ✅ SwiGLU activation
- ✅ RMSNorm normalization
- ✅ Telemetry and performance metrics

## 🛠️ Development

### Project Organization

All core definitions are in `native/niyah.h`. Specialized modules have their own headers:

- `niyah_bridge.h` - Bridge between LLM and knowledge graph
- `niyah_llm.h` - LLM generation API
- `niyah_runtime.h` - Runtime management
- `niyah_tokenizer.h` - Tokenization
- `niyah_telemetry.h` - Performance monitoring
- `niyah_attention.h` - Attention mechanisms
- `niyah_transformer_layer.h` - Transformer layers

### Build System

The project supports both CMake and Make:

**CMake** (recommended):
- Better dependency management
- Easier cross-platform builds
- Automatic test discovery

**Make**:
- Simpler for quick builds
- No CMake dependency
- Direct control over compilation

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
