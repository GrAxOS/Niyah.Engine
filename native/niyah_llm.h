#ifndef NIYAH_LLM_H
#define NIYAH_LLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_LLM_DTYPE_F32 = 1,
    NIYAH_LLM_DTYPE_F16 = 2
} NiyahLlmDType;

typedef struct {
    NiyahLlmDType dtype;
    uint32_t rank;
    uint32_t shape[4];
} NiyahLlmTensorSpec;

typedef struct {
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embedding_dim;
    uint32_t layer_count;
    uint32_t attention_heads;
    uint32_t kv_heads;
    uint32_t ffn_dim;
} NiyahLlmConfig;

bool niyah_llm_config_validate(const NiyahLlmConfig *config);
bool niyah_llm_tensor_spec_validate(const NiyahLlmTensorSpec *spec);
bool niyah_llm_tensor_element_count(const NiyahLlmTensorSpec *spec, size_t *out);
bool niyah_llm_tensor_byte_count(const NiyahLlmTensorSpec *spec, size_t *out);

#ifdef __cplusplus
}
#endif

#endif
