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

typedef struct {
    float *data;
    size_t element_count;
    uint32_t rank;
    uint32_t shape[4];
} NiyahLlmTensorF32;

typedef struct {
    uint32_t token_count;
    uint32_t head_count;
    uint32_t head_dim;
    float *keys;
    float *values;
    size_t capacity_elements;
} NiyahLlmKvCache;

typedef struct {
    float temperature;
    uint32_t top_k;
    float top_p;
    uint64_t seed;
} NiyahLlmSamplerConfig;

typedef struct {
    uint32_t token_id;
    float probability;
} NiyahLlmSample;

bool niyah_llm_config_validate(const NiyahLlmConfig *config);
bool niyah_llm_tensor_spec_validate(const NiyahLlmTensorSpec *spec);
bool niyah_llm_tensor_element_count(const NiyahLlmTensorSpec *spec, size_t *out);
bool niyah_llm_tensor_byte_count(const NiyahLlmTensorSpec *spec, size_t *out);

bool niyah_llm_tensor_f32_init(NiyahLlmTensorF32 *tensor,
                               uint32_t rank,
                               const uint32_t shape[4]);
void niyah_llm_tensor_f32_free(NiyahLlmTensorF32 *tensor);
bool niyah_llm_tensor_f32_reshape(NiyahLlmTensorF32 *tensor,
                                  uint32_t rank,
                                  const uint32_t shape[4]);

bool niyah_llm_matmul(const NiyahLlmTensorF32 *a,
                      const NiyahLlmTensorF32 *b,
                      NiyahLlmTensorF32 *out);

bool niyah_llm_apply_rope(float *query,
                          float *key,
                          uint32_t head_dim,
                          uint32_t position,
                          float theta);

bool niyah_llm_attention(const float *query,
                         const float *keys,
                         const float *values,
                         uint32_t query_heads,
                         uint32_t kv_heads,
                         uint32_t head_dim,
                         uint32_t token_count,
                         float *output);

bool niyah_llm_ffn_swiglu(const float *input,
                          const float *gate,
                          const float *up,
                          const float *down,
                          uint32_t input_dim,
                          uint32_t hidden_dim,
                          float *output,
                          float *scratch);

bool niyah_llm_kv_cache_init(NiyahLlmKvCache *cache,
                             uint32_t context_length,
                             uint32_t head_count,
                             uint32_t head_dim);
void niyah_llm_kv_cache_free(NiyahLlmKvCache *cache);
bool niyah_llm_kv_cache_append(NiyahLlmKvCache *cache,
                               const float *keys,
                               const float *values,
                               uint32_t token_count);

void niyah_llm_logits_softmax(float *logits, uint32_t count);
bool niyah_llm_sample(const float *logits,
                      uint32_t count,
                      const NiyahLlmSamplerConfig *config,
                      NiyahLlmSample *out);

#ifdef __cplusplus
}
#endif

#endif
