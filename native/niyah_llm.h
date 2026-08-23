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

typedef struct {
    const char **vocabulary;
    uint32_t vocabulary_size;
    uint32_t max_token_bytes;
} NiyahLlmWordPieceVocab;

typedef struct {
    const char **vocabulary;
    uint32_t vocabulary_size;
    const uint32_t *merge_left;
    const uint32_t *merge_right;
    uint32_t merge_count;
    uint32_t max_token_bytes;
} NiyahLlmBpeModel;

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    const float *o;
    const float *attention_norm;
    const float *ffn_gate;
    const float *ffn_up;
    const float *ffn_down;
    const float *ffn_norm;
} NiyahLlmLayerWeights;

typedef struct {
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t layer_count;
    uint32_t kv_dim;
    const float *embedding;
    const float *lm_head;
    const NiyahLlmLayerWeights *layers;
} NiyahLlmModelWeights;

typedef struct {
    void *mapping;
    size_t mapping_size;
    NiyahLlmModelWeights weights;
    NiyahLlmLayerWeights *layers;
    uint32_t *vocabulary_offsets;
    char *vocabulary_data;
    uint32_t vocabulary_size;
    uint32_t max_token_bytes;
} NiyahLlmLoadedWeights;

typedef struct {
    NiyahLlmConfig config;
    NiyahLlmModelWeights weights;
    NiyahLlmKvCache *kv_caches;
    float *hidden;
    float *normalized;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *ffn;
    float *ffn_scratch;
    float *logits;
    size_t layer_stride;
    uint32_t position;
} NiyahLlmGenerationState;

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

bool niyah_llm_wordpiece_tokenize(const char *text,
                                  const NiyahLlmWordPieceVocab *model,
                                  uint32_t *token_ids,
                                  size_t token_capacity,
                                  size_t *token_count);

bool niyah_llm_bpe_tokenize(const char *text,
                            const NiyahLlmBpeModel *model,
                            uint32_t *token_ids,
                            size_t token_capacity,
                            size_t *token_count);

void niyah_llm_logits_softmax(float *logits, uint32_t count);
bool niyah_llm_sample(const float *logits,
                      uint32_t count,
                      const NiyahLlmSamplerConfig *config,
                      NiyahLlmSample *out);

bool niyah_llm_weights_load_from_buffer(const void *buffer,
                                        size_t buffer_size,
                                        const NiyahLlmConfig *expected_config,
                                        NiyahLlmLoadedWeights *out);
void niyah_llm_weights_unload(NiyahLlmLoadedWeights *weights);
const NiyahLlmModelWeights *niyah_llm_weights_view(const NiyahLlmLoadedWeights *weights);

bool niyah_llm_generation_init(NiyahLlmGenerationState *state,
                               const NiyahLlmConfig *config,
                               const NiyahLlmModelWeights *weights);
void niyah_llm_generation_free(NiyahLlmGenerationState *state);
bool niyah_llm_generation_step(NiyahLlmGenerationState *state,
                               uint32_t token_id,
                               const NiyahLlmSamplerConfig *sampler,
                               uint32_t *next_token_id,
                               float *probability);

#ifdef __cplusplus
}
#endif

#endif
