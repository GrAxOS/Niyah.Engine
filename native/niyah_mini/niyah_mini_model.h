#ifndef NIYAH_MINI_MODEL_H
#define NIYAH_MINI_MODEL_H

#include "../niyah.h"
#include "niyah_mini_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *attn_norm;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *ffn_norm;
    float *ffn_gate;
    float *ffn_up;
    float *ffn_down;
} NiyahMiniLayerWeights;

typedef struct {
    float *embedding;
    NiyahMiniLayerWeights *layers;
    int32_t n_layers;
    float *final_norm;
    float *lm_head;
    void *memory_block;
    size_t memory_size;
    bool owns_memory;
} NiyahMiniWeights;

typedef struct {
    NiyahMiniConfig config;
    NiyahMiniWeights weights;
    void *runtime;
    float *kv_cache_k;
    float *kv_cache_v;
    int32_t kv_cache_seq_len;
    float *scratch;
    size_t scratch_size;
} NiyahMiniModel;

typedef struct {
    float *hidden;
    float *norm1;
    float *attn_out;
    float *norm2;
    float *ffn_out;
    float *q;
    float *k;
    float *v;
    float *attn_scores;
    float *attn_probs;
    float *ffn_gate_out;
    float *ffn_up_out;
    float *layer_out;
    void *memory_block;
    size_t memory_size;
} NiyahMiniForwardState;

NIYAH_API NiyahStatus niyah_mini_model_init(NiyahMiniModel *model, const NiyahMiniConfig *config);
NIYAH_API NiyahStatus niyah_mini_model_load_weights(NiyahMiniModel *model, const char *weights_path);
NIYAH_API NiyahStatus niyah_mini_model_save_weights(const NiyahMiniModel *model, const char *weights_path);
NIYAH_API NiyahStatus niyah_mini_model_load(NiyahMiniModel *model, const char *config_path, const char *weights_path);
NIYAH_API NiyahStatus niyah_mini_model_save(const NiyahMiniModel *model, const char *config_path, const char *weights_path);
NIYAH_API void niyah_mini_model_free(NiyahMiniModel *model);

NIYAH_API NiyahStatus niyah_mini_forward_state_init(NiyahMiniForwardState *state, const NiyahMiniConfig *config, int32_t max_seq_len);
NIYAH_API void niyah_mini_forward_state_free(NiyahMiniForwardState *state);
NIYAH_API NiyahStatus niyah_mini_forward_token(NiyahMiniModel *model, NiyahMiniForwardState *state, int32_t token_id, int32_t position, float *logits_out);
NIYAH_API NiyahStatus niyah_mini_forward_sequence(NiyahMiniModel *model, NiyahMiniForwardState *state, const int32_t *input_ids, int32_t seq_len, float *logits_out);

NIYAH_API void niyah_mini_weights_init_xavier(NiyahMiniWeights *weights, const NiyahMiniConfig *config);
NIYAH_API void niyah_mini_weights_init_small(NiyahMiniWeights *weights, const NiyahMiniConfig *config, float scale);
NIYAH_API NiyahStatus niyah_mini_weights_copy(NiyahMiniWeights *dst, const NiyahMiniWeights *src, const NiyahMiniConfig *config);
NIYAH_API void niyah_mini_weights_scale(NiyahMiniWeights *weights, float scale);

NIYAH_API NiyahStatus niyah_mini_generate(NiyahMiniModel *model, const int32_t *prompt_ids, int32_t prompt_len, int32_t max_tokens, float temperature, int32_t *output_ids, int32_t *output_len);
NIYAH_API NiyahStatus niyah_mini_get_logits(NiyahMiniModel *model, const int32_t *input_ids, int32_t seq_len, float *logits);
NIYAH_API void niyah_mini_reset_kv_cache(NiyahMiniModel *model);

NIYAH_API NiyahStatus niyah_mini_weights_allocate(NiyahMiniWeights *weights, const NiyahMiniConfig *config);
NIYAH_API void niyah_mini_weights_free(NiyahMiniWeights *weights);
NIYAH_API size_t niyah_mini_weights_memory_size(const NiyahMiniConfig *config);
NIYAH_API size_t niyah_mini_forward_state_memory_size(const NiyahMiniConfig *config, int32_t max_seq_len);

#ifdef __cplusplus
}
#endif

#endif
