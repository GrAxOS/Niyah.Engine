#ifndef NIYAH_MINI_TRAIN_H
#define NIYAH_MINI_TRAIN_H

#include "niyah_mini_model.h"

/* NiyahMini native training (P0). Self-contained standard multi-head
 * attention training path mirroring the numerically-verified reference.
 * Does not modify the inference forward in niyah_mini_model.c. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* attn_norm;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* ffn_norm;
    float* ffn_gate;
    float* ffn_up;
    float* ffn_down;
} NiyahMiniLayerGrads;

typedef struct {
    float* embedding;
    NiyahMiniLayerGrads* layers;
    float* final_norm;
    void* memory_block;
    size_t memory_size;
} NiyahMiniGrads;

typedef struct {
    float* m;
    float* v;
    int64_t t;
    size_t n;
} NiyahMiniOptimizerState;

NIYAH_API NiyahStatus niyah_mini_grads_allocate(NiyahMiniGrads* grads,
                                                const NiyahMiniConfig* config);
NIYAH_API void niyah_mini_grads_free(NiyahMiniGrads* grads);
NIYAH_API void niyah_mini_grads_zero(NiyahMiniGrads* grads,
                                     const NiyahMiniConfig* config);

NIYAH_API NiyahStatus niyah_mini_optim_init(NiyahMiniOptimizerState* opt,
                                            const NiyahMiniConfig* config);
NIYAH_API void niyah_mini_optim_free(NiyahMiniOptimizerState* opt);

typedef struct {
    float* x; float* h1; float* r1;
    float* q; float* k; float* v;
    float* probs; float* attn_in; float* ao;
    float* res; float* h2; float* r2;
    float* gate; float* up; float* ff; float* fo;
    float* final_pre; float* final_h; float* r_final;
    void* memory_block;
    size_t memory_size;
    int32_t seq_len;
} NiyahMiniTrainCache;

NIYAH_API NiyahStatus niyah_mini_cache_allocate(NiyahMiniTrainCache* cache,
                                                const NiyahMiniConfig* config,
                                                int32_t seq_len);
NIYAH_API void niyah_mini_cache_free(NiyahMiniTrainCache* cache);

NIYAH_API NiyahStatus niyah_mini_train_forward(
    NiyahMiniModel* model, NiyahMiniTrainCache* cache,
    const int32_t* input_ids, int32_t seq_len, float* logits_out);

NIYAH_API float niyah_mini_loss_and_dlogits(
    const float* logits, const int32_t* targets,
    int32_t seq_len, int32_t vocab, float* dlogits_out);

NIYAH_API NiyahStatus niyah_mini_train_backward(
    NiyahMiniModel* model, NiyahMiniGrads* grads, NiyahMiniTrainCache* cache,
    const int32_t* input_ids, const float* dlogits);

NIYAH_API float niyah_mini_clip_grads(NiyahMiniGrads* grads,
                                      const NiyahMiniConfig* config, float max_norm);

NIYAH_API NiyahStatus niyah_mini_step_adamw(
    NiyahMiniModel* model, NiyahMiniGrads* grads,
    NiyahMiniOptimizerState* opt, const NiyahMiniConfig* config,
    float lr, float beta1, float beta2, float eps, float weight_decay);
NIYAH_API NiyahStatus niyah_mini_step_sgd(
    NiyahMiniModel* model, NiyahMiniGrads* grads,
    NiyahMiniOptimizerState* opt, const NiyahMiniConfig* config,
    float lr, float momentum, float weight_decay);

NIYAH_API NiyahStatus niyah_mini_grad_check(
    NiyahMiniModel* model, const NiyahMiniConfig* config,
    const int32_t* input_ids, int32_t seq_len,
    float* max_rel_out, int32_t* n_checked_out);

#ifdef __cplusplus
}
#endif

#endif /* NIYAH_MINI_TRAIN_H */
