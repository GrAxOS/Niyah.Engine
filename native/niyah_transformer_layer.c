#include "niyah.h"

#include <stdlib.h>
#include <string.h>

/* Was a stub. Pre-norm transformer block, both a weightless variant (for the
 * existing state-struct API) and the real weighted decode step. */

static size_t niyah_kv_slot(const NiyahKVCache* cache,
                            int32_t layer,
                            int32_t kv_head,
                            int32_t pos)
{
    const size_t per_head = (size_t)cache->max_seq * (size_t)cache->head_dim;
    const size_t head_index =
        (size_t)layer * (size_t)cache->n_kv_head + (size_t)kv_head;
    return head_index * per_head + (size_t)pos * (size_t)cache->head_dim;
}

void niyah_transformer_layer_forward(NiyahTransformerLayerState* state,
                                     const float* x)
{
    if (!state || !x || !state->attn_out || !state->ffn_out ||
        !state->norm1_out || !state->norm2_out) {
        return;
    }

    const int32_t batch = state->batch > 0 ? state->batch : 1;
    const int32_t seq = state->seq;
    const int32_t dim = state->dim;
    const int32_t n_head = state->n_head > 0 ? state->n_head : 1;

    if (seq <= 0 || dim <= 0 || dim % n_head != 0) {
        return;
    }

    const size_t rows = (size_t)batch * (size_t)seq;
    const size_t span = rows * (size_t)dim;

    /* 1. Pre-attention RMSNorm, per token row. */
    for (size_t r = 0; r < rows; ++r) {
        niyah_rmsnorm_to(state->norm1_out + r * (size_t)dim,
                         x + r * (size_t)dim,
                         NULL, dim, 1e-5f);
    }

    /* 2. Projection-free causal self-attention. */
    float* qkv = (float*)malloc(3u * span * sizeof(float));
    if (!qkv) {
        return;
    }

    NiyahAttentionState attn;
    attn.qkv = qkv;
    attn.out = NULL;
    attn.batch = batch;
    attn.seq = seq;
    attn.dim = dim;
    attn.n_head = n_head;

    niyah_attention_forward(&attn, state->norm1_out, state->attn_out);
    free(qkv);

    /* 3. Residual. */
    for (size_t i = 0; i < span; ++i) {
        state->attn_out[i] += x[i];
    }

    /* 4. Pre-FFN RMSNorm. */
    for (size_t r = 0; r < rows; ++r) {
        niyah_rmsnorm_to(state->norm2_out + r * (size_t)dim,
                         state->attn_out + r * (size_t)dim,
                         NULL, dim, 1e-5f);
    }

    /* 5. Weightless SwiGLU + residual. */
    niyah_swiglu_to(state->ffn_out, state->norm2_out, state->norm2_out,
                    (int32_t)span);
    for (size_t i = 0; i < span; ++i) {
        state->ffn_out[i] += state->attn_out[i];
    }
}

size_t niyah_transformer_scratch_floats(const NiyahModelConfig* config)
{
    if (!config) {
        return 0;
    }

    NiyahModelConfig c = *config;
    niyah_model_config_normalize(&c);

    if (c.n_head <= 0 || c.n_embd <= 0 || c.n_embd % c.n_head != 0) {
        return 0;
    }

    const size_t dim = (size_t)c.n_embd;
    const size_t head_dim = dim / (size_t)c.n_head;
    const size_t kv_dim = (size_t)c.n_kv_head * head_dim;
    const size_t ff = (size_t)c.n_ff;
    const size_t ctx = (size_t)(c.n_ctx > 0 ? c.n_ctx : 1);

    /* norm + tmp + q + k + v + attn + gate + up + scores */
    return dim + dim + dim + kv_dim + kv_dim + dim + ff + ff + ctx;
}

NiyahStatus niyah_transformer_layer_forward_weighted(
    float* hidden,
    const NiyahLayerWeights* w,
    const NiyahModelConfig* config,
    NiyahKVCache* cache,
    int32_t layer,
    int32_t position,
    float* scratch)
{
    if (!hidden || !w || !config || !cache || !scratch) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahModelConfig c = *config;
    niyah_model_config_normalize(&c);

    if (c.n_embd <= 0 || c.n_head <= 0 || c.n_embd % c.n_head != 0) {
        return NIYAH_ERR_SHAPE;
    }
    if (c.n_head % c.n_kv_head != 0) {
        return NIYAH_ERR_SHAPE;
    }

    const int32_t dim = c.n_embd;
    const int32_t head_dim = dim / c.n_head;
    const int32_t kv_dim = c.n_kv_head * head_dim;
    const int32_t ff = c.n_ff;

    if (cache->head_dim != head_dim || cache->n_kv_head != c.n_kv_head) {
        return NIYAH_ERR_SHAPE;
    }
    if (position < 0 || position >= cache->max_seq) {
        return NIYAH_ERR_INVALID_ARG;
    }

    /* Carve the caller-provided scratch. */
    float* norm   = scratch;
    float* tmp    = norm + dim;
    float* q      = tmp + dim;
    float* k      = q + dim;
    float* v      = k + kv_dim;
    float* attn   = v + kv_dim;
    float* gate   = attn + dim;
    float* up     = gate + ff;
    float* scores = up + ff;

    /* --- Attention ------------------------------------------------------- */
    niyah_rmsnorm_to(norm, hidden, w->attn_norm, dim, c.norm_eps);

    niyah_matvec(q, w->wq, norm, dim, dim);
    niyah_matvec(k, w->wk, norm, kv_dim, dim);
    niyah_matvec(v, w->wv, norm, kv_dim, dim);

    niyah_rope_forward_ex(q, 1, dim, c.n_head, position, c.rope_theta);
    niyah_rope_forward_ex(k, 1, kv_dim, c.n_kv_head, position, c.rope_theta);

    for (int32_t h = 0; h < c.n_kv_head; ++h) {
        const size_t slot = niyah_kv_slot(cache, layer, h, position);
        memcpy(cache->k + slot,
               k + (size_t)h * (size_t)head_dim,
               (size_t)head_dim * sizeof(float));
        memcpy(cache->v + slot,
               v + (size_t)h * (size_t)head_dim,
               (size_t)head_dim * sizeof(float));
    }

    const NiyahStatus st =
        niyah_attention_decode(attn, q, cache, layer, c.n_head, position, scores);
    if (st != NIYAH_OK) {
        return st;
    }

    niyah_matvec(tmp, w->wo, attn, dim, dim);
    niyah_add_inplace(hidden, tmp, dim);

    /* --- Feed-forward ---------------------------------------------------- */
    niyah_rmsnorm_to(norm, hidden, w->ffn_norm, dim, c.norm_eps);

    niyah_matvec(gate, w->ffn_gate, norm, ff, dim);
    niyah_matvec(up, w->ffn_up, norm, ff, dim);
    niyah_swiglu_to(up, up, gate, ff);

    niyah_matvec(tmp, w->ffn_down, up, dim, ff);
    niyah_add_inplace(hidden, tmp, dim);

    return NIYAH_OK;
}
