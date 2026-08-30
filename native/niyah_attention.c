#include "niyah.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was: `// Attention stubs / TODO: Implement when types are available`.
 *
 * Causal scaled dot-product attention.
 *
 * Tensor layout for the batch path: q/k/v/out are [batch][seq][dim] with
 * dim == n_head * head_dim and heads interleaved inside each row.
 */

static size_t niyah_kv_offset(const NiyahKVCache* cache,
                              int32_t layer,
                              int32_t kv_head,
                              int32_t pos)
{
    const size_t per_head = (size_t)cache->max_seq * (size_t)cache->head_dim;
    const size_t head_index =
        (size_t)layer * (size_t)cache->n_kv_head + (size_t)kv_head;
    return head_index * per_head + (size_t)pos * (size_t)cache->head_dim;
}

void niyah_multihead_attention_forward(NiyahMultiHeadAttentionState* state)
{
    if (!state || !state->q || !state->k || !state->v || !state->out) {
        return;
    }

    const int32_t batch = state->batch > 0 ? state->batch : 1;
    const int32_t seq = state->seq;
    const int32_t dim = state->dim;
    const int32_t n_head = state->n_head > 0 ? state->n_head : 1;

    if (seq <= 0 || dim <= 0 || dim % n_head != 0) {
        return;
    }

    const int32_t head_dim = dim / n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float* scores = (float*)malloc((size_t)seq * sizeof(float));
    if (!scores) {
        return;
    }

    for (int32_t b = 0; b < batch; ++b) {
        const size_t base = (size_t)b * (size_t)seq * (size_t)dim;
        const float* q_b = state->q + base;
        const float* k_b = state->k + base;
        const float* v_b = state->v + base;
        float* out_b = state->out + base;

        for (int32_t h = 0; h < n_head; ++h) {
            const size_t head_off = (size_t)h * (size_t)head_dim;

            for (int32_t i = 0; i < seq; ++i) {
                const float* q_vec = q_b + (size_t)i * (size_t)dim + head_off;

                /* Causal mask: position i may only attend to j <= i. */
                for (int32_t j = 0; j <= i; ++j) {
                    const float* k_vec = k_b + (size_t)j * (size_t)dim + head_off;
                    float dot = 0.0f;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    scores[j] = dot * scale;
                }

                niyah_softmax(scores, i + 1);

                float* out_vec = out_b + (size_t)i * (size_t)dim + head_off;
                memset(out_vec, 0, (size_t)head_dim * sizeof(float));

                for (int32_t j = 0; j <= i; ++j) {
                    const float w = scores[j];
                    if (w == 0.0f) {
                        continue;
                    }
                    const float* v_vec = v_b + (size_t)j * (size_t)dim + head_off;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        out_vec[d] += w * v_vec[d];
                    }
                }
            }
        }
    }

    free(scores);
}

void niyah_attention_forward(NiyahAttentionState* state,
                             const float* x,
                             float* out)
{
    if (!state || !x || !out) {
        return;
    }

    const int32_t batch = state->batch > 0 ? state->batch : 1;
    const int32_t seq = state->seq;
    const int32_t dim = state->dim;
    const int32_t n_head = state->n_head > 0 ? state->n_head : 1;

    if (seq <= 0 || dim <= 0 || dim % n_head != 0 || !state->qkv) {
        return;
    }

    /*
     * There are no projection weights in NiyahAttentionState, so this entry
     * point runs projection-free self-attention (Q = K = V = x) using
     * state->qkv as the packed scratch buffer. Callers that need learned
     * projections should use niyah_transformer_layer_forward_weighted.
     */
    const size_t span = (size_t)batch * (size_t)seq * (size_t)dim;

    float* q = state->qkv;
    float* k = state->qkv + span;
    float* v = state->qkv + 2u * span;

    memcpy(q, x, span * sizeof(float));
    memcpy(k, x, span * sizeof(float));
    memcpy(v, x, span * sizeof(float));

    NiyahMultiHeadAttentionState mha;
    mha.q = q;
    mha.k = k;
    mha.v = v;
    mha.out = out;
    mha.batch = batch;
    mha.seq = seq;
    mha.dim = dim;
    mha.n_head = n_head;

    niyah_multihead_attention_forward(&mha);

    if (state->out && state->out != out) {
        memcpy(state->out, out, span * sizeof(float));
    }
}

/* ==========================================================================
 * KV cache
 * ========================================================================== */

NiyahStatus niyah_kv_cache_init(NiyahKVCache* cache,
                                int32_t n_layer,
                                int32_t n_kv_head,
                                int32_t head_dim,
                                int32_t max_seq)
{
    if (!cache || n_layer <= 0 || n_kv_head <= 0 ||
        head_dim <= 0 || max_seq <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(cache, 0, sizeof(*cache));

    const size_t count = (size_t)n_layer * (size_t)n_kv_head *
                         (size_t)max_seq * (size_t)head_dim;

    /* Guard the multiplication before handing it to calloc. */
    if (count / (size_t)n_layer / (size_t)n_kv_head / (size_t)max_seq !=
        (size_t)head_dim) {
        return NIYAH_ERR_OVERFLOW;
    }

    cache->k = (float*)calloc(count, sizeof(float));
    cache->v = (float*)calloc(count, sizeof(float));
    if (!cache->k || !cache->v) {
        free(cache->k);
        free(cache->v);
        memset(cache, 0, sizeof(*cache));
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    cache->n_layer = n_layer;
    cache->n_kv_head = n_kv_head;
    cache->head_dim = head_dim;
    cache->max_seq = max_seq;
    cache->length = 0;

    return NIYAH_OK;
}

void niyah_kv_cache_reset(NiyahKVCache* cache)
{
    if (cache) {
        cache->length = 0;
    }
}

void niyah_kv_cache_free(NiyahKVCache* cache)
{
    if (!cache) {
        return;
    }
    free(cache->k);
    free(cache->v);
    memset(cache, 0, sizeof(*cache));
}

NiyahStatus niyah_attention_decode(float* out,
                                   const float* q,
                                   NiyahKVCache* cache,
                                   int32_t layer,
                                   int32_t n_head,
                                   int32_t position,
                                   float* scratch)
{
    if (!out || !q || !cache || !cache->k || !cache->v || !scratch) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (layer < 0 || layer >= cache->n_layer ||
        position < 0 || position >= cache->max_seq ||
        n_head <= 0 || n_head % cache->n_kv_head != 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    const int32_t head_dim = cache->head_dim;
    const int32_t group = n_head / cache->n_kv_head; /* GQA sharing factor */
    const int32_t n_ctx = position + 1;
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int32_t h = 0; h < n_head; ++h) {
        const int32_t kv_head = h / group;
        const float* q_vec = q + (size_t)h * (size_t)head_dim;

        for (int32_t t = 0; t < n_ctx; ++t) {
            const float* k_vec =
                cache->k + niyah_kv_offset(cache, layer, kv_head, t);
            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            scratch[t] = dot * scale;
        }

        niyah_softmax(scratch, n_ctx);

        float* out_vec = out + (size_t)h * (size_t)head_dim;
        memset(out_vec, 0, (size_t)head_dim * sizeof(float));

        for (int32_t t = 0; t < n_ctx; ++t) {
            const float w = scratch[t];
            if (w == 0.0f) {
                continue;
            }
            const float* v_vec =
                cache->v + niyah_kv_offset(cache, layer, kv_head, t);
            for (int32_t d = 0; d < head_dim; ++d) {
                out_vec[d] += w * v_vec[d];
            }
        }
    }

    if (cache->length < n_ctx) {
        cache->length = n_ctx;
    }

    return NIYAH_OK;
}
