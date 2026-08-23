#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Causal scaled dot-product attention ───────────────────────────────── */
void niyah_attention_forward(NiyahAttentionState* state,
                             const float* x, float* out) {
    if (!state || !x || !out) return;

    int32_t seq      = state->seq;
    int32_t dim      = state->dim;
    int32_t n_head   = state->n_head;
    int32_t head_dim = dim / n_head;
    float   scale    = 1.0f / sqrtf((float)head_dim);

    float* scores = (float*)malloc((size_t)seq * sizeof(float));
    if (!scores) return;

    for (int32_t h = 0; h < n_head; h++) {
        for (int32_t i = 0; i < seq; i++) {
            const float* q = x + i * dim + h * head_dim;

            /* Attention scores with causal mask */
            for (int32_t j = 0; j <= i; j++) {
                const float* k = x + j * dim + h * head_dim;
                float dot = 0.0f;
                for (int32_t d = 0; d < head_dim; d++) dot += q[d] * k[d];
                scores[j] = dot * scale;
            }
            for (int32_t j = i + 1; j < seq; j++) scores[j] = -1e9f;

            niyah_softmax(scores, seq);

            float* o = out + i * dim + h * head_dim;
            memset(o, 0, (size_t)head_dim * sizeof(float));
            for (int32_t j = 0; j < seq; j++) {
                const float* v = x + j * dim + h * head_dim;
                for (int32_t d = 0; d < head_dim; d++)
                    o[d] += scores[j] * v[d];
            }
        }
    }

    free(scores);
}

/* ── Multi-head attention with separate Q/K/V buffers ─────────────────── */
void niyah_multihead_attention_forward(NiyahMultiHeadAttentionState* state) {
    if (!state || !state->q || !state->k || !state->v || !state->out) return;

    int32_t batch    = state->batch;
    int32_t seq      = state->seq;
    int32_t dim      = state->dim;
    int32_t n_head   = state->n_head;
    int32_t head_dim = dim / n_head;
    float   scale    = 1.0f / sqrtf((float)head_dim);

    float* scores = (float*)malloc((size_t)seq * sizeof(float));
    if (!scores) return;

    for (int32_t b = 0; b < batch; b++) {
        size_t batch_off = (size_t)b * seq * dim;

        for (int32_t h = 0; h < n_head; h++) {
            for (int32_t i = 0; i < seq; i++) {
                const float* q = state->q + batch_off + i * dim + h * head_dim;

                for (int32_t j = 0; j <= i; j++) {
                    const float* k = state->k + batch_off + j * dim + h * head_dim;
                    float dot = 0.0f;
                    for (int32_t d = 0; d < head_dim; d++) dot += q[d] * k[d];
                    scores[j] = dot * scale;
                }
                for (int32_t j = i + 1; j < seq; j++) scores[j] = -1e9f;

                niyah_softmax(scores, seq);

                float* o = state->out + batch_off + i * dim + h * head_dim;
                memset(o, 0, (size_t)head_dim * sizeof(float));
                for (int32_t j = 0; j < seq; j++) {
                    const float* v = state->v + batch_off + j * dim + h * head_dim;
                    for (int32_t d = 0; d < head_dim; d++)
                        o[d] += scores[j] * v[d];
                }
            }
        }
    }

    free(scores);
}
