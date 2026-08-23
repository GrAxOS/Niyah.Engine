#include "niyah_attention_multihead.h"
#include "niyah_attention.h"
#include "niyah_core.h"

bool niyah_attention_multihead_causal_f32(const float *q, const float *k,
                                           const float *v, float *q_head,
                                           float *k_head, float *v_head,
                                           float *out_head,
                                           float *scores_scratch, float *out,
                                           size_t seq_len, size_t num_heads,
                                           size_t head_dim) {
    size_t dim = 0u;
    size_t seq_dim = 0u;

    if (q == NULL || k == NULL || v == NULL || q_head == NULL ||
        k_head == NULL || v_head == NULL || out_head == NULL ||
        scores_scratch == NULL || out == NULL) {
        return false;
    }
    if (seq_len == 0u || num_heads == 0u || head_dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(num_heads, head_dim, &dim) ||
        !niyah_mul_size(seq_len, dim, &seq_dim)) {
        return false;
    }

    for (size_t h = 0u; h < num_heads; ++h) {
        const size_t col_offset = h * head_dim;

        for (size_t r = 0u; r < seq_len; ++r) {
            const float *q_src = q + r * dim + col_offset;
            const float *k_src = k + r * dim + col_offset;
            const float *v_src = v + r * dim + col_offset;
            float *q_dst = q_head + r * head_dim;
            float *k_dst = k_head + r * head_dim;
            float *v_dst = v_head + r * head_dim;

            for (size_t c = 0u; c < head_dim; ++c) {
                q_dst[c] = q_src[c];
                k_dst[c] = k_src[c];
                v_dst[c] = v_src[c];
            }
        }

        if (!niyah_attention_causal_f32(q_head, k_head, v_head,
                                         scores_scratch, out_head, seq_len,
                                         head_dim)) {
            return false;
        }

        for (size_t r = 0u; r < seq_len; ++r) {
            const float *src = out_head + r * head_dim;
            float *dst = out + r * dim + col_offset;
            for (size_t c = 0u; c < head_dim; ++c) {
                dst[c] = src[c];
            }
        }
    }

    (void)seq_dim;
    return true;
}
