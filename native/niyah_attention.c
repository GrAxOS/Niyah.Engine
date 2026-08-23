#include "niyah_attention.h"
#include "niyah_matmul.h"
#include "niyah_softmax.h"
#include "niyah_core.h"

#include <math.h>

bool niyah_attention_causal_f32(const float *q, const float *k,
                                 const float *v, float *scores_scratch,
                                 float *out, size_t seq_len, size_t head_dim) {
    size_t scores_elems = 0u;
    size_t qkv_elems = 0u;

    if (q == NULL || k == NULL || v == NULL || scores_scratch == NULL ||
        out == NULL) {
        return false;
    }
    if (seq_len == 0u || head_dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(seq_len, seq_len, &scores_elems) ||
        !niyah_mul_size(seq_len, head_dim, &qkv_elems)) {
        return false;
    }

    /* scores = Q * K^T. K is [seq_len x head_dim], exactly the layout
     * niyah_matmul_f32_bt expects for its "already transposed" operand. */
    if (!niyah_matmul_f32_bt(q, k, scores_scratch, seq_len, head_dim,
                              seq_len)) {
        return false;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);

    for (size_t i = 0u; i < seq_len; ++i) {
        float *row = scores_scratch + i * seq_len;
        for (size_t j = 0u; j < seq_len; ++j) {
            row[j] = (j > i) ? -INFINITY : row[j] * scale;
        }
    }

    if (!niyah_softmax_f32(scores_scratch, seq_len, seq_len)) {
        return false;
    }

    /* out = weights * V. weights is [seq_len x seq_len], V is
     * [seq_len x head_dim] — standard (non-transposed) matmul layout. */
    return niyah_matmul_f32(scores_scratch, v, out, seq_len, seq_len,
                             head_dim);
}
