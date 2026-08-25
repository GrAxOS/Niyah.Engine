#include "niyah.h"
#include <math.h>
#include <stddef.h>

/*
 * Rotary Position Embedding (RoPE).
 * Applies in-place to x[seq, dim] viewed as x[seq, n_head, head_dim].
 * Each head gets 2-D rotations on consecutive pairs of features.
 */
void niyah_rope_forward(float* x, int32_t seq, int32_t dim, int32_t n_head) {
    if (!x || seq <= 0 || dim <= 0 || n_head <= 0) return;

    int32_t head_dim = dim / n_head;

    for (int32_t pos = 0; pos < seq; pos++) {
        for (int32_t h = 0; h < n_head; h++) {
            float* head = x + pos * dim + h * head_dim;

            for (int32_t i = 0; i < head_dim / 2; i++) {
                float freq  = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
                float theta = (float)pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);

                float x0 = head[2 * i];
                float x1 = head[2 * i + 1];
                head[2 * i]     = x0 * cos_t - x1 * sin_t;
                head[2 * i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}
