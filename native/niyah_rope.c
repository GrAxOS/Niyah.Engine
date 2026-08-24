#include "niyah.h"

#include <math.h>

/*
 * Was a stub. Rotary position embeddings in the NeoX / GGUF convention:
 * within each head the first half of the channels is paired with the second
 * half, so channel i rotates against channel i + head_dim/2.
 *
 * Layout of x: [seq][dim] with dim == n_head * head_dim, contiguous.
 */

void niyah_rope_forward_ex(float* x,
                           int32_t seq,
                           int32_t dim,
                           int32_t n_head,
                           int32_t pos_offset,
                           float theta)
{
    if (!x || seq <= 0 || dim <= 0 || n_head <= 0) {
        return;
    }
    if (dim % n_head != 0) {
        return; /* shape mismatch: refuse rather than corrupt memory */
    }

    const int32_t head_dim = dim / n_head;
    const int32_t half = head_dim / 2;
    if (half <= 0) {
        return;
    }
    if (!(theta > 0.0f)) {
        theta = 10000.0f;
    }

    for (int32_t p = 0; p < seq; ++p) {
        const double position = (double)(p + pos_offset);
        float* row = x + (size_t)p * (size_t)dim;

        for (int32_t i = 0; i < half; ++i) {
            const double exponent = (2.0 * (double)i) / (double)head_dim;
            const double freq = 1.0 / pow((double)theta, exponent);
            const double angle = position * freq;
            const float cos_a = (float)cos(angle);
            const float sin_a = (float)sin(angle);

            for (int32_t h = 0; h < n_head; ++h) {
                float* head = row + (size_t)h * (size_t)head_dim;
                const float v0 = head[i];
                const float v1 = head[i + half];
                head[i]        = v0 * cos_a - v1 * sin_a;
                head[i + half] = v0 * sin_a + v1 * cos_a;
            }
        }
    }
}

void niyah_rope_forward(float* x, int32_t seq, int32_t dim, int32_t n_head)
{
    niyah_rope_forward_ex(x, seq, dim, n_head, 0, 10000.0f);
}
