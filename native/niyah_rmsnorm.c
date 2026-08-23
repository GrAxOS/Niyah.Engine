#include "niyah.h"
#include <math.h>
#include <stddef.h>

/* Root Mean Square Layer Normalisation (Llama-style) */
void niyah_rmsnorm(float* x, const float* weight, int32_t n, float eps) {
    if (!x || n <= 0) return;

    float ss = 0.0f;
    for (int32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);

    if (weight) {
        for (int32_t i = 0; i < n; i++) x[i] = weight[i] * (scale * x[i]);
    } else {
        for (int32_t i = 0; i < n; i++) x[i] *= scale;
    }
}
