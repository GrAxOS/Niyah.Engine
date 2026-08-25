#include "niyah.h"
<<<<<<< HEAD
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
=======

#include <math.h>

/*
 * Was a stub. RMSNorm as used by Llama/Qwen: rescale by the root mean square
 * (no mean subtraction), then apply the per-channel gain. `weight` may be NULL
 * for a plain normalisation.
 */

void niyah_rmsnorm_to(float* out,
                      const float* x,
                      const float* weight,
                      int32_t n,
                      float eps)
{
    if (!out || !x || n <= 0) {
        return;
    }
    if (!(eps > 0.0f)) {
        eps = 1e-5f;
    }

    /* Accumulate in double: with n_embd in the thousands and fp16-derived
     * weights, float accumulation loses meaningful precision here. */
    double sum_sq = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }

    const double mean_sq = sum_sq / (double)n;
    const float scale = (float)(1.0 / sqrt(mean_sq + (double)eps));

    if (weight) {
        for (int32_t i = 0; i < n; ++i) {
            out[i] = x[i] * scale * weight[i];
        }
    } else {
        for (int32_t i = 0; i < n; ++i) {
            out[i] = x[i] * scale;
        }
    }
}

void niyah_rmsnorm(float* x, const float* weight, int32_t n, float eps)
{
    niyah_rmsnorm_to(x, x, weight, n, eps);
}

void niyah_layernorm(float* x,
                     const float* weight,
                     const float* bias,
                     int32_t n,
                     float eps)
{
    if (!x || n <= 0) {
        return;
    }
    if (!(eps > 0.0f)) {
        eps = 1e-5f;
    }

    double mean = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        mean += (double)x[i];
    }
    mean /= (double)n;

    double variance = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double d = (double)x[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;

    const float inv_std = (float)(1.0 / sqrt(variance + (double)eps));
    const float mean_f = (float)mean;

    for (int32_t i = 0; i < n; ++i) {
        float v = (x[i] - mean_f) * inv_std;
        if (weight) {
            v *= weight[i];
        }
        if (bias) {
            v += bias[i];
        }
        x[i] = v;
>>>>>>> origin/main
    }
}
