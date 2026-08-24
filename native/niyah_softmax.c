#include "niyah.h"

#include <math.h>

/*
 * Was a stub. Numerically stable softmax: subtract the row max before exp so
 * large logits cannot overflow to inf. Degenerate rows fall back to uniform
 * rather than producing NaN, which is what the sampler needs to stay safe.
 */

void niyah_softmax(float* x, int32_t n)
{
    if (!x || n <= 0) {
        return;
    }

    float max_val = x[0];
    for (int32_t i = 1; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    if (!isfinite(max_val)) {
        /* All -inf, or a NaN slipped in: emit a uniform distribution. */
        const float uniform = 1.0f / (float)n;
        for (int32_t i = 0; i < n; ++i) {
            x[i] = uniform;
        }
        return;
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        const float e = expf(x[i] - max_val);
        x[i] = e;
        sum += e;
    }

    if (sum <= 0.0f || !isfinite(sum)) {
        const float uniform = 1.0f / (float)n;
        for (int32_t i = 0; i < n; ++i) {
            x[i] = uniform;
        }
        return;
    }

    const float inv = 1.0f / sum;
    for (int32_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

void niyah_softmax_temperature(float* x, int32_t n, float temperature)
{
    if (!x || n <= 0) {
        return;
    }

    if (temperature > 0.0f && temperature != 1.0f) {
        const float inv_t = 1.0f / temperature;
        for (int32_t i = 0; i < n; ++i) {
            x[i] *= inv_t;
        }
    }

    niyah_softmax(x, n);
}

void niyah_log_softmax(float* x, int32_t n)
{
    if (!x || n <= 0) {
        return;
    }

    /* Find max for numerical stability and record its index. */
    float   max_val = x[0];
    int32_t max_idx = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }

    if (!isfinite(max_val)) {
        /*
         * All elements are -inf or contain NaN.
         * Emit uniform log-distribution: log(1/n) for every slot.
         */
        const float uniform = -logf((float)n);
        for (int32_t i = 0; i < n; ++i) {
            x[i] = uniform;
        }
        return;
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        sum += expf(x[i] - max_val);
    }

    /*
     * Underflow guard — mirrors the equivalent check in niyah_softmax.
     *
     * In the standard path the max element contributes expf(0)=1, so sum>=1.
     * However, in pathological cases (pre-shifted arrays, compiler differences
     * in flush-to-zero mode with -ffast-math, or future callers) sum may be
     * zero or non-finite.
     *
     * Without this guard: logf(0.0f) == -HUGE_VAL (-inf), so
     *   log_sum = max_val + (-inf) = -inf
     * and then every element becomes
     *   x[i] -= (-inf)  =>  finite + inf  =>  NaN under IEEE 754.
     *
     * Degenerate fallback: place all probability mass on the argmax element
     * (log-prob 0, i.e. prob 1) and assign -INFINITY to every other slot.
     * This is the correct mathematical limit when all other classes have
     * negligible probability and is NaN-free for the sampler.
     */
    if (sum <= 0.0f || !isfinite(sum)) {
        for (int32_t i = 0; i < n; ++i) {
            x[i] = (i == max_idx) ? 0.0f : -INFINITY;
        }
        return;
    }

    const float log_sum = max_val + logf(sum);
    for (int32_t i = 0; i < n; ++i) {
        x[i] -= log_sum;
    }
}

int32_t niyah_argmax(const float* x, int32_t n)
{
    if (!x || n <= 0) {
        return -1;
    }

    int32_t best = 0;
    float best_val = x[0];
    for (int32_t i = 1; i < n; ++i) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}
