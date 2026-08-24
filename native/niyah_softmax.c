#include "niyah.h"

#include <math.h>

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

    /*
     * T < 0: no probabilistic meaning.
     * Before this fix, negative inv_t inverted the distribution
     * (highest logit -> lowest prob) before softmax -- wrong.
     */
    if (temperature < 0.0f) {
        const float uniform = 1.0f / (float)n;
        for (int32_t i = 0; i < n; ++i) {
            x[i] = uniform;
        }
        return;
    }

    /*
     * T = 0: mathematical limit as T -> 0+, all mass on argmax (one-hot).
     * Before this fix, T=0 fell through to T=1 softmax because
     * `temperature > 0.0f` was false and no scaling ran.
     */
    if (temperature == 0.0f) {
        float   max_val = x[0];
        int32_t max_idx = 0;
        for (int32_t i = 1; i < n; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
                max_idx = i;
            }
        }
        for (int32_t i = 0; i < n; ++i) {
            x[i] = (i == max_idx) ? 1.0f : 0.0f;
        }
        return;
    }

    /* T > 0, T != 1: scale then run stable softmax. */
    if (temperature != 1.0f) {
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

    float   max_val = x[0];
    int32_t max_idx = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }

    if (!isfinite(max_val)) {
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
     * Underflow guard: if sum is 0 or non-finite, logf(sum) = -inf,
     * making x[i] -= log_sum produce NaN (finite - (-inf)).
     * Fallback: argmax gets log-prob 0, all others -INFINITY.
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
