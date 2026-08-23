#include "niyah_sampler.h"

#include <math.h>

bool niyah_sampler_apply_temperature_f32(float *logits, size_t n,
                                          float temperature) {
    if (logits == NULL || n == 0u || !(temperature > 0.0f)) {
        return false;
    }
    for (size_t i = 0u; i < n; ++i) {
        logits[i] = logits[i] / temperature;
    }
    return true;
}

bool niyah_sampler_top_k_filter_f32(float *logits, float *scratch, size_t n,
                                     size_t k) {
    if (logits == NULL || scratch == NULL || n == 0u || k == 0u) {
        return false;
    }
    if (k >= n) {
        return true;
    }

    for (size_t i = 0u; i < n; ++i) {
        scratch[i] = logits[i];
    }

    float threshold = -INFINITY;
    for (size_t iter = 0u; iter < k; ++iter) {
        size_t best = 0u;
        float best_val = scratch[0];
        for (size_t i = 1u; i < n; ++i) {
            if (scratch[i] > best_val) {
                best_val = scratch[i];
                best = i;
            }
        }
        threshold = best_val;
        scratch[best] = -INFINITY;
    }

    for (size_t i = 0u; i < n; ++i) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }

    return true;
}

bool niyah_sampler_argmax_f32(const float *logits, size_t n,
                               size_t *out_index) {
    if (logits == NULL || out_index == NULL || n == 0u) {
        return false;
    }

    size_t best = 0u;
    float best_val = logits[0];
    for (size_t i = 1u; i < n; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }

    *out_index = best;
    return true;
}

bool niyah_sampler_weighted_f32(const float *probs, size_t n,
                                 float uniform_0_1, size_t *out_index) {
    if (probs == NULL || out_index == NULL || n == 0u) {
        return false;
    }
    if (uniform_0_1 < 0.0f || uniform_0_1 >= 1.0f) {
        return false;
    }

    double cumulative = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        cumulative += (double)probs[i];
        if ((double)uniform_0_1 < cumulative) {
            *out_index = i;
            return true;
        }
    }

    *out_index = n - 1u;
    return true;
}
