#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Seeded once at first call */
static int g_rand_init = 0;

static void ensure_rand(void) {
    if (!g_rand_init) {
        srand((unsigned)time(NULL));
        g_rand_init = 1;
    }
}

static float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

/* Greedy: return argmax */
static int32_t sample_greedy(const float* logits, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* Temperature: scale logits, then sample */
static int32_t sample_temperature(const float* logits, int32_t n, float temp) {
    float* probs = (float*)malloc((size_t)n * sizeof(float));
    if (!probs) return sample_greedy(logits, n);
    for (int32_t i = 0; i < n; i++) probs[i] = logits[i] / temp;
    niyah_softmax(probs, n);

    float r   = rand_float();
    float cum = 0.0f;
    int32_t token = n - 1;
    for (int32_t i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) { token = i; break; }
    }
    free(probs);
    return token;
}

/* Top-K: keep top k logits, renormalise, sample */
static int32_t sample_top_k(const float* logits, int32_t n, int32_t k) {
    if (k >= n) return sample_temperature(logits, n, 1.0f);

    /* Find k-th largest via partial sort (simple for small k) */
    float* tmp = (float*)malloc((size_t)n * sizeof(float));
    if (!tmp) return sample_greedy(logits, n);
    memcpy(tmp, logits, (size_t)n * sizeof(float));

    /* Bubble the top k values to the front */
    for (int32_t i = 0; i < k; i++)
        for (int32_t j = i + 1; j < n; j++)
            if (tmp[j] > tmp[i]) { float t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }

    float threshold = tmp[k - 1];
    for (int32_t i = 0; i < n; i++)
        if (logits[i] < threshold) tmp[i] = -1e9f;
        else                       tmp[i] = logits[i];
    niyah_softmax(tmp, n);

    float r   = rand_float();
    float cum = 0.0f;
    int32_t token = n - 1;
    for (int32_t i = 0; i < n; i++) {
        cum += tmp[i];
        if (r < cum) { token = i; break; }
    }
    free(tmp);
    return token;
}

/* Top-P (nucleus): accumulate until probability mass >= p, then sample */
static int32_t sample_top_p(const float* logits, int32_t n, float p) {
    /* Sort indices by logit descending */
    int32_t* idx = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    float*   prob = (float*)malloc((size_t)n * sizeof(float));
    if (!idx || !prob) { free(idx); free(prob); return sample_greedy(logits, n); }

    for (int32_t i = 0; i < n; i++) { idx[i] = i; prob[i] = logits[i]; }
    /* Simple insertion sort (vocab can be large – use this for correctness) */
    for (int32_t i = 1; i < n; i++) {
        int32_t key_idx = idx[i];
        float   key_val = prob[i];
        int32_t j = i - 1;
        while (j >= 0 && prob[j] < key_val) {
            prob[j + 1] = prob[j];
            idx[j + 1]  = idx[j];
            j--;
        }
        prob[j + 1] = key_val;
        idx[j + 1]  = key_idx;
    }

    niyah_softmax(prob, n);

    float cum = 0.0f;
    int32_t cutoff = n;
    for (int32_t i = 0; i < n; i++) {
        cum += prob[i];
        if (cum >= p) { cutoff = i + 1; break; }
    }

    float r      = rand_float() * cum;
    float running = 0.0f;
    int32_t token = idx[cutoff - 1];
    for (int32_t i = 0; i < cutoff; i++) {
        running += prob[i];
        if (r < running) { token = idx[i]; break; }
    }
    free(idx); free(prob);
    return token;
}

int32_t niyah_sample(const float* logits, int32_t n_vocab,
                     const NiyahSamplerConfig* config) {
    if (!logits || n_vocab <= 0) return 0;
    ensure_rand();

    if (!config) return sample_greedy(logits, n_vocab);

    switch (config->strategy) {
        case NIYAH_SAMPLE_GREEDY:
            return sample_greedy(logits, n_vocab);
        case NIYAH_SAMPLE_TOP_K:
            return sample_top_k(logits, n_vocab, config->top_k > 0 ? config->top_k : 40);
        case NIYAH_SAMPLE_TOP_P:
            return sample_top_p(logits, n_vocab, config->top_p > 0 ? config->top_p : 0.9f);
        case NIYAH_SAMPLE_TEMPERATURE:
            return sample_temperature(logits, n_vocab,
                                      config->temperature > 0 ? config->temperature : 1.0f);
        default:
            return sample_greedy(logits, n_vocab);
    }
}
