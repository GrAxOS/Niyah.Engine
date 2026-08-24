#include "niyah.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was a stub, so niyah_sample() was declared in niyah.h but never defined and
 * every executable that touched it failed at link time.
 *
 * xoshiro256** seeded through splitmix64. Deterministic for a given seed,
 * which matters for reproducible evaluation runs.
 */

static uint64_t g_state[4] = {
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
    0x2545f4914f6cdd1dULL, 0x9e3779b97f4a7c15ULL
};

static uint64_t splitmix64(uint64_t* x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void niyah_sampler_seed(uint64_t seed)
{
    uint64_t x = seed ? seed : 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; ++i) {
        g_state[i] = splitmix64(&x);
    }
}

static uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t next_u64(void)
{
    const uint64_t result = rotl(g_state[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = g_state[1] << 17;

    g_state[2] ^= g_state[0];
    g_state[3] ^= g_state[1];
    g_state[1] ^= g_state[2];
    g_state[0] ^= g_state[3];
    g_state[2] ^= t;
    g_state[3] = rotl(g_state[3], 45);

    return result;
}

/* Uniform in [0, 1). */
static float next_float(void)
{
    return (float)((next_u64() >> 11) * (1.0 / 9007199254740992.0));
}

typedef struct {
    float   prob;
    int32_t index;
} Candidate;

static int candidate_compare(const void* a, const void* b)
{
    const Candidate* ca = (const Candidate*)a;
    const Candidate* cb = (const Candidate*)b;
    if (ca->prob < cb->prob) return 1;   /* descending */
    if (ca->prob > cb->prob) return -1;
    if (ca->index < cb->index) return -1;
    if (ca->index > cb->index) return 1;
    return 0;
}

static int32_t sample_from(const Candidate* pool, int32_t count, float total)
{
    if (count <= 0) {
        return -1;
    }
    if (!(total > 0.0f)) {
        return pool[0].index;
    }

    const float target = next_float() * total;
    float running = 0.0f;

    for (int32_t i = 0; i < count; ++i) {
        running += pool[i].prob;
        if (running >= target) {
            return pool[i].index;
        }
    }

    /* Floating-point shortfall: fall back to the last candidate. */
    return pool[count - 1].index;
}

void niyah_sampler_apply_repetition_penalty(float* logits,
                                            int32_t n_vocab,
                                            const int32_t* history,
                                            int32_t history_len,
                                            float penalty)
{
    if (!logits || n_vocab <= 0 || !history || history_len <= 0) {
        return;
    }
    if (!(penalty > 0.0f) || penalty == 1.0f) {
        return;
    }

    for (int32_t i = 0; i < history_len; ++i) {
        const int32_t token = history[i];
        if (token < 0 || token >= n_vocab) {
            continue;
        }
        /* Positive logits are divided, negative ones multiplied, so the
         * penalty always pushes the token down. */
        logits[token] = logits[token] > 0.0f
            ? logits[token] / penalty
            : logits[token] * penalty;
    }
}

int32_t niyah_sample(const float* logits,
                     int32_t n_vocab,
                     const NiyahSamplerConfig* config)
{
    if (!logits || n_vocab <= 0) {
        return -1;
    }

    NiyahSamplerConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg.strategy = NIYAH_SAMPLE_GREEDY;
        cfg.temperature = 1.0f;
        cfg.top_k = 0;
        cfg.top_p = 1.0f;
    }

    if (cfg.strategy == NIYAH_SAMPLE_GREEDY || !(cfg.temperature > 0.0f)) {
        return niyah_argmax(logits, n_vocab);
    }

    float* probs = (float*)malloc((size_t)n_vocab * sizeof(float));
    if (!probs) {
        return niyah_argmax(logits, n_vocab);
    }
    memcpy(probs, logits, (size_t)n_vocab * sizeof(float));
    niyah_softmax_temperature(probs, n_vocab, cfg.temperature);

    int32_t chosen = -1;

    if (cfg.strategy == NIYAH_SAMPLE_TEMPERATURE) {
        Candidate* pool = (Candidate*)malloc((size_t)n_vocab * sizeof(Candidate));
        if (pool) {
            float total = 0.0f;
            for (int32_t i = 0; i < n_vocab; ++i) {
                pool[i].prob = probs[i];
                pool[i].index = i;
                total += probs[i];
            }
            chosen = sample_from(pool, n_vocab, total);
            free(pool);
        }
    } else if (cfg.strategy == NIYAH_SAMPLE_TOP_K) {
        int32_t k = cfg.top_k > 0 ? cfg.top_k : 40;
        if (k > n_vocab) {
            k = n_vocab;
        }

        Candidate* pool = (Candidate*)malloc((size_t)k * sizeof(Candidate));
        if (pool) {
            /* Partial selection: k passes, no full O(V log V) sort. */
            for (int32_t slot = 0; slot < k; ++slot) {
                int32_t best = -1;
                float best_p = -1.0f;
                for (int32_t i = 0; i < n_vocab; ++i) {
                    if (probs[i] < 0.0f) {
                        continue; /* already taken */
                    }
                    if (probs[i] > best_p) {
                        best_p = probs[i];
                        best = i;
                    }
                }
                if (best < 0) {
                    k = slot;
                    break;
                }
                pool[slot].prob = best_p;
                pool[slot].index = best;
                probs[best] = -1.0f; /* mark consumed */
            }

            float total = 0.0f;
            for (int32_t i = 0; i < k; ++i) {
                total += pool[i].prob;
            }
            chosen = sample_from(pool, k, total);
            free(pool);
        }
    } else { /* NIYAH_SAMPLE_TOP_P */
        const float top_p = (cfg.top_p > 0.0f && cfg.top_p <= 1.0f)
            ? cfg.top_p : 0.9f;

        Candidate* pool = (Candidate*)malloc((size_t)n_vocab * sizeof(Candidate));
        if (pool) {
            for (int32_t i = 0; i < n_vocab; ++i) {
                pool[i].prob = probs[i];
                pool[i].index = i;
            }
            qsort(pool, (size_t)n_vocab, sizeof(Candidate), candidate_compare);

            float cumulative = 0.0f;
            int32_t cutoff = 0;
            while (cutoff < n_vocab) {
                cumulative += pool[cutoff].prob;
                ++cutoff;
                if (cumulative >= top_p) {
                    break;
                }
            }
            chosen = sample_from(pool, cutoff, cumulative);
            free(pool);
        }
    }

    free(probs);

    if (chosen < 0) {
        chosen = niyah_argmax(logits, n_vocab);
    }
    return chosen;
}
