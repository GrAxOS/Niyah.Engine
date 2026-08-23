#ifndef NIYAH_SAMPLER_H
#define NIYAH_SAMPLER_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_sampler: token-selection primitives applied to the final LM-head
 * logits during autoregressive generation. No RNG is embedded here (kept
 * sovereign/deterministic and testable) — callers supply their own
 * uniform-random draw for niyah_sampler_weighted_f32.
 *
 * Caller owns all buffers; no allocation is performed here.
 */

/* logits[i] /= temperature, in place. temperature must be > 0. */
bool niyah_sampler_apply_temperature_f32(float *logits, size_t n,
                                          float temperature);

/* Keeps the top-k logits, sets every other entry to -INFINITY (so a
 * subsequent niyah_softmax_f32 call zeroes their probability). scratch is a
 * caller-owned [n] buffer used internally; its final contents are
 * unspecified. If k >= n, this is a no-op (nothing is filtered). */
bool niyah_sampler_top_k_filter_f32(float *logits, float *scratch, size_t n,
                                     size_t k);

/* Deterministic greedy selection: index of the maximum logit. */
bool niyah_sampler_argmax_f32(const float *logits, size_t n,
                               size_t *out_index);

/* Samples an index from a probability distribution (e.g. the output of
 * niyah_softmax_f32) given a caller-supplied uniform random draw in
 * [0, 1). Walks the cumulative distribution and returns the first index
 * whose cumulative probability meets or exceeds the draw. */
bool niyah_sampler_weighted_f32(const float *probs, size_t n,
                                 float uniform_0_1, size_t *out_index);

#endif
