#ifndef NIYAH_ATTENTION_H
#define NIYAH_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_attention: single-head causal self-attention.
 *
 *   scores = (Q * K^T) / sqrt(head_dim)
 *   scores[i][j] = -inf for j > i   (causal mask: position i cannot see
 *                                    future positions j > i)
 *   weights = softmax(scores, row-wise)
 *   out     = weights * V
 *
 * Composed from niyah_matmul_f32_bt, the causal mask/scale pass below, and
 * niyah_softmax_f32 — no new numerical primitives, only orchestration.
 *
 * Caller is expected to have already applied RoPE (niyah_rope_f32) to Q and
 * K before calling this function, and to provide the projected Q/K/V
 * themselves (this function does not perform the input projection matmuls).
 *
 * q, k, v: [seq_len x head_dim] row-major.
 * scores_scratch: caller-owned [seq_len x seq_len] scratch buffer (no
 *                 allocation is performed here; reused for the softmax
 *                 weights in place).
 * out: [seq_len x head_dim] row-major.
 *
 * Returns false on NULL pointers, zero dimensions, or size_t overflow.
 */
bool niyah_attention_causal_f32(const float *q, const float *k,
                                 const float *v, float *scores_scratch,
                                 float *out, size_t seq_len, size_t head_dim);

#endif
