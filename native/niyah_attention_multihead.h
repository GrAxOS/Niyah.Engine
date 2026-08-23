#ifndef NIYAH_ATTENTION_MULTIHEAD_H
#define NIYAH_ATTENTION_MULTIHEAD_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_attention_multihead: splits Q/K/V into num_heads independent heads
 * of head_dim each, runs niyah_attention_causal_f32 per head, and writes
 * results back into the same interleaved [seq_len x (num_heads*head_dim)]
 * layout used by real multi-head transformers (head h occupies columns
 * [h*head_dim, (h+1)*head_dim)).
 *
 * q, k, v, out: [seq_len x (num_heads*head_dim)] row-major.
 * q_head, k_head, v_head, out_head: caller-owned [seq_len x head_dim]
 * scratch, reused sequentially across heads (heads are processed one at a
 * time, never concurrently, so a single set of scratch buffers suffices).
 * scores_scratch: caller-owned [seq_len x seq_len] scratch, also reused
 * per head.
 *
 * Returns false on NULL pointers, zero dimensions, or size_t overflow.
 */
bool niyah_attention_multihead_causal_f32(const float *q, const float *k,
                                           const float *v, float *q_head,
                                           float *k_head, float *v_head,
                                           float *out_head,
                                           float *scores_scratch, float *out,
                                           size_t seq_len, size_t num_heads,
                                           size_t head_dim);

#endif
