#ifndef NIYAH_EMBEDDING_H
#define NIYAH_EMBEDDING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * niyah_embedding: token <-> vector bridge for a decoder transformer.
 *
 *   niyah_embedding_lookup_f32: token ids -> embedding vectors (gather).
 *   niyah_lm_head_f32:          final hidden states -> vocab logits
 *                                (thin, bounds-documented wrapper around
 *                                niyah_matmul_f32_bt).
 *
 * Caller owns all buffers; no allocation is performed here.
 */

/* table: [vocab_size x embedding_dim] row-major embedding matrix.
 * token_ids: [num_tokens], each must be < vocab_size.
 * out: [num_tokens x embedding_dim].
 * Returns false on NULL pointers, zero dimensions, out-of-range token id,
 * or size_t overflow. */
bool niyah_embedding_lookup_f32(const float *table, size_t vocab_size,
                                 size_t embedding_dim,
                                 const uint32_t *token_ids,
                                 size_t num_tokens, float *out);

/* hidden: [rows x embedding_dim]. w_head: [vocab_size x embedding_dim]
 * ("already transposed" convention, same as every other weight matrix in
 * this codebase). logits: [rows x vocab_size].
 * Returns false on NULL pointers, zero dimensions, or size_t overflow. */
bool niyah_lm_head_f32(const float *hidden, const float *w_head,
                        float *logits, size_t rows, size_t embedding_dim,
                        size_t vocab_size);

#endif
