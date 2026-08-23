#ifndef NIYAH_SOFTMAX_H
#define NIYAH_SOFTMAX_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_softmax: numerically-stable row-wise softmax, used on attention
 * scores (Q*K^T / sqrt(head_dim)) before multiplying by V.
 *
 * Applied in place, row by row: for each row, subtracts the row max before
 * exponentiating (avoids overflow for large logits, e.g. +-1000) and
 * normalizes by the row sum. Caller owns the buffer; no allocation is
 * performed here.
 */

/* Returns false on NULL pointer, zero dimensions, or size_t overflow. */
bool niyah_softmax_f32(float *x, size_t rows, size_t dim);

#endif
