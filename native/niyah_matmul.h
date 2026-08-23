#ifndef NIYAH_MATMUL_H
#define NIYAH_MATMUL_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_matmul: minimal, dependency-light F32 matrix multiply kernels.
 *
 * Design constraints (matching niyah_core.h conventions):
 *   - Caller owns all buffers. No allocation is performed here.
 *   - All dimension products are overflow-checked via niyah_add_size /
 *     niyah_mul_size before any indexing occurs.
 *   - Row-major layout throughout.
 */

/* C[M x N] = A[M x K] * B[K x N]. All matrices row-major.
 * Returns false on NULL pointers, zero dimensions, or size_t overflow. */
bool niyah_matmul_f32(const float *a, const float *b, float *c,
                       size_t m, size_t k, size_t n);

/* C[M x N] = A[M x K] * B_transposed[N x K].
 * B is stored as [N x K] (i.e. already transposed), which matches how
 * projection weight matrices (Wq/Wk/Wv/Wo) are typically stored for a
 * decoder-style transformer. Avoids a separate transpose pass. */
bool niyah_matmul_f32_bt(const float *a, const float *b_transposed, float *c,
                          size_t m, size_t k, size_t n);

#endif
