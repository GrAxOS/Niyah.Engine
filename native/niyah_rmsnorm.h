#ifndef NIYAH_RMSNORM_H
#define NIYAH_RMSNORM_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_rmsnorm: RMSNorm (root-mean-square layer normalization), the
 * normalization used before attention/FFN blocks in LLaMA-style decoder
 * transformers. No mean-subtraction (unlike LayerNorm) — only rescales by
 * the RMS of each row, then applies a learned per-channel gain.
 *
 * Caller owns all buffers; no allocation is performed here.
 */

/*
 * out[r][i] = x[r][i] / sqrt(mean_i(x[r][i]^2) + eps) * weight[i]
 *
 * x, out: [rows x dim] row-major. weight: [dim]. out may alias x.
 * Returns false on NULL pointers, zero dimensions, or size_t overflow.
 */
bool niyah_rmsnorm_f32(const float *x, const float *weight, float *out,
                        size_t rows, size_t dim, float eps);

#endif
