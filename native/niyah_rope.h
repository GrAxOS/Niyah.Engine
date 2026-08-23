#ifndef NIYAH_ROPE_H
#define NIYAH_ROPE_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_rope: Rotary Position Embedding (RoPE), applied to Q/K projections
 * before attention in decoder-style transformers (LLaMA-style convention:
 * adjacent-pair rotation, not the interleaved-half variant).
 *
 * For row r (absolute position = pos_offset + r) and pair index i in
 * [0, dim/2):
 *   theta_i   = theta_base ^ (-2*i / dim)
 *   angle     = (pos_offset + r) * theta_i
 *   x[2i]'   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
 *   x[2i+1]' = x[2i] * sin(angle) + x[2i+1] * cos(angle)
 *
 * Applied in place. Caller owns the buffer; no allocation is performed here.
 */

/* dim must be even and non-zero. theta_base is typically 10000.0f.
 * Returns false on NULL pointer, zero/odd dim, or size_t overflow. */
bool niyah_rope_f32(float *x, size_t rows, size_t dim, size_t pos_offset,
                     float theta_base);

#endif
