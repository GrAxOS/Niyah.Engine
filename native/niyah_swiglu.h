#ifndef NIYAH_SWIGLU_H
#define NIYAH_SWIGLU_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_swiglu: SwiGLU feed-forward block, as used in LLaMA-style decoder
 * transformers in place of a plain ReLU/GELU MLP.
 *
 *   gate = x * W_gate^T          [rows x hidden_dim]
 *   up   = x * W_up^T            [rows x hidden_dim]
 *   h    = SiLU(gate) * up       (elementwise; SiLU(v) = v * sigmoid(v))
 *   out  = h * W_down^T          [rows x dim]
 *
 * Weight layout convention matches niyah_matmul_f32_bt: W_gate and W_up are
 * stored as [hidden_dim x dim] ("already transposed"), W_down as
 * [dim x hidden_dim]. Composed entirely from niyah_matmul_f32_bt plus the
 * elementwise SiLU-gate pass below.
 *
 * Caller owns all buffers, including gate_scratch/up_scratch
 * ([rows x hidden_dim] each); no allocation is performed here.
 */
bool niyah_swiglu_f32(const float *x, const float *w_gate,
                       const float *w_up, const float *w_down,
                       float *gate_scratch, float *up_scratch, float *out,
                       size_t rows, size_t dim, size_t hidden_dim);

#endif
