#ifndef NIYAH_TRANSFORMER_LAYER_H
#define NIYAH_TRANSFORMER_LAYER_H

#include <stdbool.h>
#include <stddef.h>

/*
 * niyah_transformer_layer: one full decoder block, composed entirely from
 * the primitives already implemented in this directory:
 *
 *   normed1  = RMSNorm(x, rms1_weight)
 *   q,k,v    = normed1 * {W_q,W_k,W_v}^T
 *   q,k      = RoPE(q,k, pos_offset)
 *   attn_out = CausalAttention(q, k, v)
 *   x        = x + attn_out * W_o^T                (residual 1)
 *   normed2  = RMSNorm(x, rms2_weight)
 *   ffn_out  = SwiGLU(normed2, W_gate, W_up, W_down)
 *   x        = x + ffn_out                          (residual 2)
 *
 * Single-head attention: head_dim == dim (no multi-head splitting here;
 * that would be a thin wrapper that slices q/k/v into heads before calling
 * niyah_attention_causal_f32 per head — left for a follow-up once
 * multi-head weight layouts are decided).
 *
 * x is updated in place (it is both input residual-stream and output).
 * All weight matrices use the [out_dim x in_dim] "already transposed"
 * convention shared by niyah_matmul_f32_bt / niyah_attention / niyah_swiglu.
 * Caller owns every buffer in NiyahTransformerLayerScratch; no allocation
 * is performed anywhere in this function.
 */

typedef struct {
    const float *rms1_weight;  /* [dim] */
    const float *w_q;          /* [dim x dim] */
    const float *w_k;          /* [dim x dim] */
    const float *w_v;          /* [dim x dim] */
    const float *w_o;          /* [dim x dim] */
    const float *rms2_weight;  /* [dim] */
    const float *w_gate;       /* [hidden_dim x dim] */
    const float *w_up;         /* [hidden_dim x dim] */
    const float *w_down;       /* [dim x hidden_dim] */
} NiyahTransformerLayerWeights;

typedef struct {
    float *normed;    /* [rows x dim] */
    float *q;         /* [rows x dim] */
    float *k;         /* [rows x dim] */
    float *v;         /* [rows x dim] */
    float *attn_out;  /* [rows x dim] */
    float *proj;      /* [rows x dim] */
    float *scores;    /* [rows x rows] */
    float *ffn_gate;  /* [rows x hidden_dim] */
    float *ffn_up;    /* [rows x hidden_dim] */
    float *ffn_out;   /* [rows x dim] */
} NiyahTransformerLayerScratch;

/* Returns false on NULL pointers, zero dimensions, or size_t overflow. */
bool niyah_transformer_layer_f32(float *x,
                                  const NiyahTransformerLayerWeights *w,
                                  NiyahTransformerLayerScratch *scratch,
                                  size_t rows, size_t dim, size_t hidden_dim,
                                  size_t pos_offset, float rms_eps,
                                  float rope_theta_base);

#endif
