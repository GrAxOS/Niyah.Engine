#include "niyah_transformer_layer.h"
#include "niyah_matmul.h"
#include "niyah_rmsnorm.h"
#include "niyah_rope.h"
#include "niyah_attention.h"
#include "niyah_swiglu.h"
#include "niyah_core.h"

bool niyah_transformer_layer_f32(float *x,
                                  const NiyahTransformerLayerWeights *w,
                                  NiyahTransformerLayerScratch *scratch,
                                  size_t rows, size_t dim, size_t hidden_dim,
                                  size_t pos_offset, float rms_eps,
                                  float rope_theta_base) {
    size_t rows_dim = 0u;
    size_t rows_hidden = 0u;

    if (x == NULL || w == NULL || scratch == NULL) {
        return false;
    }
    if (w->rms1_weight == NULL || w->w_q == NULL || w->w_k == NULL ||
        w->w_v == NULL || w->w_o == NULL || w->rms2_weight == NULL ||
        w->w_gate == NULL || w->w_up == NULL || w->w_down == NULL) {
        return false;
    }
    if (scratch->normed == NULL || scratch->q == NULL || scratch->k == NULL ||
        scratch->v == NULL || scratch->attn_out == NULL ||
        scratch->proj == NULL || scratch->scores == NULL ||
        scratch->ffn_gate == NULL || scratch->ffn_up == NULL ||
        scratch->ffn_out == NULL) {
        return false;
    }
    if (rows == 0u || dim == 0u || hidden_dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(rows, dim, &rows_dim) ||
        !niyah_mul_size(rows, hidden_dim, &rows_hidden)) {
        return false;
    }

    if (!niyah_rmsnorm_f32(x, w->rms1_weight, scratch->normed, rows, dim,
                            rms_eps)) {
        return false;
    }

    if (!niyah_matmul_f32_bt(scratch->normed, w->w_q, scratch->q, rows, dim,
                              dim) ||
        !niyah_matmul_f32_bt(scratch->normed, w->w_k, scratch->k, rows, dim,
                              dim) ||
        !niyah_matmul_f32_bt(scratch->normed, w->w_v, scratch->v, rows, dim,
                              dim)) {
        return false;
    }

    if (!niyah_rope_f32(scratch->q, rows, dim, pos_offset, rope_theta_base) ||
        !niyah_rope_f32(scratch->k, rows, dim, pos_offset, rope_theta_base)) {
        return false;
    }

    if (!niyah_attention_causal_f32(scratch->q, scratch->k, scratch->v,
                                     scratch->scores, scratch->attn_out, rows,
                                     dim)) {
        return false;
    }

    if (!niyah_matmul_f32_bt(scratch->attn_out, w->w_o, scratch->proj, rows,
                              dim, dim)) {
        return false;
    }

    for (size_t i = 0u; i < rows_dim; ++i) {
        x[i] += scratch->proj[i];
    }

    if (!niyah_rmsnorm_f32(x, w->rms2_weight, scratch->normed, rows, dim,
                            rms_eps)) {
        return false;
    }

    if (!niyah_swiglu_f32(scratch->normed, w->w_gate, w->w_up, w->w_down,
                           scratch->ffn_gate, scratch->ffn_up,
                           scratch->ffn_out, rows, dim, hidden_dim)) {
        return false;
    }

    for (size_t i = 0u; i < rows_dim; ++i) {
        x[i] += scratch->ffn_out[i];
    }

    (void)rows_hidden;
    return true;
}
