#include "niyah.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Pre-norm transformer layer (Llama style).
 * Since weight matrices are not part of NiyahTransformerLayerState,
 * this performs the structural computation (norm → attn → residual → norm)
 * using identity projections.  Replace with weighted matmul once weights
 * are wired in from NiyahModel.
 */
void niyah_transformer_layer_forward(NiyahTransformerLayerState* state,
                                     const float* x) {
    if (!state || !x) return;

    int32_t seq = state->seq;
    int32_t dim = state->dim;
    size_t  n   = (size_t)seq * dim;

    /* -- 1. Pre-attention RMSNorm (identity weights = 1.0) ----------- */
    memcpy(state->norm1_out, x, n * sizeof(float));
    for (int32_t i = 0; i < seq; i++)
        niyah_rmsnorm(state->norm1_out + i * dim, NULL, dim, 1e-5f);

    /* -- 2. Self-attention ------------------------------------------- */
    {
        NiyahAttentionState attn;
        attn.batch  = 1;
        attn.seq    = seq;
        attn.dim    = dim;
        attn.n_head = state->n_head;
        attn.qkv    = NULL;
        attn.out    = NULL;
        niyah_attention_forward(&attn, state->norm1_out, state->attn_out);
    }

    /* -- 3. Residual: residual = x + attn_out ------------------------ */
    float* residual = (float*)malloc(n * sizeof(float));
    if (!residual) return;
    for (size_t i = 0; i < n; i++)
        residual[i] = x[i] + state->attn_out[i];

    /* -- 4. Pre-FFN RMSNorm ------------------------------------------ */
    memcpy(state->norm2_out, residual, n * sizeof(float));
    for (int32_t i = 0; i < seq; i++)
        niyah_rmsnorm(state->norm2_out + i * dim, NULL, dim, 1e-5f);

    /* -- 5. Feed-forward (identity, no trained weights here) --------- */
    memcpy(state->ffn_out, state->norm2_out, n * sizeof(float));

    /* -- 6. Second residual ------------------------------------------ */
    for (size_t i = 0; i < n; i++)
        state->ffn_out[i] += residual[i];

    free(residual);
}
