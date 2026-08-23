#include "niyah_transformer_layer.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 5e-4f;
}

static void test_transformer_layer_f32_full_pipeline(void) {
    /* rows=2, dim=2, hidden_dim=2, all projection weights = identity(2x2),
     * rms gains = 1. Expected values cross-checked against a NumPy
     * reference implementation of the exact same algorithm. */
    float x[4] = {1.0f, 2.0f, 3.0f, -1.0f};

    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float ones2[2] = {1.0f, 1.0f};

    NiyahTransformerLayerWeights weights;
    weights.rms1_weight = ones2;
    weights.w_q = identity2;
    weights.w_k = identity2;
    weights.w_v = identity2;
    weights.w_o = identity2;
    weights.rms2_weight = ones2;
    weights.w_gate = identity2;
    weights.w_up = identity2;
    weights.w_down = identity2;

    float normed[4] = {0};
    float q[4] = {0};
    float k[4] = {0};
    float v[4] = {0};
    float attn_out[4] = {0};
    float proj[4] = {0};
    float scores[4] = {0};
    float ffn_gate[4] = {0};
    float ffn_up[4] = {0};
    float ffn_out[4] = {0};

    NiyahTransformerLayerScratch scratch;
    scratch.normed = normed;
    scratch.q = q;
    scratch.k = k;
    scratch.v = v;
    scratch.attn_out = attn_out;
    scratch.proj = proj;
    scratch.scores = scores;
    scratch.ffn_gate = ffn_gate;
    scratch.ffn_up = ffn_up;
    scratch.ffn_out = ffn_out;

    assert(niyah_transformer_layer_f32(x, &weights, &scratch, 2u, 2u, 2u, 0u,
                                        1e-5f, 10000.0f));

    assert(nearly_equal(x[0], 1.89367225f));
    assert(nearly_equal(x[1], 4.51269898f));
    assert(nearly_equal(x[2], 5.57234225f));
    assert(nearly_equal(x[3], -0.62344619f));
}

static void test_transformer_layer_f32_rejects_invalid_input(void) {
    float x[2] = {1.0f, 2.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float ones2[2] = {1.0f, 1.0f};
    float scratch_buf[4] = {0};

    NiyahTransformerLayerWeights weights;
    weights.rms1_weight = ones2;
    weights.w_q = identity2;
    weights.w_k = identity2;
    weights.w_v = identity2;
    weights.w_o = identity2;
    weights.rms2_weight = ones2;
    weights.w_gate = identity2;
    weights.w_up = identity2;
    weights.w_down = identity2;

    NiyahTransformerLayerScratch scratch;
    scratch.normed = scratch_buf;
    scratch.q = scratch_buf;
    scratch.k = scratch_buf;
    scratch.v = scratch_buf;
    scratch.attn_out = scratch_buf;
    scratch.proj = scratch_buf;
    scratch.scores = scratch_buf;
    scratch.ffn_gate = scratch_buf;
    scratch.ffn_up = scratch_buf;
    scratch.ffn_out = scratch_buf;

    assert(!niyah_transformer_layer_f32(NULL, &weights, &scratch, 1u, 2u, 2u,
                                         0u, 1e-5f, 10000.0f));
    assert(!niyah_transformer_layer_f32(x, NULL, &scratch, 1u, 2u, 2u, 0u,
                                         1e-5f, 10000.0f));
    assert(!niyah_transformer_layer_f32(x, &weights, &scratch, 0u, 2u, 2u,
                                         0u, 1e-5f, 10000.0f));
}

int main(void) {
    test_transformer_layer_f32_full_pipeline();
    test_transformer_layer_f32_rejects_invalid_input();
    return 0;
}
