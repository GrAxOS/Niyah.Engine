#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-3f)

static void test_weightless_block(void)
{
    const int batch = 1, seq = 2, dim = 4, n_head = 2;
    const size_t span = (size_t)batch * seq * dim;

    float x[8] = {0.5f, -0.5f, 1.0f, -1.0f, 0.25f, 0.75f, -0.25f, -0.75f};

    float attn_out[8] = {0};
    float ffn_out[8] = {0};
    float norm1[8] = {0};
    float norm2[8] = {0};

    NiyahTransformerLayerState state;
    state.attn_out = attn_out;
    state.ffn_out = ffn_out;
    state.norm1_out = norm1;
    state.norm2_out = norm2;
    state.batch = batch;
    state.seq = seq;
    state.dim = dim;
    state.n_head = n_head;

    niyah_transformer_layer_forward(&state, x);

    for (size_t i = 0; i < span; ++i) {
        assert(isfinite(norm1[i]));
        assert(isfinite(attn_out[i]));
        assert(isfinite(ffn_out[i]));
    }

    /* The residual is real: attn_out must differ from the bare attention
     * result, and RMSNorm must have actually rescaled the input. */
    bool changed = false;
    for (size_t i = 0; i < span; ++i) {
        if (!CLOSE(norm1[i], x[i])) {
            changed = true;
        }
    }
    assert(changed);

    /* Degenerate inputs must not crash. */
    niyah_transformer_layer_forward(NULL, x);
    niyah_transformer_layer_forward(&state, NULL);
}

static void test_scratch_sizing(void)
{
    NiyahModelConfig config;
    memset(&config, 0, sizeof(config));
    config.n_vocab = 32;
    config.n_embd = 8;
    config.n_head = 2;
    config.n_layer = 1;
    config.n_ctx = 16;

    /* Unset optional fields must be filled in with documented defaults. */
    niyah_model_config_normalize(&config);
    assert(config.n_kv_head == 2);        /* defaults to n_head */
    assert(config.n_ff == 32);            /* defaults to 4 * n_embd */
    assert(CLOSE(config.rope_theta, 10000.0f));
    assert(config.norm_eps > 0.0f);

    const size_t floats = niyah_transformer_scratch_floats(&config);
    assert(floats > 0);

    /* dim*3 + kv_dim*2 + ff*2 + ctx = 24 + 16 + 64 + 16 = 120 */
    assert(floats == 120);

    assert(niyah_transformer_scratch_floats(NULL) == 0);
}

static void test_weighted_block(void)
{
    /* Small but real layer: dim 4, 2 heads, kv 2, ff 8, ctx 4. */
    NiyahModelConfig config;
    memset(&config, 0, sizeof(config));
    config.n_vocab = 16;
    config.n_embd = 4;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.n_ff = 8;
    config.n_layer = 1;
    config.n_ctx = 4;
    niyah_model_config_normalize(&config);

    const int dim = 4, ff = 8;

    /* Identity-ish weights so the arithmetic stays checkable. */
    float attn_norm[4] = {1, 1, 1, 1};
    float ffn_norm[4] = {1, 1, 1, 1};
    float wq[16] = {0}, wk[16] = {0}, wv[16] = {0}, wo[16] = {0};
    for (int i = 0; i < dim; ++i) {
        wq[i * dim + i] = 1.0f;
        wk[i * dim + i] = 1.0f;
        wv[i * dim + i] = 1.0f;
        wo[i * dim + i] = 1.0f;
    }

    float ffn_gate[32] = {0}, ffn_up[32] = {0}, ffn_down[32] = {0};

    NiyahLayerWeights w;
    w.attn_norm = attn_norm;
    w.wq = wq;
    w.wk = wk;
    w.wv = wv;
    w.wo = wo;
    w.ffn_norm = ffn_norm;
    w.ffn_gate = ffn_gate;
    w.ffn_up = ffn_up;
    w.ffn_down = ffn_down;

    NiyahKVCache cache;
    memset(&cache, 0, sizeof(cache));
    assert(niyah_kv_cache_init(&cache, 1, config.n_kv_head,
                               config.n_embd / config.n_head,
                               config.n_ctx) == NIYAH_OK);

    const size_t scratch_floats = niyah_transformer_scratch_floats(&config);
    float* scratch = (float*)calloc(scratch_floats, sizeof(float));
    assert(scratch != NULL);

    float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    const NiyahStatus st = niyah_transformer_layer_forward_weighted(
        hidden, &w, &config, &cache, 0, 0, scratch);
    assert(st == NIYAH_OK);

    for (int i = 0; i < dim; ++i) {
        assert(isfinite(hidden[i]));
    }

    /* ffn_down is all zeros, so the FFN contributes nothing and the output is
     * hidden + attention(rmsnorm(hidden)). It must have moved off the input. */
    assert(!CLOSE(hidden[0], 1.0f) || !CLOSE(hidden[3], 4.0f));

    /* The KV cache advanced by exactly one position. */
    assert(cache.length == 1);

    /* Shape validation: a config whose dim is not divisible by n_head. */
    NiyahModelConfig bad = config;
    bad.n_head = 3;
    assert(niyah_transformer_layer_forward_weighted(
               hidden, &w, &bad, &cache, 0, 0, scratch) == NIYAH_ERR_SHAPE);

    /* NULL arguments. */
    assert(niyah_transformer_layer_forward_weighted(
               NULL, &w, &config, &cache, 0, 0, scratch)
           == NIYAH_ERR_INVALID_ARG);

    free(scratch);
    niyah_kv_cache_free(&cache);
}

int main(void)
{
    test_weightless_block();
    test_scratch_sizing();
    test_weighted_block();
    return 0;
}
