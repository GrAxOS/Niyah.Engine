#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

/*
 * Tiny but structurally real model:
 *   n_vocab 8, n_embd 4, n_head 2, n_kv_head 2, n_ff 8, n_layer 2, n_ctx 8
 *   head_dim = 4 / 2 = 2
 *
 * Float counts, matching the on-disk order emitted by tools/gguf_to_niyah.py:
 *   embedding            8 * 4                    =  32
 *   per layer:
 *     attn_norm          4                        =   4
 *     wq   n_head*head_dim*n_embd   = 2*2*4       =  16
 *     wk   n_kv_head*head_dim*n_embd = 2*2*4      =  16
 *     wv                                          =  16
 *     wo   n_embd*(n_head*head_dim) = 4*4         =  16
 *     ffn_norm                                    =   4
 *     ffn_gate  n_ff*n_embd = 8*4                 =  32
 *     ffn_up                                      =  32
 *     ffn_down  n_embd*n_ff = 4*8                 =  32
 *                                        per layer = 168
 *   final_norm           4                        =   4
 *   lm_head              8 * 4                    =  32
 *
 *   total = 32 + 2*168 + 4 + 32 = 404
 */
#define TINY_TOTAL_FLOATS 404

static void fill_config(NiyahModelConfig* config)
{
    memset(config, 0, sizeof(*config));
    config->n_vocab = 8;
    config->n_embd = 4;
    config->n_head = 2;
    config->n_kv_head = 2;
    config->n_ff = 8;
    config->n_layer = 2;
    config->n_ctx = 8;
    config->eos_token_id = 7;
    niyah_model_config_normalize(config);
}

static void test_expected_floats(void)
{
    NiyahModelConfig config;
    fill_config(&config);

    assert(config.n_kv_head == 2);
    assert(config.n_ff == 8);
    assert(CLOSE(config.rope_theta, 10000.0f));
    assert(config.norm_eps > 0.0f);

    const size_t floats = niyah_model_expected_floats(&config);
    assert(floats == (size_t)TINY_TOTAL_FLOATS);

    assert(niyah_model_expected_floats(NULL) == 0);
}

static void test_weight_offsets(void)
{
    NiyahModelConfig config;
    fill_config(&config);

    float* blob = (float*)calloc(TINY_TOTAL_FLOATS, sizeof(float));
    assert(blob != NULL);
    for (int i = 0; i < TINY_TOTAL_FLOATS; ++i) {
        /* Small magnitudes so the forward pass stays numerically tame. */
        blob[i] = 0.01f * (float)((i % 7) - 3);
    }

    NiyahModel model;
    memset(&model, 0, sizeof(model));
    model.config = config;
    model.weights = blob;
    model.weights_size = (size_t)TINY_TOTAL_FLOATS * sizeof(float);

    NiyahModelWeights w;
    memset(&w, 0, sizeof(w));
    assert(niyah_model_weights_map(&w, &model) == NIYAH_OK);
    assert(w.n_layer == 2);
    assert(w.layers != NULL);

    /*
     * Hand-computed offsets. If tools/gguf_to_niyah.py ever changes the write
     * order, these assertions fail loudly instead of the engine reading
     * garbage weights and producing plausible-looking nonsense.
     */
    assert(w.embedding == blob + 0);

    assert(w.layers[0].attn_norm == blob + 32);
    assert(w.layers[0].wq        == blob + 36);
    assert(w.layers[0].wk        == blob + 52);
    assert(w.layers[0].wv        == blob + 68);
    assert(w.layers[0].wo        == blob + 84);
    assert(w.layers[0].ffn_norm  == blob + 100);
    assert(w.layers[0].ffn_gate  == blob + 104);
    assert(w.layers[0].ffn_up    == blob + 136);
    assert(w.layers[0].ffn_down  == blob + 168);

    /* Layer 1 begins exactly 168 floats after layer 0. */
    assert(w.layers[1].attn_norm == blob + 200);
    assert(w.layers[1].ffn_down  == blob + 336);

    assert(w.final_norm == blob + 368);
    assert(w.lm_head    == blob + 372);

    niyah_model_weights_unmap(&w);

    /* A blob that is too small must be refused, not mapped past its end. */
    NiyahModel truncated = model;
    truncated.weights_size = 16u * sizeof(float);
    NiyahModelWeights bad;
    memset(&bad, 0, sizeof(bad));
    assert(niyah_model_weights_map(&bad, &truncated) != NIYAH_OK);

    assert(niyah_model_weights_map(NULL, &model) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_model_weights_map(&bad, NULL) == NIYAH_ERR_INVALID_ARG);

    free(blob);
}

static void test_forward_pass(void)
{
    NiyahModelConfig config;
    fill_config(&config);

    float* blob = (float*)calloc(TINY_TOTAL_FLOATS, sizeof(float));
    assert(blob != NULL);
    for (int i = 0; i < TINY_TOTAL_FLOATS; ++i) {
        blob[i] = 0.01f * (float)((i % 7) - 3);
    }

    NiyahLLM llm;
    memset(&llm, 0, sizeof(llm));
    llm.model.config = config;
    llm.model.weights = blob;
    llm.model.weights_size = (size_t)TINY_TOTAL_FLOATS * sizeof(float);

    NiyahKVCache cache;
    memset(&cache, 0, sizeof(cache));
    assert(niyah_kv_cache_init(&cache,
                               config.n_layer,
                               config.n_kv_head,
                               config.n_embd / config.n_head,
                               config.n_ctx) == NIYAH_OK);

    const size_t scratch_floats = niyah_transformer_scratch_floats(&config);
    assert(scratch_floats > 0);
    float* scratch = (float*)calloc(scratch_floats, sizeof(float));
    assert(scratch != NULL);

    float logits[8] = {0};

    /* Real forward pass over a real (if tiny) weight blob. */
    assert(niyah_llm_forward(&llm, 0, 0, &cache, logits, scratch) == NIYAH_OK);

    for (int i = 0; i < 8; ++i) {
        assert(isfinite(logits[i]));
    }

    /* argmax over the logits must be a valid token id. */
    const int32_t next = niyah_argmax(logits, 8);
    assert(next >= 0 && next < 8);

    /* Determinism: same token and position produce the same logits. */
    float logits_again[8] = {0};
    niyah_kv_cache_reset(&cache);
    assert(niyah_llm_forward(&llm, 0, 0, &cache, logits_again, scratch)
           == NIYAH_OK);
    for (int i = 0; i < 8; ++i) {
        assert(CLOSE(logits[i], logits_again[i]));
    }

    /* A different input token must produce different logits, otherwise the
     * embedding lookup is not actually being used. */
    float other[8] = {0};
    niyah_kv_cache_reset(&cache);
    assert(niyah_llm_forward(&llm, 3, 0, &cache, other, scratch) == NIYAH_OK);
    bool differs = false;
    for (int i = 0; i < 8; ++i) {
        if (!CLOSE(logits[i], other[i])) {
            differs = true;
        }
    }
    assert(differs);

    /* Out-of-range token ids must be refused rather than read out of bounds. */
    assert(niyah_llm_forward(&llm, 999, 0, &cache, logits, scratch)
           == NIYAH_ERR_INVALID_ARG);
    assert(niyah_llm_forward(&llm, -1, 0, &cache, logits, scratch)
           == NIYAH_ERR_INVALID_ARG);

    free(scratch);
    niyah_kv_cache_free(&cache);
    free(blob);
}

int main(void)
{
    test_expected_floats();
    test_weight_offsets();
    test_forward_pass();
    return 0;
}
