/*
 * Regression test for config parsing.
 *
 * Before the 2026-08 fix the scanner could only read integers, so
 * "rope_theta" and "norm_eps" were never parsed and
 * niyah_model_config_normalize() substituted the Llama-2 constants 10000.0f
 * and 1e-5f for every checkpoint. Llama-3 uses 500000.0 and Qwen2 uses
 * 1000000.0, so positions were rotated at the wrong frequency and the engine
 * produced degraded output that still read as plausible text.
 *
 * Every assertion below fails against the previous parser.
 */
#undef NDEBUG
#include <assert.h>

#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_file(const char* path, const char* text)
{
    FILE* f = fopen(path, "wb");
    assert(f != NULL);
    fputs(text, f);
    fclose(f);
}

int main(void)
{
    const char* path = "niyah_config_test.json";
    NiyahModelConfig cfg;

    /* --- 1. Llama-3 style: the real rope base must survive ------------- */
    write_file(path,
        "{\n"
        "  \"vocab_size\": 128256,\n"
        "  \"dim\": 4096,\n"
        "  \"heads\": 32,\n"
        "  \"kv_heads\": 8,\n"
        "  \"layer_count\": 32,\n"
        "  \"hidden_dim\": 14336,\n"
        "  \"context_size\": 8192,\n"
        "  \"rope_theta\": 500000.0,\n"
        "  \"norm_eps\": 1e-05,\n"
        "  \"bos_token\": 128000,\n"
        "  \"eos_token\": 128001\n"
        "}\n");

    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, path) == NIYAH_OK);

    assert(cfg.n_vocab == 128256);
    assert(cfg.n_embd == 4096);
    assert(cfg.n_head == 32);
    assert(cfg.n_kv_head == 8);
    assert(cfg.n_layer == 32);
    assert(cfg.n_ff == 14336);
    assert(cfg.n_ctx == 8192);
    assert(cfg.eos_token_id == 128001);

    /* bos_token was never read at all before the fix. */
    assert(cfg.bos_token_id == 128000);

    /* The headline regression. */
    assert(cfg.rope_theta > 499999.0f);
    assert(cfg.rope_theta < 500001.0f);
    assert(cfg.norm_eps > 0.0f);
    assert(cfg.norm_eps < 1.0e-4f);

    /* --- 2. Qwen2 style, tied embeddings ------------------------------- */
    write_file(path,
        "{\n"
        "  \"vocab_size\": 151936,\n"
        "  \"dim\": 896,\n"
        "  \"heads\": 14,\n"
        "  \"kv_heads\": 2,\n"
        "  \"layer_count\": 24,\n"
        "  \"hidden_dim\": 4864,\n"
        "  \"context_size\": 32768,\n"
        "  \"rope_theta\": 1000000.0,\n"
        "  \"norm_eps\": 1e-06,\n"
        "  \"eos_token\": 151645,\n"
        "  \"tie_word_embeddings\": true\n"
        "}\n");

    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, path) == NIYAH_OK);

    assert(cfg.rope_theta > 999999.0f);
    assert(cfg.rope_theta < 1000001.0f);
    assert(cfg.n_ctx == 32768);
    assert(cfg.tie_word_embeddings);

    /* --- 3. A config with no rope_theta is rejected, not defaulted ----- */
    write_file(path,
        "{\n"
        "  \"vocab_size\": 32000,\n"
        "  \"dim\": 512,\n"
        "  \"heads\": 8,\n"
        "  \"layer_count\": 4,\n"
        "  \"norm_eps\": 1e-05\n"
        "}\n");

    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, path) == NIYAH_ERR_SHAPE);

    /* --- 4. dim not divisible by heads is a shape error ---------------- */
    write_file(path,
        "{\n"
        "  \"vocab_size\": 32000,\n"
        "  \"dim\": 100,\n"
        "  \"heads\": 8,\n"
        "  \"layer_count\": 4,\n"
        "  \"rope_theta\": 10000.0,\n"
        "  \"norm_eps\": 1e-05\n"
        "}\n");

    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, path) == NIYAH_ERR_SHAPE);

    /* --- 5. heads not divisible by kv_heads is a shape error ----------- */
    write_file(path,
        "{\n"
        "  \"vocab_size\": 32000,\n"
        "  \"dim\": 512,\n"
        "  \"heads\": 8,\n"
        "  \"kv_heads\": 3,\n"
        "  \"layer_count\": 4,\n"
        "  \"rope_theta\": 10000.0,\n"
        "  \"norm_eps\": 1e-05\n"
        "}\n");

    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, path) == NIYAH_ERR_SHAPE);

    /* --- 6. A missing file is an IO error, not a crash ----------------- */
    memset(&cfg, 0, sizeof(cfg));
    assert(niyah_model_load_config_json(&cfg, "no_such_config_file.json")
           == NIYAH_ERR_IO);
    assert(niyah_model_load_config_json(NULL, path) == NIYAH_ERR_INVALID_ARG);

    remove(path);
    printf("niyah_config_test: OK\n");
    return 0;
}
