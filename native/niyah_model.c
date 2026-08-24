#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was: `// Model stubs`.
 *
 * Reads the artefacts produced by tools/gguf_to_niyah.py:
 *   <name>.json  - config
 *   <name>.bin   - flat little-endian float32 blob, tensors concatenated as
 *                  embedding
 *                  per layer: attn_norm, wq, wk, wv, wo,
 *                             ffn_norm, ffn_gate, ffn_up, ffn_down
 *                  final_norm
 *                  lm_head
 */

void niyah_model_config_normalize(NiyahModelConfig* config)
{
    if (!config) {
        return;
    }
    if (config->n_kv_head <= 0) {
        config->n_kv_head = config->n_head;
    }
    if (config->n_ff <= 0) {
        config->n_ff = config->n_embd * 4;
    }
    if (!(config->rope_theta > 0.0f)) {
        config->rope_theta = 10000.0f;
    }
    if (!(config->norm_eps > 0.0f)) {
        config->norm_eps = 1e-5f;
    }
    if (config->n_ctx <= 0) {
        config->n_ctx = 2048;
    }
}

size_t niyah_model_expected_floats(const NiyahModelConfig* config)
{
    if (!config) {
        return 0;
    }

    NiyahModelConfig c = *config;
    niyah_model_config_normalize(&c);

    if (c.n_embd <= 0 || c.n_head <= 0 || c.n_layer <= 0 ||
        c.n_vocab <= 0 || c.n_embd % c.n_head != 0) {
        return 0;
    }

    const size_t dim = (size_t)c.n_embd;
    const size_t head_dim = dim / (size_t)c.n_head;
    const size_t kv_dim = (size_t)c.n_kv_head * head_dim;
    const size_t ff = (size_t)c.n_ff;
    const size_t vocab = (size_t)c.n_vocab;

    const size_t per_layer =
        dim                 /* attn_norm */
        + dim * dim         /* wq  [dim][dim]        */
        + kv_dim * dim      /* wk  [kv_dim][dim]     */
        + kv_dim * dim      /* wv  [kv_dim][dim]     */
        + dim * dim         /* wo  [dim][dim]        */
        + dim               /* ffn_norm */
        + ff * dim          /* ffn_gate [ff][dim]    */
        + ff * dim          /* ffn_up   [ff][dim]    */
        + dim * ff;         /* ffn_down [dim][ff]    */

    return vocab * dim                          /* embedding  */
         + (size_t)c.n_layer * per_layer
         + dim                                  /* final_norm */
         + vocab * dim;                         /* lm_head    */
}

/* Minimal scanner for the flat integer-valued config this project emits.
 * Deliberately not a general JSON parser. */
static bool json_find_int(const char* buf, const char* key, int32_t* out)
{
    char pattern[128];
    const int written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return false;
    }

    const char* p = strstr(buf, pattern);
    if (!p) {
        return false;
    }
    p += written;

    while (*p && *p != ':') {
        ++p;
    }
    if (*p != ':') {
        return false;
    }
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        ++p;
    }

    char* end = NULL;
    const long value = strtol(p, &end, 10);
    if (end == p) {
        return false;
    }

    *out = (int32_t)value;
    return true;
}

NiyahStatus niyah_model_load_config_json(NiyahModelConfig* config,
                                         const char* json_path)
{
    if (!config || !json_path) {
        return NIYAH_ERR_INVALID_ARG;
    }

    FILE* f = fopen(json_path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    const long size = ftell(f);
    if (size <= 0 || size > (1L << 20)) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    rewind(f);

    char* buf = (char*)malloc((size_t)size + 1u);
    if (!buf) {
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    const size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[read] = '\0';

    memset(config, 0, sizeof(*config));

    /* Key names match tools/gguf_to_niyah.py exactly. */
    json_find_int(buf, "vocab_size", &config->n_vocab);
    json_find_int(buf, "dim", &config->n_embd);
    json_find_int(buf, "heads", &config->n_head);
    json_find_int(buf, "layer_count", &config->n_layer);
    json_find_int(buf, "context_size", &config->n_ctx);
    json_find_int(buf, "kv_heads", &config->n_kv_head);
    json_find_int(buf, "hidden_dim", &config->n_ff);
    json_find_int(buf, "eos_token", &config->eos_token_id);

    free(buf);

    if (config->n_vocab <= 0 || config->n_embd <= 0 ||
        config->n_head <= 0 || config->n_layer <= 0) {
        return NIYAH_ERR_SHAPE;
    }

    niyah_model_config_normalize(config);
    return NIYAH_OK;
}

NiyahStatus niyah_model_load(NiyahModel* model,
                             const NiyahModelConfig* config,
                             const char* weights_path)
{
    if (!model || !config || !weights_path) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahModelConfig c = *config;
    niyah_model_config_normalize(&c);

    const size_t expected = niyah_model_expected_floats(&c);
    if (expected == 0) {
        return NIYAH_ERR_SHAPE;
    }

    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    const long file_size = ftell(f);
    rewind(f);

    if (file_size <= 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }

    const size_t actual_floats = (size_t)file_size / sizeof(float);
    const size_t tied = expected - (size_t)c.n_vocab * (size_t)c.n_embd;

    if (actual_floats == tied) {
        /* output.weight absent: embeddings are tied to the LM head. */
        c.tie_word_embeddings = true;
    } else if (actual_floats != expected) {
        fclose(f);
        return NIYAH_ERR_SHAPE;
    }

    const size_t want = c.tie_word_embeddings ? tied : expected;

    float* weights = (float*)malloc(want * sizeof(float));
    if (!weights) {
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    const size_t got = fread(weights, sizeof(float), want, f);
    fclose(f);

    if (got != want) {
        free(weights);
        return NIYAH_ERR_IO;
    }

    model->config = c;
    model->weights = weights;
    model->weights_size = want * sizeof(float);

    return NIYAH_OK;
}

void niyah_model_free(NiyahModel* model)
{
    if (!model) {
        return;
    }
    free(model->weights);
    model->weights = NULL;
    model->weights_size = 0;
}

NiyahStatus niyah_model_weights_map(NiyahModelWeights* out,
                                    const NiyahModel* model)
{
    if (!out || !model) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!model->weights) {
        return NIYAH_ERR_NO_WEIGHTS;
    }

    NiyahModelConfig c = model->config;
    niyah_model_config_normalize(&c);

    if (c.n_embd <= 0 || c.n_head <= 0 || c.n_embd % c.n_head != 0) {
        return NIYAH_ERR_SHAPE;
    }

    const size_t dim = (size_t)c.n_embd;
    const size_t head_dim = dim / (size_t)c.n_head;
    const size_t kv_dim = (size_t)c.n_kv_head * head_dim;
    const size_t ff = (size_t)c.n_ff;
    const size_t vocab = (size_t)c.n_vocab;

    memset(out, 0, sizeof(*out));

    out->layers = (NiyahLayerWeights*)calloc((size_t)c.n_layer,
                                             sizeof(NiyahLayerWeights));
    if (!out->layers) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    out->n_layer = c.n_layer;

    const float* p = (const float*)model->weights;

    out->embedding = p;
    p += vocab * dim;

    for (int32_t l = 0; l < c.n_layer; ++l) {
        NiyahLayerWeights* w = &out->layers[l];
        w->attn_norm = p; p += dim;
        w->wq        = p; p += dim * dim;
        w->wk        = p; p += kv_dim * dim;
        w->wv        = p; p += kv_dim * dim;
        w->wo        = p; p += dim * dim;
        w->ffn_norm  = p; p += dim;
        w->ffn_gate  = p; p += ff * dim;
        w->ffn_up    = p; p += ff * dim;
        w->ffn_down  = p; p += dim * ff;
    }

    out->final_norm = p;
    p += dim;

    out->lm_head = c.tie_word_embeddings ? out->embedding : p;

    return NIYAH_OK;
}

void niyah_model_weights_unmap(NiyahModelWeights* weights)
{
    if (!weights) {
        return;
    }
    free(weights->layers);
    memset(weights, 0, sizeof(*weights));
}
