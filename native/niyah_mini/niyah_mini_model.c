#include "niyah_mini_model.h"
#include "niyah_mini_vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0U && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int size_add_ok(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static NiyahStatus model_config_ok(const NiyahMiniConfig *config)
{
    return niyah_mini_config_validate(config);
}

static void init_xavier(float *weights, size_t n_in, size_t n_out)
{
    size_t count;
    size_t i;
    float scale;
    if (!weights || !size_mul_ok(n_in, n_out, &count) || count == 0U) return;
    if (n_in > SIZE_MAX - n_out) return;
    scale = sqrtf(2.0f / (float)(n_in + n_out));
    for (i = 0U; i < count; ++i) {
        float v = ((float)(i % 1000U) / 1000.0f) - 0.5f;
        weights[i] = 2.0f * v * scale;
    }
}

static void init_small(float *weights, size_t count, float scale)
{
    size_t i;
    if (!weights || !isfinite(scale) || scale < 0.0f) return;
    for (i = 0U; i < count; ++i) {
        float v = ((float)(i % 1000U) / 1000.0f) - 0.5f;
        weights[i] = 2.0f * v * scale;
    }
}

static void zero_weights(NiyahMiniWeights *weights)
{
    if (weights) memset(weights, 0, sizeof(*weights));
}

void niyah_mini_weights_free(NiyahMiniWeights *weights)
{
    if (!weights) return;
    if (weights->owns_memory) free(weights->memory_block);
    free(weights->layers);
    zero_weights(weights);
}

size_t niyah_mini_weights_memory_size(const NiyahMiniConfig *config)
{
    size_t dim, vocab, layers, heads, kv_heads, head_dim, kv_dim, ff;
    size_t a, per_layer, total, bytes;
    if (!config || model_config_ok(config) != NIYAH_OK) return 0U;
    dim = (size_t)config->n_dim;
    vocab = (size_t)config->n_vocab;
    layers = (size_t)config->n_layers;
    heads = (size_t)config->n_heads;
    kv_heads = (size_t)config->n_kv_heads;
    ff = (size_t)config->n_ff;
    if (heads == 0U) return 0U;
    head_dim = dim / heads;
    if (!size_mul_ok(kv_heads, head_dim, &kv_dim)) return 0U;
    total = 0U;
    if (!size_mul_ok(vocab, dim, &a) || !size_add_ok(total, a, &total)) return 0U;
    if (!size_mul_ok(dim, dim, &a)) return 0U;
    per_layer = 0U;
    if (!size_add_ok(per_layer, dim, &per_layer)) return 0U;
    if (!size_add_ok(per_layer, a, &per_layer)) return 0U;
    if (!size_mul_ok(kv_dim, dim, &a)) return 0U;
    if (!size_add_ok(per_layer, a, &per_layer) || !size_add_ok(per_layer, a, &per_layer)) return 0U;
    if (!size_mul_ok(dim, dim, &a) || !size_add_ok(per_layer, a, &per_layer)) return 0U;
    if (!size_add_ok(per_layer, dim, &per_layer)) return 0U;
    if (!size_mul_ok(ff, dim, &a)) return 0U;
    if (!size_add_ok(per_layer, a, &per_layer) || !size_add_ok(per_layer, a, &per_layer) || !size_add_ok(per_layer, a, &per_layer)) return 0U;
    if (!size_mul_ok(layers, per_layer, &a) || !size_add_ok(total, a, &total)) return 0U;
    if (!size_add_ok(total, dim, &total)) return 0U;
    if (!config->tie_word_embeddings) {
        if (!size_mul_ok(vocab, dim, &a) || !size_add_ok(total, a, &total)) return 0U;
    }
    if (!size_mul_ok(total, sizeof(float), &bytes)) return 0U;
    return bytes;
}

NiyahStatus niyah_mini_weights_allocate(NiyahMiniWeights *weights, const NiyahMiniConfig *config)
{
    size_t total_size, layer_bytes;
    size_t dim, vocab, layers, heads, kv_heads, head_dim, kv_dim, ff;
    float *ptr;
    size_t l;
    void *block;
    if (!weights || !config) return NIYAH_ERR_INVALID_ARG;
    if (model_config_ok(config) != NIYAH_OK) return model_config_ok(config);
    niyah_mini_weights_free(weights);
    total_size = niyah_mini_weights_memory_size(config);
    if (total_size == 0U) return NIYAH_ERR_OVERFLOW;
    block = calloc(1U, total_size);
    if (!block) return NIYAH_ERR_OUT_OF_MEMORY;
    if (!size_mul_ok((size_t)config->n_layers, sizeof(NiyahMiniLayerWeights), &layer_bytes)) {
        free(block);
        return NIYAH_ERR_OVERFLOW;
    }
    weights->layers = (NiyahMiniLayerWeights *)calloc(1U, layer_bytes);
    if (!weights->layers) {
        free(block);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    weights->memory_block = block;
    weights->memory_size = total_size;
    weights->owns_memory = true;
    weights->n_layers = config->n_layers;
    ptr = (float *)block;
    dim = (size_t)config->n_dim;
    vocab = (size_t)config->n_vocab;
    layers = (size_t)config->n_layers;
    heads = (size_t)config->n_heads;
    kv_heads = (size_t)config->n_kv_heads;
    ff = (size_t)config->n_ff;
    head_dim = dim / heads;
    if (!size_mul_ok(kv_heads, head_dim, &kv_dim)) {
        niyah_mini_weights_free(weights);
        return NIYAH_ERR_OVERFLOW;
    }
    if (vocab > total_size / sizeof(float) / dim) {
        niyah_mini_weights_free(weights);
        return NIYAH_ERR_OVERFLOW;
    }
    weights->embedding = ptr;
    ptr += vocab * dim;
    for (l = 0U; l < layers; ++l) {
        NiyahMiniLayerWeights *w = &weights->layers[l];
        w->attn_norm = ptr; ptr += dim;
        w->wq = ptr; ptr += dim * dim;
        w->wk = ptr; ptr += kv_dim * dim;
        w->wv = ptr; ptr += kv_dim * dim;
        w->wo = ptr; ptr += dim * dim;
        w->ffn_norm = ptr; ptr += dim;
        w->ffn_gate = ptr; ptr += ff * dim;
        w->ffn_up = ptr; ptr += ff * dim;
        w->ffn_down = ptr; ptr += dim * ff;
    }
    weights->final_norm = ptr;
    ptr += dim;
    weights->lm_head = config->tie_word_embeddings ? weights->embedding : ptr;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_model_init(NiyahMiniModel *model, const NiyahMiniConfig *config)
{
    NiyahStatus status;
    if (!model || !config) return NIYAH_ERR_INVALID_ARG;
    memset(model, 0, sizeof(*model));
    status = model_config_ok(config);
    if (status != NIYAH_OK) return status;
    model->config = *config;
    status = niyah_mini_weights_allocate(&model->weights, config);
    if (status != NIYAH_OK) {
        memset(model, 0, sizeof(*model));
        return status;
    }
    niyah_mini_weights_init_small(&model->weights, config, 0.02f);
    return NIYAH_OK;
}

void niyah_mini_model_free(NiyahMiniModel *model)
{
    if (!model) return;
    niyah_mini_weights_free(&model->weights);
    free(model->kv_cache_k);
    free(model->kv_cache_v);
    free(model->scratch);
    memset(model, 0, sizeof(*model));
}

void niyah_mini_weights_init_xavier(NiyahMiniWeights *weights, const NiyahMiniConfig *config)
{
    size_t dim, vocab, layers, heads, kv_heads, head_dim, kv_dim, ff, l;
    if (!weights || !config || model_config_ok(config) != NIYAH_OK || !weights->memory_block) return;
    dim = (size_t)config->n_dim; vocab = (size_t)config->n_vocab; layers = (size_t)config->n_layers;
    heads = (size_t)config->n_heads; kv_heads = (size_t)config->n_kv_heads; ff = (size_t)config->n_ff;
    head_dim = dim / heads; kv_dim = kv_heads * head_dim;
    init_xavier(weights->embedding, vocab, dim);
    for (l = 0U; l < layers; ++l) {
        NiyahMiniLayerWeights *w = &weights->layers[l];
        init_xavier(w->attn_norm, dim, 1U);
        init_xavier(w->wq, dim, dim);
        init_xavier(w->wk, kv_dim, dim);
        init_xavier(w->wv, kv_dim, dim);
        init_xavier(w->wo, dim, dim);
        init_xavier(w->ffn_norm, dim, 1U);
        init_xavier(w->ffn_gate, ff, dim);
        init_xavier(w->ffn_up, ff, dim);
        init_xavier(w->ffn_down, dim, ff);
    }
    init_xavier(weights->final_norm, dim, 1U);
    if (!config->tie_word_embeddings) init_xavier(weights->lm_head, vocab, dim);
}

void niyah_mini_weights_init_small(NiyahMiniWeights *weights, const NiyahMiniConfig *config, float scale)
{
    size_t floats;
    if (!weights || !config || !weights->memory_block) return;
    if (niyah_mini_weights_memory_size(config) == 0U) return;
    floats = niyah_mini_weights_memory_size(config) / sizeof(float);
    init_small((float *)weights->memory_block, floats, scale);
}

NiyahStatus niyah_mini_weights_copy(NiyahMiniWeights *dst, const NiyahMiniWeights *src, const NiyahMiniConfig *config)
{
    size_t bytes;
    if (!dst || !src || !config) return NIYAH_ERR_INVALID_ARG;
    if (model_config_ok(config) != NIYAH_OK) return model_config_ok(config);
    if (!src->memory_block || !src->layers) return NIYAH_ERR_INVALID_ARG;
    bytes = niyah_mini_weights_memory_size(config);
    if (bytes == 0U || src->memory_size != bytes) return NIYAH_ERR_SHAPE;
    if (niyah_mini_weights_allocate(dst, config) != NIYAH_OK) return NIYAH_ERR_OUT_OF_MEMORY;
    memcpy(dst->memory_block, src->memory_block, bytes);
    return NIYAH_OK;
}

void niyah_mini_weights_scale(NiyahMiniWeights *weights, float scale)
{
    size_t count, i;
    if (!weights || !weights->memory_block || !isfinite(scale)) return;
    count = weights->memory_size / sizeof(float);
    for (i = 0U; i < count; ++i) ((float *)weights->memory_block)[i] *= scale;
}

NiyahStatus niyah_mini_model_load_weights(NiyahMiniModel *model, const char *weights_path)
{
    FILE *f;
    long end_pos;
    size_t expected, bytes_read;
    if (!model || !weights_path || !model->weights.memory_block) return NIYAH_ERR_INVALID_ARG;
    expected = model->weights.memory_size;
    f = fopen(weights_path, "rb");
    if (!f) return NIYAH_ERR_IO;
    if (fseek(f, 0L, SEEK_END) != 0) { fclose(f); return NIYAH_ERR_IO; }
    end_pos = ftell(f);
    if (end_pos < 0L) { fclose(f); return NIYAH_ERR_IO; }
    if ((unsigned long)end_pos > (unsigned long)SIZE_MAX || (size_t)end_pos != expected) { fclose(f); return NIYAH_ERR_SHAPE; }
    if (fseek(f, 0L, SEEK_SET) != 0) { fclose(f); return NIYAH_ERR_IO; }
    bytes_read = fread(model->weights.memory_block, 1U, expected, f);
    fclose(f);
    return bytes_read == expected ? NIYAH_OK : NIYAH_ERR_IO;
}

NiyahStatus niyah_mini_model_save_weights(const NiyahMiniModel *model, const char *weights_path)
{
    FILE *f;
    size_t written;
    if (!model || !weights_path || !model->weights.memory_block) return NIYAH_ERR_INVALID_ARG;
    f = fopen(weights_path, "wb");
    if (!f) return NIYAH_ERR_IO;
    written = fwrite(model->weights.memory_block, 1U, model->weights.memory_size, f);
    if (fclose(f) != 0) return NIYAH_ERR_IO;
    return written == model->weights.memory_size ? NIYAH_OK : NIYAH_ERR_IO;
}

static int read_int_field(const char *text, const char *key, long *value)
{
    const char *p;
    char *end;
    if (!text || !key || !value) return 0;
    p = strstr(text, key);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    *value = strtol(p, &end, 10);
    return end != p;
}

static int read_float_field(const char *text, const char *key, float *value)
{
    const char *p;
    char *end;
    if (!text || !key || !value) return 0;
    p = strstr(text, key);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    *value = strtof(p, &end);
    return end != p;
}

static int read_bool_field(const char *text, const char *key, bool *value)
{
    const char *p;
    if (!text || !key || !value) return 0;
    p = strstr(text, key);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (strncmp(p, "true", 4U) == 0) { *value = true; return 1; }
    if (strncmp(p, "false", 5U) == 0) { *value = false; return 1; }
    return 0;
}

NiyahStatus niyah_mini_model_load(NiyahMiniModel *model, const char *config_path, const char *weights_path)
{
    FILE *f;
    long size;
    char *buf;
    size_t readn;
    long v;
    float fv;
    bool bv;
    NiyahMiniConfig cfg;
    NiyahStatus status;
    if (!model || !config_path || !weights_path) return NIYAH_ERR_INVALID_ARG;
    f = fopen(config_path, "rb");
    if (!f) return NIYAH_ERR_IO;
    if (fseek(f, 0L, SEEK_END) != 0) { fclose(f); return NIYAH_ERR_IO; }
    size = ftell(f);
    if (size < 0L || (unsigned long)size > (unsigned long)(SIZE_MAX - 1U)) { fclose(f); return NIYAH_ERR_OVERFLOW; }
    if (fseek(f, 0L, SEEK_SET) != 0) { fclose(f); return NIYAH_ERR_IO; }
    buf = (char *)malloc((size_t)size + 1U);
    if (!buf) { fclose(f); return NIYAH_ERR_OUT_OF_MEMORY; }
    readn = fread(buf, 1U, (size_t)size, f);
    fclose(f);
    if (readn != (size_t)size) { free(buf); return NIYAH_ERR_IO; }
    buf[size] = '\0';
    niyah_mini_config_init(&cfg, NIYAH_MINI_BASE);
    if (read_int_field(buf, "n_layers", &v)) cfg.n_layers = (int32_t)v;
    if (read_int_field(buf, "n_dim", &v)) cfg.n_dim = (int32_t)v;
    if (read_int_field(buf, "n_heads", &v)) cfg.n_heads = (int32_t)v;
    if (read_int_field(buf, "n_kv_heads", &v)) cfg.n_kv_heads = (int32_t)v;
    if (read_int_field(buf, "n_ff", &v)) cfg.n_ff = (int32_t)v;
    if (read_int_field(buf, "n_vocab", &v)) cfg.n_vocab = (int32_t)v;
    if (read_int_field(buf, "n_ctx", &v)) cfg.n_ctx = (int32_t)v;
    if (read_float_field(buf, "rope_theta", &fv)) cfg.rope_theta = fv;
    if (read_float_field(buf, "norm_eps", &fv)) cfg.norm_eps = fv;
    if (read_bool_field(buf, "tie_word_embeddings", &bv)) cfg.tie_word_embeddings = bv;
    free(buf);
    status = model_config_ok(&cfg);
    if (status != NIYAH_OK) return status;
    niyah_mini_model_free(model);
    status = niyah_mini_model_init(model, &cfg);
    if (status != NIYAH_OK) return status;
    return niyah_mini_model_load_weights(model, weights_path);
}

NiyahStatus niyah_mini_model_save(const NiyahMiniModel *model, const char *config_path, const char *weights_path)
{
    FILE *f;
    if (!model || !config_path || !weights_path) return NIYAH_ERR_INVALID_ARG;
    if (model_config_ok(&model->config) != NIYAH_OK || !model->weights.memory_block) return NIYAH_ERR_INVALID_ARG;
    f = fopen(config_path, "wb");
    if (!f) return NIYAH_ERR_IO;
    if (fprintf(f, "{\n  \"n_layers\": %d,\n  \"n_dim\": %d,\n  \"n_heads\": %d,\n  \"n_kv_heads\": %d,\n  \"n_ff\": %d,\n  \"n_vocab\": %d,\n  \"n_ctx\": %d,\n  \"rope_theta\": %.9g,\n  \"norm_eps\": %.9g,\n  \"tie_word_embeddings\": %s\n}\n", model->config.n_layers, model->config.n_dim, model->config.n_heads, model->config.n_kv_heads, model->config.n_ff, model->config.n_vocab, model->config.n_ctx, model->config.rope_theta, model->config.norm_eps, model->config.tie_word_embeddings ? "true" : "false") < 0) { fclose(f); return NIYAH_ERR_IO; }
    if (fclose(f) != 0) return NIYAH_ERR_IO;
    return niyah_mini_model_save_weights(model, weights_path);
}

size_t niyah_mini_forward_state_memory_size(const NiyahMiniConfig *config, int32_t max_seq_len)
{
    size_t seq, dim, ff, total = 0U, a;
    if (!config || max_seq_len <= 0 || max_seq_len > NIYAH_MAX_SEQ_LEN || model_config_ok(config) != NIYAH_OK) return 0U;
    seq = (size_t)max_seq_len; dim = (size_t)config->n_dim; ff = (size_t)config->n_ff;
#define ADD_PRODUCT(x,y) do { if (!size_mul_ok((x),(y),&a) || !size_add_ok(total,a,&total)) return 0U; } while (0)
    ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim);
    ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, dim); ADD_PRODUCT(seq, seq); ADD_PRODUCT(seq, seq);
    ADD_PRODUCT(seq, ff); ADD_PRODUCT(seq, ff); ADD_PRODUCT(seq, dim);
#undef ADD_PRODUCT
    if (!size_mul_ok(total, sizeof(float), &a)) return 0U;
    return a;
}

NiyahStatus niyah_mini_forward_state_init(NiyahMiniForwardState *state, const NiyahMiniConfig *config, int32_t max_seq_len)
{
    void *block;
    float *ptr;
    size_t total;
    size_t seq, dim, ff;
    if (!state || !config || max_seq_len <= 0 || max_seq_len > NIYAH_MAX_SEQ_LEN) return NIYAH_ERR_INVALID_ARG;
    if (model_config_ok(config) != NIYAH_OK) return model_config_ok(config);
    memset(state, 0, sizeof(*state));
    total = niyah_mini_forward_state_memory_size(config, max_seq_len);
    if (total == 0U) return NIYAH_ERR_OVERFLOW;
    block = calloc(1U, total);
    if (!block) return NIYAH_ERR_OUT_OF_MEMORY;
    seq = (size_t)max_seq_len; dim = (size_t)config->n_dim; ff = (size_t)config->n_ff; ptr = (float *)block;
    state->hidden = ptr; ptr += seq * dim;
    state->norm1 = ptr; ptr += seq * dim;
    state->attn_out = ptr; ptr += seq * dim;
    state->norm2 = ptr; ptr += seq * dim;
    state->ffn_out = ptr; ptr += seq * dim;
    state->q = ptr; ptr += seq * dim;
    state->k = ptr; ptr += seq * dim;
    state->v = ptr; ptr += seq * dim;
    state->attn_scores = ptr; ptr += seq * seq;
    state->attn_probs = ptr; ptr += seq * seq;
    state->ffn_gate_out = ptr; ptr += seq * ff;
    state->ffn_up_out = ptr; ptr += seq * ff;
    state->layer_out = ptr;
    state->memory_block = block;
    state->memory_size = total;
    return NIYAH_OK;
}

void niyah_mini_forward_state_free(NiyahMiniForwardState *state)
{
    if (!state) return;
    free(state->memory_block);
    memset(state, 0, sizeof(*state));
}

static void rmsnorm(float *out, const float *x, const float *weight, int32_t n, float eps)
{
    double sum = 0.0;
    int32_t i;
    float inv;
    if (!out || !x || n <= 0 || !isfinite(eps) || eps <= 0.0f) return;
    for (i = 0; i < n; ++i) sum += (double)x[i] * (double)x[i];
    inv = (float)(1.0 / sqrt(sum / (double)n + (double)eps));
    for (i = 0; i < n; ++i) out[i] = x[i] * inv * (weight ? weight[i] : 1.0f);
}

static void matvec(float *out, const float *w, const float *x, int rows, int cols)
{
    int r, c;
    for (r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (c = 0; c < cols; ++c) acc += (double)w[(size_t)r * (size_t)cols + (size_t)c] * (double)x[c];
        out[r] = (float)acc;
    }
}

static void rope_apply_vector(float *x, int32_t dim, int32_t n_heads, int32_t position, float theta)
{
    int32_t head_dim, h, i;
    if (!x || dim <= 0 || n_heads <= 0 || dim % n_heads != 0 || theta <= 0.0f) return;
    head_dim = dim / n_heads;
    for (h = 0; h < n_heads; ++h) {
        float *p = x + h * head_dim;
        for (i = 0; i + 1 < head_dim; i += 2) {
            float angle = (float)position / powf(theta, (float)i / (float)head_dim);
            float c = cosf(angle), s = sinf(angle);
            float a = p[i], b = p[i + 1];
            p[i] = a * c - b * s;
            p[i + 1] = a * s + b * c;
        }
    }
}

static void silu_mul(float *out, const float *gate, const float *up, int32_t n)
{
    int32_t i;
    for (i = 0; i < n; ++i) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = g * sig * up[i];
    }
}

static NiyahStatus forward_one(NiyahMiniModel *model, NiyahMiniForwardState *state, int32_t token_id, int32_t position, float *logits)
{
    int32_t layer, h, d, t;
    const int32_t dim = model->config.n_dim;
    const int32_t heads = model->config.n_heads;
    const int32_t kv_heads = model->config.n_kv_heads;
    const int32_t ff = model->config.n_ff;
    const int32_t head_dim = dim / heads;
    const int32_t kv_dim = kv_heads * head_dim;
    float *x = state->hidden;
    if (token_id < 0 || token_id >= model->config.n_vocab || position < 0 || position >= model->config.n_ctx || !logits) return NIYAH_ERR_INVALID_ARG;
    memcpy(x, model->weights.embedding + (size_t)token_id * (size_t)dim, (size_t)dim * sizeof(float));
    for (layer = 0; layer < model->config.n_layers; ++layer) {
        const NiyahMiniLayerWeights *w = &model->weights.layers[layer];
        float *q = state->q, *k = state->k, *v = state->v;
        rmsnorm(state->norm1, x, w->attn_norm, dim, model->config.norm_eps);
        matvec(q, w->wq, state->norm1, dim, dim);
        matvec(k, w->wk, state->norm1, kv_dim, dim);
        matvec(v, w->wv, state->norm1, kv_dim, dim);
        rope_apply_vector(q, dim, heads, position, model->config.rope_theta);
        rope_apply_vector(k, kv_dim, kv_heads, position, model->config.rope_theta);
        if (position < model->config.n_ctx) {
            size_t layer_stride = (size_t)model->config.n_ctx * (size_t)kv_dim;
            float *cache_k = model->kv_cache_k + (size_t)layer * layer_stride + (size_t)position * (size_t)kv_dim;
            float *cache_v = model->kv_cache_v + (size_t)layer * layer_stride + (size_t)position * (size_t)kv_dim;
            memcpy(cache_k, k, (size_t)kv_dim * sizeof(float));
            memcpy(cache_v, v, (size_t)kv_dim * sizeof(float));
            memset(state->attn_out, 0, (size_t)dim * sizeof(float));
            for (h = 0; h < heads; ++h) {
                int32_t kvh = h % kv_heads;
                double max_score = -HUGE_VAL, sum = 0.0;
                for (t = 0; t <= position; ++t) {
                    const float *kh = model->kv_cache_k + (size_t)layer * layer_stride + (size_t)t * (size_t)kv_dim + (size_t)kvh * (size_t)head_dim;
                    double score = 0.0;
                    for (d = 0; d < head_dim; ++d) score += (double)q[h * head_dim + d] * (double)kh[d];
                    state->attn_scores[t] = (float)(score / sqrt((double)head_dim));
                    if (state->attn_scores[t] > max_score) max_score = state->attn_scores[t];
                }
                for (t = 0; t <= position; ++t) {
                    double e = exp((double)state->attn_scores[t] - max_score);
                    state->attn_probs[t] = (float)e;
                    sum += e;
                }
                if (sum <= 0.0 || !isfinite(sum)) sum = 1.0;
                for (t = 0; t <= position; ++t) state->attn_probs[t] = (float)((double)state->attn_probs[t] / sum);
                for (t = 0; t <= position; ++t) {
                    const float *vh = model->kv_cache_v + (size_t)layer * layer_stride + (size_t)t * (size_t)kv_dim + (size_t)kvh * (size_t)head_dim;
                    for (d = 0; d < head_dim; ++d) state->attn_out[h * head_dim + d] += state->attn_probs[t] * vh[d];
                }
            }
        } else {
            return NIYAH_ERR_SHAPE;
        }
        matvec(state->layer_out, w->wo, state->attn_out, dim, dim);
        for (d = 0; d < dim; ++d) x[d] += state->layer_out[d];
        rmsnorm(state->norm2, x, w->ffn_norm, dim, model->config.norm_eps);
        matvec(state->ffn_gate_out, w->ffn_gate, state->norm2, ff, dim);
        matvec(state->ffn_up_out, w->ffn_up, state->norm2, ff, dim);
        silu_mul(state->ffn_out, state->ffn_gate_out, state->ffn_up_out, ff);
        matvec(state->layer_out, w->ffn_down, state->ffn_out, dim, ff);
        for (d = 0; d < dim; ++d) x[d] += state->layer_out[d];
    }
    rmsnorm(state->norm1, x, model->weights.final_norm, dim, model->config.norm_eps);
    for (t = 0; t < model->config.n_vocab; ++t) {
        double acc = 0.0;
        const float *row = model->weights.lm_head + (size_t)t * (size_t)dim;
        for (d = 0; d < dim; ++d) acc += (double)row[d] * (double)state->norm1[d];
        logits[t] = (float)acc;
    }
    return NIYAH_OK;
}

static NiyahStatus ensure_runtime(NiyahMiniModel *model, const NiyahMiniConfig *config)
{
    size_t count, bytes;
    if (!model || !config) return NIYAH_ERR_INVALID_ARG;
    if (model->kv_cache_k && model->kv_cache_v) return NIYAH_OK;
    if (!size_mul_ok((size_t)config->n_layers, (size_t)config->n_ctx, &count) || !size_mul_ok(count, (size_t)config->n_kv_heads * ((size_t)config->n_dim / (size_t)config->n_heads), &count) || !size_mul_ok(count, sizeof(float), &bytes)) return NIYAH_ERR_OVERFLOW;
    model->kv_cache_k = (float *)calloc(1U, bytes);
    model->kv_cache_v = (float *)calloc(1U, bytes);
    if (!model->kv_cache_k || !model->kv_cache_v) {
        free(model->kv_cache_k); free(model->kv_cache_v); model->kv_cache_k = NULL; model->kv_cache_v = NULL;
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    model->kv_cache_seq_len = 0;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_forward_token(NiyahMiniModel *model, NiyahMiniForwardState *state, int32_t token_id, int32_t position, float *logits_out)
{
    NiyahStatus status;
    if (!model || !state || !logits_out || !model->weights.memory_block || !state->memory_block) return NIYAH_ERR_INVALID_ARG;
    status = ensure_runtime(model, &model->config);
    if (status != NIYAH_OK) return status;
    status = forward_one(model, state, token_id, position, logits_out);
    if (status == NIYAH_OK && position + 1 > model->kv_cache_seq_len) model->kv_cache_seq_len = position + 1;
    return status;
}

NiyahStatus niyah_mini_forward_sequence(NiyahMiniModel *model, NiyahMiniForwardState *state, const int32_t *input_ids, int32_t seq_len, float *logits_out)
{
    NiyahStatus status;
    int32_t i;
    size_t offset;
    if (!model || !state || !input_ids || !logits_out || seq_len <= 0 || seq_len > model->config.n_ctx) return NIYAH_ERR_INVALID_ARG;
    niyah_mini_reset_kv_cache(model);
    status = ensure_runtime(model, &model->config);
    if (status != NIYAH_OK) return status;
    for (i = 0; i < seq_len; ++i) {
        offset = (size_t)i * (size_t)model->config.n_vocab;
        status = forward_one(model, state, input_ids[i], i, logits_out + offset);
        if (status != NIYAH_OK) return status;
    }
    model->kv_cache_seq_len = seq_len;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_get_logits(NiyahMiniModel *model, const int32_t *input_ids, int32_t seq_len, float *logits)
{
    NiyahMiniForwardState state;
    NiyahStatus status;
    if (!model || !input_ids || !logits || seq_len <= 0 || seq_len > model->config.n_ctx) return NIYAH_ERR_INVALID_ARG;
    status = niyah_mini_forward_state_init(&state, &model->config, seq_len);
    if (status != NIYAH_OK) return status;
    status = niyah_mini_forward_sequence(model, &state, input_ids, seq_len, logits);
    niyah_mini_forward_state_free(&state);
    return status;
}

void niyah_mini_reset_kv_cache(NiyahMiniModel *model)
{
    size_t count, bytes;
    if (!model) return;
    if (!model->kv_cache_k || !model->kv_cache_v) { model->kv_cache_seq_len = 0; return; }
    if (!size_mul_ok((size_t)model->config.n_layers, (size_t)model->config.n_ctx, &count) || !size_mul_ok(count, (size_t)model->config.n_kv_heads * ((size_t)model->config.n_dim / (size_t)model->config.n_heads), &count) || !size_mul_ok(count, sizeof(float), &bytes)) { model->kv_cache_seq_len = 0; return; }
    memset(model->kv_cache_k, 0, bytes);
    memset(model->kv_cache_v, 0, bytes);
    model->kv_cache_seq_len = 0;
}

NiyahStatus niyah_mini_generate(NiyahMiniModel *model, const int32_t *prompt_ids, int32_t prompt_len, int32_t max_tokens, float temperature, int32_t *output_ids, int32_t *output_len)
{
    NiyahMiniForwardState state;
    float *logits;
    NiyahStatus status;
    int32_t i, next;
    if (!model || !prompt_ids || !output_ids || !output_len || prompt_len < 0 || max_tokens < 0 || prompt_len > model->config.n_ctx || temperature <= 0.0f || !isfinite(temperature)) return NIYAH_ERR_INVALID_ARG;
    *output_len = 0;
    status = niyah_mini_forward_state_init(&state, &model->config, model->config.n_ctx);
    if (status != NIYAH_OK) return status;
    logits = (float *)malloc((size_t)model->config.n_vocab * sizeof(float));
    if (!logits) { niyah_mini_forward_state_free(&state); return NIYAH_ERR_OUT_OF_MEMORY; }
    niyah_mini_reset_kv_cache(model);
    if (prompt_len > 0) {
        for (i = 0; i < prompt_len; ++i) {
            status = niyah_mini_forward_token(model, &state, prompt_ids[i], i, logits);
            if (status != NIYAH_OK) { free(logits); niyah_mini_forward_state_free(&state); return status; }
        }
        next = 0;
        for (i = 1; i < model->config.n_vocab; ++i) if (logits[i] > logits[next]) next = i;
    } else {
        next = NIYAH_MINI_BOS_TOKEN_ID;
    }
    for (i = 0; i < max_tokens; ++i) {
        int32_t position = prompt_len + i;
        int32_t best;
        int32_t j;
        if (position >= model->config.n_ctx) break;
        output_ids[*output_len] = next;
        ++(*output_len);
        if (next == NIYAH_MINI_EOS_TOKEN_ID) break;
        status = niyah_mini_forward_token(model, &state, next, position, logits);
        if (status != NIYAH_OK) break;
        best = 0;
        for (j = 1; j < model->config.n_vocab; ++j) if (logits[j] > logits[best]) best = j;
        next = best;
    }
    free(logits);
    niyah_mini_forward_state_free(&state);
    return NIYAH_OK;
}
