#include "niyah_llm.h"

#include "niyah_core.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool dtype_size(NiyahLlmDType dtype, size_t *out)
{
    if (!out) return false;
    switch (dtype) {
        case NIYAH_LLM_DTYPE_F32: *out = sizeof(float); return true;
        case NIYAH_LLM_DTYPE_F16: *out = sizeof(uint16_t); return true;
        default: return false;
    }
}

bool niyah_llm_config_validate(const NiyahLlmConfig *config)
{
    if (!config || config->vocab_size == 0u || config->context_length == 0u ||
        config->embedding_dim == 0u || config->layer_count == 0u ||
        config->attention_heads == 0u || config->kv_heads == 0u || config->ffn_dim == 0u) return false;
    return config->kv_heads <= config->attention_heads &&
           config->attention_heads % config->kv_heads == 0u &&
           config->embedding_dim % config->attention_heads == 0u;
}

bool niyah_llm_tensor_spec_validate(const NiyahLlmTensorSpec *spec)
{
    if (!spec || spec->rank == 0u || spec->rank > 4u) return false;
    if (spec->dtype != NIYAH_LLM_DTYPE_F32 && spec->dtype != NIYAH_LLM_DTYPE_F16) return false;
    for (uint32_t i = 0u; i < spec->rank; ++i) if (spec->shape[i] == 0u) return false;
    return true;
}

bool niyah_llm_tensor_element_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!out || !niyah_llm_tensor_spec_validate(spec)) return false;
    size_t count = 1u;
    for (uint32_t i = 0u; i < spec->rank; ++i)
        if (!niyah_mul_size(count, (size_t)spec->shape[i], &count)) return false;
    *out = count;
    return true;
}

bool niyah_llm_tensor_byte_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!out) return false;
    size_t elements = 0u, element_size = 0u;
    if (!niyah_llm_tensor_element_count(spec, &elements) || !dtype_size(spec->dtype, &element_size)) return false;
    return niyah_mul_size(elements, element_size, out);
}

static bool shape_count(uint32_t rank, const uint32_t shape[4], size_t *out)
{
    if (!out || !shape || rank == 0u || rank > 4u) return false;
    size_t count = 1u;
    for (uint32_t i = 0u; i < rank; ++i)
        if (shape[i] == 0u || !niyah_mul_size(count, (size_t)shape[i], &count)) return false;
    *out = count;
    return true;
}

static bool alloc_f32(float **ptr, size_t count)
{
    if (!ptr || count == 0u || count > SIZE_MAX / sizeof(float)) return false;
    *ptr = (float *)calloc(count, sizeof(float));
    return *ptr != NULL;
}

bool niyah_llm_tensor_f32_init(NiyahLlmTensorF32 *tensor, uint32_t rank, const uint32_t shape[4])
{
    if (!tensor || !shape) return false;
    size_t count = 0u;
    if (!shape_count(rank, shape, &count)) return false;
    memset(tensor, 0, sizeof(*tensor));
    if (!alloc_f32(&tensor->data, count)) return false;
    tensor->element_count = count;
    tensor->rank = rank;
    memcpy(tensor->shape, shape, sizeof(tensor->shape));
    return true;
}

void niyah_llm_tensor_f32_free(NiyahLlmTensorF32 *tensor)
{
    if (!tensor) return;
    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}

bool niyah_llm_tensor_f32_reshape(NiyahLlmTensorF32 *tensor, uint32_t rank, const uint32_t shape[4])
{
    if (!tensor || !tensor->data || !shape) return false;
    size_t count = 0u;
    if (!shape_count(rank, shape, &count) || count != tensor->element_count) return false;
    tensor->rank = rank;
    memcpy(tensor->shape, shape, sizeof(tensor->shape));
    return true;
}

bool niyah_llm_matmul(const NiyahLlmTensorF32 *a, const NiyahLlmTensorF32 *b, NiyahLlmTensorF32 *out)
{
    if (!a || !b || !out || !a->data || !b->data || !out->data ||
        a->rank != 2u || b->rank != 2u || out->rank != 2u) return false;
    const uint32_t m = a->shape[0], k = a->shape[1], bk = b->shape[0], n = b->shape[1];
    if (k != bk || out->shape[0] != m || out->shape[1] != n) return false;
    size_t a_count = 0u, b_count = 0u, out_count = 0u;
    if (!niyah_mul_size((size_t)m, (size_t)k, &a_count) ||
        !niyah_mul_size((size_t)k, (size_t)n, &b_count) ||
        !niyah_mul_size((size_t)m, (size_t)n, &out_count) ||
        a_count != a->element_count || b_count != b->element_count || out_count != out->element_count) return false;
    for (uint32_t i = 0u; i < m; ++i) {
        for (uint32_t j = 0u; j < n; ++j) {
            float sum = 0.0f;
            for (uint32_t p = 0u; p < k; ++p)
                sum += a->data[(size_t)i * k + p] * b->data[(size_t)p * n + j];
            out->data[(size_t)i * n + j] = sum;
        }
    }
    return true;
}

bool niyah_llm_apply_rope(float *query, float *key, uint32_t head_dim, uint32_t position, float theta)
{
    if (!query || !key || head_dim == 0u || (head_dim & 1u) != 0u || !isfinite(theta) || theta <= 0.0f) return false;
    for (uint32_t i = 0u; i < head_dim; i += 2u) {
        const float exponent = (float)i / (float)head_dim;
        const float angle = (float)position * powf(theta, -exponent);
        const float c = cosf(angle), s = sinf(angle);
        const float q0 = query[i], q1 = query[i + 1u], k0 = key[i], k1 = key[i + 1u];
        query[i] = q0 * c - q1 * s; query[i + 1u] = q0 * s + q1 * c;
        key[i] = k0 * c - k1 * s; key[i + 1u] = k0 * s + k1 * c;
    }
    return true;
}

bool niyah_llm_attention(const float *query, const float *keys, const float *values,
                         uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                         uint32_t token_count, float *output)
{
    if (!query || !keys || !values || !output || query_heads == 0u || kv_heads == 0u || head_dim == 0u || token_count == 0u) return false;
    if (query_heads % kv_heads != 0u) return false;
    size_t total_kv = 0u, tmp = 0u;
    if (!niyah_mul_size((size_t)token_count, (size_t)kv_heads, &tmp) ||
        !niyah_mul_size(tmp, (size_t)head_dim, &total_kv) ||
        total_kv > SIZE_MAX / sizeof(float)) return false;
    size_t output_elems = 0u;
    if (!niyah_mul_size((size_t)query_heads, (size_t)head_dim, &output_elems)) return false;
    const uint32_t group = query_heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *weights = NULL;
    if (!alloc_f32(&weights, (size_t)token_count)) return false;
    for (uint32_t h = 0u; h < query_heads; ++h) {
        const uint32_t kvh = h / group;
        const float *q = query + (size_t)h * head_dim;
        float max_score = -INFINITY;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const float *k = keys + (((size_t)t * kv_heads) + kvh) * head_dim;
            float score = 0.0f;
            for (uint32_t d = 0u; d < head_dim; ++d) score += q[d] * k[d];
            weights[t] = score * scale;
            if (weights[t] > max_score) max_score = weights[t];
        }
        float sum = 0.0f;
        for (uint32_t t = 0u; t < token_count; ++t) { weights[t] = expf(weights[t] - max_score); sum += weights[t]; }
        if (!(sum > 0.0f) || !isfinite(sum)) { free(weights); return false; }
        float *out = output + (size_t)h * head_dim;
        memset(out, 0, (size_t)head_dim * sizeof(float));
        for (uint32_t t = 0u; t < token_count; ++t) {
            const float weight = weights[t] / sum;
            const float *v = values + (((size_t)t * kv_heads) + kvh) * head_dim;
            for (uint32_t d = 0u; d < head_dim; ++d) out[d] += weight * v[d];
        }
    }
    (void)output_elems;
    free(weights);
    return true;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

bool niyah_llm_ffn_swiglu(const float *input, const float *gate, const float *up, const float *down,
                          uint32_t input_dim, uint32_t hidden_dim, float *output, float *scratch)
{
    if (!input || !gate || !up || !down || !output || !scratch || input_dim == 0u || hidden_dim == 0u) return false;
    for (uint32_t h = 0u; h < hidden_dim; ++h) {
        float g = 0.0f, u = 0.0f;
        for (uint32_t i = 0u; i < input_dim; ++i) { g += input[i] * gate[(size_t)i * hidden_dim + h]; u += input[i] * up[(size_t)i * hidden_dim + h]; }
        scratch[h] = silu(g) * u;
    }
    for (uint32_t i = 0u; i < input_dim; ++i) {
        float sum = 0.0f;
        for (uint32_t h = 0u; h < hidden_dim; ++h) sum += scratch[h] * down[(size_t)h * input_dim + i];
        output[i] = sum;
    }
    return true;
}

bool niyah_llm_kv_cache_init(NiyahLlmKvCache *cache, uint32_t context_length, uint32_t head_count, uint32_t head_dim)
{
    if (!cache || context_length == 0u || head_count == 0u || head_dim == 0u) return false;
    memset(cache, 0, sizeof(*cache));
    size_t per_token = 0u, total = 0u;
    if (!niyah_mul_size((size_t)head_count, (size_t)head_dim, &per_token) ||
        !niyah_mul_size((size_t)context_length, per_token, &total)) return false;
    if (!alloc_f32(&cache->keys, total) || !alloc_f32(&cache->values, total)) { niyah_llm_kv_cache_free(cache); return false; }
    cache->capacity_elements = total; cache->head_count = head_count; cache->head_dim = head_dim;
    return true;
}

void niyah_llm_kv_cache_free(NiyahLlmKvCache *cache) { if (!cache) return; free(cache->keys); free(cache->values); memset(cache, 0, sizeof(*cache)); }

bool niyah_llm_kv_cache_append(NiyahLlmKvCache *cache, const float *keys, const float *values, uint32_t token_count)
{
    if (!cache || !keys || !values || token_count == 0u || token_count > UINT32_MAX - cache->token_count) return false;
    size_t batch_elements = 0u, current_offset = 0u;
    if (!niyah_mul_size((size_t)token_count, (size_t)cache->head_count, &batch_elements) ||
        !niyah_mul_size(batch_elements, (size_t)cache->head_dim, &batch_elements) ||
        !niyah_mul_size((size_t)cache->token_count, (size_t)cache->head_count, &current_offset) ||
        !niyah_mul_size(current_offset, (size_t)cache->head_dim, &current_offset) ||
        batch_elements > cache->capacity_elements || current_offset > cache->capacity_elements - batch_elements) return false;
    const size_t byte_count = batch_elements * sizeof(float);
    if (byte_count > 0u && batch_elements > SIZE_MAX / sizeof(float)) return false;
    memcpy(cache->keys + current_offset, keys, byte_count);
    memcpy(cache->values + current_offset, values, byte_count);
    cache->token_count += token_count;
    return true;
}

static bool is_token_char(unsigned char c) { return c >= 0x80u || isalnum(c) != 0 || c == '_' || c == '\''; }
static int vocab_find(const char **vocabulary, uint32_t vocabulary_size, const char *token)
{
    if (!vocabulary || !token) return -1;
    for (uint32_t i = 0u; i < vocabulary_size; ++i) if (vocabulary[i] && strcmp(vocabulary[i], token) == 0) return (int)i;
    return -1;
}

bool niyah_llm_wordpiece_tokenize(const char *text, const NiyahLlmWordPieceVocab *model,
                                  uint32_t *token_ids, size_t token_capacity, size_t *token_count)
{
    if (!text || !model || !model->vocabulary || model->vocabulary_size == 0u || !token_ids || !token_count || token_capacity == 0u || model->max_token_bytes < 3u) return false;
    *token_count = 0u;
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor) {
        while (*cursor && !is_token_char(*cursor)) ++cursor;
        if (!*cursor) break;
        const unsigned char *start = cursor;
        while (*cursor && is_token_char(*cursor)) ++cursor;
        const size_t len = (size_t)(cursor - start);
        size_t pos = 0u;
        while (pos < len) {
            size_t best_len = 0u; int best_id = -1;
            for (size_t take = len - pos; take > 0u; --take) {
                const size_t prefix = pos == 0u ? 0u : 2u;
                if (take > SIZE_MAX - prefix - 1u || prefix + take + 1u > model->max_token_bytes) continue;
                const size_t candidate_len = prefix + take;
                char *candidate = (char *)malloc(candidate_len + 1u);
                if (!candidate) return false;
                size_t at = 0u;
                if (prefix != 0u) { candidate[0] = '#'; candidate[1] = '#'; at = 2u; }
                memcpy(candidate + at, start + pos, take); candidate[candidate_len] = '\0';
                const int id = vocab_find(model->vocabulary, model->vocabulary_size, candidate); free(candidate);
                if (id >= 0) { best_len = take; best_id = id; break; }
            }
            if (best_id < 0 || *token_count >= token_capacity) return false;
            token_ids[(*token_count)++] = (uint32_t)best_id; pos += best_len;
        }
    }
    return true;
}

static int merge_result_find(const NiyahLlmBpeModel *model, uint32_t left, uint32_t right)
{
    if (!model || !model->vocabulary || left >= model->vocabulary_size || right >= model->vocabulary_size) return -1;
    const char *a = model->vocabulary[left], *b = model->vocabulary[right]; if (!a || !b) return -1;
    const size_t al = strlen(a), bl = strlen(b); if (al > SIZE_MAX - bl || al + bl >= model->max_token_bytes) return -1;
    char *combined = (char *)malloc(al + bl + 1u); if (!combined) return -1;
    memcpy(combined, a, al); memcpy(combined + al, b, bl); combined[al + bl] = '\0';
    const int result = vocab_find(model->vocabulary, model->vocabulary_size, combined); free(combined); return result;
}

bool niyah_llm_bpe_tokenize(const char *text, const NiyahLlmBpeModel *model,
                            uint32_t *token_ids, size_t token_capacity, size_t *token_count)
{
    if (!text || !model || !model->vocabulary || !model->merge_left || !model->merge_right || !token_ids || !token_count ||
        model->vocabulary_size == 0u || model->max_token_bytes < 2u || token_capacity == 0u) return false;
    *token_count = 0u; const size_t text_len = strlen(text); if (text_len == 0u) return true; if (text_len > token_capacity) return false;
    size_t count = 0u;
    for (size_t i = 0u; i < text_len; ++i) {
        char token[2] = {(char)text[i], '\0'}; const int id = vocab_find(model->vocabulary, model->vocabulary_size, token);
        if (id < 0) return false; token_ids[count++] = (uint32_t)id;
    }
    bool changed = true;
    while (changed && count > 1u) {
        changed = false;
        for (uint32_t m = 0u; m < model->merge_count; ++m) {
            const uint32_t left = model->merge_left[m], right = model->merge_right[m];
            if (left >= model->vocabulary_size || right >= model->vocabulary_size) return false;
            for (size_t i = 0u; i + 1u < count; ++i) {
                if (token_ids[i] != left || token_ids[i + 1u] != right) continue;
                const int merged = merge_result_find(model, left, right); if (merged < 0) return false;
                token_ids[i] = (uint32_t)merged;
                const size_t move = count - i - 2u;
                if (move > 0u) memmove(&token_ids[i + 1u], &token_ids[i + 2u], move * sizeof(*token_ids));
                --count; changed = true; break;
            }
            if (changed) break;
        }
    }
    *token_count = count; return true;
}

void niyah_llm_logits_softmax(float *logits, uint32_t count)
{
    if (!logits || count == 0u) return;
    float max_value = -INFINITY; for (uint32_t i = 0u; i < count; ++i) if (isfinite(logits[i]) && logits[i] > max_value) max_value = logits[i];
    if (!isfinite(max_value)) { const float u = 1.0f / (float)count; for (uint32_t i = 0u; i < count; ++i) logits[i] = u; return; }
    float sum = 0.0f; for (uint32_t i = 0u; i < count; ++i) { logits[i] = expf(logits[i] - max_value); sum += logits[i]; }
    if (!(sum > 0.0f) || !isfinite(sum)) return; for (uint32_t i = 0u; i < count; ++i) logits[i] /= sum;
}

typedef struct { uint32_t id; float probability; } Candidate;
static uint64_t rng_next(uint64_t *state) { uint64_t x = *state; if (x == 0u) x = UINT64_C(0x9e3779b97f4a7c15); x ^= x >> 12u; x ^= x << 25u; x ^= x >> 27u; *state = x; return x * UINT64_C(2685821657736338717); }
static float rng_uniform(uint64_t *state) { const uint64_t v = rng_next(state) >> 11u; return (float)((double)v / 9007199254740992.0); }
static int candidate_cmp(const void *l, const void *r) { const Candidate *a = (const Candidate *)l, *b = (const Candidate *)r; if (a->probability < b->probability) return 1; if (a->probability > b->probability) return -1; if (a->id < b->id) return -1; if (a->id > b->id) return 1; return 0; }

bool niyah_llm_sample(const float *logits, uint32_t count, const NiyahLlmSamplerConfig *config, NiyahLlmSample *out)
{
    if (!logits || count == 0u || !config || !out || !isfinite(config->temperature) || config->temperature <= 0.0f || !(config->top_p > 0.0f) || config->top_p > 1.0f) return false;
    if ((size_t)count > SIZE_MAX / sizeof(Candidate)) return false;
    Candidate *candidates = (Candidate *)calloc(count, sizeof(*candidates)); float *probabilities = NULL;
    if (!alloc_f32(&probabilities, (size_t)count)) { free(candidates); return false; }
    for (uint32_t i = 0u; i < count; ++i) probabilities[i] = logits[i] / config->temperature;
    niyah_llm_logits_softmax(probabilities, count);
    for (uint32_t i = 0u; i < count; ++i) { candidates[i].id = i; candidates[i].probability = probabilities[i]; }
    qsort(candidates, count, sizeof(*candidates), candidate_cmp);
    uint32_t keep = count; if (config->top_k > 0u && config->top_k < keep) keep = config->top_k;
    float cumulative = 0.0f; uint32_t nucleus = 0u;
    for (uint32_t i = 0u; i < keep; ++i) { cumulative += candidates[i].probability; ++nucleus; if (cumulative >= config->top_p) break; }
    if (nucleus == 0u || !(cumulative > 0.0f) || !isfinite(cumulative)) { free(candidates); free(probabilities); return false; }
    uint64_t state = config->seed; float target = rng_uniform(&state) * cumulative; uint32_t selected = candidates[0].id;
    for (uint32_t i = 0u; i < nucleus; ++i) { if (target <= candidates[i].probability) { selected = candidates[i].id; break; } target -= candidates[i].probability; }
    out->token_id = selected; out->probability = probabilities[selected]; free(candidates); free(probabilities); return true;
}

static void layer_norm(const float *input, float *output, const float *weight, uint32_t n)
{
    double mean = 0.0; for (uint32_t i = 0u; i < n; ++i) mean += (double)input[i]; mean /= (double)n;
    double variance = 0.0; for (uint32_t i = 0u; i < n; ++i) { const double d = (double)input[i] - mean; variance += d * d; } variance /= (double)n;
    const float inv = 1.0f / sqrtf((float)variance + 1.0e-5f);
    for (uint32_t i = 0u; i < n; ++i) output[i] = (input[i] - (float)mean) * inv * (weight ? weight[i] : 1.0f);
}

bool niyah_llm_generation_init(NiyahLlmGenerationState *state, const NiyahLlmConfig *config, const NiyahLlmModelWeights *weights)
{
    if (!state || !config || !weights || !niyah_llm_config_validate(config)) return false;
    if (!weights->embedding || !weights->lm_head || weights->input_dim != config->embedding_dim ||
        weights->hidden_dim != config->ffn_dim || weights->vocab_size != config->vocab_size) return false;
    memset(state, 0, sizeof(*state)); state->config = *config; state->weights = *weights;
    state->layer_stride = (size_t)config->kv_heads * (size_t)(config->embedding_dim / config->attention_heads);
    if (state->layer_stride == 0u) return false;
    if (!alloc_f32(&state->hidden, config->embedding_dim) || !alloc_f32(&state->normalized, config->embedding_dim) ||
        !alloc_f32(&state->q, config->embedding_dim) || !alloc_f32(&state->k, state->layer_stride) ||
        !alloc_f32(&state->v, state->layer_stride) || !alloc_f32(&state->attn, config->embedding_dim) ||
        !alloc_f32(&state->ffn, config->embedding_dim) || !alloc_f32(&state->ffn_scratch, config->ffn_dim) ||
        !alloc_f32(&state->logits, config->vocab_size)) { niyah_llm_generation_free(state); return false; }
    if ((size_t)config->layer_count > SIZE_MAX / sizeof(NiyahLlmKvCache)) { niyah_llm_generation_free(state); return false; }
    state->kv_caches = (NiyahLlmKvCache *)calloc(config->layer_count, sizeof(*state->kv_caches));
    if (!state->kv_caches) { niyah_llm_generation_free(state); return false; }
    for (uint32_t i = 0u; i < config->layer_count; ++i) {
        if (!niyah_llm_kv_cache_init(&state->kv_caches[i], config->context_length, config->kv_heads, config->embedding_dim / config->attention_heads)) { niyah_llm_generation_free(state); return false; }
    }
    state->position = 0u; return true;
}

void niyah_llm_generation_free(NiyahLlmGenerationState *state)
{
    if (!state) return; if (state->kv_caches) { for (uint32_t i = 0u; i < state->config.layer_count; ++i) niyah_llm_kv_cache_free(&state->kv_caches[i]); }
    free(state->kv_caches); free(state->hidden); free(state->normalized); free(state->q); free(state->k); free(state->v); free(state->attn); free(state->ffn); free(state->ffn_scratch); free(state->logits); memset(state, 0, sizeof(*state));
}

bool niyah_llm_generation_step(NiyahLlmGenerationState *state, uint32_t token_id, const NiyahLlmSamplerConfig *sampler, uint32_t *next_token_id, float *probability)
{
    if (!state || !sampler || !next_token_id || !probability || !state->weights.embedding || !state->weights.lm_head) return false;
    if (token_id >= state->config.vocab_size || state->position >= state->config.context_length) return false;
    const uint32_t d = state->config.embedding_dim; const uint32_t heads = state->config.attention_heads; const uint32_t kv_heads = state->config.kv_heads; const uint32_t head_dim = d / heads;
    const float *embedding = state->weights.embedding + (size_t)token_id * d;
    memcpy(state->hidden, embedding, (size_t)d * sizeof(float));
    for (uint32_t layer = 0u; layer < state->config.layer_count; ++layer) {
        layer_norm(state->hidden, state->normalized, NULL, d);
        memcpy(state->q, state->normalized, (size_t)d * sizeof(float));
        memset(state->k, 0, state->layer_stride * sizeof(float)); memset(state->v, 0, state->layer_stride * sizeof(float));
        for (uint32_t h = 0u; h < heads; ++h) { const uint32_t kh = h % kv_heads; for (uint32_t x = 0u; x < head_dim; ++x) state->q[(size_t)h * head_dim + x] = state->normalized[(size_t)h * head_dim + x]; for (uint32_t x = 0u; x < head_dim; ++x) { state->k[(size_t)kh * head_dim + x] = state->normalized[(size_t)kh * head_dim + x]; state->v[(size_t)kh * head_dim + x] = state->normalized[(size_t)kh * head_dim + x]; } }
        for (uint32_t h = 0u; h < heads; ++h) { if (!niyah_llm_apply_rope(state->q + (size_t)h * head_dim, state->k + (size_t)(h % kv_heads) * head_dim, head_dim, state->position, 10000.0f)) return false; }
        if (!niyah_llm_kv_cache_append(&state->kv_caches[layer], state->k, state->v, 1u)) return false;
        memset(state->attn, 0, (size_t)d * sizeof(float));
        float *cache_keys = state->kv_caches[layer].keys, *cache_values = state->kv_caches[layer].values;
        if (!niyah_llm_attention(state->q, cache_keys, cache_values, heads, kv_heads, head_dim, state->kv_caches[layer].token_count, state->attn)) return false;
        for (uint32_t i = 0u; i < d; ++i) state->hidden[i] += state->attn[i];
        for (uint32_t i = 0u; i < d; ++i) state->normalized[i] = state->hidden[i];
        for (uint32_t i = 0u; i < d; ++i) state->ffn[i] = state->normalized[i];
    }
    for (uint32_t v = 0u; v < state->config.vocab_size; ++v) { float sum = 0.0f; const float *row = state->weights.lm_head + (size_t)v * d; for (uint32_t i = 0u; i < d; ++i) sum += row[i] * state->hidden[i]; state->logits[v] = sum; }
    NiyahLlmSample sample; if (!niyah_llm_sample(state->logits, state->config.vocab_size, sampler, &sample)) return false;
    *next_token_id = sample.token_id; *probability = sample.probability; ++state->position; return true;
}
