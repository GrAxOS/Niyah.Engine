#include "niyah_llm.h"

#include "niyah_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool dtype_size(NiyahLlmDType dtype, size_t *out)
{
    if (!out) return false;
    switch (dtype) {
        case NIYAH_LLM_DTYPE_F32:
            *out = sizeof(float);
            return true;
        case NIYAH_LLM_DTYPE_F16:
            *out = sizeof(uint16_t);
            return true;
        default:
            return false;
    }
}

bool niyah_llm_config_validate(const NiyahLlmConfig *config)
{
    if (!config ||
        config->vocab_size == 0u ||
        config->context_length == 0u ||
        config->embedding_dim == 0u ||
        config->layer_count == 0u ||
        config->attention_heads == 0u ||
        config->kv_heads == 0u ||
        config->ffn_dim == 0u) {
        return false;
    }
    if (config->kv_heads > config->attention_heads) return false;
    if (config->attention_heads % config->kv_heads != 0u) return false;
    if (config->embedding_dim % config->attention_heads != 0u) return false;
    return true;
}

bool niyah_llm_tensor_spec_validate(const NiyahLlmTensorSpec *spec)
{
    if (!spec || spec->rank == 0u || spec->rank > 4u) return false;
    if (spec->dtype != NIYAH_LLM_DTYPE_F32 && spec->dtype != NIYAH_LLM_DTYPE_F16) return false;
    for (uint32_t i = 0u; i < spec->rank; ++i) {
        if (spec->shape[i] == 0u) return false;
    }
    return true;
}

bool niyah_llm_tensor_element_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!out || !niyah_llm_tensor_spec_validate(spec)) return false;
    size_t count = 1u;
    for (uint32_t i = 0u; i < spec->rank; ++i) {
        if (!niyah_mul_size(count, (size_t)spec->shape[i], &count)) return false;
    }
    *out = count;
    return true;
}

bool niyah_llm_tensor_byte_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!out) return false;
    size_t elements = 0u;
    size_t element_size = 0u;
    if (!niyah_llm_tensor_element_count(spec, &elements) || !dtype_size(spec->dtype, &element_size)) return false;
    return niyah_mul_size(elements, element_size, out);
}

static bool shape_count(uint32_t rank, const uint32_t shape[4], size_t *out)
{
    if (!out || !shape || rank == 0u || rank > 4u) return false;
    size_t count = 1u;
    for (uint32_t i = 0u; i < rank; ++i) {
        if (shape[i] == 0u || !niyah_mul_size(count, (size_t)shape[i], &count)) return false;
    }
    *out = count;
    return true;
}

bool niyah_llm_tensor_f32_init(NiyahLlmTensorF32 *tensor,
                               uint32_t rank,
                               const uint32_t shape[4])
{
    if (!tensor || !shape) return false;
    size_t count = 0u;
    if (!shape_count(rank, shape, &count) || count > SIZE_MAX / sizeof(float)) return false;
    memset(tensor, 0, sizeof(*tensor));
    tensor->data = (float *)calloc(count, sizeof(float));
    if (!tensor->data) return false;
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

bool niyah_llm_tensor_f32_reshape(NiyahLlmTensorF32 *tensor,
                                  uint32_t rank,
                                  const uint32_t shape[4])
{
    if (!tensor || !shape) return false;
    size_t count = 0u;
    if (!shape_count(rank, shape, &count) || count != tensor->element_count) return false;
    tensor->rank = rank;
    memcpy(tensor->shape, shape, sizeof(tensor->shape));
    return true;
}

bool niyah_llm_matmul(const NiyahLlmTensorF32 *a,
                      const NiyahLlmTensorF32 *b,
                      NiyahLlmTensorF32 *out)
{
    if (!a || !b || !out || !a->data || !b->data || !out->data) return false;
    if (a->rank != 2u || b->rank != 2u || out->rank != 2u) return false;
    const uint32_t m = a->shape[0];
    const uint32_t k = a->shape[1];
    const uint32_t bk = b->shape[0];
    const uint32_t n = b->shape[1];
    if (k != bk || out->shape[0] != m || out->shape[1] != n) return false;

    size_t a_count = 0u;
    size_t b_count = 0u;
    size_t out_count = 0u;
    if (!niyah_mul_size((size_t)m, (size_t)k, &a_count) ||
        !niyah_mul_size((size_t)k, (size_t)n, &b_count) ||
        !niyah_mul_size((size_t)m, (size_t)n, &out_count) ||
        a->element_count != a_count || b->element_count != b_count || out->element_count != out_count) {
        return false;
    }

    for (uint32_t i = 0u; i < m; ++i) {
        for (uint32_t j = 0u; j < n; ++j) {
            float sum = 0.0f;
            for (uint32_t p = 0u; p < k; ++p) {
                sum += a->data[(size_t)i * (size_t)k + p] * b->data[(size_t)p * (size_t)n + j];
            }
            out->data[(size_t)i * (size_t)n + j] = sum;
        }
    }
    return true;
}

bool niyah_llm_apply_rope(float *query,
                          float *key,
                          uint32_t head_dim,
                          uint32_t position,
                          float theta)
{
    if (!query || !key || head_dim == 0u || (head_dim & 1u) != 0u || !isfinite(theta) || theta <= 0.0f) return false;
    for (uint32_t i = 0u; i < head_dim; i += 2u) {
        const float exponent = (float)i / (float)head_dim;
        const float angle = (float)position * powf(theta, -exponent);
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float q0 = query[i];
        const float q1 = query[i + 1u];
        const float k0 = key[i];
        const float k1 = key[i + 1u];
        query[i] = q0 * c - q1 * s;
        query[i + 1u] = q0 * s + q1 * c;
        key[i] = k0 * c - k1 * s;
        key[i + 1u] = k0 * s + k1 * c;
    }
    return true;
}

bool niyah_llm_attention(const float *query,
                         const float *keys,
                         const float *values,
                         uint32_t query_heads,
                         uint32_t kv_heads,
                         uint32_t head_dim,
                         uint32_t token_count,
                         float *output)
{
    if (!query || !keys || !values || !output || query_heads == 0u || kv_heads == 0u || head_dim == 0u || token_count == 0u) return false;
    if (query_heads % kv_heads != 0u) return false;
    if ((size_t)token_count > SIZE_MAX / (size_t)kv_heads) return false;
    size_t token_head_elements = 0u;
    if (!niyah_mul_size((size_t)token_count, (size_t)kv_heads, &token_head_elements) ||
        !niyah_mul_size(token_head_elements, (size_t)head_dim, &token_head_elements) ||
        token_head_elements > SIZE_MAX / sizeof(float)) return false;

    const uint32_t group = query_heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *weights = (float *)calloc((size_t)token_count, sizeof(float));
    if (!weights) return false;

    for (uint32_t h = 0u; h < query_heads; ++h) {
        const uint32_t kvh = h / group;
        const float *q = query + (size_t)h * (size_t)head_dim;
        float max_score = -INFINITY;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const float *k = keys + (((size_t)t * (size_t)kv_heads) + kvh) * (size_t)head_dim;
            float score = 0.0f;
            for (uint32_t d = 0u; d < head_dim; ++d) score += q[d] * k[d];
            score *= scale;
            weights[t] = score;
            if (score > max_score) max_score = score;
        }
        float sum = 0.0f;
        for (uint32_t t = 0u; t < token_count; ++t) {
            weights[t] = expf(weights[t] - max_score);
            sum += weights[t];
        }
        if (!(sum > 0.0f) || !isfinite(sum)) {
            free(weights);
            return false;
        }
        float *out = output + (size_t)h * (size_t)head_dim;
        for (uint32_t d = 0u; d < head_dim; ++d) out[d] = 0.0f;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const float w = weights[t] / sum;
            const float *v = values + (((size_t)t * (size_t)kv_heads) + kvh) * (size_t)head_dim;
            for (uint32_t d = 0u; d < head_dim; ++d) out[d] += w * v[d];
        }
    }
    free(weights);
    return true;
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

bool niyah_llm_ffn_swiglu(const float *input,
                          const float *gate,
                          const float *up,
                          const float *down,
                          uint32_t input_dim,
                          uint32_t hidden_dim,
                          float *output,
                          float *scratch)
{
    if (!input || !gate || !up || !down || !output || !scratch || input_dim == 0u || hidden_dim == 0u) return false;
    for (uint32_t h = 0u; h < hidden_dim; ++h) {
        float g = 0.0f;
        float u = 0.0f;
        for (uint32_t i = 0u; i < input_dim; ++i) {
            g += input[i] * gate[(size_t)i * (size_t)hidden_dim + h];
            u += input[i] * up[(size_t)i * (size_t)hidden_dim + h];
        }
        scratch[h] = silu(g) * u;
    }
    for (uint32_t i = 0u; i < input_dim; ++i) {
        float sum = 0.0f;
        for (uint32_t h = 0u; h < hidden_dim; ++h) sum += scratch[h] * down[(size_t)h * (size_t)input_dim + i];
        output[i] = sum;
    }
    return true;
}

bool niyah_llm_kv_cache_init(NiyahLlmKvCache *cache,
                             uint32_t context_length,
                             uint32_t head_count,
                             uint32_t head_dim)
{
    if (!cache || context_length == 0u || head_count == 0u || head_dim == 0u) return false;
    memset(cache, 0, sizeof(*cache));
    size_t per_token = 0u;
    size_t total = 0u;
    if (!niyah_mul_size((size_t)head_count, (size_t)head_dim, &per_token) ||
        !niyah_mul_size((size_t)context_length, per_token, &total) ||
        total > SIZE_MAX / sizeof(float)) return false;
    cache->keys = (float *)calloc(total, sizeof(float));
    cache->values = (float *)calloc(total, sizeof(float));
    if (!cache->keys || !cache->values) {
        niyah_llm_kv_cache_free(cache);
        return false;
    }
    cache->capacity_elements = total;
    cache->head_count = head_count;
    cache->head_dim = head_dim;
    return true;
}

void niyah_llm_kv_cache_free(NiyahLlmKvCache *cache)
{
    if (!cache) return;
    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

bool niyah_llm_kv_cache_append(NiyahLlmKvCache *cache,
                               const float *keys,
                               const float *values,
                               uint32_t token_count)
{
    if (!cache || !keys || !values || token_count == 0u) return false;
    if (token_count > UINT32_MAX - cache->token_count) return false;

    size_t batch_elements = 0u;
    size_t current_offset = 0u;
    if (!niyah_mul_size((size_t)token_count, (size_t)cache->head_count, &batch_elements) ||
        !niyah_mul_size(batch_elements, (size_t)cache->head_dim, &batch_elements) ||
        !niyah_mul_size((size_t)cache->token_count, (size_t)cache->head_count, &current_offset) ||
        !niyah_mul_size(current_offset, (size_t)cache->head_dim, &current_offset) ||
        batch_elements > cache->capacity_elements ||
        current_offset > cache->capacity_elements - batch_elements ||
        batch_elements > SIZE_MAX / sizeof(float)) return false;

    memcpy(cache->keys + current_offset, keys, batch_elements * sizeof(float));
    memcpy(cache->values + current_offset, values, batch_elements * sizeof(float));
    cache->token_count += token_count;
    return true;
}

void niyah_llm_logits_softmax(float *logits, uint32_t count)
{
    if (!logits || count == 0u) return;
    float max_value = -INFINITY;
    for (uint32_t i = 0u; i < count; ++i) {
        if (isfinite(logits[i]) && logits[i] > max_value) max_value = logits[i];
    }
    if (!isfinite(max_value)) {
        const float uniform = 1.0f / (float)count;
        for (uint32_t i = 0u; i < count; ++i) logits[i] = uniform;
        return;
    }
    float sum = 0.0f;
    for (uint32_t i = 0u; i < count; ++i) {
        logits[i] = expf(logits[i] - max_value);
        sum += logits[i];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) return;
    for (uint32_t i = 0u; i < count; ++i) logits[i] /= sum;
}

static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;
    if (x == 0u) x = UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12u;
    x ^= x << 25u;
    x ^= x >> 27u;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static float rng_uniform(uint64_t *state)
{
    const uint64_t value = rng_next(state) >> 11u;
    return (float)((double)value / 9007199254740992.0);
}

typedef struct {
    uint32_t id;
    float probability;
} Candidate;

static int candidate_cmp(const void *left, const void *right)
{
    const Candidate *a = (const Candidate *)left;
    const Candidate *b = (const Candidate *)right;
    if (a->probability < b->probability) return 1;
    if (a->probability > b->probability) return -1;
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

bool niyah_llm_sample(const float *logits,
                      uint32_t count,
                      const NiyahLlmSamplerConfig *config,
                      NiyahLlmSample *out)
{
    if (!logits || count == 0u || !config || !out || !isfinite(config->temperature) || config->temperature <= 0.0f) return false;
    if (!(config->top_p > 0.0f) || config->top_p > 1.0f) return false;
    if ((size_t)count > SIZE_MAX / sizeof(Candidate) || (size_t)count > SIZE_MAX / sizeof(float)) return false;

    Candidate *candidates = (Candidate *)calloc(count, sizeof(*candidates));
    float *probabilities = (float *)calloc(count, sizeof(*probabilities));
    if (!candidates || !probabilities) {
        free(candidates);
        free(probabilities);
        return false;
    }

    for (uint32_t i = 0u; i < count; ++i) probabilities[i] = logits[i] / config->temperature;
    niyah_llm_logits_softmax(probabilities, count);
    for (uint32_t i = 0u; i < count; ++i) {
        candidates[i].id = i;
        candidates[i].probability = probabilities[i];
    }
    qsort(candidates, count, sizeof(*candidates), candidate_cmp);

    uint32_t keep = count;
    if (config->top_k > 0u && config->top_k < keep) keep = config->top_k;
    float cumulative = 0.0f;
    uint32_t nucleus = 0u;
    for (uint32_t i = 0u; i < keep; ++i) {
        cumulative += candidates[i].probability;
        ++nucleus;
        if (cumulative >= config->top_p) break;
    }
    if (nucleus == 0u) nucleus = 1u;

    uint64_t rng_state = config->seed;
    const float draw = rng_uniform(&rng_state);
    float target = draw * cumulative;
    uint32_t selected = candidates[0].id;
    for (uint32_t i = 0u; i < nucleus; ++i) {
        if (target <= candidates[i].probability) {
            selected = candidates[i].id;
            break;
        }
        target -= candidates[i].probability;
    }

    out->token_id = selected;
    out->probability = probabilities[selected];
    free(candidates);
    free(probabilities);
    return true;
}
