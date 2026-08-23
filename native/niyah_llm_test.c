#include "niyah_llm.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_valid_config(void)
{
    const NiyahLlmConfig config = {32000u, 4096u, 512u, 8u, 8u, 4u, 1536u};
    assert(niyah_llm_config_validate(&config));
}

static void test_invalid_config(void)
{
    const NiyahLlmConfig invalid_ratio = {32000u, 4096u, 512u, 8u, 7u, 4u, 1536u};
    const NiyahLlmConfig invalid_heads = {32000u, 4096u, 512u, 8u, 8u, 16u, 1536u};
    assert(!niyah_llm_config_validate(&invalid_ratio));
    assert(!niyah_llm_config_validate(&invalid_heads));
}

static void test_tensor_size(void)
{
    const NiyahLlmTensorSpec spec = {NIYAH_LLM_DTYPE_F32, 3u, {2u, 3u, 4u, 0u}};
    size_t elements = 0u;
    size_t bytes = 0u;
    assert(niyah_llm_tensor_spec_validate(&spec));
    assert(niyah_llm_tensor_element_count(&spec, &elements));
    assert(elements == 24u);
    assert(niyah_llm_tensor_byte_count(&spec, &bytes));
    assert(bytes == 24u * sizeof(float));
}

static void test_tensor_overflow(void)
{
    const NiyahLlmTensorSpec spec = {NIYAH_LLM_DTYPE_F32, 4u, {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}};
    size_t elements = 0u;
    size_t bytes = 0u;
    assert(niyah_llm_tensor_spec_validate(&spec));
    assert(!niyah_llm_tensor_element_count(&spec, &elements));
    assert(!niyah_llm_tensor_byte_count(&spec, &bytes));
}

static void test_tensor_matmul(void)
{
    const uint32_t a_shape[4] = {2u, 3u, 0u, 0u};
    const uint32_t b_shape[4] = {3u, 2u, 0u, 0u};
    const uint32_t o_shape[4] = {2u, 2u, 0u, 0u};
    NiyahLlmTensorF32 a = {0};
    NiyahLlmTensorF32 b = {0};
    NiyahLlmTensorF32 out = {0};
    assert(niyah_llm_tensor_f32_init(&a, 2u, a_shape));
    assert(niyah_llm_tensor_f32_init(&b, 2u, b_shape));
    assert(niyah_llm_tensor_f32_init(&out, 2u, o_shape));
    a.data[0] = 1.0f; a.data[1] = 2.0f; a.data[2] = 3.0f;
    a.data[3] = 4.0f; a.data[4] = 5.0f; a.data[5] = 6.0f;
    b.data[0] = 7.0f; b.data[1] = 8.0f; b.data[2] = 9.0f;
    b.data[3] = 10.0f; b.data[4] = 11.0f; b.data[5] = 12.0f;
    assert(niyah_llm_matmul(&a, &b, &out));
    assert(fabsf(out.data[0] - 58.0f) < 0.0001f);
    assert(fabsf(out.data[1] - 64.0f) < 0.0001f);
    assert(fabsf(out.data[2] - 139.0f) < 0.0001f);
    assert(fabsf(out.data[3] - 154.0f) < 0.0001f);
    niyah_llm_tensor_f32_free(&a);
    niyah_llm_tensor_f32_free(&b);
    niyah_llm_tensor_f32_free(&out);
}

static void test_rope(void)
{
    float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    assert(niyah_llm_apply_rope(q, k, 4u, 1u, 10000.0f));
    for (size_t i = 0u; i < 4u; ++i) assert(isfinite(q[i]) && isfinite(k[i]));
}

static void test_attention(void)
{
    const float query[2] = {1.0f, 0.0f};
    const float keys[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float values[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float output[2] = {0.0f, 0.0f};
    assert(niyah_llm_attention(query, keys, values, 1u, 1u, 2u, 2u, output));
    assert(isfinite(output[0]) && isfinite(output[1]));
}

static void test_kv_cache(void)
{
    NiyahLlmKvCache cache = {0};
    assert(niyah_llm_kv_cache_init(&cache, 4u, 2u, 2u));
    const float keys[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float values[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    assert(niyah_llm_kv_cache_append(&cache, keys, values, 1u));
    assert(cache.token_count == 1u);
    niyah_llm_kv_cache_free(&cache);
}

static void test_wordpiece(void)
{
    const char *vocabulary[] = {"hello", "world", "##s"};
    const NiyahLlmWordPieceVocab model = {vocabulary, 3u, 32u};
    uint32_t ids[4] = {0u};
    size_t count = 0u;
    assert(niyah_llm_wordpiece_tokenize("hello worlds", &model, ids, 4u, &count));
    assert(count == 3u);
    assert(ids[0] == 0u && ids[1] == 1u && ids[2] == 2u);
}

static void test_bpe(void)
{
    const char *vocabulary[] = {"h", "e", "l", "o", "he", "hel", "hello"};
    const uint32_t left[] = {0u, 4u, 5u};
    const uint32_t right[] = {1u, 2u, 3u};
    const NiyahLlmBpeModel model = {vocabulary, 7u, left, right, 3u, 32u};
    uint32_t ids[8] = {0u};
    size_t count = 0u;
    assert(niyah_llm_bpe_tokenize("hello", &model, ids, 8u, &count));
    assert(count == 1u);
    assert(ids[0] == 6u);
}

static void test_sampling(void)
{
    const float logits[4] = {4.0f, 2.0f, 1.0f, -1.0f};
    const NiyahLlmSamplerConfig config = {1.0f, 2u, 0.9f, 123u};
    NiyahLlmSample sample = {0u, 0.0f};
    assert(niyah_llm_sample(logits, 4u, &config, &sample));
    assert(sample.token_id < 4u);
    assert(sample.probability > 0.0f);
}

int main(void)
{
    test_valid_config();
    test_invalid_config();
    test_tensor_size();
    test_tensor_overflow();
    test_tensor_matmul();
    test_rope();
    test_attention();
    test_kv_cache();
    test_wordpiece();
    test_bpe();
    test_sampling();
    puts("niyah_llm: ok");
    return 0;
}
