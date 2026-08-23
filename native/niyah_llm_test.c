#include "niyah_llm.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static void test_valid_config(void)
{
    const NiyahLlmConfig config = {
        32000u,
        4096u,
        512u,
        8u,
        8u,
        4u,
        1536u
    };

    assert(niyah_llm_config_validate(&config));
}

static void test_invalid_config(void)
{
    const NiyahLlmConfig invalid_ratio = {
        32000u,
        4096u,
        512u,
        8u,
        7u,
        4u,
        1536u
    };

    const NiyahLlmConfig invalid_heads = {
        32000u,
        4096u,
        512u,
        8u,
        8u,
        16u,
        1536u
    };

    assert(!niyah_llm_config_validate(&invalid_ratio));
    assert(!niyah_llm_config_validate(&invalid_heads));
}

static void test_tensor_size(void)
{
    const NiyahLlmTensorSpec spec = {
        NIYAH_LLM_DTYPE_F32,
        3u,
        {2u, 3u, 4u, 0u}
    };

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
    const NiyahLlmTensorSpec spec = {
        NIYAH_LLM_DTYPE_F32,
        4u,
        {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}
    };

    size_t elements = 0u;
    size_t bytes = 0u;

    assert(niyah_llm_tensor_spec_validate(&spec));
    assert(!niyah_llm_tensor_element_count(&spec, &elements));
    assert(!niyah_llm_tensor_byte_count(&spec, &bytes));
}

int main(void)
{
    test_valid_config();
    test_invalid_config();
    test_tensor_size();
    test_tensor_overflow();
    puts("niyah_llm: ok");
    return 0;
}
