#include "niyah_llm.h"

#include "niyah_core.h"

#include <limits.h>

static bool dtype_size(NiyahLlmDType dtype, size_t *out)
{
    if (!out) {
        return false;
    }

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

    if (config->kv_heads > config->attention_heads) {
        return false;
    }

    if (config->attention_heads % config->kv_heads != 0u) {
        return false;
    }

    if (config->embedding_dim % config->attention_heads != 0u) {
        return false;
    }

    return true;
}

bool niyah_llm_tensor_spec_validate(const NiyahLlmTensorSpec *spec)
{
    if (!spec || spec->rank == 0u || spec->rank > 4u) {
        return false;
    }

    if (spec->dtype != NIYAH_LLM_DTYPE_F32 &&
        spec->dtype != NIYAH_LLM_DTYPE_F16) {
        return false;
    }

    for (uint32_t i = 0u; i < spec->rank; ++i) {
        if (spec->shape[i] == 0u) {
            return false;
        }
    }

    return true;
}

bool niyah_llm_tensor_element_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!niyah_llm_tensor_spec_validate(spec) || !out) {
        return false;
    }

    size_t count = 1u;
    for (uint32_t i = 0u; i < spec->rank; ++i) {
        if (!niyah_mul_size(count, (size_t)spec->shape[i], &count)) {
            return false;
        }
    }

    *out = count;
    return true;
}

bool niyah_llm_tensor_byte_count(const NiyahLlmTensorSpec *spec, size_t *out)
{
    if (!out) {
        return false;
    }

    size_t elements = 0u;
    size_t element_size = 0u;
    if (!niyah_llm_tensor_element_count(spec, &elements) ||
        !dtype_size(spec->dtype, &element_size)) {
        return false;
    }

    if (!niyah_mul_size(elements, element_size, out)) {
        return false;
    }

    return true;
}
