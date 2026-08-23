#include "niyah_bridge.h"
#include "niyah_llm.h"
#include "niyah_search.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * NiyahBridge structure
 * ============================================================================ */

struct NiyahBridge {
    NiyahSearch *search;
    NiyahLlmGenerationState *llm_generation;
    NiyahLlmModelWeights *weights;
    NiyahLlmConfig config;
};

/* ============================================================================
 * Bridge lifecycle
 * ============================================================================ */

NiyahBridgeStatus niyah_bridge_create(NiyahBridge **out) {
    if (!out) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    NiyahBridge *bridge = calloc(1, sizeof(NiyahBridge));
    if (!bridge) {
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    bridge->search = NULL;
    bridge->llm_generation = NULL;
    bridge->weights = NULL;
    memset(&bridge->config, 0, sizeof(bridge->config));

    *out = bridge;
    return NIYAH_BRIDGE_OK;
}

void niyah_bridge_destroy(NiyahBridge *bridge) {
    if (!bridge) {
        return;
    }

    if (bridge->search) {
        niyah_search_destroy(bridge->search);
    }

    if (bridge->llm_generation) {
        niyah_llm_generation_free(bridge->llm_generation);
    }

    if (bridge->weights) {
        niyah_llm_weights_unload(bridge->weights);
        free(bridge->weights);
    }

    free(bridge);
}

/* ============================================================================
 * Search functionality (unchanged)
 * ============================================================================ */

NiyahBridgeStatus niyah_bridge_add_document(
    NiyahBridge *bridge,
    const char *id,
    const char *url,
    const char *title,
    const char *text)
{
    if (!bridge || !id || !text) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    if (!bridge->search) {
        NiyahSearch *search = NULL;
        NiyahSearchStatus status = niyah_search_create(&search, 10000);
        if (status != NIYAH_SEARCH_OK) {
            return NIYAH_BRIDGE_OUT_OF_MEMORY;
        }
        bridge->search = search;
    }

    NiyahSearchStatus status = niyah_search_add_document(
        bridge->search,
        id,
        url,
        title,
        text
    );

    switch (status) {
        case NIYAH_SEARCH_OK:
            return NIYAH_BRIDGE_OK;
        case NIYAH_SEARCH_OUT_OF_MEMORY:
            return NIYAH_BRIDGE_OUT_OF_MEMORY;
        default:
            return NIYAH_BRIDGE_ERROR;
    }
}

NiyahBridgeStatus niyah_bridge_search(
    NiyahBridge *bridge,
    const char *query,
    char **results,
    size_t *result_count,
    size_t max_results)
{
    if (!bridge || !query || !results || !result_count || !bridge->search) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    NiyahSearchResult *search_results = NULL;
    NiyahSearchStatus status = niyah_search_query(
        bridge->search,
        query,
        &search_results,
        result_count,
        max_results
    );

    if (status != NIYAH_SEARCH_OK || !search_results) {
        return NIYAH_BRIDGE_ERROR;
    }

    /* Format results as tab-separated "id\tscore" */
    size_t total_size = 0;
    for (size_t i = 0; i < *result_count; ++i) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s\t%.4f\n", 
                 search_results[i].id, search_results[i].score);
        total_size += strlen(buffer);
    }

    *results = malloc(total_size + 1);
    if (!*results) {
        free(search_results);
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    (*results)[0] = '\0';
    for (size_t i = 0; i < *result_count; ++i) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s\t%.4f\n", 
                 search_results[i].id, search_results[i].score);
        strcat(*results, buffer);
    }

    free(search_results);
    return NIYAH_BRIDGE_OK;
}

/* ============================================================================
 * Model configuration (unchanged)
 * ============================================================================ */

NiyahBridgeStatus niyah_bridge_model_validate(
    NiyahBridge *bridge,
    const NiyahLlmConfig *config)
{
    if (!bridge || !config) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    if (!niyah_llm_config_validate(config)) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    memcpy(&bridge->config, config, sizeof(bridge->config));
    return NIYAH_BRIDGE_OK;
}

/* ============================================================================
 * Generation (FIXED: now connects to LLM generation state)
 * ============================================================================ */

struct NiyahBridgeGeneration {
    NiyahBridge *bridge;
    uint32_t *prompt_tokens;
    size_t prompt_count;
    size_t prompt_index;
    size_t maximum_tokens;
    size_t produced_tokens;
    bool cancelled;
};

NiyahBridgeStatus niyah_bridge_generation_create(
    NiyahBridgeGeneration **out,
    NiyahBridge *bridge,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    size_t maximum_tokens)
{
    if (!out || !bridge || !prompt_tokens || prompt_count == 0) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    NiyahBridgeGeneration *gen = calloc(1, sizeof(NiyahBridgeGeneration));
    if (!gen) {
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    gen->bridge = bridge;
    gen->prompt_tokens = malloc(prompt_count * sizeof(uint32_t));
    if (!gen->prompt_tokens) {
        free(gen);
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    memcpy(gen->prompt_tokens, prompt_tokens, prompt_count * sizeof(uint32_t));
    gen->prompt_count = prompt_count;
    gen->prompt_index = 0;
    gen->maximum_tokens = maximum_tokens;
    gen->produced_tokens = 0;
    gen->cancelled = false;

    /* FIXED: Initialize LLM generation state */
    if (bridge->llm_generation) {
        niyah_llm_generation_free(bridge->llm_generation);
    }

    bridge->llm_generation = niyah_llm_generation_init(
        &bridge->config,
        bridge->weights,
        prompt_tokens[0],  /* Start with first token */
        maximum_tokens,
        10000.0f,  /* RoPE theta */
        0.8f,      /* Temperature */
        0.95f,     /* Top-p */
        40         /* Top-k */
    );

    if (!bridge->llm_generation) {
        free(gen->prompt_tokens);
        free(gen);
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    *out = gen;
    return NIYAH_BRIDGE_OK;
}

NiyahBridgeStatus niyah_bridge_generation_next(
    NiyahBridgeGeneration *generation,
    uint32_t *token_id,
    float *probability,
    bool *finished)
{
    if (!generation || !token_id || !probability || !finished) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    *token_id = 0u;
    *probability = 0.0f;
    *finished = false;

    if (generation->cancelled) {
        *finished = true;
        return NIYAH_BRIDGE_CANCELLED;
    }

    if (generation->produced_tokens >= generation->maximum_tokens) {
        *finished = true;
        return NIYAH_BRIDGE_OK;
    }

    /* FIXED: Use LLM generation state to produce new tokens */
    if (generation->bridge->llm_generation) {
        uint32_t next_token;
        float prob;
        
        if (niyah_llm_generation_step(
                generation->bridge->llm_generation,
                &next_token,
                &prob)) {
            
            *token_id = next_token;
            *probability = prob;
            generation->produced_tokens++;
            return NIYAH_BRIDGE_OK;
        }
    }

    /* Fallback: echo prompt tokens (for testing without weights) */
    if (generation->prompt_index < generation->prompt_count) {
        *token_id = generation->prompt_tokens[generation->prompt_index++];
        *probability = 1.0f;
        generation->produced_tokens++;
        return NIYAH_BRIDGE_OK;
    }

    *finished = true;
    return NIYAH_BRIDGE_UNAVAILABLE;
}

NiyahBridgeStatus niyah_bridge_generation_cancel(NiyahBridgeGeneration *generation) {
    if (!generation) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    generation->cancelled = true;
    return NIYAH_BRIDGE_OK;
}

void niyah_bridge_generation_destroy(NiyahBridgeGeneration *generation) {
    if (!generation) {
        return;
    }

    if (generation->bridge && generation->bridge->llm_generation) {
        niyah_llm_generation_free(generation->bridge->llm_generation);
        generation->bridge->llm_generation = NULL;
    }

    free(generation->prompt_tokens);
    free(generation);
}

/* ============================================================================
 * Weight loading (NEW: exposed to C#)
 * ============================================================================ */

NiyahBridgeStatus niyah_bridge_weights_load(
    NiyahBridge *bridge,
    const uint8_t *buffer,
    size_t buffer_size)
{
    if (!bridge || !buffer || buffer_size == 0) {
        return NIYAH_BRIDGE_INVALID_ARGS;
    }

    if (bridge->weights) {
        niyah_llm_weights_unload(bridge->weights);
        free(bridge->weights);
    }

    bridge->weights = malloc(sizeof(NiyahLlmModelWeights));
    if (!bridge->weights) {
        return NIYAH_BRIDGE_OUT_OF_MEMORY;
    }

    memset(bridge->weights, 0, sizeof(NiyahLlmModelWeights));

    if (!niyah_llm_weights_load_from_buffer(
            bridge->weights,
            &bridge->config,
            buffer,
            buffer_size)) {
        
        free(bridge->weights);
        bridge->weights = NULL;
        return NIYAH_BRIDGE_ERROR;
    }

    return NIYAH_BRIDGE_OK;
}
