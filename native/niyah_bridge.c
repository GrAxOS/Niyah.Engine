#include "niyah_bridge.h"

#include "niyah_llm.h"
#include "niyah_search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct NiyahBridge {
    NiyahSearchIndex *search;
};

NiyahBridgeStatus niyah_bridge_create(NiyahBridge **out_bridge) {
    if (!out_bridge) return NIYAH_BRIDGE_INVALID;
    *out_bridge = NULL;

    NiyahBridge *bridge = (NiyahBridge *)calloc(1u, sizeof(*bridge));
    if (!bridge) return NIYAH_BRIDGE_INTERNAL;

    bridge->search = niyah_search_create();
    if (!bridge->search) {
        free(bridge);
        return NIYAH_BRIDGE_INTERNAL;
    }

    *out_bridge = bridge;
    return NIYAH_BRIDGE_OK;
}

void niyah_bridge_destroy(NiyahBridge *bridge) {
    if (!bridge) return;
    niyah_search_free(bridge->search);
    free(bridge);
}

NiyahBridgeStatus niyah_bridge_add_document(
    NiyahBridge *bridge,
    uint64_t document_id,
    const char *url,
    const char *title,
    const char *text) {

    if (!bridge || !bridge->search || document_id == 0u || !text)
        return NIYAH_BRIDGE_INVALID;

    const NiyahSearchDocument document = {
        document_id,
        url,
        title,
        text
    };

    return niyah_search_add(bridge->search, &document)
        ? NIYAH_BRIDGE_OK
        : NIYAH_BRIDGE_CAPACITY;
}

NiyahBridgeStatus niyah_bridge_search(
    const NiyahBridge *bridge,
    const char *query,
    size_t limit,
    char *output,
    size_t output_size) {

    if (!bridge || !bridge->search || !query || !output || output_size == 0u)
        return NIYAH_BRIDGE_INVALID;

    output[0] = '\0';
    if (limit == 0u) return NIYAH_BRIDGE_OK;

    if (limit > SIZE_MAX / sizeof(NiyahSearchHit))
        return NIYAH_BRIDGE_CAPACITY;

    NiyahSearchHit *hits = (NiyahSearchHit *)calloc(limit, sizeof(*hits));
    if (!hits) return NIYAH_BRIDGE_INTERNAL;

    const size_t count = niyah_search_query(bridge->search, query, hits, limit);

    size_t used = 0u;
    for (size_t i = 0u; i < count; ++i) {
        char line[128];
        const int written = snprintf(
            line,
            sizeof(line),
            "%llu\t%.17g\n",
            (unsigned long long)hits[i].document_id,
            hits[i].score);
        if (written < 0) {
            free(hits);
            return NIYAH_BRIDGE_INTERNAL;
        }

        const size_t line_size = (size_t)written;
        if (line_size >= sizeof(line) || used >= output_size ||
            line_size >= output_size - used) {
            free(hits);
            output[used < output_size ? used : output_size - 1u] = '\0';
            return NIYAH_BRIDGE_CAPACITY;
        }

        memcpy(output + used, line, line_size);
        used += line_size;
    }

    output[used] = '\0';
    free(hits);
    return NIYAH_BRIDGE_OK;
}

NiyahBridgeStatus niyah_bridge_model_validate(
    const NiyahBridge *bridge,
    uint32_t vocab_size,
    uint32_t context_length,
    uint32_t embedding_dim,
    uint32_t layer_count,
    uint32_t attention_heads,
    uint32_t kv_heads,
    uint32_t ffn_dim) {

    if (!bridge) return NIYAH_BRIDGE_INVALID;

    const NiyahLlmConfig config = {
        vocab_size,
        context_length,
        embedding_dim,
        layer_count,
        attention_heads,
        kv_heads,
        ffn_dim
    };

    return niyah_llm_config_validate(&config)
        ? NIYAH_BRIDGE_OK
        : NIYAH_BRIDGE_INVALID;
}

const char *niyah_bridge_version(void) {
    return "niyah-engine-bridge/1";
}
