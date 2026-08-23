#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahBridge NiyahBridge;
typedef struct NiyahBridgeGeneration NiyahBridgeGeneration;

typedef enum {
    NIYAH_BRIDGE_OK = 0,
    NIYAH_BRIDGE_INVALID = 1,
    NIYAH_BRIDGE_CAPACITY = 2,
    NIYAH_BRIDGE_INTERNAL = 3,
    NIYAH_BRIDGE_CANCELLED = 4,
    NIYAH_BRIDGE_UNAVAILABLE = 5
} NiyahBridgeStatus;

NiyahBridgeStatus niyah_bridge_create(NiyahBridge **out_bridge);
void niyah_bridge_destroy(NiyahBridge *bridge);

NiyahBridgeStatus niyah_bridge_add_document(
    NiyahBridge *bridge,
    uint64_t document_id,
    const char *url,
    const char *title,
    const char *text);

NiyahBridgeStatus niyah_bridge_search(
    const NiyahBridge *bridge,
    const char *query,
    size_t limit,
    char *output,
    size_t output_size);

NiyahBridgeStatus niyah_bridge_model_validate(
    const NiyahBridge *bridge,
    uint32_t vocab_size,
    uint32_t context_length,
    uint32_t embedding_dim,
    uint32_t layer_count,
    uint32_t attention_heads,
    uint32_t kv_heads,
    uint32_t ffn_dim);

NiyahBridgeStatus niyah_bridge_generation_create(
    NiyahBridge *bridge,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    size_t maximum_tokens,
    NiyahBridgeGeneration **out_generation);

NiyahBridgeStatus niyah_bridge_generation_next(
    NiyahBridgeGeneration *generation,
    uint32_t *token_id,
    bool *finished);

void niyah_bridge_generation_cancel(NiyahBridgeGeneration *generation);
void niyah_bridge_generation_destroy(NiyahBridgeGeneration *generation);

const char *niyah_bridge_version(void);

#ifdef __cplusplus
}
#endif

#endif
