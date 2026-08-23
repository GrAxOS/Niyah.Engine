#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
    #ifdef NIYAH_BRIDGE_EXPORTS
        #define NIYAH_BRIDGE_API __declspec(dllexport)
    #else
        #define NIYAH_BRIDGE_API __declspec(dllimport)
    #endif
#else
    #define NIYAH_BRIDGE_API __attribute__((visibility("default")))
#endif

typedef enum {
    NIYAH_BRIDGE_OK = 0,
    NIYAH_BRIDGE_ERROR = 1,
    NIYAH_BRIDGE_OUT_OF_MEMORY = 2,
    NIYAH_BRIDGE_INVALID_ARGS = 3,
    NIYAH_BRIDGE_CANCELLED = 4,
    NIYAH_BRIDGE_UNAVAILABLE = 5
} NiyahBridgeStatus;

typedef struct NiyahBridge NiyahBridge;
typedef struct NiyahBridgeGeneration NiyahBridgeGeneration;

/* Forward declare types from niyah_llm.h */
typedef struct NiyahLlmConfig NiyahLlmConfig;

/* ============================================================================
 * Bridge lifecycle
 * ============================================================================ */

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_create(NiyahBridge **out);
NIYAH_BRIDGE_API void niyah_bridge_destroy(NiyahBridge *bridge);

/* ============================================================================
 * Search functionality
 * ============================================================================ */

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_add_document(
    NiyahBridge *bridge,
    const char *id,
    const char *url,
    const char *title,
    const char *text);

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_search(
    NiyahBridge *bridge,
    const char *query,
    char **results,
    size_t *result_count,
    size_t max_results);

/* ============================================================================
 * Model configuration
 * ============================================================================ */

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_model_validate(
    NiyahBridge *bridge,
    const NiyahLlmConfig *config);

/* ============================================================================
 * Weight loading (NEW)
 * ============================================================================ */

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_weights_load(
    NiyahBridge *bridge,
    const uint8_t *buffer,
    size_t buffer_size);

/* ============================================================================
 * Generation
 * ============================================================================ */

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_generation_create(
    NiyahBridgeGeneration **out,
    NiyahBridge *bridge,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    size_t maximum_tokens);

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_generation_next(
    NiyahBridgeGeneration *generation,
    uint32_t *token_id,
    float *probability,
    bool *finished);

NIYAH_BRIDGE_API NiyahBridgeStatus niyah_bridge_generation_cancel(
    NiyahBridgeGeneration *generation);

NIYAH_BRIDGE_API void niyah_bridge_generation_destroy(
    NiyahBridgeGeneration *generation);

#endif /* NIYAH_BRIDGE_H */
