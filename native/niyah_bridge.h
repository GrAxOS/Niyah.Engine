#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable C ABI consumed by ui/Niyah.App via P/Invoke.
 *
 * Every exported symbol here now reflects real state. The previous version
 * answered niyah_bridge_search() with a static {"doc1","doc2","doc3"} array
 * and hard-coded 0.95/0.87/0.76 scores, so the UI displayed results for
 * documents that had never been indexed.
 */

typedef struct {
    char* doc_id;
    char* snippet;
    float score;
} NiyahBridgeHit;

typedef struct {
    NiyahBridgeHit* hits;
    int32_t         count;
} NiyahBridgeResults;

NIYAH_API const char* niyah_bridge_version(void);
NIYAH_API const char* niyah_get_version(void);
NIYAH_API const char* niyah_get_truth_string(NiyahTruth truth);

/* Adds a document to the process-wide store. On success *doc_id points at a
 * heap string owned by the caller; release it with niyah_bridge_free_string. */
NIYAH_API int32_t niyah_bridge_add_document(const char* content,
                                            const char** doc_id);

/* Legacy signature retained for the existing P/Invoke declaration.
 * *results receives a heap NiyahBridgeResults*; free it with
 * niyah_bridge_free_results. Returns 0 on success. */
NIYAH_API int32_t niyah_bridge_search(const char* query,
                                       void** results,
                                       int* count);

/* Preferred entry point for managed callers: returns a JSON array string. */
NIYAH_API char* niyah_bridge_search_json(const char* query, int32_t max_hits);

NIYAH_API int32_t niyah_bridge_document_count(void);
NIYAH_API void    niyah_bridge_clear(void);
NIYAH_API void    niyah_bridge_free_results(void* results);
NIYAH_API void    niyah_bridge_free_string(char* text);

#ifdef __cplusplus
}
#endif

#endif /* NIYAH_BRIDGE_H */
