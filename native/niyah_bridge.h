#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"

<<<<<<< HEAD
#ifdef _WIN32
#  ifdef NIYAH_DLL_EXPORTS
#    define NIYAH_API __declspec(dllexport)
#  else
#    define NIYAH_API __declspec(dllimport)
#  endif
#else
#  define NIYAH_API __attribute__((visibility("default")))
#endif

/* ── Search result item (matches C# struct layout exactly) ─────────────── */
#pragma pack(push, 1)
typedef struct {
    char  doc_id[64];
    char  snippet[200];
    float score;
} BridgeResultItem;
#pragma pack(pop)

/* ── Version ─────────────────────────────────────────────────────────────── */
NIYAH_API const char* niyah_bridge_version(void);

/* ── Document management ─────────────────────────────────────────────────── */
NIYAH_API int         niyah_bridge_add_document(const char* content, const char** doc_id_out);
NIYAH_API int         niyah_bridge_delete_document(const char* doc_id);
NIYAH_API const char* niyah_bridge_get_document(const char* doc_id);
NIYAH_API int         niyah_bridge_doc_count(void);

/* ── Search ──────────────────────────────────────────────────────────────── */
NIYAH_API int niyah_bridge_search(const char* query,
                                   BridgeResultItem** out_results,
                                   int* out_count);

/* ── LLM generation ──────────────────────────────────────────────────────── */
NIYAH_API char* niyah_bridge_generate(const char* prompt,
                                       const char* model_path,
                                       int max_tokens);
NIYAH_API void  niyah_bridge_free_string(char* s);

/* ── Managed context ─────────────────────────────────────────────────────── */
NIYAH_API NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm);
NIYAH_API void                niyah_bridge_destroy(NiyahBridgeContext* ctx);
=======
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
>>>>>>> origin/main

#endif /* NIYAH_BRIDGE_H */
