#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"

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

#endif /* NIYAH_BRIDGE_H */
