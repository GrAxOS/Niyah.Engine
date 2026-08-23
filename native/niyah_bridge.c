#ifdef _WIN32
#  define NIYAH_DLL_EXPORTS
#endif
#include "niyah_bridge.h"
#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define EXPORT NIYAH_API

/* ============================================================
 * In-memory document store with TF-IDF search
 * ============================================================ */

#define BRIDGE_MAX_DOCS      4096
#define BRIDGE_MAX_WORDS     1024   /* unique words per doc   */
#define BRIDGE_MAX_WORD_LEN  64
#define BRIDGE_MAX_RESULTS   32
#define BRIDGE_SNIPPET_LEN   200

typedef struct {
    char word[BRIDGE_MAX_WORD_LEN];
    int  freq;
} WordFreq;

typedef struct {
    char      id[64];
    char*     content;
    WordFreq  wf[BRIDGE_MAX_WORDS];
    int       wf_count;     /* unique word count */
    int       total_words;  /* total word count  */
    time_t    created_at;
} BridgeDoc;

static BridgeDoc        g_docs[BRIDGE_MAX_DOCS];
static int              g_doc_count    = 0;
static BridgeResultItem g_results[BRIDGE_MAX_RESULTS];
static int              g_result_count = 0;
static unsigned int     g_id_counter   = 0;

/* ── Helpers ────────────────────────────────────────────────────────────── */

static void normalise_word(char* dst, const char* src, int max) {
    int i = 0;
    for (; *src && i < max - 1; src++) {
        char c = (char)tolower((unsigned char)*src);
        if (isalpha((unsigned char)c) || isdigit((unsigned char)c)) dst[i++] = c;
    }
    dst[i] = '\0';
}

static void index_doc(BridgeDoc* doc) {
    doc->wf_count    = 0;
    doc->total_words = 0;

    const char* p     = doc->content;
    char        raw[BRIDGE_MAX_WORD_LEN * 2];
    int         wi    = 0;
    int         in_w  = 0;

    while (*p) {
        if (isspace((unsigned char)*p) || ispunct((unsigned char)*p)) {
            if (in_w) {
                raw[wi] = '\0';
                char w[BRIDGE_MAX_WORD_LEN];
                normalise_word(w, raw, BRIDGE_MAX_WORD_LEN);
                if (w[0]) {
                    doc->total_words++;
                    /* Find or insert in wf array */
                    int found = 0;
                    for (int i = 0; i < doc->wf_count; i++) {
                        if (strcmp(doc->wf[i].word, w) == 0) {
                            doc->wf[i].freq++;
                            found = 1;
                            break;
                        }
                    }
                    if (!found && doc->wf_count < BRIDGE_MAX_WORDS) {
                        strncpy(doc->wf[doc->wf_count].word, w, BRIDGE_MAX_WORD_LEN - 1);
                        doc->wf[doc->wf_count].freq = 1;
                        doc->wf_count++;
                    }
                }
                wi    = 0;
                in_w  = 0;
            }
        } else if (wi < (int)sizeof(raw) - 1) {
            raw[wi++] = *p;
            in_w = 1;
        }
        p++;
    }
    if (in_w) { /* flush last word */
        raw[wi] = '\0';
        char w[BRIDGE_MAX_WORD_LEN];
        normalise_word(w, raw, BRIDGE_MAX_WORD_LEN);
        if (w[0]) {
            doc->total_words++;
            int found = 0;
            for (int i = 0; i < doc->wf_count; i++) {
                if (strcmp(doc->wf[i].word, w) == 0) { doc->wf[i].freq++; found = 1; break; }
            }
            if (!found && doc->wf_count < BRIDGE_MAX_WORDS) {
                strncpy(doc->wf[doc->wf_count].word, w, BRIDGE_MAX_WORD_LEN - 1);
                doc->wf[doc->wf_count].freq = 1;
                doc->wf_count++;
            }
        }
    }
}

static int doc_contains_word(const BridgeDoc* doc, const char* word) {
    for (int i = 0; i < doc->wf_count; i++)
        if (strcmp(doc->wf[i].word, word) == 0) return doc->wf[i].freq;
    return 0;
}

static int df_count(const char* word) {
    int n = 0;
    for (int i = 0; i < g_doc_count; i++)
        if (doc_contains_word(&g_docs[i], word) > 0) n++;
    return n;
}

static void make_snippet(const BridgeDoc* doc, const char* query,
                         char* out, int max_len) {
    /* Find first occurrence of any query word */
    const char* p   = doc->content;
    const char* best = p;

    /* Tokenise query and find earliest hit */
    char q_copy[256];
    strncpy(q_copy, query, sizeof(q_copy) - 1);
    char* token = strtok(q_copy, " \t\n\r");
    while (token) {
        char qw[BRIDGE_MAX_WORD_LEN];
        normalise_word(qw, token, BRIDGE_MAX_WORD_LEN);
        if (qw[0]) {
            /* Case-insensitive search */
            const char* found = strcasestr ? strcasestr(doc->content, qw) : NULL;
#ifndef strcasestr
            /* Manual fallback */
            for (const char* s = doc->content; *s; s++) {
                if (tolower((unsigned char)*s) == tolower((unsigned char)qw[0])) {
                    int match = 1;
                    for (int i = 1; qw[i]; i++) {
                        if (tolower((unsigned char)s[i]) != tolower((unsigned char)qw[i])) {
                            match = 0; break;
                        }
                    }
                    if (match) { found = s; break; }
                }
            }
#endif
            if (found && (found < best || best == p))
                best = found;
        }
        token = strtok(NULL, " \t\n\r");
    }

    /* Start snippet up to 40 chars before the hit */
    if (best > doc->content + 40) best -= 40;
    int written = snprintf(out, (size_t)max_len, "...%.197s", best);
    if (written >= max_len) { out[max_len - 4] = '.'; out[max_len - 3] = '.'; out[max_len - 2] = '.'; out[max_len - 1] = '\0'; }
}

/* ── TF-IDF search ─────────────────────────────────────────────────────── */
static int result_cmp(const void* a, const void* b) {
    const BridgeResultItem* ra = (const BridgeResultItem*)a;
    const BridgeResultItem* rb = (const BridgeResultItem*)b;
    return (rb->score > ra->score) ? 1 : (rb->score < ra->score ? -1 : 0);
}

static int tfidf_search(const char* query) {
    if (!query || g_doc_count == 0) return 0;

    /* Tokenise query */
    char q_words[64][BRIDGE_MAX_WORD_LEN];
    int  q_count = 0;

    char q_copy[1024];
    strncpy(q_copy, query, sizeof(q_copy) - 1);
    char* raw = strtok(q_copy, " \t\n\r");
    while (raw && q_count < 64) {
        normalise_word(q_words[q_count], raw, BRIDGE_MAX_WORD_LEN);
        if (q_words[q_count][0]) q_count++;
        raw = strtok(NULL, " \t\n\r");
    }
    if (q_count == 0) return 0;

    /* Precompute IDF for each query term */
    float idf[64];
    for (int qi = 0; qi < q_count; qi++) {
        int df = df_count(q_words[qi]);
        idf[qi] = df > 0 ? logf((float)g_doc_count / (float)df) : 0.0f;
    }

    /* Score each document */
    BridgeResultItem tmp_results[BRIDGE_MAX_DOCS];
    int res_count = 0;

    for (int d = 0; d < g_doc_count; d++) {
        float score = 0.0f;
        for (int qi = 0; qi < q_count; qi++) {
            int freq = doc_contains_word(&g_docs[d], q_words[qi]);
            if (freq == 0) continue;
            float tf = (float)freq / (float)(g_docs[d].total_words > 0 ? g_docs[d].total_words : 1);
            score += tf * idf[qi];
        }
        if (score > 1e-9f) {
            strncpy(tmp_results[res_count].doc_id, g_docs[d].id, 63);
            make_snippet(&g_docs[d], query, tmp_results[res_count].snippet, BRIDGE_SNIPPET_LEN);
            tmp_results[res_count].score = score;
            res_count++;
        }
    }

    qsort(tmp_results, (size_t)res_count, sizeof(BridgeResultItem), result_cmp);

    int out_count = res_count < BRIDGE_MAX_RESULTS ? res_count : BRIDGE_MAX_RESULTS;
    memcpy(g_results, tmp_results, (size_t)out_count * sizeof(BridgeResultItem));
    g_result_count = out_count;
    return out_count;
}

/* ============================================================
 * Public DLL exports (P/Invoke API)
 * ============================================================ */

EXPORT const char* niyah_bridge_version(void) {
    return niyah_version();
}

EXPORT int niyah_bridge_doc_count(void) {
    return g_doc_count;
}

/*
 * Search documents using TF-IDF.
 * Returns pointer to internal static array of BridgeResultItem.
 * Caller must NOT free; valid until next call to niyah_bridge_search.
 */
EXPORT int niyah_bridge_search(
        const char*        query,
        BridgeResultItem** out_results,
        int*               out_count) {

    if (!query || !out_results || !out_count) return -1;

    int n = tfidf_search(query);
    *out_results = g_results;
    *out_count   = n;
    return 0;
}

/*
 * Add a document to the in-memory store.
 * doc_id_out receives a pointer to the document's ID string (owned by store).
 */
EXPORT int niyah_bridge_add_document(
        const char*  content,
        const char** doc_id_out) {

    if (!content || !doc_id_out) return -1;
    if (g_doc_count >= BRIDGE_MAX_DOCS) return -2;

    BridgeDoc* doc = &g_docs[g_doc_count];
    memset(doc, 0, sizeof(BridgeDoc));

    /* Generate a deterministic short ID */
    snprintf(doc->id, sizeof(doc->id), "doc_%08x", ++g_id_counter);

    doc->content = _strdup(content);
    if (!doc->content) return -3;

    doc->created_at = time(NULL);
    index_doc(doc);

    *doc_id_out = doc->id;
    g_doc_count++;
    return 0;
}

/*
 * Delete a document by ID.
 */
EXPORT int niyah_bridge_delete_document(const char* doc_id) {
    if (!doc_id) return -1;
    for (int i = 0; i < g_doc_count; i++) {
        if (strcmp(g_docs[i].id, doc_id) == 0) {
            free(g_docs[i].content);
            /* Shift remaining docs down */
            memmove(&g_docs[i], &g_docs[i + 1],
                    (size_t)(g_doc_count - i - 1) * sizeof(BridgeDoc));
            g_doc_count--;
            return 0;
        }
    }
    return -1; /* not found */
}

/*
 * Get document content by ID (returns pointer into internal store).
 */
EXPORT const char* niyah_bridge_get_document(const char* doc_id) {
    if (!doc_id) return NULL;
    for (int i = 0; i < g_doc_count; i++)
        if (strcmp(g_docs[i].id, doc_id) == 0)
            return g_docs[i].content;
    return NULL;
}

/*
 * Run LLM generation (requires a loaded model path).
 * Returns heap-allocated string; caller must call niyah_bridge_free_string.
 */
EXPORT char* niyah_bridge_generate(const char* prompt, const char* model_path,
                                    int max_tokens) {
    if (!prompt) return _strdup("Error: no prompt.");

    NiyahLLM llm;
    memset(&llm, 0, sizeof(llm));
    llm.sampler.strategy    = NIYAH_SAMPLE_TOP_K;
    llm.sampler.temperature = 0.8f;
    llm.sampler.top_k       = 40;
    llm.sampler.top_p       = 0.9f;

    if (model_path && *model_path)
        niyah_model_load(&llm.model, model_path);

    NiyahLLMOutput out = niyah_llm_generate(&llm, prompt, max_tokens > 0 ? max_tokens : 128);
    niyah_model_free(&llm.model);

    return out.text; /* caller owns */
}

EXPORT void niyah_bridge_free_string(char* s) {
    free(s);
}

/* ── Managed bridge context (wraps LLM + knowledge graph) ─────────────── */

NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm) {
    NiyahBridgeContext* ctx = (NiyahBridgeContext*)calloc(1, sizeof(NiyahBridgeContext));
    if (!ctx) return NULL;
    ctx->llm            = llm;
    ctx->knowledge_graph = NULL;
    ctx->search_results = NULL;
    return ctx;
}

void niyah_bridge_destroy(NiyahBridgeContext* ctx) {
    free(ctx);
}
