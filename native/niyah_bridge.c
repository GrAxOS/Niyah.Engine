<<<<<<< HEAD
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
=======
#include "niyah_bridge.h"

#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * REMOVED FAKE CODE
 * -----------------
 * The previous implementation of this file was:
 *
 *     static const char* dummy[] = {"doc1", "doc2", "doc3"};
 *     static float scores[] = {0.95f, 0.87f, 0.76f};
 *     *count = 3;
 *     *results = (void*)dummy;
 *
 * `scores` was assigned but never returned, `niyah_bridge_add_document`
 * discarded its input and always replied "doc_new", and nothing was stored
 * anywhere. The UI was displaying fabricated relevance numbers.
 *
 * Below is a real store: documents are copied and owned, ids are generated and
 * unique, and scores come from actual term frequency over actual text.
 *
 * THREAD SAFETY (2026-08-24)
 * --------------------------
 * g_store is now protected by g_store_mutex (a plain, non-recursive
 * PTHREAD_MUTEX_INITIALIZER).  niyah_bridge_search_json calls
 * bridge_search_locked() directly so the lock is never taken twice on the
 * same thread, avoiding any need for a recursive mutex.
 */

typedef struct {
    char*  id;
    char*  content;
    size_t length;
} BridgeDoc;
>>>>>>> origin/main

typedef struct {
    BridgeDoc* docs;
    int32_t    count;
    int32_t    capacity;
    int32_t    next_id;
} BridgeStore;

static BridgeStore     g_store       = {NULL, 0, 0, 1};
static pthread_mutex_t g_store_mutex = PTHREAD_MUTEX_INITIALIZER;

struct NiyahBridgeContext {
    NiyahLLM*   llm;
    NiyahGraph* graph;
};

/* -------------------------------------------------------------------------- */

static char* dup_string(const char* s, size_t len)
{
    char* out = (char*)malloc(len + 1u);
    if (!out) {
        return NULL;
    }
    if (len) {
        memcpy(out, s, len);
    }
    out[len] = '\0';
    return out;
}

static void lowercase_into(char* dst, const char* src, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[len] = '\0';
}

/* Case-insensitive occurrence count of `needle` in `haystack`. */
static int32_t count_occurrences(const char* haystack, const char* needle)
{
    const size_t nlen = strlen(needle);
    if (nlen == 0) {
        return 0;
    }

    int32_t hits = 0;
    const char* p = haystack;
    while ((p = strstr(p, needle)) != NULL) {
        ++hits;
        p += nlen;
    }
    return hits;
}

/* -------------------------------------------------------------------------- */

const char* niyah_bridge_version(void)
{
    return NIYAH_VERSION;
}

const char* niyah_get_version(void)
{
    return niyah_version();
}

<<<<<<< HEAD
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
=======
const char* niyah_get_truth_string(NiyahTruth truth)
{
    return niyah_truth_to_string(truth);
>>>>>>> origin/main
}

int32_t niyah_bridge_document_count(void)
{
    pthread_mutex_lock(&g_store_mutex);
    const int32_t n = g_store.count;
    pthread_mutex_unlock(&g_store_mutex);
    return n;
}

int32_t niyah_bridge_add_document(const char* content, const char** doc_id)
{
    if (!content) {
        return NIYAH_ERR_INVALID_ARG;
    }

    pthread_mutex_lock(&g_store_mutex);

    if (g_store.count >= g_store.capacity) {
        const int32_t next = g_store.capacity ? g_store.capacity * 2 : 16;
        BridgeDoc* grown = (BridgeDoc*)realloc(
            g_store.docs, (size_t)next * sizeof(BridgeDoc));
        if (!grown) {
            pthread_mutex_unlock(&g_store_mutex);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        g_store.docs = grown;
        g_store.capacity = next;
    }

    const size_t len = strlen(content);

    char id_buf[32];
    snprintf(id_buf, sizeof(id_buf), "doc_%d", g_store.next_id);

    char* id_copy      = dup_string(id_buf, strlen(id_buf));
    char* content_copy = dup_string(content, len);
    if (!id_copy || !content_copy) {
        free(id_copy);
        free(content_copy);
        pthread_mutex_unlock(&g_store_mutex);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    BridgeDoc* slot = &g_store.docs[g_store.count];
    slot->id      = id_copy;
    slot->content = content_copy;
    slot->length  = len;

    ++g_store.count;
    ++g_store.next_id;

    if (doc_id) {
        /* Hand back an independent copy so the caller's lifetime is its own. */
        char* out_id = dup_string(id_copy, strlen(id_copy));
        if (!out_id) {
            pthread_mutex_unlock(&g_store_mutex);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        *doc_id = out_id;
    }

    pthread_mutex_unlock(&g_store_mutex);
    return NIYAH_OK;
}

void niyah_bridge_clear(void)
{
    pthread_mutex_lock(&g_store_mutex);
    for (int32_t i = 0; i < g_store.count; ++i) {
        free(g_store.docs[i].id);
        free(g_store.docs[i].content);
    }
    free(g_store.docs);
    g_store.docs     = NULL;
    g_store.count    = 0;
    g_store.capacity = 0;
    g_store.next_id  = 1;
    pthread_mutex_unlock(&g_store_mutex);
}

/*
 * Internal search core — caller MUST hold g_store_mutex.
 * Extracted so that both niyah_bridge_search (public) and
 * niyah_bridge_search_json can call it without a recursive lock.
 */
static int32_t bridge_search_locked(const char* query,
                                    void**      results,
                                    int*        count)
{
    *results = NULL;
    *count   = 0;

    if (g_store.count == 0) {
        /* Genuinely empty: report zero hits instead of inventing three. */
        return NIYAH_OK;
    }

    const size_t qlen   = strlen(query);
    char*        q_lower = (char*)malloc(qlen + 1u);
    if (!q_lower) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    lowercase_into(q_lower, query, qlen);

    NiyahBridgeResults* out =
        (NiyahBridgeResults*)calloc(1, sizeof(NiyahBridgeResults));
    if (!out) {
        free(q_lower);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    out->hits = (NiyahBridgeHit*)calloc((size_t)g_store.count,
                                        sizeof(NiyahBridgeHit));
    if (!out->hits) {
        free(out);
        free(q_lower);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < g_store.count; ++i) {
        const BridgeDoc* doc = &g_store.docs[i];

        char* d_lower = (char*)malloc(doc->length + 1u);
        if (!d_lower) {
            continue;
        }
        lowercase_into(d_lower, doc->content, doc->length);

        const int32_t tf = count_occurrences(d_lower, q_lower);
        free(d_lower);

        if (tf == 0) {
            continue;
        }

        /* Length-normalised term frequency: a real, explainable score. */
        const float score =
            (float)tf / (1.0f + (float)doc->length / 1000.0f);

        const size_t snippet_len = doc->length < 160u ? doc->length : 160u;

        NiyahBridgeHit* hit = &out->hits[out->count];
        hit->doc_id  = dup_string(doc->id, strlen(doc->id));
        hit->snippet = dup_string(doc->content, snippet_len);
        hit->score   = score;

        if (!hit->doc_id || !hit->snippet) {
            free(hit->doc_id);
            free(hit->snippet);
            continue;
        }

        ++out->count;
    }

    free(q_lower);

    /* Insertion sort by descending score; hit counts here are small. */
    for (int32_t i = 1; i < out->count; ++i) {
        const NiyahBridgeHit key = out->hits[i];
        int32_t j = i - 1;
        while (j >= 0 && out->hits[j].score < key.score) {
            out->hits[j + 1] = out->hits[j];
            --j;
        }
        out->hits[j + 1] = key;
    }

    *results = out;
    *count   = (int)out->count;
    return NIYAH_OK;
}

int32_t niyah_bridge_search(const char* query, void** results, int* count)
{
    if (!query || !results || !count) {
        return NIYAH_ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&g_store_mutex);
    const int32_t ret = bridge_search_locked(query, results, count);
    pthread_mutex_unlock(&g_store_mutex);
    return ret;
}

char* niyah_bridge_search_json(const char* query, int32_t max_hits)
{
    if (!query) {
        return dup_string("[]", 2);
    }

    pthread_mutex_lock(&g_store_mutex);

    void* raw  = NULL;
    int   count = 0;

    if (bridge_search_locked(query, &raw, &count) != NIYAH_OK) {
        pthread_mutex_unlock(&g_store_mutex);
        return dup_string("[]", 2);
    }

    NiyahBridgeResults* res = (NiyahBridgeResults*)raw;
    if (!res || count == 0) {
        pthread_mutex_unlock(&g_store_mutex);
        niyah_bridge_free_results(raw);
        return dup_string("[]", 2);
    }

    if (max_hits > 0 && count > max_hits) {
        count = max_hits;
    }

    /*
     * Reserve 4 bytes per snippet byte (worst-case Unicode escape \uXXXX)
     * plus generous fixed overhead, so the escape loop rarely needs realloc.
     */
    size_t capacity = 256u;
    for (int i = 0; i < count; ++i) {
        capacity += strlen(res->hits[i].doc_id) * 2u
                  + strlen(res->hits[i].snippet) * 4u + 64u;
    }

    char* json = (char*)malloc(capacity);
    if (!json) {
        pthread_mutex_unlock(&g_store_mutex);
        niyah_bridge_free_results(raw);
        return NULL;
    }

    size_t used = 0;
    json[used++] = '[';

    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            json[used++] = ',';
        }

        used += (size_t)snprintf(json + used, capacity - used,
                                 "{\"id\":\"%s\",\"score\":%.6f,\"snippet\":\"",
                                 res->hits[i].doc_id, res->hits[i].score);

        /* Escape the snippet. Grow the buffer if needed. */
        const char* s = res->hits[i].snippet;
        for (; *s; ++s) {
            if (used + 8u >= capacity) {
                size_t next  = capacity * 2u;
                char*  grown = (char*)realloc(json, next);
                if (!grown) {
                    /* On OOM, truncate gracefully and close the object. */
                    break;
                }
                json     = grown;
                capacity = next;
            }
            switch (*s) {
                case '"':  json[used++] = '\\'; json[used++] = '"';  break;
                case '\\': json[used++] = '\\'; json[used++] = '\\'; break;
                case '\n': json[used++] = '\\'; json[used++] = 'n';  break;
                case '\r': json[used++] = '\\'; json[used++] = 'r';  break;
                case '\t': json[used++] = '\\'; json[used++] = 't';  break;
                default:
                    if ((unsigned char)*s >= 0x20u) {
                        json[used++] = *s;
                    }
                    break;
            }
        }

        /* Ensure room for closing `"}` and the final `]\0`. */
        if (used + 4u >= capacity) {
            char* grown = (char*)realloc(json, capacity + 16u);
            if (grown) { json = grown; capacity += 16u; }
        }
        json[used++] = '"';
        json[used++] = '}';
    }

    json[used++] = ']';
    json[used]   = '\0';

    pthread_mutex_unlock(&g_store_mutex);
    niyah_bridge_free_results(raw);
    return json;
}

void niyah_bridge_free_results(void* results)
{
    NiyahBridgeResults* res = (NiyahBridgeResults*)results;
    if (!res) {
        return;
    }
    for (int32_t i = 0; i < res->count; ++i) {
        free(res->hits[i].doc_id);
        free(res->hits[i].snippet);
    }
    free(res->hits);
    free(res);
}

void niyah_bridge_free_string(char* text)
{
    free(text);
}

/* -------------------------------------------------------------------------- */

NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm)
{
    NiyahBridgeContext* ctx =
        (NiyahBridgeContext*)calloc(1, sizeof(NiyahBridgeContext));
    if (!ctx) {
        return NULL;
    }
    ctx->llm   = llm;
    ctx->graph = niyah_graph_create();
    if (!ctx->graph) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void niyah_bridge_destroy(NiyahBridgeContext* ctx)
{
    if (!ctx) {
        return;
    }
    niyah_graph_destroy(ctx->graph);
    free(ctx);
}

NiyahGraph* niyah_bridge_graph(NiyahBridgeContext* ctx)
{
    return ctx ? ctx->graph : NULL;
}
