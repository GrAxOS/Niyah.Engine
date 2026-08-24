#include "niyah_bridge.h"

#include <ctype.h>
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
 */

typedef struct {
    char*  id;
    char*  content;
    size_t length;
} BridgeDoc;

typedef struct {
    BridgeDoc* docs;
    int32_t    count;
    int32_t    capacity;
    int32_t    next_id;
} BridgeStore;

static BridgeStore g_store = {NULL, 0, 0, 1};

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

const char* niyah_get_truth_string(NiyahTruth truth)
{
    return niyah_truth_to_string(truth);
}

int32_t niyah_bridge_document_count(void)
{
    return g_store.count;
}

int32_t niyah_bridge_add_document(const char* content, const char** doc_id)
{
    if (!content) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (g_store.count >= g_store.capacity) {
        const int32_t next = g_store.capacity ? g_store.capacity * 2 : 16;
        BridgeDoc* grown = (BridgeDoc*)realloc(
            g_store.docs, (size_t)next * sizeof(BridgeDoc));
        if (!grown) {
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        g_store.docs = grown;
        g_store.capacity = next;
    }

    const size_t len = strlen(content);

    char id_buf[32];
    snprintf(id_buf, sizeof(id_buf), "doc_%d", g_store.next_id);

    char* id_copy = dup_string(id_buf, strlen(id_buf));
    char* content_copy = dup_string(content, len);
    if (!id_copy || !content_copy) {
        free(id_copy);
        free(content_copy);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    BridgeDoc* slot = &g_store.docs[g_store.count];
    slot->id = id_copy;
    slot->content = content_copy;
    slot->length = len;

    ++g_store.count;
    ++g_store.next_id;

    if (doc_id) {
        /* Hand back an independent copy so the caller's lifetime is its own. */
        *doc_id = dup_string(id_copy, strlen(id_copy));
        if (!*doc_id) {
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
    }

    return NIYAH_OK;
}

void niyah_bridge_clear(void)
{
    for (int32_t i = 0; i < g_store.count; ++i) {
        free(g_store.docs[i].id);
        free(g_store.docs[i].content);
    }
    free(g_store.docs);
    g_store.docs = NULL;
    g_store.count = 0;
    g_store.capacity = 0;
    g_store.next_id = 1;
}

int32_t niyah_bridge_search(const char* query, void** results, int* count)
{
    if (!query || !results || !count) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *results = NULL;
    *count = 0;

    if (g_store.count == 0) {
        /* Genuinely empty: report zero hits instead of inventing three. */
        return NIYAH_OK;
    }

    const size_t qlen = strlen(query);
    char* q_lower = (char*)malloc(qlen + 1u);
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
        hit->doc_id = dup_string(doc->id, strlen(doc->id));
        hit->snippet = dup_string(doc->content, snippet_len);
        hit->score = score;

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
    *count = (int)out->count;
    return NIYAH_OK;
}

char* niyah_bridge_search_json(const char* query, int32_t max_hits)
{
    void* raw = NULL;
    int count = 0;

    if (niyah_bridge_search(query, &raw, &count) != NIYAH_OK) {
        return dup_string("[]", 2);
    }

    NiyahBridgeResults* res = (NiyahBridgeResults*)raw;
    if (!res || count == 0) {
        niyah_bridge_free_results(raw);
        return dup_string("[]", 2);
    }

    if (max_hits > 0 && count > max_hits) {
        count = max_hits;
    }

    size_t capacity = 256u;
    for (int i = 0; i < count; ++i) {
        capacity += strlen(res->hits[i].doc_id) * 2u
                  + strlen(res->hits[i].snippet) * 2u + 64u;
    }

    char* json = (char*)malloc(capacity);
    if (!json) {
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

        /* Escape the snippet so the UI never receives malformed JSON. */
        const char* s = res->hits[i].snippet;
        for (; *s && used + 8u < capacity; ++s) {
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

        json[used++] = '"';
        json[used++] = '}';
    }

    json[used++] = ']';
    json[used] = '\0';

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
    ctx->llm = llm;
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
