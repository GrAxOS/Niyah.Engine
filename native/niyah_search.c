#include "niyah.h"

#include <stdlib.h>
#include <string.h>

/*
 * Was: `// Search stubs`.
 *
 * This is only the in-process result carrier. The real BM25 inverted index
 * lives in search/niyah_index.c and is built as its own target.
 */

static char* dup_str(const char* s)
{
    if (!s) {
        return NULL;
    }
    const size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) {
        memcpy(out, s, n);
    }
    return out;
}

NiyahStatus niyah_search_response_init(NiyahSearchResponse* response,
                                       int32_t capacity)
{
    if (!response || capacity <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(response, 0, sizeof(*response));
    response->results = (NiyahSearchResult*)calloc((size_t)capacity,
                                                   sizeof(NiyahSearchResult));
    if (!response->results) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    response->capacity = capacity;
    return NIYAH_OK;
}

NiyahStatus niyah_search_response_push(NiyahSearchResponse* response,
                                        const char* title,
                                        const char* snippet,
                                        const char* url,
                                        float score)
{
    if (!response) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (response->n_results >= response->capacity) {
        const int32_t next = response->capacity ? response->capacity * 2 : 16;
        NiyahSearchResult* grown = (NiyahSearchResult*)realloc(
            response->results, (size_t)next * sizeof(NiyahSearchResult));
        if (!grown) {
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        memset(grown + response->capacity, 0,
               (size_t)(next - response->capacity) * sizeof(NiyahSearchResult));
        response->results = grown;
        response->capacity = next;
    }

    NiyahSearchResult* slot = &response->results[response->n_results];
    slot->title = dup_str(title);
    slot->snippet = dup_str(snippet);
    slot->url = dup_str(url);
    slot->score = score;

    ++response->n_results;
    return NIYAH_OK;
}

static int result_compare(const void* a, const void* b)
{
    const NiyahSearchResult* ra = (const NiyahSearchResult*)a;
    const NiyahSearchResult* rb = (const NiyahSearchResult*)b;
    if (ra->score < rb->score) return 1;   /* descending */
    if (ra->score > rb->score) return -1;
    return 0;
}

void niyah_search_response_sort(NiyahSearchResponse* response)
{
    if (!response || !response->results || response->n_results <= 1) {
        return;
    }
    qsort(response->results, (size_t)response->n_results,
          sizeof(NiyahSearchResult), result_compare);
}

void niyah_search_response_free(NiyahSearchResponse* response)
{
    if (!response) {
        return;
    }
    for (int32_t i = 0; i < response->n_results; ++i) {
        free(response->results[i].title);
        free(response->results[i].snippet);
        free(response->results[i].url);
    }
    free(response->results);
    memset(response, 0, sizeof(*response));
}
