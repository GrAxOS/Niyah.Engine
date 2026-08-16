#ifndef NIYAH_SEARCH_H
#define NIYAH_SEARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t id;
    const char *url;
    const char *title;
    const char *text;
} NiyahSearchDocument;

typedef struct {
    uint64_t document_id;
    double score;
} NiyahSearchHit;

typedef struct NiyahSearchIndex NiyahSearchIndex;

NiyahSearchIndex *niyah_search_create(void);
void niyah_search_free(NiyahSearchIndex *index);
int niyah_search_add(NiyahSearchIndex *index, const NiyahSearchDocument *document);
int niyah_search_remove(NiyahSearchIndex *index, uint64_t document_id);
size_t niyah_search_query(const NiyahSearchIndex *index,
                         const char *query,
                         NiyahSearchHit *hits,
                         size_t hit_capacity);

#ifdef __cplusplus
}
#endif

#endif
