#ifndef NIYAH_INDEX_H
#define NIYAH_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_TERM_MAX 64
#define NIYAH_INDEX_TEXT_MAX 4096

typedef struct {
    uint64_t document_id;
    char url[2048];
    char title[256];
    char text[NIYAH_INDEX_TEXT_MAX];
    uint32_t term_count;
} NiyahDocument;

typedef struct {
    uint64_t document_id;
    uint32_t term_frequency;
} NiyahPosting;

typedef struct {
    char term[NIYAH_TERM_MAX];
    NiyahPosting *postings;
    size_t posting_count;
    size_t posting_capacity;
    uint32_t document_frequency;
} NiyahTermEntry;

typedef struct {
    NiyahTermEntry *terms;
    size_t term_count;
    size_t term_capacity;
    NiyahDocument *documents;
    size_t document_count;
    size_t document_capacity;
    double average_document_length;
    double k1;
    double b;
} NiyahInvertedIndex;

typedef struct {
    uint64_t document_id;
    double score;
} NiyahSearchHit;

void niyah_index_init(NiyahInvertedIndex *index, double k1, double b);
void niyah_index_free(NiyahInvertedIndex *index);
bool niyah_index_add_document(NiyahInvertedIndex *index, const NiyahDocument *document);
size_t niyah_index_search(const NiyahInvertedIndex *index,
                          const char *query,
                          NiyahSearchHit *hits,
                          size_t hit_capacity);
const NiyahDocument *niyah_index_document(const NiyahInvertedIndex *index,
                                           uint64_t document_id);

#ifdef __cplusplus
}
#endif

#endif
