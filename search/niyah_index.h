#ifndef NIYAH_INDEX_H
#define NIYAH_INDEX_H

/*
 * BM25 inverted index.
 *
 * This file used to be a truncated copy of niyah_index.c whose first line was
 * `#include "niyah_index.h"` -- it included itself, and it was cut off in the
 * middle of a statement. It now contains declarations only; the definitions
 * live in niyah_index.c.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef NIYAH_DOCUMENT_H
#error "search/niyah_index.h and native/niyah_document.h both define NiyahDocument. \
They are different concepts: this one is an indexed record with a BM25 term \
count, the native one is a parsed text/code/latex segment list. Do not include \
both in the same translation unit."
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum bytes retained per term, including the terminating NUL. */
#ifndef NIYAH_TERM_MAX
#define NIYAH_TERM_MAX 64u
#endif

/*
 * A document as stored by the index.
 *
 * `text` is borrowed, not copied: it must outlive the index.
 * `document_id` must be non-zero; zero is reserved as the "absent" sentinel.
 * `term_count` is filled in by niyah_index_add_document with the number of
 * tokens actually indexed, and is used as the BM25 document length.
 */
typedef struct {
    uint64_t    document_id;
    const char* text;
    uint32_t    term_count;
} NiyahDocument;

/* One (document, term frequency) pair in a term's postings list. */
typedef struct {
    uint64_t document_id;
    uint32_t term_frequency;
} NiyahPosting;

/* A single vocabulary entry and its postings list. */
typedef struct {
    char          term[NIYAH_TERM_MAX];
    NiyahPosting* postings;
    size_t        posting_count;
    size_t        posting_capacity;
    uint32_t      document_frequency;
} NiyahTermEntry;

/* A scored search result. */
typedef struct {
    uint64_t document_id;
    double   score;
} NiyahSearchHit;

typedef struct {
    NiyahTermEntry* terms;
    size_t          term_count;
    size_t          term_capacity;

    NiyahDocument*  documents;
    size_t          document_count;
    size_t          document_capacity;

    double          average_document_length;

    /* BM25 parameters. niyah_index_init clamps these to sane defaults. */
    double          k1;
    double          b;

    /*
     * Internal open-addressing hash tables.  Do not access these fields
     * directly; they are managed exclusively by niyah_index.c.
     *
     * term_ht: maps term string (NUL-terminated) → index into terms[].
     * doc_ht:  maps document_id (uint64_t)        → index into documents[].
     *
     * Each slot holds SIZE_MAX when empty.  Capacity is always a power of two
     * and is kept at least 2× the element count so the load factor stays < 0.5.
     */
    size_t*  term_ht;       /* parallel-key slot array; value is terms[] index  */
    char   (*term_ht_keys)[NIYAH_TERM_MAX]; /* key mirror for collision probe    */
    size_t   term_ht_cap;   /* must be a power of two                            */

    size_t*   doc_ht;       /* value is documents[] index                        */
    uint64_t* doc_ht_keys;  /* key mirror                                        */
    size_t    doc_ht_cap;   /* must be a power of two                            */
} NiyahInvertedIndex;

/*
 * Initialises `index` in place. Out-of-range parameters fall back to the
 * standard BM25 defaults: k1 = 1.2 (requires k1 > 0) and b = 0.75 (requires
 * 0 <= b <= 1).
 */
void niyah_index_init(NiyahInvertedIndex* index, double k1, double b);

/* Releases every allocation and zeroes the struct. Safe on a NULL pointer. */
void niyah_index_free(NiyahInvertedIndex* index);

/*
 * Indexes `document`. Returns false on allocation failure, on a NULL or
 * zero-id document, or if the id is already present -- ids are unique.
 *
 * The document struct is copied by value, but `text` itself is borrowed.
 */
bool niyah_index_add_document(NiyahInvertedIndex* index,
                              const NiyahDocument* document);

/*
 * Scores `query` against the index with BM25 and writes at most
 * `hit_capacity` results into `hits`, sorted by descending score with the
 * document id as a stable tiebreaker. Returns the number written.
 */
size_t niyah_index_search(const NiyahInvertedIndex* index,
                          const char* query,
                          NiyahSearchHit* hits,
                          size_t hit_capacity);

/* Looks up an indexed document by id, or NULL if it is not present. */
const NiyahDocument* niyah_index_document(const NiyahInvertedIndex* index,
                                          uint64_t document_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIYAH_INDEX_H */
