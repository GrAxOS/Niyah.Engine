#include "niyah_search.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_DOC_CAP 32u
#define INITIAL_TERM_CAP 64u
#define TOKEN_MAX 128u
#define BM25_K1 1.2
#define BM25_B 0.75
#define TITLE_BOOST 0.25
#define URL_BOOST 0.10

typedef struct {
    char *term;
    uint64_t *doc_ids;
    uint32_t *tf;
    size_t count;
    size_t doc_ids_capacity;
    size_t tf_capacity;
} TermEntry;

typedef struct {
    uint64_t id;
    char *url;
    char *title;
    char *text;
    size_t length;
    bool active;
} StoredDoc;

struct NiyahSearchIndex {
    StoredDoc *docs;
    size_t doc_count;
    size_t active_doc_count;
    size_t doc_capacity;
    TermEntry *terms;
    size_t term_count;
    size_t term_capacity;
    size_t total_length;
};

static char *dup_string(const char *s)
{
    if (!s) return NULL;
    const size_t n = strlen(s);
    if (n == SIZE_MAX) return NULL;
    char *p = (char *)malloc(n + 1u);
    if (!p) return NULL;
    memcpy(p, s, n + 1u);
    return p;
}

static int reserve(void **ptr, size_t *capacity, size_t need, size_t elem_size)
{
    if (!ptr || !capacity || elem_size == 0u) return 0;
    if (need <= *capacity) return 1;

    size_t cap = *capacity ? *capacity : 1u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return 0;
        cap *= 2u;
    }
    if (cap > SIZE_MAX / elem_size) return 0;

    void *p = realloc(*ptr, cap * elem_size);
    if (!p) return 0;
    *ptr = p;
    *capacity = cap;
    return 1;
}

static int next_token(const char **cursor, char token[TOKEN_MAX])
{
    if (!cursor || !*cursor || !token) return 0;

    const unsigned char *p = (const unsigned char *)*cursor;
    while (*p != '\0' && !isalnum(*p)) ++p;
    if (*p == '\0') {
        *cursor = (const char *)p;
        return 0;
    }

    size_t n = 0u;
    while (*p != '\0' && isalnum(*p)) {
        if (n + 1u < TOKEN_MAX)
            token[n++] = (char)tolower(*p);
        ++p;
    }
    token[n] = '\0';
    *cursor = (const char *)p;
    return n > 0u;
}

static TermEntry *find_term(NiyahSearchIndex *index, const char *term)
{
    if (!index || !term) return NULL;
    for (size_t i = 0u; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0)
            return &index->terms[i];
    return NULL;
}

static const TermEntry *find_term_const(const NiyahSearchIndex *index,
                                        const char *term)
{
    if (!index || !term) return NULL;
    for (size_t i = 0u; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0)
            return &index->terms[i];
    return NULL;
}

static StoredDoc *find_doc(NiyahSearchIndex *index, uint64_t id)
{
    if (!index || id == 0u) return NULL;
    for (size_t i = 0u; i < index->doc_count; ++i)
        if (index->docs[i].id == id)
            return &index->docs[i];
    return NULL;
}

static const StoredDoc *find_doc_const(const NiyahSearchIndex *index, uint64_t id)
{
    if (!index || id == 0u) return NULL;
    for (size_t i = 0u; i < index->doc_count; ++i)
        if (index->docs[i].id == id)
            return &index->docs[i];
    return NULL;
}

static void free_term_entry(TermEntry *entry)
{
    if (!entry) return;
    free(entry->term);
    free(entry->doc_ids);
    free(entry->tf);
    memset(entry, 0, sizeof(*entry));
}

static bool posting_find(const TermEntry *entry, uint64_t doc_id, size_t *position)
{
    if (!entry || !position) return false;
    for (size_t i = 0u; i < entry->count; ++i) {
        if (entry->doc_ids[i] == doc_id) {
            *position = i;
            return true;
        }
    }
    return false;
}

static void posting_remove(TermEntry *entry, size_t position)
{
    if (!entry || position >= entry->count) return;
    const size_t remaining = entry->count - position - 1u;
    if (remaining > 0u) {
        memmove(&entry->doc_ids[position],
                &entry->doc_ids[position + 1u],
                remaining * sizeof(*entry->doc_ids));
        memmove(&entry->tf[position],
                &entry->tf[position + 1u],
                remaining * sizeof(*entry->tf));
    }
    --entry->count;
}

static int add_posting(NiyahSearchIndex *index, const char *term, uint64_t doc_id)
{
    if (!index || !term || doc_id == 0u) return 0;

    TermEntry *entry = find_term(index, term);
    if (!entry) {
        if (!reserve((void **)&index->terms,
                     &index->term_capacity,
                     index->term_count + 1u,
                     sizeof(*index->terms))) {
            return 0;
        }
        entry = &index->terms[index->term_count++];
        memset(entry, 0, sizeof(*entry));
        entry->term = dup_string(term);
        if (!entry->term) {
            --index->term_count;
            return 0;
        }
    }

    size_t position = 0u;
    if (posting_find(entry, doc_id, &position)) {
        if (entry->tf[position] == UINT32_MAX) return 0;
        ++entry->tf[position];
        return 1;
    }

    if (!reserve((void **)&entry->doc_ids,
                 &entry->doc_ids_capacity,
                 entry->count + 1u,
                 sizeof(*entry->doc_ids)) ||
        !reserve((void **)&entry->tf,
                 &entry->tf_capacity,
                 entry->count + 1u,
                 sizeof(*entry->tf))) {
        return 0;
    }

    entry->doc_ids[entry->count] = doc_id;
    entry->tf[entry->count] = 1u;
    ++entry->count;
    return 1;
}

static void rollback_document_postings(NiyahSearchIndex *index, uint64_t doc_id)
{
    if (!index || doc_id == 0u) return;

    size_t term = 0u;
    while (term < index->term_count) {
        TermEntry *entry = &index->terms[term];
        size_t position = 0u;
        if (posting_find(entry, doc_id, &position)) {
            posting_remove(entry, position);
        }

        if (entry->count == 0u) {
            free_term_entry(entry);
            if (term + 1u < index->term_count) {
                memmove(&index->terms[term],
                        &index->terms[term + 1u],
                        (index->term_count - term - 1u) * sizeof(*index->terms));
            }
            --index->term_count;
            continue;
        }
        ++term;
    }
}

NiyahSearchIndex *niyah_search_create(void)
{
    NiyahSearchIndex *index = (NiyahSearchIndex *)calloc(1, sizeof(*index));
    if (!index) return NULL;

    index->doc_capacity = INITIAL_DOC_CAP;
    index->docs = (StoredDoc *)calloc(index->doc_capacity, sizeof(*index->docs));
    if (!index->docs) {
        free(index);
        return NULL;
    }

    index->term_capacity = INITIAL_TERM_CAP;
    index->terms = (TermEntry *)calloc(index->term_capacity, sizeof(*index->terms));
    if (!index->terms) {
        free(index->docs);
        free(index);
        return NULL;
    }
    return index;
}

void niyah_search_free(NiyahSearchIndex *index)
{
    if (!index) return;
    for (size_t i = 0u; i < index->doc_count; ++i) {
        free(index->docs[i].url);
        free(index->docs[i].title);
        free(index->docs[i].text);
    }
    for (size_t i = 0u; i < index->term_count; ++i)
        free_term_entry(&index->terms[i]);
    free(index->docs);
    free(index->terms);
    free(index);
}

int niyah_search_add(NiyahSearchIndex *index, const NiyahSearchDocument *document)
{
    if (!index || !document || document->id == 0u || !document->text)
        return 0;
    if (find_doc(index, document->id))
        return 0;

    if (index->doc_count == index->doc_capacity) {
        const size_t old = index->doc_capacity;
        if (!reserve((void **)&index->docs,
                     &index->doc_capacity,
                     old + 1u,
                     sizeof(*index->docs))) {
            return 0;
        }
        memset(index->docs + old,
               0,
               (index->doc_capacity - old) * sizeof(*index->docs));
    }

    StoredDoc *doc = &index->docs[index->doc_count];
    memset(doc, 0, sizeof(*doc));
    doc->id = document->id;
    doc->url = dup_string(document->url ? document->url : "");
    doc->title = dup_string(document->title ? document->title : "");
    doc->text = dup_string(document->text);

    if (!doc->url || !doc->title || !doc->text) {
        free(doc->url);
        free(doc->title);
        free(doc->text);
        memset(doc, 0, sizeof(*doc));
        return 0;
    }

    doc->active = true;
    uint64_t doc_id = doc->id;
    size_t length = 0u;
    const char *cursor = doc->text;
    char token[TOKEN_MAX];

    while (next_token(&cursor, token)) {
        if (length == SIZE_MAX || !add_posting(index, token, doc_id)) {
            rollback_document_postings(index, doc_id);
            free(doc->url);
            free(doc->title);
            free(doc->text);
            memset(doc, 0, sizeof(*doc));
            return 0;
        }
        ++length;
    }

    if (index->total_length > SIZE_MAX - length) {
        rollback_document_postings(index, doc_id);
        free(doc->url);
        free(doc->title);
        free(doc->text);
        memset(doc, 0, sizeof(*doc));
        return 0;
    }

    doc->length = length;
    index->total_length += length;
    ++index->doc_count;
    ++index->active_doc_count;
    return 1;
}

int niyah_search_remove(NiyahSearchIndex *index, uint64_t document_id)
{
    if (!index) return 0;

    StoredDoc *doc = find_doc(index, document_id);
    if (!doc || !doc->active) return 0;

    for (size_t term = 0u; term < index->term_count;) {
        TermEntry *entry = &index->terms[term];
        size_t position = 0u;
        if (posting_find(entry, document_id, &position))
            posting_remove(entry, position);

        if (entry->count == 0u) {
            free_term_entry(entry);
            if (term + 1u < index->term_count) {
                memmove(&index->terms[term],
                        &index->terms[term + 1u],
                        (index->term_count - term - 1u) * sizeof(*index->terms));
            }
            --index->term_count;
            continue;
        }
        ++term;
    }

    doc->active = false;
    if (index->total_length >= doc->length)
        index->total_length -= doc->length;
    if (index->active_doc_count > 0u)
        --index->active_doc_count;
    return 1;
}

typedef struct {
    uint64_t id;
    double score;
} Candidate;

static size_t token_occurrences(const char *text, const char *wanted)
{
    if (!text || !wanted || wanted[0] == '\0') return 0u;
    size_t count = 0u;
    const char *cursor = text;
    char token[TOKEN_MAX];
    while (next_token(&cursor, token)) {
        if (strcmp(token, wanted) == 0)
            ++count;
    }
    return count;
}

static bool candidate_less(const Candidate *a, const Candidate *b)
{
    if (a->score != b->score)
        return a->score > b->score;
    return a->id < b->id;
}

size_t niyah_search_query(const NiyahSearchIndex *index,
                         const char *query,
                         NiyahSearchHit *hits,
                         size_t hit_capacity)
{
    if (!index || !query || !hits || hit_capacity == 0u || index->active_doc_count == 0u)
        return 0u;

    Candidate *candidates =
        (Candidate *)calloc(index->active_doc_count, sizeof(*candidates));
    if (!candidates) return 0u;

    size_t candidate_count = 0u;
    const double avgdl =
        index->active_doc_count > 0u
            ? (double)index->total_length / (double)index->active_doc_count
            : 0.0;

    const char *cursor = query;
    char token[TOKEN_MAX];

    while (next_token(&cursor, token)) {
        const TermEntry *entry = find_term_const(index, token);
        if (!entry) continue;

        const double df = (double)entry->count;
        const double documents = (double)index->active_doc_count;
        const double idf = log1p((documents - df + 0.5) / (df + 0.5));

        for (size_t p = 0u; p < entry->count; ++p) {
            const StoredDoc *doc = find_doc_const(index, entry->doc_ids[p]);
            if (!doc || !doc->active) continue;

            size_t cidx = 0u;
            while (cidx < candidate_count && candidates[cidx].id != doc->id)
                ++cidx;
            if (cidx == candidate_count) {
                if (candidate_count >= index->active_doc_count)
                    break;
                candidates[candidate_count].id = doc->id;
                candidates[candidate_count].score = 0.0;
                ++candidate_count;
            }

            const double tf = (double)entry->tf[p];
            const double norm = avgdl > 0.0
                ? (1.0 - BM25_B + BM25_B * ((double)doc->length / avgdl))
                : 1.0;
            candidates[cidx].score += idf *
                ((tf * (BM25_K1 + 1.0)) /
                 (tf + BM25_K1 * norm));

            candidates[cidx].score +=
                TITLE_BOOST * (double)token_occurrences(doc->title, token);
            candidates[cidx].score +=
                URL_BOOST * (double)token_occurrences(doc->url, token);
        }
    }

    for (size_t i = 1u; i < candidate_count; ++i) {
        Candidate key = candidates[i];
        size_t j = i;
        while (j > 0u && candidate_less(&key, &candidates[j - 1u])) {
            candidates[j] = candidates[j - 1u];
            --j;
        }
        candidates[j] = key;
    }

    const size_t n = candidate_count < hit_capacity ? candidate_count : hit_capacity;
    for (size_t i = 0u; i < n; ++i) {
        hits[i].document_id = candidates[i].id;
        hits[i].score = candidates[i].score;
    }

    free(candidates);
    return n;
}
