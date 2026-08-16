#include "niyah_search.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_DOC_CAP 32u
#define INITIAL_TERM_CAP 64u
#define TOKEN_MAX 128u
#define BM25_K1 1.2
#define BM25_B 0.75

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
    size_t doc_capacity;
    TermEntry *terms;
    size_t term_count;
    size_t term_capacity;
    size_t total_length;
};

static char *dup_string(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1u);
    if (!p) return NULL;
    memcpy(p, s, n + 1u);
    return p;
}

static int reserve(void **ptr, size_t *capacity, size_t need, size_t elem_size) {
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

static int next_token(const char **cursor, char token[TOKEN_MAX]) {
    const unsigned char *p = (const unsigned char *)*cursor;
    while (*p && !isalnum(*p)) ++p;
    if (!*p) { *cursor = (const char *)p; return 0; }
    size_t n = 0;
    while (*p && isalnum(*p)) {
        if (n + 1u < TOKEN_MAX) token[n++] = (char)tolower(*p);
        ++p;
    }
    token[n] = '\0';
    *cursor = (const char *)p;
    return n > 0u;
}

static TermEntry *find_term(NiyahSearchIndex *index, const char *term) {
    for (size_t i = 0; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    return NULL;
}

static const TermEntry *find_term_const(const NiyahSearchIndex *index, const char *term) {
    for (size_t i = 0; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    return NULL;
}

static int add_posting(NiyahSearchIndex *index, const char *term, uint64_t doc_id) {
    TermEntry *entry = find_term(index, term);
    if (!entry) {
        if (!reserve((void **)&index->terms, &index->term_capacity,
                     index->term_count + 1u, sizeof(*index->terms))) return 0;
        entry = &index->terms[index->term_count++];
        memset(entry, 0, sizeof(*entry));
        entry->term = dup_string(term);
        if (!entry->term) return 0;
    }

    if (entry->count > 0u && entry->doc_ids[entry->count - 1u] == doc_id) {
        entry->tf[entry->count - 1u]++;
        return 1;
    }

    if (!reserve((void **)&entry->doc_ids, &entry->doc_ids_capacity,
                 entry->count + 1u, sizeof(*entry->doc_ids))) return 0;
    if (!reserve((void **)&entry->tf, &entry->tf_capacity,
                 entry->count + 1u, sizeof(*entry->tf))) return 0;
    entry->doc_ids[entry->count] = doc_id;
    entry->tf[entry->count] = 1u;
    entry->count++;
    return 1;
}

NiyahSearchIndex *niyah_search_create(void) {
    NiyahSearchIndex *index = (NiyahSearchIndex *)calloc(1, sizeof(*index));
    if (!index) return NULL;
    index->doc_capacity = INITIAL_DOC_CAP;
    index->docs = (StoredDoc *)calloc(index->doc_capacity, sizeof(*index->docs));
    if (!index->docs) { free(index); return NULL; }
    index->term_capacity = INITIAL_TERM_CAP;
    index->terms = (TermEntry *)calloc(index->term_capacity, sizeof(*index->terms));
    if (!index->terms) { free(index->docs); free(index); return NULL; }
    return index;
}

void niyah_search_free(NiyahSearchIndex *index) {
    if (!index) return;
    for (size_t i = 0; i < index->doc_count; ++i) {
        free(index->docs[i].url);
        free(index->docs[i].title);
        free(index->docs[i].text);
    }
    for (size_t i = 0; i < index->term_count; ++i) {
        free(index->terms[i].term);
        free(index->terms[i].doc_ids);
        free(index->terms[i].tf);
    }
    free(index->docs);
    free(index->terms);
    free(index);
}

static StoredDoc *find_doc(NiyahSearchIndex *index, uint64_t id) {
    for (size_t i = 0; i < index->doc_count; ++i)
        if (index->docs[i].id == id) return &index->docs[i];
    return NULL;
}

int niyah_search_add(NiyahSearchIndex *index, const NiyahSearchDocument *document) {
    if (!index || !document || document->id == 0 || !document->text) return 0;
    if (find_doc(index, document->id)) return 0;
    if (index->doc_count == index->doc_capacity) {
        size_t old = index->doc_capacity;
        if (!reserve((void **)&index->docs, &index->doc_capacity, old + 1u, sizeof(*index->docs))) return 0;
        memset(index->docs + old, 0, (index->doc_capacity - old) * sizeof(*index->docs));
    }
    StoredDoc *doc = &index->docs[index->doc_count++];
    doc->id = document->id;
    doc->url = dup_string(document->url ? document->url : "");
    doc->title = dup_string(document->title ? document->title : "");
    doc->text = dup_string(document->text);
    if (!doc->url || !doc->title || !doc->text) return 0;
    doc->length = 0;
    doc->active = true;

    const char *cursor = doc->text;
    char token[TOKEN_MAX];
    while (next_token(&cursor, token)) {
        ++doc->length;
        if (!add_posting(index, token, doc->id)) return 0;
    }
    index->total_length += doc->length;
    return 1;
}

int niyah_search_remove(NiyahSearchIndex *index, uint64_t document_id) {
    if (!index) return 0;
    StoredDoc *doc = find_doc(index, document_id);
    if (!doc || !doc->active) return 0;
    doc->active = false;
    if (index->total_length >= doc->length) index->total_length -= doc->length;
    return 1;
}

typedef struct {
    uint64_t id;
    double score;
} Candidate;

size_t niyah_search_query(const NiyahSearchIndex *index,
                         const char *query,
                         NiyahSearchHit *hits,
                         size_t hit_capacity) {
    if (!index || !query || !hits || hit_capacity == 0u || index->doc_count == 0u) return 0u;
    Candidate *candidates = (Candidate *)calloc(index->doc_count, sizeof(*candidates));
    if (!candidates) return 0u;
    size_t candidate_count = 0u;
    double avgdl = index->doc_count > 0u ? (double)index->total_length / (double)index->doc_count : 0.0;
    const char *cursor = query;
    char token[TOKEN_MAX];

    while (next_token(&cursor, token)) {
        const TermEntry *entry = find_term_const(index, token);
        if (!entry) continue;
        double df = (double)entry->count;
        double idf = log(1.0 + ((double)index->doc_count - df + 0.5) / (df + 0.5));
        for (size_t p = 0; p < entry->count; ++p) {
            StoredDoc *doc = NULL;
            for (size_t d = 0; d < index->doc_count; ++d)
                if (index->docs[d].id == entry->doc_ids[p]) { doc = &index->docs[d]; break; }
            if (!doc || !doc->active) continue;
            size_t cidx = 0u;
            while (cidx < candidate_count && candidates[cidx].id != doc->id) ++cidx;
            if (cidx == candidate_count) {
                candidates[candidate_count].id = doc->id;
                candidates[candidate_count].score = 0.0;
                ++candidate_count;
            }
            double tf = (double)entry->tf[p];
            double norm = avgdl > 0.0 ? (1.0 - BM25_B + BM25_B * ((double)doc->length / avgdl)) : 1.0;
            candidates[cidx].score += idf * ((tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * norm));
        }
    }

    for (size_t i = 0; i < candidate_count; ++i) {
        for (size_t j = i + 1u; j < candidate_count; ++j) {
            if (candidates[j].score > candidates[i].score) {
                Candidate tmp = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp;
            }
        }
    }

    size_t n = candidate_count < hit_capacity ? candidate_count : hit_capacity;
    for (size_t i = 0; i < n; ++i) {
        hits[i].document_id = candidates[i].id;
        hits[i].score = candidates[i].score;
    }
    free(candidates);
    return n;
}
