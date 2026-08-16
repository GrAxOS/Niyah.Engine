#include "niyah_index.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int term_compare(const void *a, const void *b) {
    const NiyahSearchHit *ha = (const NiyahSearchHit *)a;
    const NiyahSearchHit *hb = (const NiyahSearchHit *)b;
    if (ha->score < hb->score) return 1;
    if (ha->score > hb->score) return -1;
    if (ha->document_id < hb->document_id) return -1;
    if (ha->document_id > hb->document_id) return 1;
    return 0;
}

static bool token_byte(unsigned char c) {
    return isalnum(c) != 0 || c >= 0x80u || c == '_' || c == '-';
}

static size_t tokenize(const char *text, char tokens[][NIYAH_TERM_MAX], size_t max_tokens) {
    if (!text || max_tokens == 0) return 0;
    size_t count = 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p && count < max_tokens) {
        while (*p && !token_byte(*p)) ++p;
        if (!*p) break;

        size_t len = 0;
        while (*p && token_byte(*p)) {
            if (len + 1u < NIYAH_TERM_MAX) {
                unsigned char c = *p;
                tokens[count][len++] = (char)(c < 0x80u ? tolower(c) : c);
            }
            ++p;
        }
        tokens[count][len] = '\0';
        if (len > 0) ++count;
    }
    return count;
}

static NiyahTermEntry *find_term(NiyahInvertedIndex *index, const char *term) {
    for (size_t i = 0; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    return NULL;
}

static const NiyahTermEntry *find_term_const(const NiyahInvertedIndex *index, const char *term) {
    for (size_t i = 0; i < index->term_count; ++i)
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    return NULL;
}

static bool grow_terms(NiyahInvertedIndex *index) {
    size_t next = index->term_capacity ? index->term_capacity * 2u : 128u;
    if (next < index->term_capacity || next > SIZE_MAX / sizeof(*index->terms)) return false;
    NiyahTermEntry *p = realloc(index->terms, next * sizeof(*p));
    if (!p) return false;
    memset(p + index->term_capacity, 0, (next - index->term_capacity) * sizeof(*p));
    index->terms = p;
    index->term_capacity = next;
    return true;
}

static bool grow_documents(NiyahInvertedIndex *index) {
    size_t next = index->document_capacity ? index->document_capacity * 2u : 64u;
    if (next < index->document_capacity || next > SIZE_MAX / sizeof(*index->documents)) return false;
    NiyahDocument *p = realloc(index->documents, next * sizeof(*p));
    if (!p) return false;
    index->documents = p;
    index->document_capacity = next;
    return true;
}

static bool grow_postings(NiyahTermEntry *entry) {
    size_t next = entry->posting_capacity ? entry->posting_capacity * 2u : 16u;
    if (next < entry->posting_capacity || next > SIZE_MAX / sizeof(*entry->postings)) return false;
    NiyahPosting *p = realloc(entry->postings, next * sizeof(*p));
    if (!p) return false;
    entry->postings = p;
    entry->posting_capacity = next;
    return true;
}

void niyah_index_init(NiyahInvertedIndex *index, double k1, double b) {
    if (!index) return;
    memset(index, 0, sizeof(*index));
    index->k1 = (k1 > 0.0) ? k1 : 1.2;
    index->b = (b >= 0.0 && b <= 1.0) ? b : 0.75;
}

void niyah_index_free(NiyahInvertedIndex *index) {
    if (!index) return;
    for (size_t i = 0; i < index->term_count; ++i)
        free(index->terms[i].postings);
    free(index->terms);
    free(index->documents);
    memset(index, 0, sizeof(*index));
}

bool niyah_index_add_document(NiyahInvertedIndex *index, const NiyahDocument *document) {
    if (!index || !document || document->document_id == 0) return false;
    if (niyah_index_document(index, document->document_id)) return false;

    char tokens[1024][NIYAH_TERM_MAX];
    size_t n = tokenize(document->text, tokens, 1024);
    if (n > UINT32_MAX) return false;

    if (index->document_count == index->document_capacity && !grow_documents(index)) return false;

    size_t unique_count = 0;
    for (size_t i = 0; i < n; ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j) {
            if (strcmp(tokens[i], tokens[j]) == 0) {
                seen = true;
                break;
            }
        }
        if (!seen) ++unique_count;
    }

    size_t missing_terms = 0;
    for (size_t i = 0; i < n; ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j) {
            if (strcmp(tokens[i], tokens[j]) == 0) {
                seen = true;
                break;
            }
        }
        if (!seen && !find_term(index, tokens[i])) ++missing_terms;
    }

    while (index->term_capacity - index->term_count < missing_terms) {
        if (!grow_terms(index)) return false;
    }

    for (size_t i = 0; i < n; ++i) {
        NiyahTermEntry *entry = find_term(index, tokens[i]);
        if (!entry) continue;
        if (!find_term(index, tokens[i])) return false;
        if (entry->posting_count == entry->posting_capacity && !grow_postings(entry)) return false;
    }

    NiyahDocument *dst = &index->documents[index->document_count++];
    *dst = *document;
    dst->term_count = (uint32_t)n;

    if (n > 0) {
        index->average_document_length =
            ((index->average_document_length * (double)(index->document_count - 1u)) +
             (double)n) / (double)index->document_count;
    }

    for (size_t i = 0; i < n; ++i) {
        NiyahTermEntry *entry = find_term(index, tokens[i]);
        if (!entry) {
            entry = &index->terms[index->term_count++];
            strncpy(entry->term, tokens[i], sizeof(entry->term) - 1u);
            entry->term[sizeof(entry->term) - 1u] = '\0';
        }

        size_t posting_index = SIZE_MAX;
        for (size_t j = 0; j < entry->posting_count; ++j) {
            if (entry->postings[j].document_id == document->document_id) {
                posting_index = j;
                break;
            }
        }

        if (posting_index == SIZE_MAX) {
            posting_index = entry->posting_count++;
            entry->postings[posting_index].document_id = document->document_id;
            entry->postings[posting_index].term_frequency = 0;
            entry->document_frequency++;
        }
        entry->postings[posting_index].term_frequency++;
    }

    return true;
}

static double bm25_score(const NiyahInvertedIndex *index,
                         const NiyahTermEntry *entry,
                         uint32_t tf,
                         uint64_t doc_id,
                         uint32_t doc_len) {
    if (!index || !entry || tf == 0 || doc_len == 0 || index->document_count == 0) return 0.0;
    double n = (double)index->document_count;
    double df = (double)entry->document_frequency;
    double idf = log(1.0 + (n - df + 0.5) / (df + 0.5));
    double avg = index->average_document_length > 0.0 ? index->average_document_length : 1.0;
    double norm = index->k1 * (1.0 - index->b + index->b * ((double)doc_len / avg));
    (void)doc_id;
    return idf * (((double)tf * (index->k1 + 1.0)) / ((double)tf + norm));
}

size_t niyah_index_search(const NiyahInvertedIndex *index,
                          const char *query,
                          NiyahSearchHit *hits,
                          size_t hit_capacity) {
    if (!index || !query || !hits || hit_capacity == 0 || index->document_count == 0) return 0;

    double *scores = calloc(index->document_count, sizeof(*scores));
    if (!scores) return 0;

    char qtokens[128][NIYAH_TERM_MAX];
    size_t qn = tokenize(query, qtokens, 128);
    for (size_t qi = 0; qi < qn; ++qi) {
        const NiyahTermEntry *entry = find_term_const(index, qtokens[qi]);
        if (!entry) continue;
        for (size_t pi = 0; pi < entry->posting_count; ++pi) {
            const NiyahPosting *posting = &entry->postings[pi];
            const NiyahDocument *doc = niyah_index_document(index, posting->document_id);
            if (!doc) continue;
            size_t index_pos = (size_t)(doc - index->documents);
            scores[index_pos] += bm25_score(index, entry, posting->term_frequency, doc->document_id, doc->term_count);
        }
    }

    size_t count = 0;
    for (size_t i = 0; i < index->document_count; ++i) {
        if (scores[i] <= 0.0) continue;
        if (count < hit_capacity) {
            hits[count].document_id = index->documents[i].document_id;
            hits[count].score = scores[i];
            ++count;
        } else {
            size_t worst = 0;
            for (size_t j = 1; j < hit_capacity; ++j)
                if (hits[j].score < hits[worst].score) worst = j;
            if (scores[i] > hits[worst].score) {
                hits[worst].document_id = index->documents[i].document_id;
                hits[worst].score = scores[i];
            }
        }
    }

    qsort(hits, count, sizeof(*hits), term_compare);
    free(scores);
    return count;
}

const NiyahDocument *niyah_index_document(const NiyahInvertedIndex *index, uint64_t document_id) {
    if (!index || document_id == 0) return NULL;
    for (size_t i = 0; i < index->document_count; ++i)
        if (index->documents[i].document_id == document_id) return &index->documents[i];
    return NULL;
}
