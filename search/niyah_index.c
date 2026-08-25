#include "niyah_index.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIYAH_DOCUMENT_TOKEN_LIMIT 1024u
#define NIYAH_QUERY_TOKEN_LIMIT 128u
#define NIYAH_INITIAL_TERM_CAPACITY 128u
#define NIYAH_INITIAL_DOCUMENT_CAPACITY 64u
#define NIYAH_INITIAL_POSTING_CAPACITY 16u

/* Minimum hash-table capacity (power of two). */
#define NIYAH_HT_MIN_CAP 16u

/* -------------------------------------------------------------------------
 * FNV-1a hash helpers
 * ---------------------------------------------------------------------- */

static size_t fnv1a_str(const char *s) {
    uint64_t h = UINT64_C(14695981039346656037);
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= UINT64_C(1099511628211);
    }
    return (size_t)h;
}

static size_t fnv1a_u64(uint64_t v) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 0; i < 8; ++i) {
        h ^= (unsigned char)(v & 0xFF);
        h *= UINT64_C(1099511628211);
        v >>= 8;
    }
    return (size_t)h;
}

/* -------------------------------------------------------------------------
 * Term hash table  (open addressing, linear probing)
 * keys  : term_ht_keys[][NIYAH_TERM_MAX]  (copied strings)
 * values: term_ht[]                        (index into index->terms[])
 * empty slot marker: SIZE_MAX
 * ---------------------------------------------------------------------- */

static bool term_ht_alloc(NiyahInvertedIndex *index, size_t cap) {
    size_t *ht = malloc(cap * sizeof(*ht));
    if (!ht) return false;

    char (*keys)[NIYAH_TERM_MAX] = malloc(cap * sizeof(*keys));
    if (!keys) { free(ht); return false; }

    for (size_t i = 0; i < cap; ++i) {
        ht[i] = SIZE_MAX;
        keys[i][0] = '\0';
    }

    free(index->term_ht);
    free(index->term_ht_keys);
    index->term_ht      = ht;
    index->term_ht_keys = keys;
    index->term_ht_cap  = cap;
    return true;
}

/* Rebuild the term hash table from scratch.
 * cap is sized to hold at least `min_count` entries at load < 0.5. */
static bool term_ht_rebuild_for(NiyahInvertedIndex *index, size_t min_count) {
    if (!index) return false;

    /* Pick capacity: next power of two >= 2 * min_count, minimum NIYAH_HT_MIN_CAP. */
    size_t cap = NIYAH_HT_MIN_CAP;
    const size_t base = min_count > index->term_count ? min_count : index->term_count;
    if (base > SIZE_MAX / 2u) return false;
    const size_t need = base * 2u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return false;
        cap *= 2u;
    }

    if (!term_ht_alloc(index, cap)) return false;

    for (size_t i = 0; i < index->term_count; ++i) {
        const char *key = index->terms[i].term;
        size_t slot = fnv1a_str(key) & (cap - 1u);
        while (index->term_ht[slot] != SIZE_MAX) {
            slot = (slot + 1u) & (cap - 1u);
        }
        index->term_ht[slot] = i;
        memcpy(index->term_ht_keys[slot], key, NIYAH_TERM_MAX);
    }
    return true;
}

static bool term_ht_rebuild(NiyahInvertedIndex *index) {
    return term_ht_rebuild_for(index, 0);
}

/* Insert a new term at terms[new_index] into the hash table.
 * Assumes capacity is sufficient (caller must ensure load < 0.5). */
static void term_ht_insert(NiyahInvertedIndex *index, size_t new_index) {
    const char *key = index->terms[new_index].term;
    size_t slot = fnv1a_str(key) & (index->term_ht_cap - 1u);
    while (index->term_ht[slot] != SIZE_MAX) {
        slot = (slot + 1u) & (index->term_ht_cap - 1u);
    }
    index->term_ht[slot]      = new_index;
    memcpy(index->term_ht_keys[slot], key, NIYAH_TERM_MAX);
}

static NiyahTermEntry *term_ht_find(
    NiyahInvertedIndex *index,
    const char *term
) {
    if (!index->term_ht || index->term_ht_cap == 0) return NULL;
    size_t slot = fnv1a_str(term) & (index->term_ht_cap - 1u);
    for (;;) {
        size_t idx = index->term_ht[slot];
        if (idx == SIZE_MAX) return NULL;
        if (strcmp(index->term_ht_keys[slot], term) == 0) {
            return &index->terms[idx];
        }
        slot = (slot + 1u) & (index->term_ht_cap - 1u);
    }
}

static const NiyahTermEntry *term_ht_find_const(
    const NiyahInvertedIndex *index,
    const char *term
) {
    if (!index->term_ht || index->term_ht_cap == 0) return NULL;
    size_t slot = fnv1a_str(term) & (index->term_ht_cap - 1u);
    for (;;) {
        size_t idx = index->term_ht[slot];
        if (idx == SIZE_MAX) return NULL;
        if (strcmp(index->term_ht_keys[slot], term) == 0) {
            return &index->terms[idx];
        }
        slot = (slot + 1u) & (index->term_ht_cap - 1u);
    }
}

/* -------------------------------------------------------------------------
 * Document hash table  (open addressing, linear probing)
 * keys  : doc_ht_keys[] (uint64_t document_id)
 * values: doc_ht[]       (index into index->documents[])
 * empty slot marker: SIZE_MAX  (key 0 is already forbidden by the API)
 * ---------------------------------------------------------------------- */

static bool doc_ht_alloc(NiyahInvertedIndex *index, size_t cap) {
    size_t   *ht   = malloc(cap * sizeof(*ht));
    if (!ht) return false;
    uint64_t *keys = malloc(cap * sizeof(*keys));
    if (!keys) { free(ht); return false; }

    for (size_t i = 0; i < cap; ++i) {
        ht[i]   = SIZE_MAX;
        keys[i] = 0u;
    }

    free(index->doc_ht);
    free(index->doc_ht_keys);
    index->doc_ht      = ht;
    index->doc_ht_keys = keys;
    index->doc_ht_cap  = cap;
    return true;
}

static bool doc_ht_rebuild_for(NiyahInvertedIndex *index, size_t min_count) {
    if (!index) return false;

    size_t cap = NIYAH_HT_MIN_CAP;
    const size_t need = (min_count > index->document_count ? min_count : index->document_count) * 2u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return false;
        cap *= 2u;
    }

    if (!doc_ht_alloc(index, cap)) return false;

    for (size_t i = 0; i < index->document_count; ++i) {
        uint64_t key = index->documents[i].document_id;
        size_t slot = fnv1a_u64(key) & (cap - 1u);
        while (index->doc_ht[slot] != SIZE_MAX) {
            slot = (slot + 1u) & (cap - 1u);
        }
        index->doc_ht[slot]      = i;
        index->doc_ht_keys[slot] = key;
    }
    return true;
}

static bool doc_ht_rebuild(NiyahInvertedIndex *index) {
    return doc_ht_rebuild_for(index, 0);
}

static void doc_ht_insert(NiyahInvertedIndex *index, size_t new_index) {
    uint64_t key  = index->documents[new_index].document_id;
    size_t slot = fnv1a_u64(key) & (index->doc_ht_cap - 1u);
    while (index->doc_ht[slot] != SIZE_MAX) {
        slot = (slot + 1u) & (index->doc_ht_cap - 1u);
    }
    index->doc_ht[slot]      = new_index;
    index->doc_ht_keys[slot] = key;
}

static size_t doc_ht_find(
    const NiyahInvertedIndex *index,
    uint64_t document_id
) {
    if (!index->doc_ht || index->doc_ht_cap == 0) return SIZE_MAX;
    size_t slot = fnv1a_u64(document_id) & (index->doc_ht_cap - 1u);
    for (;;) {
        if (index->doc_ht[slot] == SIZE_MAX) return SIZE_MAX;
        if (index->doc_ht_keys[slot] == document_id) {
            return index->doc_ht[slot];
        }
        slot = (slot + 1u) & (index->doc_ht_cap - 1u);
    }
}

/* -------------------------------------------------------------------------
 * Original helpers that remain (unchanged from original)
 * ---------------------------------------------------------------------- */

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
    return isalnum(c) != 0 ||
           c >= 0x80u ||
           c == '_' ||
           c == '-';
}

static size_t tokenize(
    const char *text,
    char tokens[][NIYAH_TERM_MAX],
    size_t max_tokens
) {
    if (!text || !tokens || max_tokens == 0) {
        return 0;
    }

    size_t count = 0;
    const unsigned char *cursor =
        (const unsigned char *)text;

    while (*cursor && count < max_tokens) {
        while (*cursor && !token_byte(*cursor)) {
            ++cursor;
        }

        if (!*cursor) {
            break;
        }

        size_t length = 0;

        while (*cursor && token_byte(*cursor)) {
            if (length + 1u < NIYAH_TERM_MAX) {
                const unsigned char byte = *cursor;

                tokens[count][length++] =
                    (char)(byte < 0x80u
                        ? tolower(byte)
                        : byte);
            }

            ++cursor;
        }

        tokens[count][length] = '\0';

        if (length > 0) {
            ++count;
        }
    }

    return count;
}

/*
 * find_term / find_term_const: O(1) hash-map lookup replacing the former
 * linear scan.
 */
static NiyahTermEntry *find_term(
    NiyahInvertedIndex *index,
    const char *term
) {
    if (!index || !term) return NULL;
    return term_ht_find(index, term);
}

static const NiyahTermEntry *find_term_const(
    const NiyahInvertedIndex *index,
    const char *term
) {
    if (!index || !term) return NULL;
    return term_ht_find_const(index, term);
}

static bool checked_capacity_growth(
    size_t current,
    size_t initial,
    size_t element_size,
    size_t *next_out
) {
    if (!next_out || element_size == 0) {
        return false;
    }

    size_t next = current == 0
        ? initial
        : current * 2u;

    if (next < current) {
        return false;
    }

    if (next > SIZE_MAX / element_size) {
        return false;
    }

    *next_out = next;
    return true;
}

static bool grow_terms(NiyahInvertedIndex *index) {
    if (!index) {
        return false;
    }

    size_t next = 0;

    if (!checked_capacity_growth(
            index->term_capacity,
            NIYAH_INITIAL_TERM_CAPACITY,
            sizeof(*index->terms),
            &next)) {
        return false;
    }

    NiyahTermEntry *terms =
        realloc(index->terms, next * sizeof(*terms));

    if (!terms) {
        return false;
    }

    if (next > index->term_capacity) {
        memset(
            terms + index->term_capacity,
            0,
            (next - index->term_capacity) * sizeof(*terms)
        );
    }

    index->terms = terms;
    index->term_capacity = next;

    /*
     * The terms array may have moved; rebuild the hash table so its stored
     * indices remain valid.  (The indices themselves don't change — only the
     * base pointer changed — so a rebuild is not strictly necessary for
     * correctness here, but a fresh rehash also doubles the HT capacity to
     * keep load < 0.5 for the upcoming insertions.)
     */
    return term_ht_rebuild(index);
}

static bool grow_documents(NiyahInvertedIndex *index) {
    if (!index) {
        return false;
    }

    size_t next = 0;

    if (!checked_capacity_growth(
            index->document_capacity,
            NIYAH_INITIAL_DOCUMENT_CAPACITY,
            sizeof(*index->documents),
            &next)) {
        return false;
    }

    NiyahDocument *documents =
        realloc(index->documents, next * sizeof(*documents));

    if (!documents) {
        return false;
    }

    index->documents = documents;
    index->document_capacity = next;

    /* Rebuild doc HT after potential realloc (same reason as term HT above). */
    return doc_ht_rebuild(index);
}

static bool grow_postings(NiyahTermEntry *entry) {
    if (!entry) {
        return false;
    }

    size_t next = 0;

    if (!checked_capacity_growth(
            entry->posting_capacity,
            NIYAH_INITIAL_POSTING_CAPACITY,
            sizeof(*entry->postings),
            &next)) {
        return false;
    }

    NiyahPosting *postings =
        realloc(entry->postings, next * sizeof(*postings));

    if (!postings) {
        return false;
    }

    entry->postings = postings;
    entry->posting_capacity = next;

    return true;
}

static bool token_seen_before(
    char tokens[][NIYAH_TERM_MAX],
    size_t index,
    const char *token
) {
    if (!tokens || !token) {
        return false;
    }

    for (size_t i = 0; i < index; ++i) {
        if (strcmp(tokens[i], token) == 0) {
            return true;
        }
    }

    return false;
}

/*
 * Number of occurrences of tokens[first_index] in tokens[first_index ..
 * token_count). Callers only invoke this on the first occurrence of a term,
 * so the result is the document term frequency. Saturates at UINT32_MAX.
 */
static uint32_t count_occurrences(
    char tokens[][NIYAH_TERM_MAX],
    size_t token_count,
    size_t first_index
) {
    if (!tokens || first_index >= token_count) {
        return 0;
    }

    uint32_t frequency = 0;

    for (size_t i = first_index; i < token_count; ++i) {
        if (strcmp(tokens[i], tokens[first_index]) != 0) {
            continue;
        }

        if (frequency == UINT32_MAX) {
            break;
        }

        ++frequency;
    }

    return frequency;
}

/*
 * document_position: O(1) hash-map lookup replacing the former linear scan.
 */
static size_t document_position(
    const NiyahInvertedIndex *index,
    uint64_t document_id
) {
    if (!index || document_id == 0) {
        return SIZE_MAX;
    }

    return doc_ht_find(index, document_id);
}

static bool ensure_term_capacity(
    NiyahInvertedIndex *index,
    size_t additional_terms
) {
    if (!index) {
        return false;
    }

    if (additional_terms >
        SIZE_MAX - index->term_count) {
        return false;
    }

    const size_t required =
        index->term_count + additional_terms;

    while (index->term_capacity < required) {
        if (!grow_terms(index)) {
            return false;
        }
    }

    /* Also ensure the term HT has enough headroom (load < 0.5). */
    if (index->term_ht_cap < required * 2u) {
        if (!term_ht_rebuild_for(index, required)) {
            return false;
        }
    }

    return true;
}

void niyah_index_init(
    NiyahInvertedIndex *index,
    double k1,
    double b
) {
    if (!index) {
        return;
    }

    memset(index, 0, sizeof(*index));

    index->k1 = k1 > 0.0 ? k1 : 1.2;
    index->b =
        b >= 0.0 && b <= 1.0
            ? b
            : 0.75;

    /* Allocate initial hash tables; failures are tolerated here — the tables
     * will be (re-)allocated on first insertion if needed. */
    (void)term_ht_alloc(index, NIYAH_HT_MIN_CAP);
    (void)doc_ht_alloc(index, NIYAH_HT_MIN_CAP);
}

void niyah_index_free(
    NiyahInvertedIndex *index
) {
    if (!index) {
        return;
    }

    for (size_t i = 0; i < index->term_count; ++i) {
        free(index->terms[i].postings);
        index->terms[i].postings = NULL;
        index->terms[i].posting_count = 0;
        index->terms[i].posting_capacity = 0;
    }

    free(index->terms);
    free(index->documents);
    free(index->term_ht);
    free(index->term_ht_keys);
    free(index->doc_ht);
    free(index->doc_ht_keys);

    memset(index, 0, sizeof(*index));
}

bool niyah_index_add_document(
    NiyahInvertedIndex *index,
    const NiyahDocument *document
) {
    if (!index ||
        !document ||
        document->document_id == 0) {
        return false;
    }

    if (niyah_index_document(
            index,
            document->document_id)) {
        return false;
    }

    char tokens[
        NIYAH_DOCUMENT_TOKEN_LIMIT
    ][NIYAH_TERM_MAX];

    const size_t token_count =
        tokenize(
            document->text,
            tokens,
            NIYAH_DOCUMENT_TOKEN_LIMIT
        );

    if (token_count > UINT32_MAX) {
        return false;
    }

    size_t unique_missing_terms = 0;

    for (size_t i = 0; i < token_count; ++i) {
        if (token_seen_before(tokens, i, tokens[i])) {
            continue;
        }

        if (!find_term(index, tokens[i])) {
            ++unique_missing_terms;
        }
    }

    if (!ensure_term_capacity(
            index,
            unique_missing_terms)) {
        return false;
    }

    /* Ensure doc HT has headroom for one more entry. */
    if (index->doc_ht_cap < (index->document_count + 1u) * 2u) {
        if (!doc_ht_rebuild_for(index, index->document_count + 1u)) {
            return false;
        }
    }

    if (index->document_count ==
        index->document_capacity) {
        if (!grow_documents(index)) {
            return false;
        }
    }

    NiyahDocument *destination =
        &index->documents[index->document_count];

    *destination = *document;
    destination->term_count =
        (uint32_t)token_count;

    /* Insert into doc HT before incrementing document_count so rebuild
     * (if triggered inside grow_documents) sees consistent state. */
    doc_ht_insert(index, index->document_count);
    ++index->document_count;

    if (index->document_count == 1) {
        index->average_document_length =
            (double)token_count;
    } else {
        const double previous_count =
            (double)(index->document_count - 1u);

        index->average_document_length =
            (
                index->average_document_length *
                previous_count +
                (double)token_count
            ) / (double)index->document_count;
    }

    for (size_t i = 0; i < token_count; ++i) {
        if (token_seen_before(tokens, i, tokens[i])) {
            continue;
        }

        NiyahTermEntry *entry =
            find_term(index, tokens[i]);

        if (!entry) {
            if (index->term_count >=
                index->term_capacity) {
                return false;
            }

            const size_t new_index = index->term_count;
            entry = &index->terms[new_index];

            const size_t source_length =
                strlen(tokens[i]);

            const size_t copy_length =
                source_length < sizeof(entry->term) - 1u
                    ? source_length
                    : sizeof(entry->term) - 1u;

            memcpy(
                entry->term,
                tokens[i],
                copy_length
            );

            entry->term[copy_length] = '\0';
            ++index->term_count;

            /* Insert new term into the hash table. */
            term_ht_insert(index, new_index);
        }

        if (entry->posting_count ==
            entry->posting_capacity) {
            if (!grow_postings(entry)) {
                return false;
            }
        }

        NiyahPosting *posting =
            &entry->postings[entry->posting_count++];

        posting->document_id =
            document->document_id;

        /*
         * Real term frequency: repeated tokens are skipped when building the
         * postings list, so the count is taken from the token buffer instead
         * of being pinned to 1. Without this the BM25 saturation term
         * degenerates and scoring becomes binary presence weighted by IDF and
         * document length.
         */
        posting->term_frequency =
            count_occurrences(tokens, token_count, i);

        entry->document_frequency++;
    }

    return true;
}

static double bm25_score(
    const NiyahInvertedIndex *index,
    const NiyahTermEntry *entry,
    uint32_t term_frequency,
    uint32_t document_length
) {
    if (!index ||
        !entry ||
        term_frequency == 0 ||
        document_length == 0 ||
        index->document_count == 0) {
        return 0.0;
    }

    const double documents =
        (double)index->document_count;

    const double document_frequency =
        (double)entry->document_frequency;

    const double denominator =
        document_frequency + 0.5;

    if (denominator <= 0.0) {
        return 0.0;
    }

    const double idf =
        log1p(
            (documents - document_frequency + 0.5) /
            denominator
        );

    const double average_length =
        index->average_document_length > 0.0
            ? index->average_document_length
            : 1.0;

    const double normalization =
        index->k1 *
        (
            1.0 - index->b +
            index->b *
            (
                (double)document_length /
                average_length
            )
        );

    const double tf =
        (double)term_frequency;

    const double numerator =
        tf * (index->k1 + 1.0);

    const double score_denominator =
        tf + normalization;

    if (score_denominator <= 0.0) {
        return 0.0;
    }

    return idf * (numerator / score_denominator);
}

size_t niyah_index_search(
    const NiyahInvertedIndex *index,
    const char *query,
    NiyahSearchHit *hits,
    size_t hit_capacity
) {
    if (!index ||
        !query ||
        !hits ||
        hit_capacity == 0 ||
        index->document_count == 0) {
        return 0;
    }

    double *scores =
        calloc(index->document_count, sizeof(*scores));

    if (!scores) {
        return 0;
    }

    char query_tokens[
        NIYAH_QUERY_TOKEN_LIMIT
    ][NIYAH_TERM_MAX];

    const size_t query_token_count =
        tokenize(
            query,
            query_tokens,
            NIYAH_QUERY_TOKEN_LIMIT
        );

    for (size_t query_index = 0;
         query_index < query_token_count;
         ++query_index) {
        if (token_seen_before(
                query_tokens,
                query_index,
                query_tokens[query_index])) {
            continue;
        }

        const NiyahTermEntry *entry =
            find_term_const(
                index,
                query_tokens[query_index]
            );

        if (!entry) {
            continue;
        }

        for (size_t posting_index = 0;
             posting_index < entry->posting_count;
             ++posting_index) {
            const NiyahPosting *posting =
                &entry->postings[posting_index];

            const size_t document_index =
                document_position(
                    index,
                    posting->document_id
                );

            if (document_index == SIZE_MAX) {
                continue;
            }

            const NiyahDocument *document =
                &index->documents[document_index];

            scores[document_index] +=
                bm25_score(
                    index,
                    entry,
                    posting->term_frequency,
                    document->term_count
                );
        }
    }

    size_t hit_count = 0;

    for (size_t document_index = 0;
         document_index < index->document_count;
         ++document_index) {
        const double score =
            scores[document_index];

        if (!(score > 0.0) || !isfinite(score)) {
            continue;
        }

        if (hit_count < hit_capacity) {
            hits[hit_count].document_id =
                index->documents[document_index].document_id;

            hits[hit_count].score = score;
            ++hit_count;
            continue;
        }

        size_t worst_index = 0;

        for (size_t hit_index = 1;
             hit_index < hit_capacity;
             ++hit_index) {
            if (hits[hit_index].score <
                hits[worst_index].score) {
                worst_index = hit_index;
            }
        }

        if (score > hits[worst_index].score) {
            hits[worst_index].document_id =
                index->documents[document_index].document_id;

            hits[worst_index].score = score;
        }
    }

    qsort(
        hits,
        hit_count,
        sizeof(*hits),
        term_compare
    );

    free(scores);
    return hit_count;
}

const NiyahDocument *niyah_index_document(
    const NiyahInvertedIndex *index,
    uint64_t document_id
) {
    if (!index || document_id == 0) {
        return NULL;
    }

    const size_t position =
        document_position(index, document_id);

    if (position == SIZE_MAX) {
        return NULL;
    }

    return &index->documents[position];
}

