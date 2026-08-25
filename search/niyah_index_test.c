#include "niyah_index.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/*
 * NiyahDocument borrows `text`; it has no url/title members. The previous
 * version of this file wrote into document.url/.title/.text with snprintf and
 * could not compile against niyah_index.h.
 */
static NiyahDocument make_document(
    uint64_t id,
    const char *text
) {
    NiyahDocument document;

    memset(&document, 0, sizeof(document));

    document.document_id = id;
    document.text = text ? text : "";

    return document;
}

static const NiyahPosting *find_posting(
    const NiyahInvertedIndex *index,
    const char *term,
    uint64_t document_id
) {
    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) != 0) {
            continue;
        }

        const NiyahTermEntry *entry = &index->terms[i];

        for (size_t j = 0; j < entry->posting_count; ++j) {
            if (entry->postings[j].document_id == document_id) {
                return &entry->postings[j];
            }
        }
    }

    return NULL;
}

static void test_empty_index(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    assert(index.term_count == 0);
    assert(index.document_count == 0);
    assert(index.average_document_length == 0.0);

    niyah_index_free(&index);
}

static void test_add_and_find_document(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument document =
        make_document(
            1,
            "C programming systems programming"
        );

    assert(niyah_index_add_document(
        &index,
        &document
    ));

    assert(index.document_count == 1);
    assert(index.term_count == 3);
    assert(index.documents[0].term_count == 4);

    const NiyahDocument *found =
        niyah_index_document(&index, 1);

    assert(found != NULL);
    assert(found->document_id == 1);

    assert(niyah_index_document(
        &index,
        999
    ) == NULL);

    niyah_index_free(&index);
}

/* A term repeated in a document must carry its real frequency. */
static void test_term_frequency_is_counted(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument document =
        make_document(
            1,
            "cache cache cache miss"
        );

    assert(niyah_index_add_document(
        &index,
        &document
    ));

    const NiyahPosting *repeated =
        find_posting(&index, "cache", 1);

    const NiyahPosting *single =
        find_posting(&index, "miss", 1);

    assert(repeated != NULL);
    assert(single != NULL);
    assert(repeated->term_frequency == 3);
    assert(single->term_frequency == 1);

    /* One posting per (term, document) pair, not one per occurrence. */
    for (size_t i = 0; i < index.term_count; ++i) {
        assert(index.terms[i].posting_count == 1);
        assert(index.terms[i].document_frequency == 1);
    }

    niyah_index_free(&index);
}

/*
 * Same document length, same document frequency: the only difference is the
 * term frequency, so BM25 must rank the denser document first. This assertion
 * fails if term_frequency is pinned to a constant.
 */
static void test_term_frequency_affects_ranking(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument dense =
        make_document(1, "cache cache cache miss");

    NiyahDocument sparse =
        make_document(2, "cache miss miss miss");

    assert(niyah_index_add_document(&index, &dense));
    assert(niyah_index_add_document(&index, &sparse));

    NiyahSearchHit hits[2];
    memset(hits, 0, sizeof(hits));

    const size_t count =
        niyah_index_search(&index, "cache", hits, 2);

    assert(count == 2);
    assert(hits[0].document_id == 1);
    assert(hits[1].document_id == 2);
    assert(hits[0].score > hits[1].score);

    niyah_index_free(&index);
}

static void test_duplicate_document_rejected(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument first =
        make_document(1, "alpha beta");

    NiyahDocument duplicate =
        make_document(1, "different text");

    assert(niyah_index_add_document(
        &index,
        &first
    ));

    assert(!niyah_index_add_document(
        &index,
        &duplicate
    ));

    assert(index.document_count == 1);
    assert(index.term_count == 2);

    niyah_index_free(&index);
}

static void test_search_is_deterministic(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument first =
        make_document(1, "systems programming");

    NiyahDocument second =
        make_document(2, "systems programming");

    NiyahDocument third =
        make_document(3, "unrelated topic");

    assert(niyah_index_add_document(
        &index,
        &first
    ));

    assert(niyah_index_add_document(
        &index,
        &second
    ));

    assert(niyah_index_add_document(
        &index,
        &third
    ));

    NiyahSearchHit hits[2];
    memset(hits, 0, sizeof(hits));

    const size_t count =
        niyah_index_search(
            &index,
            "systems",
            hits,
            2
        );

    assert(count == 2);
    assert(hits[0].document_id == 1);
    assert(hits[1].document_id == 2);
    assert(hits[0].score >= hits[1].score);

    niyah_index_free(&index);
}

static void test_empty_query(void) {
    NiyahInvertedIndex index;

    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument document =
        make_document(1, "alpha beta");

    assert(niyah_index_add_document(
        &index,
        &document
    ));

    NiyahSearchHit hit;
    memset(&hit, 0, sizeof(hit));

    assert(niyah_index_search(
        &index,
        "",
        &hit,
        1
    ) == 0);

    niyah_index_free(&index);
}

/*
 * Insert 3 000 unique single-token documents and verify that every document
 * can be looked up by id and that a search for the shared prefix token
 * returns a non-zero hit count.  This exercises hash-table growth (many
 * rehashes) and validates correctness at scale.
 */
static void test_scale_many_documents(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    /* Texts must outlive the index (borrowed). Use static storage. */
    static char texts[3000][32];
    for (int i = 0; i < 3000; ++i) {
        /* Each document has one unique term plus a shared "common" term. */
        snprintf(texts[i], sizeof(texts[i]), "common token_%d", i);
        NiyahDocument doc;
        doc.document_id = (uint64_t)(i + 1);
        doc.text        = texts[i];
        doc.term_count  = 0;
        assert(niyah_index_add_document(&index, &doc));
    }

    assert(index.document_count == 3000u);

    /* Every inserted document must be retrievable. */
    for (int i = 0; i < 3000; ++i) {
        const NiyahDocument *found =
            niyah_index_document(&index, (uint64_t)(i + 1));
        assert(found != NULL);
        assert(found->document_id == (uint64_t)(i + 1));
    }

    /* A doc that was never inserted must not be found. */
    assert(niyah_index_document(&index, 99999u) == NULL);

    /* Search for the shared term; must get hits (up to hit_capacity). */
    NiyahSearchHit hits[10];
    memset(hits, 0, sizeof(hits));
    const size_t count =
        niyah_index_search(&index, "common", hits, 10);
    assert(count == 10u);

    niyah_index_free(&index);
}

/*
 * Index a single document with 2 000 distinct tokens (unique terms).
 * Verifies that the term hash table grows correctly without corruption.
 */
static void test_scale_many_terms(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    /* Build a single document text with 2 000 unique space-separated tokens. */
    static char big_text[2000 * 12]; /* "termXXXX " * 2000 < 90 000 bytes */
    size_t pos = 0;
    for (int i = 0; i < 2000; ++i) {
        int written = snprintf(big_text + pos,
                               sizeof(big_text) - pos,
                               "term%04d ", i);
        assert(written > 0);
        pos += (size_t)written;
    }

    NiyahDocument doc;
    doc.document_id = 1;
    doc.text        = big_text;
    doc.term_count  = 0;
    assert(niyah_index_add_document(&index, &doc));

    /* All terms up to the token limit must be indexed. */
    assert(index.term_count == 1024u); /* NIYAH_DOCUMENT_TOKEN_LIMIT = 1024 */

    /* Spot-check: look up a few terms via search. */
    NiyahSearchHit hit;
    memset(&hit, 0, sizeof(hit));
    assert(niyah_index_search(&index, "term0000", &hit, 1) == 1u);
    assert(hit.document_id == 1u);

    memset(&hit, 0, sizeof(hit));
    assert(niyah_index_search(&index, "term1023", &hit, 1) == 1u); /* last indexed token */
    assert(hit.document_id == 1u);

    niyah_index_free(&index);
}

/*
 * Correctness under intentional hash collisions.
 *
 * We insert a set of terms that are guaranteed to collide in the hash table
 * (they are chosen to share the same slot modulo any power-of-two capacity
 * from 16 upwards using FNV-1a) and verify that all of them are found and
 * scored correctly afterwards.
 *
 * Rather than hard-coding FNV-1a collision pairs (fragile), we insert enough
 * terms that the open-addressing table must resolve many collisions — the
 * load factor reaches ~0.4 before a rehash — and check that every term is
 * still retrievable with the right document frequency.
 */
static void test_hash_collision_correctness(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    /*
     * Build 60 documents, each containing all of terms t00..t29.  After
     * insertion, every term has document_frequency == 60, and we can verify
     * that searching for any term returns all 60 documents.
     */
    static char col_texts[60][256];
    for (int d = 0; d < 60; ++d) {
        size_t p = 0;
        for (int t = 0; t < 30; ++t) {
            int w = snprintf(col_texts[d] + p,
                             sizeof(col_texts[d]) - p,
                             "t%02d ", t);
            assert(w > 0);
            p += (size_t)w;
        }
        NiyahDocument doc;
        doc.document_id = (uint64_t)(d + 1);
        doc.text        = col_texts[d];
        doc.term_count  = 0;
        assert(niyah_index_add_document(&index, &doc));
    }

    assert(index.document_count == 60u);
    assert(index.term_count == 30u);

    /* Every term must be found in all 60 documents. */
    for (size_t i = 0; i < index.term_count; ++i) {
        assert(index.terms[i].document_frequency == 60u);
    }

    /* A search for t00 must return 60 hits (exact, up to hit_capacity). */
    NiyahSearchHit hits[60];
    memset(hits, 0, sizeof(hits));
    const size_t count =
        niyah_index_search(&index, "t00", hits, 60);
    assert(count == 60u);

    niyah_index_free(&index);
}

int main(void) {
    test_empty_index();
    test_add_and_find_document();
    test_term_frequency_is_counted();
    test_term_frequency_affects_ranking();
    test_duplicate_document_rejected();
    test_search_is_deterministic();
    test_empty_query();
    test_scale_many_terms();
    test_scale_many_documents();
    test_hash_collision_correctness();

    return 0;
}
