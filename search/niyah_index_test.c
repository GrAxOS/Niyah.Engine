#include "niyah_index.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static NiyahDocument make_document(
    uint64_t id,
    const char *text
) {
    NiyahDocument document;

    memset(&document, 0, sizeof(document));

    document.document_id = id;

    snprintf(
        document.url,
        sizeof(document.url),
        "https://example.test/%llu",
        (unsigned long long)id
    );

    snprintf(
        document.title,
        sizeof(document.title),
        "Document %llu",
        (unsigned long long)id
    );

    snprintf(
        document.text,
        sizeof(document.text),
        "%s",
        text ? text : ""
    );

    return document;
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

int main(void) {
    test_empty_index();
    test_add_and_find_document();
    test_duplicate_document_rejected();
    test_search_is_deterministic();
    test_empty_query();

    return 0;
}