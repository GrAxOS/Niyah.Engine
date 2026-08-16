#include "niyah_index.h"

#include <assert.h>
#include <string.h>

int main(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    NiyahDocument a = { .document_id = 1 };
    strcpy(a.url, "https://example.test/a");
    strcpy(a.title, "Systems Programming");
    strcpy(a.text, "C programming systems memory kernel");

    NiyahDocument b = { .document_id = 2 };
    strcpy(b.url, "https://example.test/b");
    strcpy(b.title, "Search Engine");
    strcpy(b.text, "search indexing ranking crawler");

    assert(niyah_index_add_document(&index, &a));
    assert(niyah_index_add_document(&index, &b));
    assert(!niyah_index_add_document(&index, &a));

    NiyahSearchHit hits[4];
    size_t count = niyah_index_search(&index, "kernel memory", hits, 4);
    assert(count == 1);
    assert(hits[0].document_id == 1);
    assert(hits[0].score > 0.0);

    niyah_index_free(&index);
    return 0;
}
