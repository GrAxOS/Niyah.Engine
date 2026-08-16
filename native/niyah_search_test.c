#include "niyah_search.h"

#include <assert.h>
#include <stddef.h>

int main(void) {
    NiyahSearchIndex *index = niyah_search_create();
    assert(index != NULL);

    NiyahSearchDocument a = {
        1, "https://example.test/a", "Linux", "Linux kernel networking socket programming"
    };
    NiyahSearchDocument b = {
        2, "https://example.test/b", "Windows", "Windows networking sockets and programming"
    };
    NiyahSearchDocument c = {
        3, "https://example.test/c", "Math", "linear algebra and numerical methods"
    };

    assert(niyah_search_add(index, &a));
    assert(niyah_search_add(index, &b));
    assert(niyah_search_add(index, &c));
    assert(!niyah_search_add(index, &a));

    NiyahSearchHit hits[3] = {0};
    size_t count = niyah_search_query(index, "linux networking", hits, 3);
    assert(count >= 1u);
    assert(hits[0].document_id == 1u);
    assert(hits[0].score > 0.0);

    assert(niyah_search_remove(index, 1u));
    count = niyah_search_query(index, "linux networking", hits, 3);
    assert(count == 0u);

    niyah_search_free(index);
    return 0;
}
