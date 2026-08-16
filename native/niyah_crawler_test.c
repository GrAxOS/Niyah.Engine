#include "niyah_crawler.h"

#include <assert.h>
#include <string.h>

int main(void) {
    NiyahCrawlFrontier frontier;
    NiyahCrawlTarget target = {0};
    niyah_crawl_frontier_init(&frontier);

    assert(niyah_crawl_enqueue(&frontier, "https://example.test/a", 0));
    assert(!niyah_crawl_enqueue(&frontier, "https://example.test/a", 1));
    assert(niyah_crawl_enqueue(&frontier, "https://example.test/b", 1));

    assert(niyah_crawl_pop(&frontier, &target));
    assert(strcmp(target.url, "https://example.test/a") == 0);
    assert(target.depth == 0u);
    free(target.url);

    assert(niyah_crawl_pop(&frontier, &target));
    assert(strcmp(target.url, "https://example.test/b") == 0);
    free(target.url);

    assert(!niyah_crawl_pop(&frontier, &target));
    niyah_crawl_frontier_free(&frontier);
    return 0;
}
