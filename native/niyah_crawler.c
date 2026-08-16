#include "niyah_crawler.h"

#include <stdlib.h>
#include <string.h>

static char *dup_string(const char *value) {
    size_t n;
    char *copy;
    if (!value) return NULL;
    n = strlen(value);
    copy = (char *)malloc(n + 1u);
    if (!copy) return NULL;
    memcpy(copy, value, n + 1u);
    return copy;
}

void niyah_crawl_frontier_init(NiyahCrawlFrontier *frontier) {
    if (!frontier) return;
    memset(frontier, 0, sizeof(*frontier));
}

void niyah_crawl_frontier_free(NiyahCrawlFrontier *frontier) {
    size_t i;
    if (!frontier) return;
    for (i = 0u; i < frontier->count; ++i) free(frontier->items[i].url);
    free(frontier->items);
    memset(frontier, 0, sizeof(*frontier));
}

int niyah_crawl_enqueue(NiyahCrawlFrontier *frontier,
                        const char *url,
                        uint32_t depth) {
    size_t i;
    char *copy;
    if (!frontier || !url || url[0] == '\0') return 0;
    for (i = 0u; i < frontier->count; ++i) {
        if (strcmp(frontier->items[i].url, url) == 0 &&
            frontier->items[i].depth <= depth) {
            return 0;
        }
    }
    if (frontier->count == frontier->capacity) {
        size_t next = frontier->capacity ? frontier->capacity * 2u : 64u;
        if (next < frontier->capacity || next > SIZE_MAX / sizeof(*frontier->items)) return 0;
        NiyahCrawlTarget *items = (NiyahCrawlTarget *)realloc(
            frontier->items, next * sizeof(*frontier->items));
        if (!items) return 0;
        frontier->items = items;
        frontier->capacity = next;
    }
    copy = dup_string(url);
    if (!copy) return 0;
    frontier->items[frontier->count].url = copy;
    frontier->items[frontier->count].depth = depth;
    frontier->count++;
    return 1;
}

int niyah_crawl_pop(NiyahCrawlFrontier *frontier, NiyahCrawlTarget *out) {
    if (!frontier || !out || frontier->count == 0u) return 0;
    *out = frontier->items[0];
    if (frontier->count > 1u) {
        memmove(frontier->items,
                frontier->items + 1u,
                (frontier->count - 1u) * sizeof(*frontier->items));
    }
    frontier->count--;
    return 1;
}
