#ifndef NIYAH_CRAWLER_H
#define NIYAH_CRAWLER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *url;
    uint32_t depth;
} NiyahCrawlTarget;

typedef struct {
    size_t max_pages;
    size_t max_bytes_per_page;
    uint32_t max_depth;
    unsigned request_timeout_ms;
    bool obey_robots;
} NiyahCrawlPolicy;

typedef struct {
    NiyahCrawlTarget *items;
    size_t count;
    size_t capacity;
} NiyahCrawlFrontier;

void niyah_crawl_frontier_init(NiyahCrawlFrontier *frontier);
void niyah_crawl_frontier_free(NiyahCrawlFrontier *frontier);
int niyah_crawl_enqueue(NiyahCrawlFrontier *frontier,
                        const char *url,
                        uint32_t depth);
int niyah_crawl_pop(NiyahCrawlFrontier *frontier,
                    NiyahCrawlTarget *out);

#ifdef __cplusplus
}
#endif

#endif
