#ifndef NIYAH_CRAWLER_H
#define NIYAH_CRAWLER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_FETCH_ALLOW = 0,
    NIYAH_FETCH_BLOCKED_ROBOTS = 1,
    NIYAH_FETCH_INVALID_URL = 2,
    NIYAH_FETCH_FRONTIER_FULL = 3,
    NIYAH_FETCH_DUPLICATE = 4
} NiyahFetchDecision;

typedef struct {
    char url[2048];
    uint32_t depth;
    uint64_t discovered_unix_ms;
} NiyahCrawlItem;

typedef struct {
    NiyahCrawlItem *items;
    size_t count;
    size_t capacity;
} NiyahCrawlFrontier;

void niyah_crawl_frontier_init(NiyahCrawlFrontier *frontier,
                               NiyahCrawlItem *storage,
                               size_t capacity);
NiyahFetchDecision niyah_crawl_enqueue(NiyahCrawlFrontier *frontier,
                                       const char *url,
                                       uint32_t depth,
                                       uint64_t now_unix_ms);
bool niyah_crawl_pop(NiyahCrawlFrontier *frontier, NiyahCrawlItem *out);

/* Minimal robots policy hook. Full robots.txt parsing lives above this ABI. */
bool niyah_robots_path_allowed(const char *path,
                               const char *robots_body,
                               const char *user_agent);

#ifdef __cplusplus
}
#endif

#endif
