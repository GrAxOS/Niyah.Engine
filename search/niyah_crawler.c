#include "niyah_crawler.h"

#include <stdio.h>
#include <string.h>

static bool prefix_match(const char *path, const char *rule) {
    if (!rule || !*rule) return false;
    size_t n = strlen(rule);
    return strncmp(path, rule, n) == 0;
}

void niyah_crawl_frontier_init(NiyahCrawlFrontier *frontier,
                               NiyahCrawlItem *storage,
                               size_t capacity) {
    if (!frontier) return;
    frontier->items = storage;
    frontier->count = 0;
    frontier->capacity = capacity;
}

NiyahFetchDecision niyah_crawl_enqueue(NiyahCrawlFrontier *frontier,
                                       const char *url,
                                       uint32_t depth,
                                       uint64_t now_unix_ms) {
    if (!frontier || !frontier->items || !url || !*url) return NIYAH_FETCH_INVALID_URL;
    if (frontier->count >= frontier->capacity) return NIYAH_FETCH_FRONTIER_FULL;
    size_t n = strlen(url);
    if (n == 0 || n >= sizeof(frontier->items[0].url)) return NIYAH_FETCH_INVALID_URL;

    for (size_t i = 0; i < frontier->count; ++i) {
        if (strcmp(frontier->items[i].url, url) == 0) return NIYAH_FETCH_DUPLICATE;
    }

    NiyahCrawlItem *item = &frontier->items[frontier->count++];
    memcpy(item->url, url, n + 1);
    item->depth = depth;
    item->discovered_unix_ms = now_unix_ms;
    return NIYAH_FETCH_ALLOW;
}

bool niyah_crawl_pop(NiyahCrawlFrontier *frontier, NiyahCrawlItem *out) {
    if (!frontier || !out || frontier->count == 0) return false;
    *out = frontier->items[0];
    if (frontier->count > 1) {
        memmove(frontier->items,
                frontier->items + 1,
                (frontier->count - 1) * sizeof(*frontier->items));
    }
    frontier->count--;
    return true;
}

bool niyah_robots_path_allowed(const char *path,
                               const char *robots_body,
                               const char *user_agent) {
    if (!path || !robots_body) return true;
    const char *ua = (user_agent && *user_agent) ? user_agent : "*";
    const char *cursor = robots_body;
    bool active = false;
    bool matched_ua = false;

    while (*cursor) {
        const char *line_end = strchr(cursor, '\n');
        size_t line_len = line_end ? (size_t)(line_end - cursor) : strlen(cursor);
        char line[1024];
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, cursor, line_len);
        line[line_len] = '\0';

        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';

        char field[256] = {0};
        char value[768] = {0};
        if (sscanf(line, " %255[^:]: %767[^\n]", field, value) == 2) {
            if (strcasecmp(field, "User-agent") == 0) {
                active = false;
                if (strcasecmp(value, ua) == 0 || strcmp(value, "*") == 0) {
                    active = true;
                    matched_ua = true;
                }
            } else if (active && strcasecmp(field, "Disallow") == 0 && matched_ua) {
                if (value[0] == '\0') {
                    /* Empty Disallow means allow all. */
                } else if (prefix_match(path, value)) {
                    return false;
                }
            }
        }

        if (!line_end) break;
        cursor = line_end + 1;
    }
    return true;
}
