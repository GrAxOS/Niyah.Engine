#include "niyah.h"

#include <stdlib.h>
#include <string.h>

/*
 * Was: `// Crawler stubs`.
 *
 * Page-collection bookkeeping only. Actual HTTP lives in search/http_fetch.cpp;
 * this keeps the crawl frontier and enforces the configured max_pages limit.
 */

NiyahStatus niyah_crawler_init(NiyahCrawler* crawler,
                               const NiyahCrawlerConfig* config)
{
    if (!crawler) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(crawler, 0, sizeof(*crawler));

    if (config) {
        crawler->config = *config;
    } else {
        crawler->config.max_depth = 2;
        crawler->config.max_pages = 64;
        crawler->config.follow_links = true;
    }

    if (crawler->config.max_pages <= 0) {
        crawler->config.max_pages = 64;
    }
    if (crawler->config.max_depth <= 0) {
        crawler->config.max_depth = 1;
    }

    const int32_t initial = crawler->config.max_pages < 16
        ? crawler->config.max_pages : 16;

    crawler->pages = (NiyahWebPage**)calloc((size_t)initial,
                                            sizeof(NiyahWebPage*));
    if (!crawler->pages) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    crawler->page_capacity = initial;

    return NIYAH_OK;
}

NiyahStatus niyah_crawler_add_page(NiyahCrawler* crawler, NiyahWebPage* page)
{
    if (!crawler || !page) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (crawler->n_pages >= crawler->config.max_pages) {
        return NIYAH_ERR_OVERFLOW; /* budget reached; caller should stop */
    }

    if (crawler->n_pages >= crawler->page_capacity) {
        int32_t next = crawler->page_capacity ? crawler->page_capacity * 2 : 16;
        if (next > crawler->config.max_pages) {
            next = crawler->config.max_pages;
        }
        NiyahWebPage** grown = (NiyahWebPage**)realloc(
            crawler->pages, (size_t)next * sizeof(NiyahWebPage*));
        if (!grown) {
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        crawler->pages = grown;
        crawler->page_capacity = next;
    }

    crawler->pages[crawler->n_pages++] = page;
    return NIYAH_OK;
}

void niyah_crawler_free(NiyahCrawler* crawler)
{
    if (!crawler) {
        return;
    }
    for (int32_t i = 0; i < crawler->n_pages; ++i) {
        if (crawler->pages[i]) {
            niyah_webpage_free(crawler->pages[i]);
            free(crawler->pages[i]);
        }
    }
    free(crawler->pages);
    memset(crawler, 0, sizeof(*crawler));
}
