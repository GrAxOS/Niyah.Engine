#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "niyah.h"

static NiyahWebPage* make_page(const char* url)
{
    NiyahWebPage* p = (NiyahWebPage*)calloc(1, sizeof(NiyahWebPage));
    assert(p != NULL);
    p->url = (char*)malloc(strlen(url) + 1u);
    assert(p->url != NULL);
    strcpy(p->url, url);
    p->content = NULL;
    p->title = NULL;
    p->timestamp = 0;
    return p;
}

int main(void)
{
    NiyahCrawler crawler;
    memset(&crawler, 0, sizeof(crawler));

    /* A NULL config must produce documented defaults, not zeros -- a
     * max_pages of 0 would make the crawler silently useless. */
    assert(niyah_crawler_init(&crawler, NULL) == NIYAH_OK);
    assert(crawler.config.max_depth == 2);
    assert(crawler.config.max_pages == 64);
    assert(crawler.config.follow_links == true);
    assert(crawler.n_pages == 0);
    niyah_crawler_free(&crawler);

    /* Explicit config is respected. */
    NiyahCrawlerConfig config;
    config.max_depth = 1;
    config.max_pages = 3;
    config.follow_links = false;

    memset(&crawler, 0, sizeof(crawler));
    assert(niyah_crawler_init(&crawler, &config) == NIYAH_OK);
    assert(crawler.config.max_pages == 3);
    assert(crawler.config.follow_links == false);

    /* Pages accumulate up to the budget. */
    assert(niyah_crawler_add_page(&crawler, make_page("https://a")) == NIYAH_OK);
    assert(crawler.n_pages == 1);
    assert(niyah_crawler_add_page(&crawler, make_page("https://b")) == NIYAH_OK);
    assert(niyah_crawler_add_page(&crawler, make_page("https://c")) == NIYAH_OK);
    assert(crawler.n_pages == 3);

    /* The budget is enforced: the 4th page is refused rather than silently
     * accepted or written out of bounds. */
    NiyahWebPage* overflow = make_page("https://d");
    assert(niyah_crawler_add_page(&crawler, overflow) == NIYAH_ERR_OVERFLOW);
    assert(crawler.n_pages == 3);

    /* Refused pages are not owned by the crawler, so the caller frees them. */
    niyah_webpage_free(overflow);

    /* Stored pages are reachable and intact. */
    assert(crawler.pages != NULL);
    assert(crawler.pages[0] != NULL);
    assert(strcmp(crawler.pages[0]->url, "https://a") == 0);
    assert(strcmp(crawler.pages[2]->url, "https://c") == 0);

    /* free releases the owned pages and resets the counter. */
    niyah_crawler_free(&crawler);
    assert(crawler.n_pages == 0);
    assert(crawler.pages == NULL);

    /* Degenerate inputs. */
    assert(niyah_crawler_init(NULL, &config) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_crawler_add_page(NULL, NULL) == NIYAH_ERR_INVALID_ARG);
    niyah_crawler_free(NULL);
    niyah_webpage_free(NULL);

    return 0;
}
