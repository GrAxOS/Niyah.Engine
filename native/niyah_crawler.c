#include "niyah.h"

#include <stdlib.h>
#include <string.h>

<<<<<<< HEAD
/* ── Web crawler (HTTP GET via platform API) ──────────────────────────── */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <wininet.h>
#  pragma comment(lib, "wininet.lib")

static char* http_get(const char* url, size_t* out_len) {
    HINTERNET hInternet = InternetOpenA(
        "NiyahCrawler/1.0",
        INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);
    if (!hInternet) return NULL;

    HINTERNET hUrl = InternetOpenUrlA(hInternet, url, NULL, 0,
        INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE, 0);
    if (!hUrl) { InternetCloseHandle(hInternet); return NULL; }

    size_t cap  = 65536;
    size_t used = 0;
    char*  buf  = (char*)malloc(cap);
    if (!buf) { InternetCloseHandle(hUrl); InternetCloseHandle(hInternet); return NULL; }

    DWORD bytes_read = 0;
    while (InternetReadFile(hUrl, buf + used, (DWORD)(cap - used - 1), &bytes_read)
           && bytes_read > 0) {
        used += bytes_read;
        if (used >= cap - 1024) {
            cap *= 2;
            char* tmp = (char*)realloc(buf, cap);
            if (!tmp) break;
            buf = tmp;
        }
    }
    buf[used] = '\0';
    if (out_len) *out_len = used;

    InternetCloseHandle(hUrl);
    InternetCloseHandle(hInternet);
    return buf;
}

#else
/* POSIX: minimal HTTP/1.0 GET via sockets */
#  include <sys/socket.h>
#  include <netdb.h>
#  include <unistd.h>

static char* http_get(const char* url, size_t* out_len) {
    /* Parse http://host/path */
    if (strncmp(url, "http://", 7) != 0) return NULL;
    const char* host_start = url + 7;
    const char* path_start = strchr(host_start, '/');
    if (!path_start) path_start = "/";

    char host[256];
    size_t hlen = (size_t)(path_start - host_start);
    if (hlen >= sizeof(host)) return NULL;
    memcpy(host, host_start, hlen);
    host[hlen] = '\0';

    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, "80", &hints, &res) != 0) return NULL;

    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd < 0) { freeaddrinfo(res); return NULL; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        close(fd); freeaddrinfo(res); return NULL;
    }
    freeaddrinfo(res);

    char req[1024];
    snprintf(req, sizeof(req),
             "GET %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n",
             path_start, host);
    write(fd, req, strlen(req));

    size_t cap = 65536, used = 0;
    char*  buf = (char*)malloc(cap);
    if (!buf) { close(fd); return NULL; }

    ssize_t n;
    while ((n = read(fd, buf + used, cap - used - 1)) > 0) {
        used += (size_t)n;
        if (used >= cap - 1024) {
            cap *= 2;
            char* tmp = (char*)realloc(buf, cap);
            if (!tmp) break;
            buf = tmp;
        }
    }
    buf[used] = '\0';
    close(fd);

    /* Skip HTTP headers */
    char* body = strstr(buf, "\r\n\r\n");
    if (body) { body += 4; memmove(buf, body, strlen(body) + 1); used = strlen(buf); }

    if (out_len) *out_len = used;
    return buf;
}
#endif

/* ── Strip HTML tags ───────────────────────────────────────────────────── */
static char* strip_html(const char* html) {
    if (!html) return NULL;
    size_t len = strlen(html);
    char*  out = (char*)malloc(len + 1);
    if (!out) return NULL;

    size_t j      = 0;
    int    in_tag = 0;
    for (size_t i = 0; i < len; i++) {
        if (html[i] == '<') { in_tag = 1; continue; }
        if (html[i] == '>') { in_tag = 0; out[j++] = ' '; continue; }
        if (!in_tag) out[j++] = html[i];
    }
    out[j] = '\0';
    return out;
}

/* ── Public crawler API ─────────────────────────────────────────────────── */

NiyahWebPage* niyah_crawler_fetch(const char* url) {
    if (!url) return NULL;

    size_t raw_len = 0;
    char*  raw     = http_get(url, &raw_len);
    if (!raw) return NULL;

    NiyahWebPage* page = (NiyahWebPage*)calloc(1, sizeof(NiyahWebPage));
    if (!page) { free(raw); return NULL; }

    page->url     = _strdup(url);
    page->content = strip_html(raw);
    page->title   = _strdup(""); /* TODO: extract <title> */

    /* Extract title tag */
    char* ts = strstr(raw, "<title");
    if (ts) {
        ts = strchr(ts, '>');
        if (ts++) {
            char* te = strstr(ts, "</title");
            if (te) {
                size_t tlen = (size_t)(te - ts);
                free(page->title);
                page->title = (char*)malloc(tlen + 1);
                if (page->title) { memcpy(page->title, ts, tlen); page->title[tlen] = '\0'; }
            }
        }
    }

    free(raw);
    return page;
}

void niyah_crawler_free_page(NiyahWebPage* page) {
    if (!page) return;
    free(page->url);
    free(page->content);
    free(page->title);
    free(page);
=======
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
>>>>>>> origin/main
}
