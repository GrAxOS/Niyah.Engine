#ifndef NIYAH_WEB_H
#define NIYAH_WEB_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *url;
    char *canonical_url;
    char *content_type;
    char *body;
    size_t body_size;
    long status_code;
} NiyahWebDocument;

typedef struct {
    size_t max_bytes;
    unsigned timeout_ms;
    bool follow_redirects;
} NiyahWebFetchOptions;

void niyah_web_document_free(NiyahWebDocument *document);
int niyah_web_fetch(const char *url,
                    const NiyahWebFetchOptions *options,
                    NiyahWebDocument *out);

char *niyah_web_canonicalize_url(const char *url);

#ifdef __cplusplus
}
#endif

#endif
