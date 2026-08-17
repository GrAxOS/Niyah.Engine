#ifndef NIYAH_URL_H
#define NIYAH_URL_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Normalize absolute HTTP(S) URLs for crawl deduplication. */
bool niyah_url_canonicalize(const char *input, char *output, size_t output_size);

/* Resolve an absolute HTTP(S) base URL and an RFC-style link reference.
 * Fragments are removed from the resulting crawl identity. */
bool niyah_url_resolve(const char *base,
                       const char *reference,
                       char *output,
                       size_t output_size);

#ifdef __cplusplus
}
#endif

#endif
