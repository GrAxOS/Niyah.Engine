#ifndef NIYAH_URL_H
#define NIYAH_URL_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Normalize absolute HTTP(S) URLs for crawl deduplication.
 * Returns false for unsupported/invalid schemes or output overflow. */
bool niyah_url_canonicalize(const char *input, char *output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif
