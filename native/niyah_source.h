#ifndef NIYAH_SOURCE_H
#define NIYAH_SOURCE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_SOURCE_UNKNOWN = 0,
    NIYAH_SOURCE_WEB = 1,
    NIYAH_SOURCE_LOCAL_FILE = 2,
    NIYAH_SOURCE_DATABASE = 3,
    NIYAH_SOURCE_CODE = 4
} NiyahSourceKind;

typedef struct {
    NiyahSourceKind kind;
    const char *uri;
    const char *title;
    const uint8_t *content_hash; /* optional 32-byte SHA-256 */
    bool verified;
} NiyahSource;

bool niyah_source_is_usable(const NiyahSource *source);

#ifdef __cplusplus
}
#endif

#endif
