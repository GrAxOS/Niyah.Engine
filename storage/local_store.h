#ifndef NIYAH_LOCAL_STORE_H
#define NIYAH_LOCAL_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahStore NiyahStore;

typedef enum {
    NIYAH_STORE_OK = 0,
    NIYAH_STORE_INVALID = 1,
    NIYAH_STORE_IO = 2,
    NIYAH_STORE_SCHEMA = 3,
    NIYAH_STORE_BUSY = 4
} NiyahStoreStatus;

NiyahStoreStatus niyah_store_open(const char *path, NiyahStore **out_store);
void niyah_store_close(NiyahStore *store);
NiyahStoreStatus niyah_store_init_schema(NiyahStore *store);
NiyahStoreStatus niyah_store_insert_source(
    NiyahStore *store,
    const char *id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    const char *source_kind);

#ifdef __cplusplus
}
#endif

#endif
