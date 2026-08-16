#ifndef NIYAH_STORAGE_H
#define NIYAH_STORAGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t schema_version;
    const char *path;
    uint64_t opened_unix_ms;
    bool ready;
} NiyahStorage;

bool niyah_storage_open(NiyahStorage *storage,
                        const char *path,
                        uint64_t now_unix_ms);

void niyah_storage_close(NiyahStorage *storage);

bool niyah_storage_is_ready(const NiyahStorage *storage);

#ifdef __cplusplus
}
#endif

#endif
