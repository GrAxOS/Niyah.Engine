#include "niyah_storage.h"

#include <stdio.h>
#include <string.h>

#define NIYAH_STORAGE_SCHEMA_VERSION 1u



bool niyah_storage_open(NiyahStorage *storage,
                        const char *path,
                        uint64_t now_unix_ms) {
    if (!storage || !path || path[0] == '\0')
        return false;

    memset(storage, 0, sizeof(*storage));
    storage->schema_version = NIYAH_STORAGE_SCHEMA_VERSION;
    storage->path = path;
    storage->opened_unix_ms = now_unix_ms;

    FILE *fp = fopen(path, "ab+");
    if (!fp)
        return false;

    fclose(fp);
    storage->ready = true;
    return true;
}

void niyah_storage_close(NiyahStorage *storage) {
    if (!storage)
        return;
    storage->ready = false;
}

bool niyah_storage_is_ready(const NiyahStorage *storage) {
    return storage && storage->ready &&
           storage->schema_version == NIYAH_STORAGE_SCHEMA_VERSION;
}

