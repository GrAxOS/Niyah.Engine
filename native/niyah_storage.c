#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Was: `// Storage stubs`. */

NiyahStatus niyah_storage_read(NiyahStorage* storage, const char* path)
{
    if (!storage || !path) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(storage, 0, sizeof(*storage));

    FILE* f = fopen(path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    const long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    rewind(f);

    /* Trailing NUL so text callers can treat data as a C string. */
    unsigned char* buffer = (unsigned char*)malloc((size_t)size + 1u);
    if (!buffer) {
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    const size_t read = (size > 0)
        ? fread(buffer, 1, (size_t)size, f) : 0u;
    fclose(f);

    if (size > 0 && read != (size_t)size) {
        free(buffer);
        return NIYAH_ERR_IO;
    }

    buffer[read] = '\0';

    const size_t path_len = strlen(path) + 1u;
    storage->path = (char*)malloc(path_len);
    if (!storage->path) {
        free(buffer);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    memcpy(storage->path, path, path_len);

    storage->data = buffer;
    storage->size = read;

    return NIYAH_OK;
}

NiyahStatus niyah_storage_write(const char* path, const void* data, size_t size)
{
    if (!path || (!data && size > 0)) {
        return NIYAH_ERR_INVALID_ARG;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        return NIYAH_ERR_IO;
    }

    if (size > 0) {
        const size_t written = fwrite(data, 1, size, f);
        if (written != size) {
            fclose(f);
            return NIYAH_ERR_IO;
        }
    }

    if (fclose(f) != 0) {
        return NIYAH_ERR_IO;
    }
    return NIYAH_OK;
}

void niyah_storage_free(NiyahStorage* storage)
{
    if (!storage) {
        return;
    }
    free(storage->path);
    free(storage->data);
    memset(storage, 0, sizeof(*storage));
}
