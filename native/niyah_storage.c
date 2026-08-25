#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Simple flat-file binary storage ───────────────────────────────────── */
/*
 * File layout:
 *   [uint32 magic=0x4E595148][uint32 version=1][uint32 count]
 *   For each record:
 *     [uint32 key_len][key bytes][uint32 val_len][val bytes]
 */
#define NIYAH_STORAGE_MAGIC   0x4E595148u
#define NIYAH_STORAGE_VERSION 1u

static void write_u32(FILE* f, uint32_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

static uint32_t read_u32(FILE* f) {
    uint32_t v = 0;
    fread(&v, sizeof(v), 1, f);
    return v;
}

/* Write key=value pair to storage file (append mode) */
int niyah_storage_write(NiyahStorage* s, const char* key, const void* data, size_t len) {
    if (!s || !s->path || !key || !data) return -1;

    FILE* f = fopen(s->path, "ab");
    if (!f) return -1;

    /* On first write, check/write header (file size == 0) */
    long pos = ftell(f);
    if (pos == 0) {
        fseek(f, 0, SEEK_SET);
        write_u32(f, NIYAH_STORAGE_MAGIC);
        write_u32(f, NIYAH_STORAGE_VERSION);
        write_u32(f, 0); /* count placeholder */
    }
    fseek(f, 0, SEEK_END);

    uint32_t klen = (uint32_t)strlen(key);
    uint32_t vlen = (uint32_t)len;
    write_u32(f, klen);
    fwrite(key, 1, klen, f);
    write_u32(f, vlen);
    fwrite(data, 1, vlen, f);

    fclose(f);
    return 0;
}

/* Read first value matching key; caller must free result */
void* niyah_storage_read(NiyahStorage* s, const char* key, size_t* out_len) {
    if (!s || !s->path || !key) return NULL;

    FILE* f = fopen(s->path, "rb");
    if (!f) return NULL;

    uint32_t magic   = read_u32(f);
    uint32_t version = read_u32(f);
    (void)version;
    if (magic != NIYAH_STORAGE_MAGIC) { fclose(f); return NULL; }
    read_u32(f); /* count – unused */

    size_t key_len = strlen(key);
    char   kbuf[1024];
    void*  result  = NULL;

    while (!feof(f)) {
        uint32_t klen = read_u32(f);
        if (feof(f) || klen == 0 || klen >= sizeof(kbuf)) break;

        if (fread(kbuf, 1, klen, f) != klen) break;
        kbuf[klen] = '\0';

        uint32_t vlen = read_u32(f);
        if (feof(f)) break;

        if (klen == key_len && strcmp(kbuf, key) == 0) {
            result = malloc(vlen + 1);
            if (result) {
                fread(result, 1, vlen, f);
                ((char*)result)[vlen] = '\0';
                if (out_len) *out_len = vlen;
            }
            break;
        } else {
            fseek(f, (long)vlen, SEEK_CUR);
        }
    }

    fclose(f);
    return result;
}
