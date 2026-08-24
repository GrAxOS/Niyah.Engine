#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "niyah.h"

int main(void)
{
    const char* path = "niyah_storage_test_file.bin";
    const char* payload = "niyah storage round trip";
    const size_t len = strlen(payload);

    /* Write then read back. */
    assert(niyah_storage_write(path, payload, len) == NIYAH_OK);

    NiyahStorage s;
    memset(&s, 0, sizeof(s));
    assert(niyah_storage_read(&s, path) == NIYAH_OK);
    assert(s.size == len);
    assert(s.data != NULL);
    assert(memcmp(s.data, payload, len) == 0);
    assert(s.path != NULL && strcmp(s.path, path) == 0);

    /* The reader appends a NUL so the buffer is safe to treat as a string,
     * without counting it in size. */
    assert(((const char*)s.data)[len] == '\0');
    assert(strcmp((const char*)s.data, payload) == 0);

    niyah_storage_free(&s);
    assert(s.data == NULL);
    assert(s.size == 0);

    /* Binary payload containing embedded NULs must survive intact. */
    const unsigned char binary[6] = {0x00, 0x01, 0x00, 0xFF, 0x00, 0x7F};
    assert(niyah_storage_write(path, binary, sizeof(binary)) == NIYAH_OK);

    NiyahStorage bin;
    memset(&bin, 0, sizeof(bin));
    assert(niyah_storage_read(&bin, path) == NIYAH_OK);
    assert(bin.size == sizeof(binary));
    assert(memcmp(bin.data, binary, sizeof(binary)) == 0);
    niyah_storage_free(&bin);

    /* Zero-length write then read yields an empty but valid buffer. */
    assert(niyah_storage_write(path, "", 0) == NIYAH_OK);
    NiyahStorage zero;
    memset(&zero, 0, sizeof(zero));
    assert(niyah_storage_read(&zero, path) == NIYAH_OK);
    assert(zero.size == 0);
    niyah_storage_free(&zero);

    /* Missing file is an IO error. */
    NiyahStorage missing;
    memset(&missing, 0, sizeof(missing));
    assert(niyah_storage_read(&missing, "no_such_niyah_file.bin")
           == NIYAH_ERR_IO);
    assert(missing.data == NULL);

    /* Degenerate inputs. */
    assert(niyah_storage_read(NULL, path) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_storage_read(&missing, NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_storage_write(NULL, "x", 1) == NIYAH_ERR_INVALID_ARG);
    niyah_storage_free(NULL);

    remove(path);
    return 0;
}
