#include "niyah_storage.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    const char *path = "niyah-storage-test.tmp";
    remove(path);

    NiyahStorage storage;
    memset(&storage, 0, sizeof(storage));
    assert(niyah_storage_open(&storage, path, 1234u));
    assert(niyah_storage_is_ready(&storage));
    assert(storage.opened_unix_ms == 1234u);

    niyah_storage_close(&storage);
    assert(!niyah_storage_is_ready(&storage));
    assert(remove(path) == 0);
    return 0;
}
