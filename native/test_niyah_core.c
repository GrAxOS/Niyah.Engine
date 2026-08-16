#include "niyah_core.h"

#include <assert.h>
#include <stdint.h>

static void test_size_helpers(void) {
    size_t out = 0u;
    assert(niyah_mul_size(4u, 8u, &out) && out == 32u);
    assert(niyah_add_size(20u, 12u, &out) && out == 32u);
    assert(!niyah_mul_size(SIZE_MAX, 2u, &out));
    assert(!niyah_add_size(SIZE_MAX, 1u, &out));
}

static void test_pool(void) {
    _Alignas(64) uint8_t storage[256];
    NiyahPool pool;
    niyah_pool_init(&pool, storage, sizeof(storage));

    void *a = niyah_pool_alloc(&pool, 16u, 16u);
    assert(a != NULL);
    assert(((uintptr_t)a % 16u) == 0u);

    void *bad_alignment = niyah_pool_alloc(&pool, 8u, 24u);
    assert(bad_alignment == NULL);

    void *b = niyah_pool_alloc(&pool, 64u, 64u);
    assert(b != NULL);
    assert(((uintptr_t)b % 64u) == 0u);

    assert(niyah_pool_remaining(&pool) > 0u);
    niyah_pool_reset(&pool);
    assert(pool.used == 0u);
    assert(niyah_pool_remaining(&pool) == sizeof(storage));
}

int main(void) {
    test_size_helpers();
    test_pool();
    return 0;
}
