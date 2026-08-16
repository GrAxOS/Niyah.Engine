#include "niyah_core.h"

#include <stdint.h>

static bool is_power_of_two(size_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

void niyah_pool_init(NiyahPool *pool, void *buffer, size_t size) {
    if (pool == NULL) {
        return;
    }

    pool->base = (uint8_t *)buffer;
    pool->size = size;
    pool->used = 0u;
}

void *niyah_pool_alloc(NiyahPool *pool, size_t size, size_t alignment) {
    if (pool == NULL || pool->base == NULL || !is_power_of_two(alignment)) {
        return NULL;
    }

    const uintptr_t address = (uintptr_t)(pool->base + pool->used);
    const size_t mask = alignment - 1u;
    const size_t misalignment = (size_t)(address & mask);
    const size_t padding = misalignment == 0u ? 0u : alignment - misalignment;

    size_t required = 0u;
    if (!niyah_add_size(padding, size, &required)) {
        return NULL;
    }
    if (required > pool->size - pool->used) {
        return NULL;
    }

    uint8_t *result = pool->base + pool->used + padding;
    pool->used += required;
    return result;
}

void niyah_pool_reset(NiyahPool *pool) {
    if (pool != NULL) {
        pool->used = 0u;
    }
}

size_t niyah_pool_remaining(const NiyahPool *pool) {
    if (pool == NULL || pool->used > pool->size) {
        return 0u;
    }
    return pool->size - pool->used;
}

bool niyah_mul_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return false;
    }
    if (a != 0u && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool niyah_add_size(size_t a, size_t b, size_t *out) {
    if (out == NULL || b > SIZE_MAX - a) {
        return false;
    }
    *out = a + b;
    return true;
}
