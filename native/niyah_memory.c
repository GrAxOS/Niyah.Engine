#include "niyah_memory.h"

#include <stdint.h>

static int is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

int niyah_pool_init(NiyahPool *pool, void *buffer, size_t capacity) {
    if (pool == NULL || buffer == NULL || capacity == 0) {
        return -1;
    }

    pool->base = (uint8_t *)buffer;
    pool->capacity = capacity;
    pool->used = 0;
    return 0;
}

void *niyah_pool_alloc(NiyahPool *pool, size_t size, size_t alignment) {
    if (pool == NULL || pool->base == NULL || size == 0 || !is_power_of_two(alignment)) {
        return NULL;
    }

    uintptr_t current = (uintptr_t)pool->base + pool->used;
    size_t mask = alignment - 1;
    size_t padding = (size_t)(-(uintptr_t)current) & mask;

    if (padding > pool->capacity - pool->used) {
        return NULL;
    }

    size_t remaining = pool->capacity - pool->used - padding;
    if (size > remaining) {
        return NULL;
    }

    uint8_t *ptr = pool->base + pool->used + padding;
    pool->used += padding + size;
    return ptr;
}

size_t niyah_pool_remaining(const NiyahPool *pool) {
    if (pool == NULL || pool->used > pool->capacity) {
        return 0;
    }
    return pool->capacity - pool->used;
}

void niyah_pool_reset(NiyahPool *pool) {
    if (pool != NULL) {
        pool->used = 0;
    }
}
