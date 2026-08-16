#ifndef NIYAH_MEMORY_H
#define NIYAH_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *base;
    size_t capacity;
    size_t used;
} NiyahPool;

int niyah_pool_init(NiyahPool *pool, void *buffer, size_t capacity);
void *niyah_pool_alloc(NiyahPool *pool, size_t size, size_t alignment);
size_t niyah_pool_remaining(const NiyahPool *pool);
void niyah_pool_reset(NiyahPool *pool);

#ifdef __cplusplus
}
#endif

#endif
