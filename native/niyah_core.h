#ifndef NIYAH_CORE_H
#define NIYAH_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *base;
    size_t size;
    size_t used;
} NiyahPool;

/* Caller owns buffer. Pool never calls malloc/free. */
void niyah_pool_init(NiyahPool *pool, void *buffer, size_t size);
void *niyah_pool_alloc(NiyahPool *pool, size_t size, size_t alignment);
void niyah_pool_reset(NiyahPool *pool);
size_t niyah_pool_remaining(const NiyahPool *pool);

bool niyah_mul_size(size_t a, size_t b, size_t *out);
bool niyah_add_size(size_t a, size_t b, size_t *out);

#ifdef __cplusplus
}
#endif

#endif
