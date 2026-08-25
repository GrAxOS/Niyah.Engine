#include "niyah.h"

#include <stdlib.h>
#include <string.h>

<<<<<<< HEAD
/* ── Runtime memory pool ───────────────────────────────────────────────── */

NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* cfg) {
    NiyahRuntime* rt = (NiyahRuntime*)calloc(1, sizeof(NiyahRuntime));
    if (!rt) return NULL;
    if (cfg) rt->config = *cfg;

    if (rt->config.memory_size > 0) {
        rt->config.memory_pool = malloc(rt->config.memory_size);
        if (!rt->config.memory_pool) {
            free(rt);
            return NULL;
        }
    }
    rt->context = NULL;
    return rt;
}

void niyah_runtime_destroy(NiyahRuntime* rt) {
    if (!rt) return;
    free(rt->config.memory_pool);
    free(rt);
}

/* Simple bump allocator within the memory pool */
void* niyah_runtime_alloc(NiyahRuntime* rt, size_t size) {
    if (!rt) return malloc(size);
    if (!rt->config.memory_pool) return malloc(size);

    /* Use context pointer as bump offset */
    size_t used = rt->context ? (size_t)((char*)rt->context -
                                          (char*)rt->config.memory_pool) : 0;
    size_t aligned = (size + 15) & ~(size_t)15;
    if (used + aligned > rt->config.memory_size) return NULL;

    void* ptr   = (char*)rt->config.memory_pool + used;
    rt->context = (char*)rt->config.memory_pool + used + aligned;
    return ptr;
}

void niyah_runtime_reset(NiyahRuntime* rt) {
    if (!rt) return;
    rt->context = NULL;
=======
/*
 * Was a stub: niyah_runtime.h declared create/destroy but nothing defined them.
 *
 * Bump allocator over a single pool. Inference does thousands of short-lived
 * tensor allocations per token; a reset-per-token arena avoids malloc churn
 * and makes peak memory a hard, observable number.
 */

#define NIYAH_ARENA_ALIGN 64u
#define NIYAH_ARENA_DEFAULT (64u * 1024u * 1024u)

typedef struct {
    unsigned char* base;
    size_t         capacity;
    size_t         used;
    bool           owns_memory;
} NiyahArena;

NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* config)
{
    NiyahRuntime* runtime = (NiyahRuntime*)calloc(1, sizeof(NiyahRuntime));
    if (!runtime) {
        return NULL;
    }

    if (config) {
        runtime->config = *config;
    } else {
        runtime->config.memory_pool = NULL;
        runtime->config.memory_size = NIYAH_ARENA_DEFAULT;
        runtime->config.device_id = 0;
        runtime->config.use_gpu = false;
    }

    if (runtime->config.memory_size == 0) {
        runtime->config.memory_size = NIYAH_ARENA_DEFAULT;
    }

    /* GPU execution is not implemented; say so instead of pretending. */
    if (runtime->config.use_gpu) {
        runtime->config.use_gpu = false;
    }

    NiyahArena* arena = (NiyahArena*)calloc(1, sizeof(NiyahArena));
    if (!arena) {
        free(runtime);
        return NULL;
    }

    if (runtime->config.memory_pool) {
        arena->base = (unsigned char*)runtime->config.memory_pool;
        arena->owns_memory = false;
    } else {
        arena->base = (unsigned char*)malloc(runtime->config.memory_size);
        if (!arena->base) {
            free(arena);
            free(runtime);
            return NULL;
        }
        arena->owns_memory = true;
        runtime->config.memory_pool = arena->base;
    }

    arena->capacity = runtime->config.memory_size;
    arena->used = 0;
    runtime->context = arena;

    return runtime;
}

void niyah_runtime_destroy(NiyahRuntime* runtime)
{
    if (!runtime) {
        return;
    }
    NiyahArena* arena = (NiyahArena*)runtime->context;
    if (arena) {
        if (arena->owns_memory) {
            free(arena->base);
        }
        free(arena);
    }
    free(runtime);
}

void* niyah_runtime_alloc(NiyahRuntime* runtime, size_t bytes)
{
    if (!runtime || !runtime->context || bytes == 0) {
        return NULL;
    }

    NiyahArena* arena = (NiyahArena*)runtime->context;

    const size_t misalign = arena->used % NIYAH_ARENA_ALIGN;
    const size_t padding = misalign ? (NIYAH_ARENA_ALIGN - misalign) : 0u;

    if (padding > arena->capacity - arena->used) {
        return NULL;
    }
    const size_t offset = arena->used + padding;
    if (bytes > arena->capacity - offset) {
        return NULL; /* pool exhausted */
    }

    void* ptr = arena->base + offset;
    arena->used = offset + bytes;
    return ptr;
}

float* niyah_runtime_alloc_floats(NiyahRuntime* runtime, size_t count)
{
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    float* ptr = (float*)niyah_runtime_alloc(runtime, count * sizeof(float));
    if (ptr) {
        memset(ptr, 0, count * sizeof(float));
    }
    return ptr;
}

void niyah_runtime_reset(NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return;
    }
    ((NiyahArena*)runtime->context)->used = 0;
}

size_t niyah_runtime_used(const NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return 0;
    }
    return ((const NiyahArena*)runtime->context)->used;
}

size_t niyah_runtime_capacity(const NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return 0;
    }
    return ((const NiyahArena*)runtime->context)->capacity;
>>>>>>> origin/main
}
