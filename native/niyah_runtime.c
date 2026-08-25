#include "niyah.h"

#include <stdlib.h>
#include <string.h>

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
}
