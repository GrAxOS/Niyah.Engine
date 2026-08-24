#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "niyah.h"

int main(void)
{
    NiyahRuntimeConfig config;
    memset(&config, 0, sizeof(config));
    config.memory_size = 4096;
    config.device_id = 0;
    config.use_gpu = true;   /* must be forced off: there is no GPU backend */

    NiyahRuntime* rt = niyah_runtime_create(&config);
    assert(rt != NULL);

    /*
     * The header advertises use_gpu but no GPU path exists. Reporting true
     * would be a false capability claim, so create() must clear it.
     */
    assert(rt->config.use_gpu == false);

    assert(niyah_runtime_capacity(rt) >= 4096);
    assert(niyah_runtime_used(rt) == 0);

    /* Allocations are 64-byte aligned for SIMD-friendly access. */
    void* a = niyah_runtime_alloc(rt, 100);
    assert(a != NULL);
    assert(((uintptr_t)a % 64u) == 0);
    assert(niyah_runtime_used(rt) >= 100);

    void* b = niyah_runtime_alloc(rt, 100);
    assert(b != NULL);
    assert(b != a);
    assert(((uintptr_t)b % 64u) == 0);

    /* Distinct allocations must not overlap. */
    assert((char*)b >= (char*)a + 100);

    float* f = niyah_runtime_alloc_floats(rt, 16);
    assert(f != NULL);
    for (int i = 0; i < 16; ++i) {
        f[i] = (float)i;   /* writable */
    }
    assert(f[15] == 15.0f);

    /* Exhaustion returns NULL instead of overrunning the arena. */
    void* huge = niyah_runtime_alloc(rt, 1024u * 1024u * 1024u);
    assert(huge == NULL);

    /* A zero-byte request is not an error but must not hand back the arena. */
    const size_t used_before = niyah_runtime_used(rt);
    niyah_runtime_alloc(rt, 0);
    assert(niyah_runtime_used(rt) >= used_before);

    /* Reset rewinds the bump pointer without freeing the backing block. */
    niyah_runtime_reset(rt);
    assert(niyah_runtime_used(rt) == 0);
    assert(niyah_runtime_capacity(rt) >= 4096);

    void* after_reset = niyah_runtime_alloc(rt, 100);
    assert(after_reset == a);   /* same arena, rewound */

    niyah_runtime_destroy(rt);

    /* A NULL config uses the default arena size. */
    NiyahRuntime* def = niyah_runtime_create(NULL);
    assert(def != NULL);
    assert(niyah_runtime_capacity(def) > 0);
    niyah_runtime_destroy(def);

    /* Degenerate inputs. */
    assert(niyah_runtime_alloc(NULL, 10) == NULL);
    assert(niyah_runtime_used(NULL) == 0);
    assert(niyah_runtime_capacity(NULL) == 0);
    niyah_runtime_destroy(NULL);
    niyah_runtime_reset(NULL);

    return 0;
}
