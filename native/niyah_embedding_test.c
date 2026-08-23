#include "niyah_embedding.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_embedding_lookup_basic(void) {
    const float table[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    const uint32_t ids[3] = {2u, 0u, 1u};
    float out[6] = {0};

    assert(niyah_embedding_lookup_f32(table, 3u, 2u, ids, 3u, out));
    assert(nearly_equal(out[0], 5.0f));
    assert(nearly_equal(out[1], 6.0f));
    assert(nearly_equal(out[2], 1.0f));
    assert(nearly_equal(out[3], 2.0f));
    assert(nearly_equal(out[4], 3.0f));
    assert(nearly_equal(out[5], 4.0f));
}

static void test_embedding_lookup_rejects_out_of_range(void) {
    const float table[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const uint32_t bad_ids[1] = {5u};
    float out[2] = {0};

    assert(!niyah_embedding_lookup_f32(table, 2u, 2u, bad_ids, 1u, out));
}

static void test_lm_head_basic(void) {
    const float hidden[2] = {1.0f, 2.0f};
    const float w_head[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float logits[3] = {0};

    assert(niyah_lm_head_f32(hidden, w_head, logits, 1u, 2u, 3u));
    assert(nearly_equal(logits[0], 1.0f));
    assert(nearly_equal(logits[1], 2.0f));
    assert(nearly_equal(logits[2], 3.0f));
}

static void test_rejects_invalid_input(void) {
    float out[1] = {0};
    const uint32_t ids[1] = {0u};

    assert(!niyah_embedding_lookup_f32(NULL, 1u, 1u, ids, 1u, out));
    assert(!niyah_lm_head_f32(NULL, out, out, 1u, 1u, 1u));
}

int main(void) {
    test_embedding_lookup_basic();
    test_embedding_lookup_rejects_out_of_range();
    test_lm_head_basic();
    test_rejects_invalid_input();
    return 0;
}
