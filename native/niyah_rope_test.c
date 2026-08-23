#include "niyah_rope.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_rope_f32_position_one(void) {
    /* dim=4, theta_base=10000, position=1 (pos_offset=1, single row) */
    float x[4] = {1.0f, 0.0f, 1.0f, 0.0f};

    assert(niyah_rope_f32(x, 1u, 4u, 1u, 10000.0f));
    assert(nearly_equal(x[0], 0.54030231f));
    assert(nearly_equal(x[1], 0.84147098f));
    assert(nearly_equal(x[2], 0.99995000f));
    assert(nearly_equal(x[3], 0.00999983f));
}

static void test_rope_f32_position_zero_is_identity(void) {
    /* Position 0 must leave the vector unchanged (angle = 0 for every pair). */
    float x[4] = {1.0f, 0.0f, 1.0f, 0.0f};

    assert(niyah_rope_f32(x, 1u, 4u, 0u, 10000.0f));
    assert(nearly_equal(x[0], 1.0f));
    assert(nearly_equal(x[1], 0.0f));
    assert(nearly_equal(x[2], 1.0f));
    assert(nearly_equal(x[3], 0.0f));
}

static void test_rope_f32_rejects_invalid_input(void) {
    float x[3] = {1.0f, 0.0f, 1.0f};

    assert(!niyah_rope_f32(NULL, 1u, 4u, 0u, 10000.0f));
    assert(!niyah_rope_f32(x, 0u, 4u, 0u, 10000.0f));
    assert(!niyah_rope_f32(x, 1u, 3u, 0u, 10000.0f)); /* odd dim rejected */
}

int main(void) {
    test_rope_f32_position_one();
    test_rope_f32_position_zero_is_identity();
    test_rope_f32_rejects_invalid_input();
    return 0;
}
