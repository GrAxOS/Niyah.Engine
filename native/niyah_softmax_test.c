#include "niyah_softmax.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_softmax_f32_basic(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};

    assert(niyah_softmax_f32(x, 1u, 3u));
    assert(nearly_equal(x[0], 0.09003057f));
    assert(nearly_equal(x[1], 0.24472847f));
    assert(nearly_equal(x[2], 0.66524096f));
    assert(nearly_equal(x[0] + x[1] + x[2], 1.0f));
}

static void test_softmax_f32_uniform_row(void) {
    float x[3] = {5.0f, 5.0f, 5.0f};

    assert(niyah_softmax_f32(x, 1u, 3u));
    assert(nearly_equal(x[0], 1.0f / 3.0f));
    assert(nearly_equal(x[1], 1.0f / 3.0f));
    assert(nearly_equal(x[2], 1.0f / 3.0f));
}

static void test_softmax_f32_extreme_values_no_overflow(void) {
    /* Large logit gap must not produce inf/nan (max-subtraction stability). */
    float x[2] = {-1000.0f, 1000.0f};

    assert(niyah_softmax_f32(x, 1u, 2u));
    assert(nearly_equal(x[0], 0.0f));
    assert(nearly_equal(x[1], 1.0f));
}

static void test_softmax_f32_rejects_invalid_input(void) {
    float x[1] = {1.0f};

    assert(!niyah_softmax_f32(NULL, 1u, 1u));
    assert(!niyah_softmax_f32(x, 0u, 1u));
    assert(!niyah_softmax_f32(x, 1u, 0u));
}

int main(void) {
    test_softmax_f32_basic();
    test_softmax_f32_uniform_row();
    test_softmax_f32_extreme_values_no_overflow();
    test_softmax_f32_rejects_invalid_input();
    return 0;
}
