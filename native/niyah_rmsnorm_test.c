#include "niyah_rmsnorm.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_rmsnorm_f32_basic(void) {
    /* x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-5
     * mean(x^2) = 7.5 -> inv_rms ~= 0.36514813 */
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {0};

    assert(niyah_rmsnorm_f32(x, weight, out, 1u, 4u, 1e-5f));
    assert(nearly_equal(out[0], 0.36514813f));
    assert(nearly_equal(out[1], 0.73029626f));
    assert(nearly_equal(out[2], 1.09544438f));
    assert(nearly_equal(out[3], 1.46059251f));
}

static void test_rmsnorm_f32_applies_gain(void) {
    /* Same input, but weight scales channel 0 by 2 and zeroes channel 3. */
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[4] = {2.0f, 1.0f, 1.0f, 0.0f};
    float out[4] = {0};

    assert(niyah_rmsnorm_f32(x, weight, out, 1u, 4u, 1e-5f));
    assert(nearly_equal(out[0], 0.73029626f));
    assert(nearly_equal(out[3], 0.0f));
}

static void test_rmsnorm_f32_rejects_invalid_input(void) {
    float out[1] = {0};
    const float w[1] = {1.0f};

    assert(!niyah_rmsnorm_f32(NULL, w, out, 1u, 1u, 1e-5f));
    assert(!niyah_rmsnorm_f32((const float *)out, w, out, 0u, 1u, 1e-5f));
    assert(!niyah_rmsnorm_f32((const float *)out, NULL, out, 1u, 1u, 1e-5f));
}

int main(void) {
    test_rmsnorm_f32_basic();
    test_rmsnorm_f32_applies_gain();
    test_rmsnorm_f32_rejects_invalid_input();
    return 0;
}
