#include "niyah_matmul.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

static void test_matmul_f32_basic(void) {
    /* A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> C = [[19, 22], [43, 50]] */
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {0};

    assert(niyah_matmul_f32(a, b, c, 2u, 2u, 2u));
    assert(nearly_equal(c[0], 19.0f));
    assert(nearly_equal(c[1], 22.0f));
    assert(nearly_equal(c[2], 43.0f));
    assert(nearly_equal(c[3], 50.0f));
}

static void test_matmul_f32_bt_matches_transposed_reference(void) {
    /* A = [[1, 2, 3]] (1x3), W stored as [N x K] = [[1, 0, 0], [0, 1, 0]] (2x3)
     * -> C = A * W^T = [[1, 2]] */
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float w[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float c[2] = {0};

    assert(niyah_matmul_f32_bt(a, w, c, 1u, 3u, 2u));
    assert(nearly_equal(c[0], 1.0f));
    assert(nearly_equal(c[1], 2.0f));
}

static void test_matmul_f32_rejects_invalid_input(void) {
    float c[1] = {0};

    assert(!niyah_matmul_f32(NULL, NULL, c, 1u, 1u, 1u));
    assert(!niyah_matmul_f32((const float *)c, (const float *)c, c, 0u, 1u, 1u));
    assert(!niyah_matmul_f32_bt(NULL, NULL, c, 1u, 1u, 1u));
}

int main(void) {
    test_matmul_f32_basic();
    test_matmul_f32_bt_matches_transposed_reference();
    test_matmul_f32_rejects_invalid_input();
    return 0;
}
