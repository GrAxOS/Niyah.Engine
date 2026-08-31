#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

/*
 * Scalar reference implementation of niyah_matvec used for SIMD validation.
 * Intentionally kept simple (four-way unroll with independent accumulators)
 * so it is easy to audit for correctness.
 */
static void matvec_scalar_ref(float* out,
                               const float* w,
                               const float* x,
                               int32_t n_out, int32_t n_in)
{
    for (int32_t i = 0; i < n_out; ++i) {
        const float* row = w + (size_t)i * (size_t)n_in;
        float s = 0.0f;
        for (int32_t p = 0; p < n_in; ++p) {
            s += row[p] * x[p];
        }
        out[i] = s;
    }
}

int main(void)
{
    /* a is 2x3, b is 3x2, so out is 2x2. Worked by hand:
     *   a = [[1,2,3],
     *        [4,5,6]]
     *   b = [[7, 8],
     *        [9,10],
     *        [11,12]]
     *   out[0][0] = 1*7 + 2*9  + 3*11 = 58
     *   out[0][1] = 1*8 + 2*10 + 3*12 = 64
     *   out[1][0] = 4*7 + 5*9  + 6*11 = 139
     *   out[1][1] = 4*8 + 5*10 + 6*12 = 154
     */
    const float a[6] = {1, 2, 3, 4, 5, 6};
    const float b[6] = {7, 8, 9, 10, 11, 12};
    float out[4] = {0};

    niyah_matmul(out, a, b, 2, 3, 2);
    assert(CLOSE(out[0], 58.0f));
    assert(CLOSE(out[1], 64.0f));
    assert(CLOSE(out[2], 139.0f));
    assert(CLOSE(out[3], 154.0f));

    /* Multiplying by the identity must be a no-op. */
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float ident_out[6] = {0};
    niyah_matmul(ident_out, a, identity, 2, 3, 3);
    for (int i = 0; i < 6; ++i) {
        assert(CLOSE(ident_out[i], a[i]));
    }

    /* matmul_bt takes b already transposed; results must agree with matmul.
     * bt = b^T = [[7,9,11],[8,10,12]] stored as [n][k]. */
    const float bt[6] = {7, 9, 11, 8, 10, 12};
    float out_bt[4] = {0};
    niyah_matmul_bt(out_bt, a, bt, 2, 3, 2);
    for (int i = 0; i < 4; ++i) {
        assert(CLOSE(out_bt[i], out[i]));
    }

    /* matvec: w is [2][3], x is [3]. */
    const float x[3] = {1, 1, 1};
    float mv[2] = {0};
    niyah_matvec(mv, a, x, 2, 3);
    assert(CLOSE(mv[0], 6.0f));    /* 1+2+3 */
    assert(CLOSE(mv[1], 15.0f));   /* 4+5+6 */

    /* Elementwise helpers. */
    float acc[3] = {1, 2, 3};
    const float add[3] = {10, 20, 30};
    niyah_add_inplace(acc, add, 3);
    assert(CLOSE(acc[0], 11.0f) && CLOSE(acc[1], 22.0f) && CLOSE(acc[2], 33.0f));

    niyah_scale_inplace(acc, 0.5f, 3);
    assert(CLOSE(acc[0], 5.5f) && CLOSE(acc[1], 11.0f) && CLOSE(acc[2], 16.5f));

    /* NULL and non-positive dimensions must not crash. */
    niyah_matmul(NULL, a, b, 2, 3, 2);
    niyah_matmul(out, a, b, 0, 3, 2);
    niyah_matvec(mv, a, x, -1, 3);

    /* ------------------------------------------------------------------
     * SIMD path validation: niyah_matvec must agree with the scalar
     * reference to within a tight tolerance across a range of dimension
     * lengths, including non-multiple-of-8 and non-multiple-of-4 cases
     * (to exercise the scalar tail in both AVX2 and NEON paths).
     * ------------------------------------------------------------------ */
    {
        /* Test several dimension pairs.  Sizes chosen to cover:
         *   - dim < 4 (no SIMD lanes at all)
         *   - 4 <= dim < 8 (NEON fills, AVX2 scalar-only)
         *   - dim = 8 (exactly one AVX2 round)
         *   - dim with tail (e.g. 13: 8+4+1 for AVX2, 12+1 for NEON)
         *   - large dim (256) for full-pipeline exercise
         */
        const int32_t test_dims[][2] = {
            {3,  3},
            {4,  7},
            {5,  8},
            {8,  9},
            {7,  13},
            {16, 16},
            {17, 33},
            {64, 256},
        };
        const size_t n_cases =
            sizeof(test_dims) / sizeof(test_dims[0]);

        /* Use a simple LCG seeded at a fixed value for reproducibility. */
        uint32_t rng = 0xDEADBEEFu;

        for (size_t tc = 0; tc < n_cases; ++tc) {
            const int32_t rows = test_dims[tc][0];
            const int32_t cols = test_dims[tc][1];

            float* W    = (float*)malloc((size_t)rows * (size_t)cols * sizeof(float));
            float* vec  = (float*)malloc((size_t)cols * sizeof(float));
            float* got  = (float*)malloc((size_t)rows * sizeof(float));
            float* want = (float*)malloc((size_t)rows * sizeof(float));
            assert(W && vec && got && want);

            for (int32_t i = 0; i < rows * cols; ++i) {
                rng = rng * 1664525u + 1013904223u;
                W[i] = (float)(int32_t)(rng >> 16) * (1.0f / 32768.0f);
            }
            for (int32_t i = 0; i < cols; ++i) {
                rng = rng * 1664525u + 1013904223u;
                vec[i] = (float)(int32_t)(rng >> 16) * (1.0f / 32768.0f);
            }

            niyah_matvec(got,  W, vec, rows, cols);
            matvec_scalar_ref(want, W, vec, rows, cols);

            /* Tolerance: 1e-3 relative to the maximum expected magnitude.
             * Floating-point reassociation in SIMD can shift results slightly. */
            for (int32_t i = 0; i < rows; ++i) {
                const float diff = fabsf(got[i] - want[i]);
                const float mag  = fabsf(want[i]) + 1.0f;
                assert(diff / mag < 1e-3f);
            }

            free(W);
            free(vec);
            free(got);
            free(want);
        }
    }

    return 0;
}
