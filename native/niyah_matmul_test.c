#undef NDEBUG
#include <assert.h>
#include <math.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

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

    return 0;
}
