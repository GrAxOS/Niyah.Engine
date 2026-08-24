#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    /*
     * Position 0 gives angle 0 for every frequency, so cos=1 and sin=0 and
     * rotation is the identity. seq=1 means only position 0 exists.
     */
    float identity[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    niyah_rope_forward(identity, 1, 4, 1);
    assert(CLOSE(identity[0], 1.0f));
    assert(CLOSE(identity[1], 2.0f));
    assert(CLOSE(identity[2], 3.0f));
    assert(CLOSE(identity[3], 4.0f));

    /*
     * Rotation preserves the length of each rotated pair. With head_dim = 4 the
     * NeoX/GGUF convention pairs channel i with channel i + head_dim/2, so the
     * pairs are (0,2) and (1,3).
     */
    float x[8];
    const float original[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(x, original, sizeof(x));

    niyah_rope_forward(x, 2, 4, 1);

    for (int p = 0; p < 2; ++p) {
        const float* row = x + p * 4;
        const float* orig = original + p * 4;

        const float pair0_before = sqrtf(orig[0] * orig[0] + orig[2] * orig[2]);
        const float pair0_after = sqrtf(row[0] * row[0] + row[2] * row[2]);
        assert(CLOSE(pair0_before, pair0_after));

        const float pair1_before = sqrtf(orig[1] * orig[1] + orig[3] * orig[3]);
        const float pair1_after = sqrtf(row[1] * row[1] + row[3] * row[3]);
        assert(CLOSE(pair1_before, pair1_after));
    }

    /* Position 1 must actually change the values. */
    assert(!CLOSE(x[4], original[4]) || !CLOSE(x[6], original[6]));

    /* pos_offset shifts which position a single row is rotated for: rotating
     * row 0 with offset 1 equals rotating position 1 in the batch call. */
    float single[4] = {5, 6, 7, 8};
    niyah_rope_forward_ex(single, 1, 4, 1, 1, 10000.0f);
    assert(CLOSE(single[0], x[4]));
    assert(CLOSE(single[1], x[5]));
    assert(CLOSE(single[2], x[6]));
    assert(CLOSE(single[3], x[7]));

    /* Shape mismatch (dim not divisible by n_head) must be refused, not
     * allowed to walk off the end of the buffer. */
    float bad[5] = {1, 2, 3, 4, 5};
    niyah_rope_forward(bad, 1, 5, 2);
    assert(CLOSE(bad[0], 1.0f) && CLOSE(bad[4], 5.0f));

    /* Degenerate inputs. */
    niyah_rope_forward(NULL, 1, 4, 1);
    niyah_rope_forward(x, 0, 4, 1);

    return 0;
}
