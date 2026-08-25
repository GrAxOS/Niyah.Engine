#undef NDEBUG
#include <assert.h>
#include <math.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-3f)

int main(void)
{
    /*
     * x = [3,4], mean square = (9+16)/2 = 12.5, rms = 3.53553.
     * Normalised: [0.84853, 1.13137].
     */
    float x[2] = {3.0f, 4.0f};
    niyah_rmsnorm(x, NULL, 2, 1e-5f);
    assert(CLOSE(x[0], 0.84853f));
    assert(CLOSE(x[1], 1.13137f));

    /* After RMSNorm the RMS of the output is 1. */
    float y[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    niyah_rmsnorm(y, NULL, 4, 1e-5f);
    float ms = 0.0f;
    for (int i = 0; i < 4; ++i) {
        ms += y[i] * y[i];
    }
    assert(CLOSE(sqrtf(ms / 4.0f), 1.0f));

    /* The weight vector scales per channel. */
    float w_in[2] = {3.0f, 4.0f};
    const float weight[2] = {2.0f, 0.5f};
    niyah_rmsnorm(w_in, weight, 2, 1e-5f);
    assert(CLOSE(w_in[0], 0.84853f * 2.0f));
    assert(CLOSE(w_in[1], 1.13137f * 0.5f));

    /* Out-of-place must match in-place and must not modify the input. */
    const float src[2] = {3.0f, 4.0f};
    float dst[2] = {0.0f, 0.0f};
    niyah_rmsnorm_to(dst, src, NULL, 2, 1e-5f);
    assert(CLOSE(dst[0], 0.84853f));
    assert(CLOSE(dst[1], 1.13137f));
    assert(CLOSE(src[0], 3.0f) && CLOSE(src[1], 4.0f));

    /* An all-zero vector must stay finite rather than divide by zero. */
    float zeros[3] = {0.0f, 0.0f, 0.0f};
    niyah_rmsnorm(zeros, NULL, 3, 1e-5f);
    for (int i = 0; i < 3; ++i) {
        assert(isfinite(zeros[i]));
        assert(CLOSE(zeros[i], 0.0f));
    }

    /* LayerNorm centres as well as scales: mean 0, variance 1. */
    float ln[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    niyah_layernorm(ln, NULL, NULL, 4, 1e-5f);
    float mean = 0.0f;
    for (int i = 0; i < 4; ++i) {
        mean += ln[i];
    }
    assert(CLOSE(mean / 4.0f, 0.0f));

    /* Degenerate inputs. */
    niyah_rmsnorm(NULL, NULL, 4, 1e-5f);
    niyah_rmsnorm(x, NULL, 0, 1e-5f);

    return 0;
}
