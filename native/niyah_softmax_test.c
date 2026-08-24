#undef NDEBUG
#include <assert.h>
#include <math.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    /* Uniform input yields a uniform distribution. */
    float uniform[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    niyah_softmax(uniform, 4);
    for (int i = 0; i < 4; ++i) {
        assert(CLOSE(uniform[i], 0.25f));
    }

    /* Probabilities sum to 1 and preserve ordering. */
    float x[3] = {1.0f, 2.0f, 3.0f};
    niyah_softmax(x, 3);
    assert(CLOSE(x[0] + x[1] + x[2], 1.0f));
    assert(x[0] < x[1] && x[1] < x[2]);

    /* Known values for [1,2,3]. */
    assert(CLOSE(x[0], 0.09003f));
    assert(CLOSE(x[1], 0.24473f));
    assert(CLOSE(x[2], 0.66524f));

    /*
     * Numerical stability. A naive expf() without max-subtraction overflows to
     * inf here and produces NaN. This is the regression the max-subtraction
     * guards against.
     */
    float big[3] = {1000.0f, 1000.0f, 1000.0f};
    niyah_softmax(big, 3);
    for (int i = 0; i < 3; ++i) {
        assert(isfinite(big[i]));
        assert(CLOSE(big[i], 1.0f / 3.0f));
    }

    /* Very negative values must not produce NaN either. */
    float small[2] = {-1000.0f, -1000.0f};
    niyah_softmax(small, 2);
    assert(isfinite(small[0]) && CLOSE(small[0], 0.5f));

    /* Temperature: higher temperature flattens the distribution. */
    float hot[3] = {1.0f, 2.0f, 3.0f};
    niyah_softmax_temperature(hot, 3, 10.0f);
    assert(CLOSE(hot[0] + hot[1] + hot[2], 1.0f));
    assert(hot[2] - hot[0] < 0.66524f - 0.09003f);

    /* log_softmax equals log of softmax. */
    float logits[3] = {1.0f, 2.0f, 3.0f};
    niyah_log_softmax(logits, 3);
    assert(CLOSE(expf(logits[0]), 0.09003f));
    assert(CLOSE(expf(logits[1]), 0.24473f));
    assert(CLOSE(expf(logits[2]), 0.66524f));

    /* argmax. */
    const float pick[5] = {-1.0f, 3.5f, 0.0f, 3.6f, 2.0f};
    assert(niyah_argmax(pick, 5) == 3);
    assert(niyah_argmax(NULL, 5) == -1);
    assert(niyah_argmax(pick, 0) == -1);

    /* Degenerate inputs must not crash. */
    niyah_softmax(NULL, 4);
    niyah_softmax(x, 0);

    return 0;
}
