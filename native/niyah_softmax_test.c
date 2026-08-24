#undef NDEBUG
#include <assert.h>
#include <math.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* niyah_softmax                                                       */
    /* ------------------------------------------------------------------ */

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
     * Numerical stability (overflow side).
     * A naive expf() without max-subtraction overflows to inf here and
     * produces NaN. Max-subtraction keeps all exp inputs at 0.
     */
    float big[3] = {1000.0f, 1000.0f, 1000.0f};
    niyah_softmax(big, 3);
    for (int i = 0; i < 3; ++i) {
        assert(isfinite(big[i]));
        assert(CLOSE(big[i], 1.0f / 3.0f));
    }

    /*
     * Symmetric underflow: both elements equal, so expf(0)=1 for both.
     * This always works even without a sum guard — it is NOT a stress test.
     */
    float sym[2] = {-1000.0f, -1000.0f};
    niyah_softmax(sym, 2);
    assert(isfinite(sym[0]) && CLOSE(sym[0], 0.5f));

    /*
     * Asymmetric underflow — the real regression.
     * x[1]-max = -1000 - 0 = -1000 => expf(-1000) = 0 (underflow).
     * sum = expf(0) + 0 = 1.0f, which is positive, so the guard does NOT
     * fire and the result must be {1.0f, 0.0f}, not the uniform fallback.
     * A bug that replaces any underflow with uniform would fail here.
     */
    float asym[2] = {0.0f, -1000.0f};
    niyah_softmax(asym, 2);
    assert(isfinite(asym[0]) && CLOSE(asym[0], 1.0f));
    assert(isfinite(asym[1]) && CLOSE(asym[1], 0.0f));

    /* ------------------------------------------------------------------ */
    /* niyah_softmax_temperature                                           */
    /* ------------------------------------------------------------------ */

    /* Higher temperature flattens the distribution. */
    float hot[3] = {1.0f, 2.0f, 3.0f};
    niyah_softmax_temperature(hot, 3, 10.0f);
    assert(CLOSE(hot[0] + hot[1] + hot[2], 1.0f));
    assert(hot[2] - hot[0] < 0.66524f - 0.09003f);

    /*
     * T = 0  =>  argmax: all probability mass on the maximum element.
     * Before the fix, T=0 silently fell through to T=1 softmax.
     */
    float tzero[3] = {1.0f, 5.0f, 2.0f};   /* max at index 1 */
    niyah_softmax_temperature(tzero, 3, 0.0f);
    assert(CLOSE(tzero[0], 0.0f));
    assert(CLOSE(tzero[1], 1.0f));           /* winner */
    assert(CLOSE(tzero[2], 0.0f));

    /*
     * T < 0  =>  no probabilistic meaning; safe uniform fallback.
     * Before the fix, negative inv_t inverted the distribution (high logit
     * -> low probability) and then applied softmax, producing nonsense.
     */
    float tneg[3] = {1.0f, 5.0f, 2.0f};
    niyah_softmax_temperature(tneg, 3, -1.0f);
    assert(CLOSE(tneg[0] + tneg[1] + tneg[2], 1.0f));
    assert(CLOSE(tneg[0], 1.0f / 3.0f));    /* uniform */
    assert(CLOSE(tneg[1], 1.0f / 3.0f));
    assert(CLOSE(tneg[2], 1.0f / 3.0f));

    /* ------------------------------------------------------------------ */
    /* niyah_log_softmax                                                   */
    /* ------------------------------------------------------------------ */

    /* Standard values: log(softmax([1,2,3])). */
    float logits[3] = {1.0f, 2.0f, 3.0f};
    niyah_log_softmax(logits, 3);
    assert(CLOSE(expf(logits[0]), 0.09003f));
    assert(CLOSE(expf(logits[1]), 0.24473f));
    assert(CLOSE(expf(logits[2]), 0.66524f));

    /*
     * Overflow side: large equal values.
     * Without max-subtraction expf(1000) = inf, sum = inf, log(inf) = inf,
     * result is -inf for all (inf - inf = NaN on some paths).
     * With max-subtraction: expf(0)=1 each, sum=3, log_sum = log(3).
     */
    float lbig[3] = {1000.0f, 1000.0f, 1000.0f};
    niyah_log_softmax(lbig, 3);
    for (int i = 0; i < 3; ++i) {
        assert(isfinite(lbig[i]));
        assert(CLOSE(expf(lbig[i]), 1.0f / 3.0f));
    }

    /*
     * Underflow / sum-zero path.
     * x[0]=0, x[1]=-1000: after max-subtraction, expf(-1000)=0 (underflow).
     * sum = 1 + 0 = 1.0f > 0, so the guard does NOT fire.
     * Correct result: log-prob 0 for winner, -1000 (effectively -inf) for
     * the other (expf(-1000) rounds to 0 on output as well).
     *
     * Also verify that NO result is NaN — that was the original bug.
     */
    float lsmall[2] = {0.0f, -1000.0f};
    niyah_log_softmax(lsmall, 2);
    assert(!isnan(lsmall[0]) && !isnan(lsmall[1]));
    assert(isfinite(lsmall[0]));
    assert(CLOSE(expf(lsmall[0]), 1.0f));  /* winner: log-prob ~0 */
    assert(expf(lsmall[1]) < 1e-6f);       /* loser: near zero prob */

    /*
     * Degenerate sum-zero guard (all elements -inf).
     * Should fall through to uniform log-distribution, not NaN.
     */
    float ninf[3] = {-INFINITY, -INFINITY, -INFINITY};
    niyah_log_softmax(ninf, 3);
    for (int i = 0; i < 3; ++i) {
        assert(!isnan(ninf[i]));
    }

    /* ------------------------------------------------------------------ */
    /* niyah_argmax                                                        */
    /* ------------------------------------------------------------------ */

    const float pick[5] = {-1.0f, 3.5f, 0.0f, 3.6f, 2.0f};
    assert(niyah_argmax(pick, 5) == 3);
    assert(niyah_argmax(NULL, 5) == -1);
    assert(niyah_argmax(pick, 0) == -1);

    /* ------------------------------------------------------------------ */
    /* Degenerate inputs must not crash                                    */
    /* ------------------------------------------------------------------ */
    niyah_softmax(NULL, 4);
    niyah_softmax(x, 0);

    return 0;
}
