#undef NDEBUG
#include <assert.h>
#include <math.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    /* silu(x) = x * sigmoid(x). silu(0) = 0. */
    assert(CLOSE(niyah_silu(0.0f), 0.0f));

    /* silu(1) = 1 * 0.7310586 = 0.7310586. */
    assert(CLOSE(niyah_silu(1.0f), 0.7310586f));

    /* silu(-1) = -1 * 0.2689414 = -0.2689414. Note silu is not monotonic. */
    assert(CLOSE(niyah_silu(-1.0f), -0.2689414f));

    /* Saturation: large positive approaches identity, large negative -> 0. */
    assert(CLOSE(niyah_silu(20.0f), 20.0f));
    assert(fabsf(niyah_silu(-20.0f)) < 1e-4f);

    /* The sigmoid is branch-split on sign, so extreme inputs stay finite
     * instead of overflowing expf(). */
    assert(isfinite(niyah_silu(1000.0f)));
    assert(isfinite(niyah_silu(-1000.0f)));

    /* gelu(0) = 0 and gelu is close to silu near the origin. */
    assert(CLOSE(niyah_gelu(0.0f), 0.0f));
    assert(isfinite(niyah_gelu(-1000.0f)));

    /* SwiGLU in place: x[i] = silu(gate[i]) * x[i]. */
    float x[3] = {2.0f, 4.0f, 6.0f};
    const float gate[3] = {0.0f, 1.0f, -1.0f};
    niyah_swiglu_forward(x, gate, 3);
    assert(CLOSE(x[0], 0.0f));                      /* silu(0) * 2 */
    assert(CLOSE(x[1], 0.7310586f * 4.0f));
    assert(CLOSE(x[2], -0.2689414f * 6.0f));

    /* Out-of-place variant agrees and leaves inputs untouched. */
    const float up[2] = {3.0f, 5.0f};
    const float g2[2] = {1.0f, 0.0f};
    float out[2] = {0.0f, 0.0f};
    niyah_swiglu_to(out, up, g2, 2);
    assert(CLOSE(out[0], 0.7310586f * 3.0f));
    assert(CLOSE(out[1], 0.0f));
    assert(CLOSE(up[0], 3.0f));

    /* Degenerate inputs. */
    niyah_swiglu_forward(NULL, gate, 3);
    niyah_swiglu_forward(x, NULL, 3);
    niyah_swiglu_forward(x, gate, 0);

    return 0;
}
