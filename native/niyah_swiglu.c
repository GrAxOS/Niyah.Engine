#include "niyah.h"

#include <math.h>

/*
 * Was a stub. SwiGLU: SiLU(gate) elementwise-times up-projection.
 * The sigmoid is branch-split on the sign of x so exp() never overflows for
 * large-magnitude activations.
 */

static float niyah_sigmoid(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

float niyah_silu(float x)
{
    return x * niyah_sigmoid(x);
}

float niyah_gelu(float x)
{
    /* tanh approximation, matching the usual transformer implementation. */
    static const float k0 = 0.7978845608028654f;   /* sqrt(2/pi) */
    static const float k1 = 0.044715f;
    const float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void niyah_swiglu_to(float* out,
                     const float* up,
                     const float* gate,
                     int32_t n)
{
    if (!out || !up || !gate || n <= 0) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        out[i] = niyah_silu(gate[i]) * up[i];
    }
}

void niyah_swiglu_forward(float* x, const float* gate, int32_t n)
{
    /* x carries the up-projection and is overwritten with the gated result. */
    niyah_swiglu_to(x, x, gate, n);
}
