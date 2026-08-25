#include "niyah.h"
#include <math.h>
#include <stddef.h>

/* SiLU activation: x * sigmoid(x) */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/*
 * SwiGLU gated feed-forward activation.
 * x[i]    = value stream
 * gate[i] = gate stream
 * result stored in x[i] = SiLU(gate[i]) * x[i]
 */
void niyah_swiglu_forward(float* x, const float* gate, int32_t n) {
    if (!x || !gate || n <= 0) return;
    for (int32_t i = 0; i < n; i++)
        x[i] = silu(gate[i]) * x[i];
}
