#include "niyah_swiglu.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_swiglu_f32_basic(void) {
    /* rows=1, dim=2, hidden_dim=2. x=[1,2].
     * w_gate = w_up = identity(2x2) -> gate=up=[1,2].
     * h = SiLU(gate)*up = [SiLU(1)*1, SiLU(2)*2].
     * w_down = [[1,1],[0,1]] ([dim x hidden]) -> out = h * w_down^T. */
    const float x[2] = {1.0f, 2.0f};
    const float w_gate[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float w_up[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float w_down[4] = {1.0f, 1.0f, 0.0f, 1.0f};
    float gate_scratch[2] = {0};
    float up_scratch[2] = {0};
    float out[2] = {0};

    assert(niyah_swiglu_f32(x, w_gate, w_up, w_down, gate_scratch,
                             up_scratch, out, 1u, 2u, 2u));

    assert(nearly_equal(out[0], 4.25424689f));
    assert(nearly_equal(out[1], 3.52318831f));
}

static void test_swiglu_f32_rejects_invalid_input(void) {
    const float w[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float scratch[2] = {0};
    float out[2] = {0};

    assert(!niyah_swiglu_f32(NULL, w, w, w, scratch, scratch, out, 1u, 2u,
                              2u));
    assert(!niyah_swiglu_f32(w, w, w, w, scratch, scratch, out, 0u, 2u, 2u));
    assert(!niyah_swiglu_f32(w, w, w, w, scratch, scratch, out, 1u, 0u, 2u));
}

int main(void) {
    test_swiglu_f32_basic();
    test_swiglu_f32_rejects_invalid_input();
    return 0;
}
