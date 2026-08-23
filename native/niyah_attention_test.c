#include "niyah_attention.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_attention_causal_f32_basic(void) {
    /* seq_len=2, head_dim=2. Q=K=identity rows, V = [[1,2],[3,4]].
     * Row 0 can only see itself (causal) -> out[0] = V[0] exactly.
     * Row 1 sees both positions with scores [0, 1/sqrt(2)] -> softmax
     * mixes V[0] and V[1]. */
    const float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scores[4] = {0};
    float out[4] = {0};

    assert(niyah_attention_causal_f32(q, k, v, scores, out, 2u, 2u));

    assert(nearly_equal(out[0], 1.0f));
    assert(nearly_equal(out[1], 2.0f));

    assert(nearly_equal(out[2], 2.33952310f));
    assert(nearly_equal(out[3], 3.33952310f));
}

static void test_attention_causal_f32_rejects_invalid_input(void) {
    const float q[1] = {1.0f};
    float scores[1] = {0};
    float out[1] = {0};

    assert(!niyah_attention_causal_f32(NULL, q, q, scores, out, 1u, 1u));
    assert(!niyah_attention_causal_f32(q, q, q, scores, out, 0u, 1u));
    assert(!niyah_attention_causal_f32(q, q, q, scores, out, 1u, 0u));
}

int main(void) {
    test_attention_causal_f32_basic();
    test_attention_causal_f32_rejects_invalid_input();
    return 0;
}
