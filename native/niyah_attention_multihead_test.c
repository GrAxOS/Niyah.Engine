#include "niyah_attention_multihead.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_multihead_causal_f32_basic(void) {
    const float q[8] = {1.0f, 0.0f, 0.5f, 0.5f,
                         0.0f, 1.0f, -0.5f, 0.5f};
    const float k[8] = {1.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f, 1.0f};
    const float v[8] = {1.0f, 2.0f, 5.0f, 6.0f,
                         3.0f, 4.0f, 7.0f, 8.0f};

    float q_head[4] = {0};
    float k_head[4] = {0};
    float v_head[4] = {0};
    float out_head[4] = {0};
    float scores[4] = {0};
    float out[8] = {0};

    assert(niyah_attention_multihead_causal_f32(q, k, v, q_head, k_head,
                                                 v_head, out_head, scores,
                                                 out, 2u, 2u, 2u));

    assert(nearly_equal(out[0], 1.0f));
    assert(nearly_equal(out[1], 2.0f));
    assert(nearly_equal(out[2], 5.0f));
    assert(nearly_equal(out[3], 6.0f));

    assert(nearly_equal(out[4], 2.33952310f));
    assert(nearly_equal(out[5], 3.33952310f));
    assert(nearly_equal(out[6], 6.33952310f));
    assert(nearly_equal(out[7], 7.33952310f));
}

static void test_multihead_causal_f32_rejects_invalid_input(void) {
    float buf4[4] = {0};
    float buf8[8] = {0};

    assert(!niyah_attention_multihead_causal_f32(NULL, buf8, buf8, buf4,
                                                  buf4, buf4, buf4, buf4,
                                                  buf8, 2u, 2u, 2u));
    assert(!niyah_attention_multihead_causal_f32(buf8, buf8, buf8, buf4,
                                                  buf4, buf4, buf4, buf4,
                                                  buf8, 0u, 2u, 2u));
}

int main(void) {
    test_multihead_causal_f32_basic();
    test_multihead_causal_f32_rejects_invalid_input();
    return 0;
}
