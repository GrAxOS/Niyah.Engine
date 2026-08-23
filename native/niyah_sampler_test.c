#include "niyah_sampler.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static void test_apply_temperature(void) {
    float logits[3] = {2.0f, 4.0f, 6.0f};
    assert(niyah_sampler_apply_temperature_f32(logits, 3u, 2.0f));
    assert(nearly_equal(logits[0], 1.0f));
    assert(nearly_equal(logits[1], 2.0f));
    assert(nearly_equal(logits[2], 3.0f));
    assert(!niyah_sampler_apply_temperature_f32(logits, 3u, 0.0f));
}

static void test_top_k_filter(void) {
    float logits[5] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    float scratch[5] = {0};

    assert(niyah_sampler_top_k_filter_f32(logits, scratch, 5u, 2u));
    assert(isinf(logits[0]) && logits[0] < 0.0f);
    assert(nearly_equal(logits[1], 5.0f));
    assert(isinf(logits[2]) && logits[2] < 0.0f);
    assert(isinf(logits[3]) && logits[3] < 0.0f);
    assert(nearly_equal(logits[4], 4.0f));
}

static void test_top_k_filter_noop_when_k_covers_all(void) {
    float logits[3] = {1.0f, 2.0f, 3.0f};
    float scratch[3] = {0};

    assert(niyah_sampler_top_k_filter_f32(logits, scratch, 3u, 3u));
    assert(nearly_equal(logits[0], 1.0f));
    assert(nearly_equal(logits[1], 2.0f));
    assert(nearly_equal(logits[2], 3.0f));
}

static void test_argmax(void) {
    const float logits[4] = {0.1f, 0.9f, 0.5f, 0.3f};
    size_t idx = 0u;

    assert(niyah_sampler_argmax_f32(logits, 4u, &idx));
    assert(idx == 1u);
}

static void test_weighted_sample(void) {
    const float probs[3] = {0.1f, 0.2f, 0.7f};
    size_t idx = 0u;

    assert(niyah_sampler_weighted_f32(probs, 3u, 0.05f, &idx));
    assert(idx == 0u);

    assert(niyah_sampler_weighted_f32(probs, 3u, 0.25f, &idx));
    assert(idx == 1u);

    assert(niyah_sampler_weighted_f32(probs, 3u, 0.95f, &idx));
    assert(idx == 2u);
}

static void test_rejects_invalid_input(void) {
    float logits[1] = {1.0f};
    float scratch[1] = {0};
    size_t idx = 0u;

    assert(!niyah_sampler_apply_temperature_f32(NULL, 1u, 1.0f));
    assert(!niyah_sampler_top_k_filter_f32(NULL, scratch, 1u, 1u));
    assert(!niyah_sampler_argmax_f32(NULL, 1u, &idx));
    assert(!niyah_sampler_weighted_f32(logits, 1u, 1.5f, &idx));
}

int main(void) {
    test_apply_temperature();
    test_top_k_filter();
    test_top_k_filter_noop_when_k_covers_all();
    test_argmax();
    test_weighted_sample();
    test_rejects_invalid_input();
    return 0;
}
