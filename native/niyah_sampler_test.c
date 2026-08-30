#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    const float logits[5] = {0.1f, 0.2f, 5.0f, 0.3f, 0.4f};

    /* Greedy always returns the argmax. */
    NiyahSamplerConfig greedy;
    greedy.strategy = NIYAH_SAMPLE_GREEDY;
    greedy.temperature = 1.0f;
    greedy.top_k = 0;
    greedy.top_p = 1.0f;
    for (int i = 0; i < 20; ++i) {
        assert(niyah_sample(logits, 5, &greedy) == 2);
    }

    /* A NULL config defaults to greedy rather than crashing. */
    assert(niyah_sample(logits, 5, NULL) == 2);

    /* top_k = 1 is greedy by construction. */
    NiyahSamplerConfig top1;
    top1.strategy = NIYAH_SAMPLE_TOP_K;
    top1.temperature = 1.0f;
    top1.top_k = 1;
    top1.top_p = 1.0f;
    for (int i = 0; i < 20; ++i) {
        assert(niyah_sample(logits, 5, &top1) == 2);
    }

    /* Seeding makes sampling reproducible: same seed, same sequence. */
    NiyahSamplerConfig stochastic;
    stochastic.strategy = NIYAH_SAMPLE_TOP_P;
    stochastic.temperature = 1.0f;
    stochastic.top_k = 0;
    stochastic.top_p = 0.95f;

    /* Flat logits keep multiple candidates inside the nucleus so this
     * actually tests PRNG seeding rather than deterministic top-p collapse. */
    const float stochastic_logits[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int32_t first[16];
    niyah_sampler_seed(12345u);
    for (int i = 0; i < 16; ++i) {
        first[i] = niyah_sample(stochastic_logits, 5, &stochastic);
    }

    niyah_sampler_seed(12345u);
    for (int i = 0; i < 16; ++i) {
        assert(niyah_sample(stochastic_logits, 5, &stochastic) == first[i]);
    }

    /* Every sample must be a valid vocab index. */
    for (int i = 0; i < 16; ++i) {
        assert(first[i] >= 0 && first[i] < 5);
    }

    /* A different seed should not reproduce the same 16 draws. */
    niyah_sampler_seed(999u);
    bool differs = false;
    for (int i = 0; i < 16; ++i) {
        if (niyah_sample(stochastic_logits, 5, &stochastic) != first[i]) {
            differs = true;
        }
    }
    assert(differs);

    /* Temperature sampling stays in range. */
    NiyahSamplerConfig temp;
    temp.strategy = NIYAH_SAMPLE_TEMPERATURE;
    temp.temperature = 0.8f;
    temp.top_k = 0;
    temp.top_p = 1.0f;
    for (int i = 0; i < 32; ++i) {
        const int32_t t = niyah_sample(logits, 5, &temp);
        assert(t >= 0 && t < 5);
    }

    /* Temperature 0 degenerates to greedy instead of dividing by zero. */
    temp.temperature = 0.0f;
    assert(niyah_sample(logits, 5, &temp) == 2);

    /* Repetition penalty pushes a repeated token down. */
    float penalised[3] = {2.0f, 1.0f, -1.0f};
    const int32_t history[2] = {0, 2};
    niyah_sampler_apply_repetition_penalty(penalised, 3, history, 2, 2.0f);
    assert(CLOSE(penalised[0], 1.0f));    /* positive: divided */
    assert(CLOSE(penalised[1], 1.0f));    /* untouched */
    assert(CLOSE(penalised[2], -2.0f));   /* negative: multiplied */

    /* Out-of-range history indices are ignored, not written through. */
    const int32_t bad_history[2] = {-1, 99};
    niyah_sampler_apply_repetition_penalty(penalised, 3, bad_history, 2, 2.0f);
    assert(CLOSE(penalised[0], 1.0f));

    /* Degenerate inputs. */
    assert(niyah_sample(NULL, 5, &greedy) == -1);
    assert(niyah_sample(logits, 0, &greedy) == -1);

    return 0;
}
