#undef NDEBUG
#include <assert.h>
#include <string.h>

#include "niyah.h"

/*
 * The single most important test in this repository.
 *
 * niyah_llm_generate used to return a hard-coded string regardless of whether
 * any weights were loaded, which meant a caller could not distinguish a real
 * completion from a placeholder. For an engine whose stated purpose is
 * epistemic honesty, silently fabricating model output is the worst possible
 * failure mode.
 *
 * The contract asserted here: no weights => NIYAH_ERR_NO_WEIGHTS and a NULL
 * text pointer. Never a fabricated string.
 */
int main(void)
{
    NiyahLLM llm;
    memset(&llm, 0, sizeof(llm));

    assert(llm.model.weights == NULL);

    NiyahLLMOutput out = niyah_llm_generate(&llm, "What is the capital of Japan?", 16);

    assert(out.status == NIYAH_ERR_NO_WEIGHTS);
    assert(out.text == NULL);
    assert(out.n_tokens == 0);
    assert(out.logits == NULL);

    niyah_llm_output_free(&out);
    assert(out.text == NULL);

    /* A NULL llm is an argument error, not a crash and not a fake answer. */
    NiyahLLMOutput null_out = niyah_llm_generate(NULL, "hello", 8);
    assert(null_out.status == NIYAH_ERR_INVALID_ARG);
    assert(null_out.text == NULL);
    niyah_llm_output_free(&null_out);

    /* A NULL prompt likewise. */
    NiyahLLMOutput no_prompt = niyah_llm_generate(&llm, NULL, 8);
    assert(no_prompt.status != NIYAH_OK);
    assert(no_prompt.text == NULL);
    niyah_llm_output_free(&no_prompt);

    /* max_tokens <= 0 must not produce output either. */
    NiyahLLMOutput no_budget = niyah_llm_generate(&llm, "hello", 0);
    assert(no_budget.status != NIYAH_OK);
    assert(no_budget.text == NULL);
    niyah_llm_output_free(&no_budget);

    /* Forward pass with no weights must also refuse. */
    float logits[8] = {0};
    assert(niyah_llm_forward(&llm, 0, 0, NULL, logits, NULL)
           != NIYAH_OK);

    /* Freeing a zeroed output is safe and idempotent. */
    NiyahLLMOutput blank;
    memset(&blank, 0, sizeof(blank));
    niyah_llm_output_free(&blank);
    niyah_llm_output_free(&blank);
    niyah_llm_output_free(NULL);

    return 0;
}
