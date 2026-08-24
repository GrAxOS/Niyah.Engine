#include "niyah.h"

#include <stdlib.h>
#include <string.h>

/*
 * Was: `// LLM implementation stubs / TODO: Implement actual inference kernels`.
 *
 * Real prefill + autoregressive decode. The single most important property of
 * this file: with no weights loaded it reports NIYAH_ERR_NO_WEIGHTS and returns
 * text = NULL. It never invents output text.
 */

NiyahStatus niyah_llm_forward(NiyahLLM* llm,
                             int32_t token,
                             int32_t position,
                             NiyahKVCache* cache,
                             float* logits,
                             float* scratch)
{
    if (!llm || !cache || !logits || !scratch) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!llm->model.weights) {
        return NIYAH_ERR_NO_WEIGHTS;
    }

    NiyahModelConfig c = llm->model.config;
    niyah_model_config_normalize(&c);

    if (token < 0 || token >= c.n_vocab) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahModelWeights w;
    const NiyahStatus map_status = niyah_model_weights_map(&w, &llm->model);
    if (map_status != NIYAH_OK) {
        return map_status;
    }

    const int32_t dim = c.n_embd;

    float* hidden = (float*)malloc((size_t)dim * sizeof(float));
    if (!hidden) {
        niyah_model_weights_unmap(&w);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    /* Embedding lookup. */
    memcpy(hidden,
           w.embedding + (size_t)token * (size_t)dim,
           (size_t)dim * sizeof(float));

    NiyahStatus status = NIYAH_OK;
    for (int32_t l = 0; l < c.n_layer; ++l) {
        status = niyah_transformer_layer_forward_weighted(
            hidden, &w.layers[l], &c, cache, l, position, scratch);
        if (status != NIYAH_OK) {
            break;
        }
    }

    if (status == NIYAH_OK) {
        niyah_rmsnorm(hidden, w.final_norm, dim, c.norm_eps);
        niyah_matvec(logits, w.lm_head, hidden, c.n_vocab, dim);
    }

    free(hidden);
    niyah_model_weights_unmap(&w);
    return status;
}

NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm,
                                  const char* prompt,
                                  int32_t max_tokens)
{
    NiyahLLMOutput output;
    memset(&output, 0, sizeof(output));
    output.status = NIYAH_ERR_INVALID_ARG;

    if (!llm || !prompt || max_tokens <= 0) {
        return output;
    }

    /*
     * No weights means no inference. Returning NULL here is the whole point:
     * a caller must be able to distinguish "model not loaded" from "model
     * produced an empty answer".
     */
    if (!llm->model.weights) {
        output.status = NIYAH_ERR_NO_WEIGHTS;
        return output;
    }

    NiyahModelConfig c = llm->model.config;
    niyah_model_config_normalize(&c);

    if (c.n_embd <= 0 || c.n_head <= 0 || c.n_embd % c.n_head != 0) {
        output.status = NIYAH_ERR_SHAPE;
        return output;
    }

    const int32_t head_dim = c.n_embd / c.n_head;
    const int32_t n_ctx = c.n_ctx;

    niyah_telemetry_start(&output.telemetry);

    /* --- Tokenise the prompt -------------------------------------------- */
    int32_t* tokens = (int32_t*)malloc((size_t)n_ctx * sizeof(int32_t));
    float*   logits = (float*)malloc((size_t)c.n_vocab * sizeof(float));
    const size_t scratch_floats = niyah_transformer_scratch_floats(&c);
    float* scratch = scratch_floats
        ? (float*)malloc(scratch_floats * sizeof(float)) : NULL;

    NiyahKVCache cache;
    memset(&cache, 0, sizeof(cache));

    /*
     * Only treat a NULL scratch as OOM when scratch_floats > 0; when
     * scratch_floats == 0 the engine does not need a scratch buffer and
     * NULL is the correct value.
     */
    if (!tokens || !logits || (scratch_floats > 0 && !scratch)) {
        free(tokens);
        free(logits);
        free(scratch);
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        return output;
    }

    NiyahStatus status = niyah_kv_cache_init(&cache, c.n_layer, c.n_kv_head,
                                            head_dim, n_ctx);
    if (status != NIYAH_OK) {
        free(tokens);
        free(logits);
        free(scratch);
        output.status = status;
        return output;
    }

    int32_t n_prompt = niyah_tokenize(&llm->tokenizer, prompt, tokens, n_ctx);
    if (n_prompt <= 0) {
        /* Empty prompt: seed with BOS if the model defines one. */
        if (c.bos_token_id > 0 && c.bos_token_id < c.n_vocab) {
            tokens[0] = c.bos_token_id;
            n_prompt = 1;
        } else {
            niyah_kv_cache_free(&cache);
            free(tokens);
            free(logits);
            free(scratch);
            output.status = NIYAH_ERR_INVALID_ARG;
            return output;
        }
    }

    /* --- Prefill --------------------------------------------------------- */
    int32_t position = 0;
    for (; position < n_prompt && position < n_ctx; ++position) {
        status = niyah_llm_forward(llm, tokens[position], position,
                                  &cache, logits, scratch);
        if (status != NIYAH_OK) {
            break;
        }
    }

    /* --- Decode ---------------------------------------------------------- */
    int32_t total     = n_prompt;
    int32_t generated = 0;

    if (status == NIYAH_OK) {
        while (generated < max_tokens && total < n_ctx) {
            niyah_sampler_apply_repetition_penalty(logits, c.n_vocab,
                                                   tokens, total, 1.1f);

            const int32_t next = niyah_sample(logits, c.n_vocab, &llm->sampler);
            if (next < 0) {
                status = NIYAH_ERR_SHAPE;
                break;
            }

            tokens[total++] = next;
            ++generated;

            if (c.eos_token_id > 0 && next == c.eos_token_id) {
                break;
            }
            if (total >= n_ctx) {
                break;
            }

            status = niyah_llm_forward(llm, next, total - 1,
                                       &cache, logits, scratch);
            if (status != NIYAH_OK) {
                break;
            }
        }
    }

    niyah_telemetry_end(&output.telemetry);
    output.telemetry.tokens_processed = generated;
    output.telemetry.memory_used = (int64_t)llm->model.weights_size;

    if (status == NIYAH_OK && generated > 0) {
        output.text     = niyah_detokenize(&llm->tokenizer,
                                           tokens + n_prompt, generated);
        output.n_tokens = generated;
        if (output.text) {
            /*
             * Ownership of logits transfers to the caller.
             * Do NOT free logits here.
             */
            output.logits = logits;
            output.status = NIYAH_OK;
        } else {
            /*
             * niyah_detokenize returned NULL: out of memory.
             * Free logits here so the caller never receives a non-NULL
             * logits pointer paired with an error status, eliminating the
             * double-free / leak ambiguity.
             */
            free(logits);
            output.logits   = NULL;
            output.n_tokens = 0;
            output.status   = NIYAH_ERR_OUT_OF_MEMORY;
        }
    } else {
        free(logits);
        output.text     = NULL;
        output.n_tokens = 0;
        output.logits   = NULL;
        output.status   = (status == NIYAH_OK) ? NIYAH_ERR_SHAPE : status;
    }

    niyah_kv_cache_free(&cache);
    free(tokens);
    free(scratch);

    return output;
}

void niyah_llm_output_free(NiyahLLMOutput* output)
{
    if (!output) {
        return;
    }
    free(output->text);
    free(output->logits);
    output->text     = NULL;
    output->logits   = NULL;
    output->n_tokens = 0;
}
