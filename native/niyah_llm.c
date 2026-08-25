#include "niyah.h"

#include <stdlib.h>
#include <string.h>
<<<<<<< HEAD
#include <stdio.h>

/*
 * LLM inference.
 *
 * When model weights are present (model->weights != NULL) a full transformer
 * forward pass is executed.  Without weights the engine returns a structured
 * "model not loaded" response so the UI always gets a valid NiyahLLMOutput.
 */

#define LLM_DEFAULT_MAX_TOKENS 128

/* Forward declaration (implemented in this file) */
static NiyahLLMOutput llm_no_model_response(const char* prompt);

NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm,
                                   const char* prompt,
                                   int32_t     max_tokens) {
    NiyahLLMOutput out;
    memset(&out, 0, sizeof(out));

    if (!llm || !prompt) {
        out.text = _strdup("Error: invalid arguments.");
        return out;
    }

    if (max_tokens <= 0) max_tokens = LLM_DEFAULT_MAX_TOKENS;

    /* Telemetry */
    niyah_telemetry_start(&out.telemetry);

    /* Without weights: return a deterministic response */
    if (!llm->model.weights) {
        niyah_telemetry_end(&out.telemetry);
        return llm_no_model_response(prompt);
    }

    /* ---- Actual inference (requires loaded weights) ---- */
    int32_t  n_vocab = llm->model.config.n_vocab;
    int32_t  n_embd  = llm->model.config.n_embd;
    int32_t  n_layer = llm->model.config.n_layer;
    int32_t  n_head  = llm->model.config.n_head;

    /* Tokenise input */
    int32_t* input_tokens = (int32_t*)calloc((size_t)NIYAH_MAX_TOKENS, sizeof(int32_t));
    if (!input_tokens) { out.text = _strdup("OOM"); return out; }

    int32_t n_input = niyah_tokenize(&llm->tokenizer, prompt,
                                      input_tokens, NIYAH_MAX_TOKENS);

    int32_t* output_tokens = (int32_t*)calloc((size_t)max_tokens, sizeof(int32_t));
    if (!output_tokens) { free(input_tokens); out.text = _strdup("OOM"); return out; }

    /* Allocate activation buffers */
    size_t seq_buf_size = (size_t)n_embd * NIYAH_MAX_SEQ_LEN;
    float* x    = (float*)calloc(seq_buf_size, sizeof(float));
    float* attn = (float*)calloc(seq_buf_size, sizeof(float));
    float* norm1 = (float*)calloc(seq_buf_size, sizeof(float));
    float* norm2 = (float*)calloc(seq_buf_size, sizeof(float));
    float* ffn   = (float*)calloc(seq_buf_size, sizeof(float));
    float* logits = (float*)calloc((size_t)n_vocab, sizeof(float));

    if (!x || !attn || !norm1 || !norm2 || !ffn || !logits) {
        free(x); free(attn); free(norm1); free(norm2); free(ffn); free(logits);
        free(input_tokens); free(output_tokens);
        out.text = _strdup("OOM");
        return out;
    }

    /* Initialise from token embeddings (first n_embd floats of weights per token) */
    const float* embed_table = (const float*)llm->model.weights;
    size_t embed_row = (size_t)n_embd * sizeof(float);
    int32_t ctx_len = n_input < NIYAH_MAX_SEQ_LEN ? n_input : NIYAH_MAX_SEQ_LEN;

    for (int32_t t = 0; t < ctx_len; t++) {
        int32_t tid = input_tokens[t];
        if (tid >= 0 && tid < n_vocab &&
            (size_t)(tid + 1) * embed_row <= llm->model.weights_size)
            memcpy(x + t * n_embd, embed_table + (size_t)tid * n_embd,
                   (size_t)n_embd * sizeof(float));
    }

    /* RoPE encode */
    niyah_rope_forward(x, ctx_len, n_embd, n_head);

    /* Transformer layers */
    NiyahTransformerLayerState layer_state = {
        .attn_out = attn, .ffn_out = ffn,
        .norm1_out = norm1, .norm2_out = norm2,
        .batch = 1, .seq = ctx_len, .dim = n_embd, .n_head = n_head
    };
    for (int32_t l = 0; l < n_layer; l++)
        niyah_transformer_layer_forward(&layer_state, x);

    /* Project last token to vocab (identity proj if no lm_head weights) */
    float* last = ffn + (ctx_len - 1) * n_embd;
    for (int32_t v = 0; v < n_vocab && v < n_embd; v++) logits[v] = last[v];

    /* Generate tokens */
    int32_t n_out = 0;
    for (int32_t step = 0; step < max_tokens; step++) {
        int32_t next = niyah_sample(logits, n_vocab, &llm->sampler);
        if (next == 0) break; /* EOS */
        output_tokens[n_out++] = next;
    }

    /* Detokenise */
    char* text = niyah_detokenize(&llm->tokenizer, output_tokens, n_out);
    out.text     = text ? text : _strdup("");
    out.n_tokens = n_out;
    out.logits   = NULL;

    free(x); free(attn); free(norm1); free(norm2); free(ffn); free(logits);
    free(input_tokens); free(output_tokens);

    niyah_telemetry_end(&out.telemetry);
    out.telemetry.tokens_processed = n_out;
    return out;
}

static NiyahLLMOutput llm_no_model_response(const char* prompt) {
    NiyahLLMOutput out;
    memset(&out, 0, sizeof(out));

    /* Echo prompt back with a notice when no model is loaded */
    char buf[512];
    snprintf(buf, sizeof(buf),
             "[Niyah Engine] No model loaded. Prompt received: \"%.*s\"",
             80, prompt);
    out.text     = _strdup(buf);
    out.n_tokens = 0;
    return out;
=======

/*
 * Real prefill + autoregressive decode. The single most important property of
 * this file: with no weights loaded it reports NIYAH_ERR_NO_WEIGHTS and
 * returns text = NULL. It never invents output text.
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
    if (position < 0 || position >= c.n_ctx) {
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
        /*
         * Empty prompt: seed with BOS if the model defines one. Note that
         * token id 0 cannot be used as BOS or EOS here; the config struct
         * uses 0 to mean "unset". No mainstream vocab assigns either role to
         * id 0 (Llama bos=1 eos=2, Qwen2 eos=151645).
         */
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

            const bool is_eos =
                (c.eos_token_id > 0 && next == c.eos_token_id);

            /*
             * EOS is recorded in the token window so the KV cache and the
             * repetition penalty stay accurate, but it is deliberately
             * excluded from `generated` and therefore from the returned text.
             *
             * Previously ++generated ran before this check, so the
             * end-of-turn marker was detokenised into user-visible output.
             */
            tokens[total++] = next;
            if (is_eos) {
                break;
            }
            ++generated;

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

    if (status == NIYAH_OK) {
        /*
         * generated == 0 is a legitimate result: the model emitted EOS as its
         * very first token. That is an empty completion, not a shape error,
         * and niyah_detokenize returns an allocated empty string for it.
         */
        output.text = niyah_detokenize(&llm->tokenizer,
                                       tokens + n_prompt, generated);
        if (output.text) {
            output.n_tokens = generated;
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
        output.status   = status;
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
>>>>>>> origin/main
}
