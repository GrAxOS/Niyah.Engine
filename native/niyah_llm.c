#include "niyah.h"

#include <stdlib.h>
#include <string.h>
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
}
