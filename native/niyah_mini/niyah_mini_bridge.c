#include "niyah_mini_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/* Portable strdup that never relies on a POSIX extension. */
static char *bridge_strdup(const char *s)
{
    size_t len;
    char *copy;
    if (!s) return NULL;
    len = strlen(s);
    copy = (char *)malloc(len + 1U);
    if (copy) memcpy(copy, s, len + 1U);
    return copy;
}

/* ==========================================================================
 * NiyahMiniWrappedModel
 * ========================================================================== */

NiyahStatus niyah_mini_wrapped_init(
    NiyahMiniWrappedModel *wrapped,
    const NiyahMiniConfig *config)
{
    NiyahStatus status;
    if (!wrapped || !config) return NIYAH_ERR_INVALID_ARG;
    memset(wrapped, 0, sizeof(*wrapped));
    wrapped->config = *config;
    status = niyah_mini_model_init(&wrapped->mini_model, config);
    if (status != NIYAH_OK) return status;
    wrapped->weights_loaded = false;
    return NIYAH_OK;
}

/*
 * Stub fix #1 -- niyah_mini_wrapped_load
 *
 * The previous implementation opened the config file, discarded its
 * contents and initialised from NIYAH_MINI_TINY defaults, so every load
 * silently ignored the on-disk config.
 *
 * niyah_mini_model_load() in niyah_mini_model.c already contains a working
 * key:value JSON parser for all NiyahMiniConfig fields.  Delegate to it.
 */
NiyahStatus niyah_mini_wrapped_load(
    NiyahMiniWrappedModel *wrapped,
    const char *config_path,
    const char *weights_path)
{
    NiyahStatus status;
    if (!wrapped || !config_path || !weights_path) return NIYAH_ERR_INVALID_ARG;

    /* niyah_mini_model_load frees any prior state, parses the JSON config,
     * allocates weights and fills them from the binary file. */
    status = niyah_mini_model_load(&wrapped->mini_model, config_path, weights_path);
    if (status != NIYAH_OK) {
        wrapped->weights_loaded = false;
        return status;
    }

    /* Mirror the loaded config so wrapped->config stays in sync. */
    wrapped->config = wrapped->mini_model.config;
    wrapped->weights_loaded = true;
    return NIYAH_OK;
}

void niyah_mini_wrapped_free(NiyahMiniWrappedModel *wrapped)
{
    if (!wrapped) return;
    niyah_mini_model_free(&wrapped->mini_model);
    memset(wrapped, 0, sizeof(*wrapped));
}

NiyahStatus niyah_mini_to_niyah_model(
    NiyahModel *niyah_model,
    const NiyahMiniWrappedModel *wrapped)
{
    if (!niyah_model || !wrapped) return NIYAH_ERR_INVALID_ARG;
    niyah_model->config.n_vocab           = wrapped->config.n_vocab;
    niyah_model->config.n_embd            = wrapped->config.n_dim;
    niyah_model->config.n_head            = wrapped->config.n_heads;
    niyah_model->config.n_layer           = wrapped->config.n_layers;
    niyah_model->config.n_ctx             = wrapped->config.n_ctx;
    niyah_model->config.type              = 0;
    niyah_model->config.n_kv_head         = wrapped->config.n_kv_heads;
    niyah_model->config.n_ff              = wrapped->config.n_ff;
    niyah_model->config.eos_token_id      = NIYAH_MINI_EOS_TOKEN_ID;
    niyah_model->config.bos_token_id      = NIYAH_MINI_BOS_TOKEN_ID;
    niyah_model->config.rope_theta        = wrapped->config.rope_theta;
    niyah_model->config.norm_eps          = wrapped->config.norm_eps;
    niyah_model->config.tie_word_embeddings = wrapped->config.tie_word_embeddings;
    niyah_model->weights      = wrapped->mini_model.weights.memory_block;
    niyah_model->weights_size = wrapped->mini_model.weights.memory_size;
    return NIYAH_OK;
}

/*
 * Stub fix #2 -- niyah_mini_wrapped_generate
 *
 * The previous implementation tokenised the prompt then immediately memcpy'd
 * it back as the output text -- it never called niyah_mini_generate().
 *
 * New implementation:
 *   1. Guard: NIYAH_ERR_NO_WEIGHTS + text=NULL when weights absent.
 *   2. Tokenise prompt via NiyahTokenizer (if provided) or byte-level fallback.
 *   3. Call niyah_mini_generate() to produce output token IDs.
 *   4. Detokenise with niyah_detokenize() (if tokenizer provided) or
 *      byte-level fallback.
 *   5. Return a fully populated NiyahLLMOutput.
 */
NiyahLLMOutput niyah_mini_wrapped_generate(
    const NiyahMiniWrappedModel *wrapped,
    const NiyahTokenizer *tokenizer,
    const char *prompt,
    int32_t max_tokens)
{
    NiyahLLMOutput output;
    int32_t *prompt_ids   = NULL;
    int32_t *output_ids   = NULL;
    int32_t  prompt_len   = 0;
    int32_t  output_len   = 0;
    NiyahStatus status;
    size_t i;

    memset(&output, 0, sizeof(output));

    if (!wrapped || !prompt || max_tokens <= 0) {
        output.status = NIYAH_ERR_INVALID_ARG;
        return output;
    }

    /*
     * Critical contract (matches niyah_llm_generate):
     * NIYAH_ERR_NO_WEIGHTS + text=NULL when weights are not loaded.
     * Never fabricate output.
     */
    if (!wrapped->weights_loaded || !wrapped->mini_model.weights.memory_block) {
        output.status = NIYAH_ERR_NO_WEIGHTS;
        output.text   = NULL;
        return output;
    }

    niyah_telemetry_start(&output.telemetry);

    /* ------------------------------------------------------------------ */
    /* Step 1: Tokenise                                                    */
    /* ------------------------------------------------------------------ */
    if (tokenizer) {
        NiyahTokenizer *mut_tok = (NiyahTokenizer *)(uintptr_t)tokenizer;
        prompt_len = niyah_tokenize(mut_tok, prompt, NULL, 0);
        if (prompt_len < 0) prompt_len = 0;
        if (prompt_len > 0) {
            prompt_ids = (int32_t *)malloc((size_t)prompt_len * sizeof(int32_t));
            if (!prompt_ids) { output.status = NIYAH_ERR_OUT_OF_MEMORY; goto done; }
            niyah_tokenize(mut_tok, prompt, prompt_ids, prompt_len);
        }
    } else {
        /* Byte-level fallback: each byte becomes one token ID. */
        size_t len = strlen(prompt);
        if (len > (size_t)wrapped->config.n_ctx)
            len = (size_t)wrapped->config.n_ctx;
        prompt_len = (int32_t)len;
        if (prompt_len > 0) {
            prompt_ids = (int32_t *)malloc((size_t)prompt_len * sizeof(int32_t));
            if (!prompt_ids) { output.status = NIYAH_ERR_OUT_OF_MEMORY; goto done; }
            for (i = 0U; i < (size_t)prompt_len; ++i)
                prompt_ids[i] = (int32_t)(unsigned char)prompt[i];
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 2: Generate                                                    */
    /* ------------------------------------------------------------------ */
    if (max_tokens > wrapped->config.n_ctx) max_tokens = wrapped->config.n_ctx;
    output_ids = (int32_t *)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!output_ids) { output.status = NIYAH_ERR_OUT_OF_MEMORY; goto done; }

    status = niyah_mini_generate(
        (NiyahMiniModel *)(uintptr_t)&wrapped->mini_model,
        prompt_ids, prompt_len,
        max_tokens, 1.0f,          /* temperature = 1.0; caller scales before */
        output_ids, &output_len);

    if (status != NIYAH_OK) {
        output.status = status;
        goto done;
    }
    if (output_len <= 0) {
        output.status = NIYAH_ERR_SHAPE;
        goto done;
    }

    /* ------------------------------------------------------------------ */
    /* Step 3: Detokenise                                                  */
    /* ------------------------------------------------------------------ */
    if (tokenizer) {
        NiyahTokenizer *mut_tok = (NiyahTokenizer *)(uintptr_t)tokenizer;
        output.text = niyah_detokenize(mut_tok, output_ids, output_len);
    } else {
        /* Byte-level fallback: token IDs 0-255 map directly to bytes. */
        char *buf = (char *)malloc((size_t)output_len + 1U);
        if (buf) {
            for (i = 0U; i < (size_t)output_len; ++i)
                buf[i] = (output_ids[i] >= 0 && output_ids[i] <= 255)
                          ? (char)(unsigned char)output_ids[i] : '?';
            buf[output_len] = '\0';
        }
        output.text = buf;
    }

    if (!output.text) {
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        goto done;
    }

    output.n_tokens = output_len;
    output.status   = NIYAH_OK;

done:
    niyah_telemetry_end(&output.telemetry);
    output.telemetry.tokens_processed = output_len;
    free(prompt_ids);
    free(output_ids);
    return output;
}

/* ==========================================================================
 * Evidence Envelope
 * ========================================================================== */

NiyahStatus niyah_mini_evidence_envelope_init(NiyahMiniEvidenceEnvelope *envelope)
{
    if (!envelope) return NIYAH_ERR_INVALID_ARG;
    memset(envelope, 0, sizeof(*envelope));
    envelope->label = NIYAH_MINI_EVIDENCE_UNKNOWN;
    return NIYAH_OK;
}

void niyah_mini_evidence_envelope_free(NiyahMiniEvidenceEnvelope *envelope)
{
    int32_t i;
    if (!envelope) return;
    free(envelope->answer);
    if (envelope->source_ids) {
        for (i = 0; i < envelope->n_source_ids; ++i) free(envelope->source_ids[i]);
        free(envelope->source_ids);
    }
    if (envelope->limitations) {
        for (i = 0; i < envelope->n_limitations; ++i) free(envelope->limitations[i]);
        free(envelope->limitations);
    }
    if (envelope->verification_steps) {
        for (i = 0; i < envelope->n_verification_steps; ++i) free(envelope->verification_steps[i]);
        free(envelope->verification_steps);
    }
    memset(envelope, 0, sizeof(*envelope));
}

NiyahStatus niyah_mini_format_evidence_output(
    NiyahMiniEvidenceEnvelope *envelope,
    const char *text,
    const char **source_ids,
    int32_t n_source_ids)
{
    int32_t i;
    if (!envelope || !text) return NIYAH_ERR_INVALID_ARG;
    niyah_mini_evidence_envelope_free(envelope);
    envelope->answer = bridge_strdup(text);
    if (!envelope->answer) return NIYAH_ERR_OUT_OF_MEMORY;
    if (source_ids && n_source_ids > 0) {
        envelope->source_ids = (char **)calloc((size_t)n_source_ids, sizeof(char *));
        if (!envelope->source_ids) { free(envelope->answer); envelope->answer = NULL; return NIYAH_ERR_OUT_OF_MEMORY; }
        for (i = 0; i < n_source_ids; ++i) {
            envelope->source_ids[i] = bridge_strdup(source_ids[i]);
            if (!envelope->source_ids[i]) {
                for (int32_t j = 0; j < i; ++j) free(envelope->source_ids[j]);
                free(envelope->source_ids); free(envelope->answer);
                envelope->source_ids = NULL; envelope->answer = NULL;
                return NIYAH_ERR_OUT_OF_MEMORY;
            }
        }
        envelope->n_source_ids = n_source_ids;
    }
    /* Heuristic label detection (keyword scan). */
    if (strstr(text, "FACT") || strstr(text, "\xd8\xad\xd9\x82\xd9\x8a\xd9\x82\xd8\xa9")) {
        envelope->label = NIYAH_MINI_EVIDENCE_FACT;
        envelope->lvu_agreement = 1.0f;
        envelope->peer_prediction_consistent = true;
    } else if (strstr(text, "INFERENCE") || strstr(text, "\xd8\xa7\xd8\xb3\xd8\xaa\xd8\xaf\xd9\x84\xd8\xa7\xd9\x84")) {
        envelope->label = NIYAH_MINI_EVIDENCE_INFERENCE;
        envelope->lvu_agreement = 0.8f;
        envelope->peer_prediction_consistent = true;
    } else if (strstr(text, "CONFLICTED") || strstr(text, "\xd9\x85\xd8\xaa\xd8\xb6\xd8\xa7\xd8\xb1\xd8\xa8")) {
        envelope->label = NIYAH_MINI_EVIDENCE_CONFLICTED;
        envelope->lvu_agreement = 0.5f;
        envelope->peer_prediction_consistent = false;
    } else {
        envelope->label = NIYAH_MINI_EVIDENCE_UNKNOWN;
        envelope->lvu_agreement = 0.0f;
        envelope->peer_prediction_consistent = false;
    }
    return NIYAH_OK;
}

/* ==========================================================================
 * Integration with Niyah.Engine
 * ========================================================================== */

/*
 * Stub fix #3 -- niyah_mini_to_niyah_evidence
 *
 * Was: *niyah_evidence_out = NULL; return OK -- a complete no-op.
 *
 * Now: heap-allocates a NiyahTruth value and maps the evidence label:
 *   FACT       -> NIYAH_TRUE    (verified, high confidence)
 *   INFERENCE  -> NIYAH_UNKNOWN (derived, not confirmed)
 *   UNKNOWN    -> NIYAH_UNKNOWN (no information)
 *   CONFLICTED -> NIYAH_FALSE   (contradicting signals; cannot assert truth)
 *
 * Caller receives ownership and must free() the returned pointer.
 */
NiyahStatus niyah_mini_to_niyah_evidence(
    void **niyah_evidence_out,
    const NiyahMiniEvidenceEnvelope *mini_envelope)
{
    NiyahTruth *truth;
    if (!niyah_evidence_out || !mini_envelope) return NIYAH_ERR_INVALID_ARG;
    truth = (NiyahTruth *)malloc(sizeof(NiyahTruth));
    if (!truth) return NIYAH_ERR_OUT_OF_MEMORY;
    switch (mini_envelope->label) {
        case NIYAH_MINI_EVIDENCE_FACT:
            *truth = NIYAH_TRUE;    break;
        case NIYAH_MINI_EVIDENCE_CONFLICTED:
            *truth = NIYAH_FALSE;   break;
        case NIYAH_MINI_EVIDENCE_INFERENCE:
        case NIYAH_MINI_EVIDENCE_UNKNOWN:
        default:
            *truth = NIYAH_UNKNOWN; break;
    }
    *niyah_evidence_out = truth;
    return NIYAH_OK;
}

/*
 * Stub fix #4 -- niyah_mini_attach_evidence
 *
 * Was: bare return OK placeholder.
 *
 * Now: appends a compact evidence tag to output->text:
 *   "<original answer> [FACT|lvu=1.00|peer=1]"
 * The existing string is reallocated; ownership stays with output->text.
 * Gracefully handles output->text == NULL by creating a tag-only string.
 */
NiyahStatus niyah_mini_attach_evidence(
    NiyahLLMOutput *output,
    const NiyahMiniEvidenceEnvelope *envelope)
{
    static const char *const label_names[] = {
        "FACT", "INFERENCE", "UNKNOWN", "CONFLICTED"
    };
    const char *label_str;
    char tag[64];
    size_t tag_len, base_len, total;
    char *merged;

    if (!output || !envelope) return NIYAH_ERR_INVALID_ARG;

    label_str = (envelope->label <= NIYAH_MINI_EVIDENCE_CONFLICTED)
                ? label_names[envelope->label] : "UNKNOWN";

    /* Build tag: " [LABEL|lvu=X.XX|peer=P]" */
    snprintf(tag, sizeof(tag), " [%s|lvu=%.2f|peer=%d]",
             label_str,
             (double)envelope->lvu_agreement,
             envelope->peer_prediction_consistent ? 1 : 0);
    tag_len  = strlen(tag);
    base_len = output->text ? strlen(output->text) : 0U;

    /* Check overflow before adding 1 for NUL. */
    if (tag_len > SIZE_MAX - base_len - 1U) return NIYAH_ERR_OVERFLOW;
    total = base_len + tag_len + 1U;

    merged = (char *)realloc(output->text, total);
    if (!merged) return NIYAH_ERR_OUT_OF_MEMORY;

    if (base_len == 0U) merged[0] = '\0';
    memcpy(merged + base_len, tag, tag_len + 1U);   /* includes NUL */

    output->text = merged;
    return NIYAH_OK;
}
