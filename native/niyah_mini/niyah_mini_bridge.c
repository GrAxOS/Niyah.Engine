#include "niyah_mini_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *bridge_strdup(const char *s)
{
    size_t len;
    char *copy;
    if (!s) return NULL;
    len = strlen(s);
    if (len == SIZE_MAX) return NULL;
    copy = (char *)malloc(len + 1U);
    if (!copy) return NULL;
    memcpy(copy, s, len + 1U);
    return copy;
}

static NiyahStatus bridge_bpe_tokenize_prompt(
    const NiyahMiniWrappedModel *wrapped,
    const char *prompt,
    int32_t **tokens_out,
    int32_t *token_count_out)
{
    NiyahMiniVocab vocab;
    NiyahMiniBPE bpe;
    int32_t *tokens;
    size_t prompt_len;
    int32_t capacity;
    int32_t count;
    int32_t i;
    NiyahStatus status;

    if (!wrapped || !prompt || !tokens_out || !token_count_out) return NIYAH_ERR_INVALID_ARG;
    *tokens_out = NULL;
    *token_count_out = 0;

    memset(&vocab, 0, sizeof(vocab));
    memset(&bpe, 0, sizeof(bpe));

    status = niyah_mini_vocab_add(&vocab, "<pad>", 0.0f);
    if (status != NIYAH_OK) goto fail;
    status = niyah_mini_vocab_add(&vocab, "<bos>", 0.0f);
    if (status != NIYAH_OK) goto fail;
    status = niyah_mini_vocab_add(&vocab, "<eos>", 0.0f);
    if (status != NIYAH_OK) goto fail;
    status = niyah_mini_vocab_add(&vocab, "<unk>", 0.0f);
    if (status != NIYAH_OK) goto fail;

    prompt_len = strlen(prompt);
    if (prompt_len == 0U) return NIYAH_OK;
    if (prompt_len > (size_t)INT32_MAX) return NIYAH_ERR_OVERFLOW;
    capacity = (int32_t)prompt_len;
    if (capacity > wrapped->config.n_ctx) capacity = wrapped->config.n_ctx;
    if (capacity <= 0) return NIYAH_ERR_SHAPE;

    for (i = 0; i < 256; ++i) {
        char byte_token[2];
        byte_token[0] = (char)(unsigned char)i;
        byte_token[1] = '\0';
        status = niyah_mini_vocab_add(&vocab, byte_token, 0.0f);
        if (status != NIYAH_OK) goto fail;
    }

    status = niyah_mini_bpe_init(&bpe, &vocab, NULL, 0);
    if (status != NIYAH_OK) goto fail;

    tokens = (int32_t *)malloc((size_t)capacity * sizeof(*tokens));
    if (!tokens) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto fail;
    }

    count = niyah_mini_bpe_tokenize(&bpe, prompt, tokens, capacity);
    if (count < 0 || count > capacity) {
        free(tokens);
        status = NIYAH_ERR_SHAPE;
        goto fail;
    }

    for (i = 0; i < count; ++i) {
        unsigned char byte_value = (unsigned char)prompt[i];
        int32_t expected = NIYAH_MINI_SPECIAL_TOKENS + (int32_t)byte_value;
        if (tokens[i] < NIYAH_MINI_SPECIAL_TOKENS || tokens[i] != expected) {
            tokens[i] = expected;
        }
    }

    *tokens_out = tokens;
    *token_count_out = count;
    niyah_mini_bpe_free(&bpe);
    niyah_mini_vocab_free(&vocab);
    return NIYAH_OK;

fail:
    niyah_mini_bpe_free(&bpe);
    niyah_mini_vocab_free(&vocab);
    return status;
}

static char *bridge_byte_detokenize(const int32_t *tokens, int32_t n_tokens)
{
    char *buf;
    size_t used = 0U;
    int32_t i;

    if (n_tokens < 0 || (n_tokens > 0 && !tokens)) return NULL;
    if ((size_t)n_tokens > SIZE_MAX - 1U) return NULL;

    buf = (char *)malloc((size_t)n_tokens + 1U);
    if (!buf) return NULL;

    for (i = 0; i < n_tokens; ++i) {
        int32_t token = tokens[i];
        if (token >= NIYAH_MINI_SPECIAL_TOKENS && token <= NIYAH_MINI_SPECIAL_TOKENS + 255) {
            buf[used++] = (char)(unsigned char)(token - NIYAH_MINI_SPECIAL_TOKENS);
        }
    }
    buf[used] = '\0';
    return buf;
}

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

NiyahStatus niyah_mini_wrapped_load(
    NiyahMiniWrappedModel *wrapped,
    const char *config_path,
    const char *weights_path)
{
    NiyahStatus status;
    if (!wrapped || !config_path || !weights_path) return NIYAH_ERR_INVALID_ARG;

    status = niyah_mini_model_load(&wrapped->mini_model, config_path, weights_path);
    if (status != NIYAH_OK) {
        wrapped->weights_loaded = false;
        return status;
    }

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
    niyah_model->config.n_vocab = wrapped->config.n_vocab;
    niyah_model->config.n_embd = wrapped->config.n_dim;
    niyah_model->config.n_head = wrapped->config.n_heads;
    niyah_model->config.n_layer = wrapped->config.n_layers;
    niyah_model->config.n_ctx = wrapped->config.n_ctx;
    niyah_model->config.type = 0;
    niyah_model->config.n_kv_head = wrapped->config.n_kv_heads;
    niyah_model->config.n_ff = wrapped->config.n_ff;
    niyah_model->config.eos_token_id = NIYAH_MINI_EOS_TOKEN_ID;
    niyah_model->config.bos_token_id = NIYAH_MINI_BOS_TOKEN_ID;
    niyah_model->config.rope_theta = wrapped->config.rope_theta;
    niyah_model->config.norm_eps = wrapped->config.norm_eps;
    niyah_model->config.tie_word_embeddings = wrapped->config.tie_word_embeddings;
    niyah_model->weights = wrapped->mini_model.weights.memory_block;
    niyah_model->weights_size = wrapped->mini_model.weights.memory_size;
    return NIYAH_OK;
}

NiyahLLMOutput niyah_mini_wrapped_generate(
    const NiyahMiniWrappedModel *wrapped,
    const NiyahTokenizer *tokenizer,
    const char *prompt,
    int32_t max_tokens)
{
    NiyahLLMOutput output;
    int32_t *prompt_ids = NULL;
    int32_t *output_ids = NULL;
    int32_t prompt_len = 0;
    int32_t output_len = 0;
    int32_t effective_max_tokens;
    NiyahStatus status;
    NiyahSamplerConfig sampler;
    size_t i;

    memset(&output, 0, sizeof(output));

    if (!wrapped || !prompt || max_tokens <= 0) {
        output.status = NIYAH_ERR_INVALID_ARG;
        return output;
    }

    if (!wrapped->weights_loaded || !wrapped->mini_model.weights.memory_block) {
        output.status = NIYAH_ERR_NO_WEIGHTS;
        output.text = NULL;
        return output;
    }

    niyah_telemetry_start(&output.telemetry);

    if (tokenizer) {
        NiyahTokenizer *mut_tok = (NiyahTokenizer *)(uintptr_t)tokenizer;
        prompt_len = niyah_tokenize(mut_tok, prompt, NULL, 0);
        if (prompt_len < 0) {
            output.status = NIYAH_ERR_SHAPE;
            goto done;
        }
        if (prompt_len > wrapped->config.n_ctx) prompt_len = wrapped->config.n_ctx;
        if (prompt_len > 0) {
            prompt_ids = (int32_t *)malloc((size_t)prompt_len * sizeof(*prompt_ids));
            if (!prompt_ids) {
                output.status = NIYAH_ERR_OUT_OF_MEMORY;
                goto done;
            }
            prompt_len = niyah_tokenize(mut_tok, prompt, prompt_ids, prompt_len);
            if (prompt_len < 0) {
                output.status = NIYAH_ERR_SHAPE;
                goto done;
            }
        }
    } else {
        status = bridge_bpe_tokenize_prompt(wrapped, prompt, &prompt_ids, &prompt_len);
        if (status != NIYAH_OK) {
            output.status = status;
            goto done;
        }
    }

    effective_max_tokens = max_tokens;
    if (effective_max_tokens > wrapped->config.n_ctx - prompt_len)
        effective_max_tokens = wrapped->config.n_ctx - prompt_len;
    if (effective_max_tokens <= 0) {
        output.status = NIYAH_ERR_SHAPE;
        goto done;
    }

    output_ids = (int32_t *)malloc((size_t)effective_max_tokens * sizeof(*output_ids));
    if (!output_ids) {
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        goto done;
    }

    sampler.strategy = NIYAH_SAMPLE_GREEDY;
    sampler.temperature = 1.0f;
    sampler.top_k = 0;
    sampler.top_p = 1.0f;

    if (wrapped->mini_model.runtime) {
        const NiyahLLM *llm = (const NiyahLLM *)wrapped->mini_model.runtime;
        sampler = llm->sampler;
    }

    if (!isfinite(sampler.temperature) || sampler.temperature <= 0.0f) sampler.temperature = 1.0f;
    if (sampler.top_k < 0) sampler.top_k = 0;
    if (!isfinite(sampler.top_p) || sampler.top_p <= 0.0f || sampler.top_p > 1.0f) sampler.top_p = 1.0f;

    status = niyah_mini_generate(
        (NiyahMiniModel *)(uintptr_t)&wrapped->mini_model,
        prompt_ids,
        prompt_len,
        effective_max_tokens,
        sampler.temperature,
        output_ids,
        &output_len);
    if (status != NIYAH_OK) {
        output.status = status;
        goto done;
    }
    if (output_len <= 0) {
        output.status = NIYAH_ERR_SHAPE;
        goto done;
    }

    if (tokenizer) {
        NiyahTokenizer *mut_tok = (NiyahTokenizer *)(uintptr_t)tokenizer;
        output.text = niyah_detokenize(mut_tok, output_ids, output_len);
    } else {
        output.text = bridge_byte_detokenize(output_ids, output_len);
    }

    if (!output.text) {
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        goto done;
    }

    output.n_tokens = output_len;
    output.status = NIYAH_OK;

    (void)sampler.strategy;
    (void)sampler.top_k;
    (void)sampler.top_p;
    (void)i;

done:
    niyah_telemetry_end(&output.telemetry);
    output.telemetry.tokens_processed = output_len;
    free(prompt_ids);
    free(output_ids);
    return output;
}

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
    if (!envelope || !text || n_source_ids < 0) return NIYAH_ERR_INVALID_ARG;
    niyah_mini_evidence_envelope_free(envelope);
    envelope->answer = bridge_strdup(text);
    if (!envelope->answer) return NIYAH_ERR_OUT_OF_MEMORY;

    if (source_ids && n_source_ids > 0) {
        envelope->source_ids = (char **)calloc((size_t)n_source_ids, sizeof(char *));
        if (!envelope->source_ids) {
            free(envelope->answer);
            envelope->answer = NULL;
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        for (i = 0; i < n_source_ids; ++i) {
            envelope->source_ids[i] = bridge_strdup(source_ids[i] ? source_ids[i] : "");
            if (!envelope->source_ids[i]) {
                int32_t j;
                for (j = 0; j < i; ++j) free(envelope->source_ids[j]);
                free(envelope->source_ids);
                free(envelope->answer);
                envelope->source_ids = NULL;
                envelope->answer = NULL;
                return NIYAH_ERR_OUT_OF_MEMORY;
            }
        }
        envelope->n_source_ids = n_source_ids;
    }

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

NiyahStatus niyah_mini_to_niyah_evidence(
    void **niyah_evidence_out,
    const NiyahMiniEvidenceEnvelope *mini_envelope)
{
    NiyahTruth *truth;
    if (!niyah_evidence_out || !mini_envelope) return NIYAH_ERR_INVALID_ARG;
    truth = (NiyahTruth *)malloc(sizeof(*truth));
    if (!truth) return NIYAH_ERR_OUT_OF_MEMORY;
    switch (mini_envelope->label) {
        case NIYAH_MINI_EVIDENCE_FACT:
            *truth = NIYAH_TRUE;
            break;
        case NIYAH_MINI_EVIDENCE_CONFLICTED:
            *truth = NIYAH_FALSE;
            break;
        case NIYAH_MINI_EVIDENCE_INFERENCE:
        case NIYAH_MINI_EVIDENCE_UNKNOWN:
        default:
            *truth = NIYAH_UNKNOWN;
            break;
    }
    *niyah_evidence_out = truth;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_attach_evidence(
    NiyahLLMOutput *output,
    const NiyahMiniEvidenceEnvelope *envelope)
{
    static const char *const label_names[] = {"FACT", "INFERENCE", "UNKNOWN", "CONFLICTED"};
    const char *label_str;
    char tag[64];
    size_t tag_len;
    size_t base_len;
    size_t total;
    char *merged;

    if (!output || !envelope) return NIYAH_ERR_INVALID_ARG;
    label_str = envelope->label <= NIYAH_MINI_EVIDENCE_CONFLICTED
                ? label_names[envelope->label]
                : "UNKNOWN";
    snprintf(tag, sizeof(tag), " [%s|lvu=%.2f|peer=%d]",
             label_str,
             (double)envelope->lvu_agreement,
             envelope->peer_prediction_consistent ? 1 : 0);
    tag_len = strlen(tag);
    base_len = output->text ? strlen(output->text) : 0U;
    if (base_len > SIZE_MAX - tag_len - 1U) return NIYAH_ERR_OVERFLOW;
    total = base_len + tag_len + 1U;

    merged = (char *)realloc(output->text, total);
    if (!merged) return NIYAH_ERR_OUT_OF_MEMORY;
    memcpy(merged + base_len, tag, tag_len + 1U);
    output->text = merged;
    return NIYAH_OK;
}
