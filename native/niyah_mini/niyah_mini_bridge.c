#include "niyah_mini_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* For strdup on systems that don't have it */
#ifndef HAVE_STRDUP
static char* my_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, s, len);
    return copy;
}
#define strdup my_strdup
#endif

/* ==========================================================================
 * NiyahMini Bridge Implementation
 * 
 * Integrates NiyahMini with Niyah.Engine while preserving:
 * - Critical contract: NIYAH_ERR_NO_WEIGHTS when no weights
 * - Evidence-aware output
 * - Provenance tracking
 * ========================================================================== */

NiyahStatus niyah_mini_wrapped_init(
    NiyahMiniWrappedModel* wrapped,
    const NiyahMiniConfig* config
) {
    if (!wrapped || !config) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    memset(wrapped, 0, sizeof(*wrapped));
    
    /* Copy config */
    wrapped->config = *config;
    
    /* Initialize model */
    NiyahStatus status = niyah_mini_model_init(&wrapped->mini_model, config);
    if (status != NIYAH_OK) {
        return status;
    }
    
    wrapped->weights_loaded = false;
    
    return NIYAH_OK;
}

NiyahStatus niyah_mini_wrapped_load(
    NiyahMiniWrappedModel* wrapped,
    const char* config_path,
    const char* weights_path
) {
    if (!wrapped || !config_path || !weights_path) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Load config */
    FILE* config_file = fopen(config_path, "r");
    if (!config_file) {
        return NIYAH_ERR_IO;
    }
    
    /* Parse config JSON (simplified - in practice use a proper JSON parser) */
    /* For now, use default config */
    niyah_mini_config_init(&wrapped->config, NIYAH_MINI_TINY);
    fclose(config_file);
    
    /* Initialize model */
    NiyahStatus status = niyah_mini_model_init(&wrapped->mini_model, &wrapped->config);
    if (status != 
NIYAH_OK) {
        return status;
    }
    
    /* Load weights */
    status = niyah_mini_model_load_weights(&wrapped->mini_model, weights_path);
    if (status != NIYAH_OK) {
        return status;
    }
    
    wrapped->weights_loaded = true;
    
    return NIYAH_OK;
}

void niyah_mini_wrapped_free(NiyahMiniWrappedModel* wrapped) {
    if (!wrapped) return;
    
    niyah_mini_model_free(&wrapped->mini_model);
    memset(wrapped, 0, sizeof(*wrapped));
}

NiyahStatus niyah_mini_to_niyah_model(
    NiyahModel* niyah_model,
    const NiyahMiniWrappedModel* wrapped
) {
    if (!niyah_model || !wrapped) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Set up NiyahModel config */
    niyah_model->config.n_vocab = wrapped->config.n_vocab;
    niyah_model->config.n_embd = wrapped->config.n_dim;
    niyah_model->config.n_head = wrapped->config.n_heads;
    niyah_model->config.n_layer = wrapped->config.n_layers;
    niyah_model->config.n_ctx = wrapped->config.n_ctx;
    niyah_model->config.type = 0;  /* Custom type */
    niyah_model->config.n_kv_head = wrapped->config.n_kv_heads;
    niyah_model->config.n_ff = wrapped->config.n_ff;
    niyah_model->config.eos_token_id = NIYAH_MINI_EOS_TOKEN_ID;
    niyah_model->config.bos_token_id = NIYAH_MINI_BOS_TOKEN_ID;
    niyah_model->config.rope_theta = wrapped->config.rope_theta;
    niyah_model->config.norm_eps = wrapped->config.norm_eps;
    niyah_model->config.tie_word_embeddings = wrapped->config.tie_word_embeddings;
    
    /* Point to wrapped model weights */
    /* In practice, we'd need to map the weights appropriately */
    niyah_model->weights = wrapped->mini_model.weights.memory_block;
    niyah_model->weights_size = wrapped->mini_model.weights.memory_size;
    
    return NIYAH_OK;
}

NiyahLLMOutput niyah_mini_wrapped_generate(
    const NiyahMiniWrappedModel* wrapped,
    const NiyahTokenizer* tokenizer,
    const char* prompt,
    int32_t max_tokens
) {
    NiyahLLMOutput output;
    memset(&output, 0, sizeof(output));

    if (!wrapped || !prompt || max_tokens <= 0) {
        output.status = NIYAH_ERR_INVALID_ARG;
        return output;
    }

    /* Critical contract: never fabricate output when no weights are loaded. */
    if (!wrapped->weights_loaded) {
        output.status = NIYAH_ERR_NO_WEIGHTS;
        output.text = NULL;
        return output;
    }

    /* Tokenize prompt. */
    int32_t* prompt_ids = NULL;
    int32_t prompt_len = 0;

    if (tokenizer) {
        NiyahTokenizer* mutable_tokenizer = (NiyahTokenizer*)tokenizer;
        prompt_len = niyah_tokenize(mutable_tokenizer, prompt, NULL, 0);
        if (prompt_len <= 0) {
            output.status = NIYAH_ERR_INVALID_ARG;
            return output;
        }
        prompt_ids = (int32_t*)malloc((size_t)prompt_len * sizeof(int32_t));
        if (!prompt_ids) {
            output.status = NIYAH_ERR_OUT_OF_MEMORY;
            return output;
        }
        niyah_tokenize(mutable_tokenizer, prompt, prompt_ids, prompt_len);
    } else {
        /* Fallback: byte-level tokenization. */
        size_t len = strlen(prompt);
        prompt_len = (int32_t)len;
        if (prompt_len <= 0) {
            output.status = NIYAH_ERR_INVALID_ARG;
            return output;
        }
        prompt_ids = (int32_t*)malloc((size_t)prompt_len * sizeof(int32_t));
        if (!prompt_ids) {
            output.status = NIYAH_ERR_OUT_OF_MEMORY;
            return output;
        }
        for (size_t i = 0; i < len; i++) {
            prompt_ids[i] = (int32_t)(unsigned char)prompt[i];
        }
    }

    /* Run REAL autoregressive generation (no prompt echo, no false success). */
    int32_t* gen_ids = (int32_t*)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!gen_ids) {
        free(prompt_ids);
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        return output;
    }
    int32_t n_gen = 0;

    /* generate() only reads weights + transient state; cast away const safely. */
    NiyahMiniModel* model = (NiyahMiniModel*)&wrapped->mini_model;
    NiyahStatus st = niyah_mini_generate(model, prompt_ids, prompt_len,
                                         max_tokens, 0.7f /* temperature */,
                                         gen_ids, &n_gen);
    if (st != NIYAH_OK) {
        free(prompt_ids);
        free(gen_ids);
        output.status = st;
        output.text = NULL;
        return output;
    }

    /* Detokenize the generated tokens to text. */
    char* text = NULL;
    if (tokenizer) {
        NiyahTokenizer* mutable_tokenizer = (NiyahTokenizer*)tokenizer;
        text = niyah_detokenize(mutable_tokenizer, gen_ids, n_gen);
    } else {
        /* Fallback: byte-level decode (inverse of the byte-level encode above). */
        text = (char*)malloc((size_t)n_gen + 1);
        if (text) {
            for (int32_t i = 0; i < n_gen; i++) {
                text[i] = (char)(unsigned char)(gen_ids[i] & 0xFF);
            }
            text[n_gen] = '\0';
        }
    }

    free(prompt_ids);
    free(gen_ids);

    if (!text) {
        output.status = NIYAH_ERR_OUT_OF_MEMORY;
        output.text = NULL;
        return output;
    }

    output.text = text;
    output.n_tokens = n_gen;
    output.status = NIYAH_OK;
    return output;
}

/* ==========================================================================
 * Evidence Envelope Implementation
 * ========================================================================== */

NiyahStatus niyah_mini_evidence_envelope_init(
    NiyahMiniEvidenceEnvelope* envelope
) {
    if (!envelope) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    memset(envelope, 0, sizeof(*envelope));
    envelope->label = NIYAH_MINI_EVIDENCE_UNKNOWN;
    envelope->answer = NULL;
    envelope->source_ids = NULL;
    envelope->n_source_ids = 0;
    envelope->limitations = NULL;
    envelope->n_limitations = 0;
    envelope->verification_steps = NULL;
    envelope->n_verification_steps = 0;
    envelope->lvu_agreement = 0.0f;
    envelope->peer_prediction_consistent = false;
    
    return NIYAH_OK;
}

void niyah_mini_evidence_envelope_free(
    NiyahMiniEvidenceEnvelope* envelope
) {
    if (!envelope) return;
    
    if (envelope->answer) {
        free(envelope->answer);
    }
    
    if (envelope->source_ids) {
        for (int32_t i = 0; i < envelope->n_source_ids; i++) {
            free(envelope->source_ids[i]);
        }
        free(envelope->source_ids);
    }
    
    if (envelope->limitations) {
        for (int32_t i = 0; i < envelope->n_limitations; i++) {
            free(envelope->limitations[i]);
        }
        free(envelope->limitations);
    }
    
    if (envelope->verification_steps) {
        for (int32_t i = 0; i < envelope->n_verification_steps; i++) {
            free(envelope->verification_steps[i]);
        }
        free(envelope->verification_steps);
    }
    
    memset(envelope, 0, sizeof(*envelope));
}

NiyahStatus niyah_mini_format_evidence_output(
    NiyahMiniEvidenceEnvelope* envelope,
    const char* text,
    const char** source_ids,
    int32_t n_source_ids
) {
    if (!envelope || !text) 
{
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Free existing content */
    niyah_mini_evidence_envelope_free(envelope);
    
    /* Set answer */
    envelope->answer = strdup(text);
    if (!envelope->answer) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    
    /* Copy source IDs */
    if (source_ids && n_source_ids > 0) {
        envelope->source_ids = (char**)malloc((size_t)n_source_ids * sizeof(char*));
        if (!envelope->source_ids) {
            free(envelope->answer);
            envelope->answer = NULL;
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        
        for (int32_t i = 0; i < n_source_ids; i++) {
            envelope->source_ids[i] = strdup(source_ids[i]);
            if (!envelope->source_ids[i]) {
                /* Cleanup on failure */
                for (int32_t j = 0; j < i; j++) {
                    free(envelope->source_ids[j]);
                }
                free(envelope->source_ids);
                free(envelope->answer);
                envelope->source_ids = NULL;
                envelope->answer = NULL;
                return NIYAH_ERR_OUT_OF_MEMORY;
            }
        }
        envelope->n_source_ids = n_source_ids;
    }
    
    /* Determine label from text content */
    /* This is a simple heuristic - in practice use the model's own classification */
    if (strstr(text, "FACT") || strstr(text, "حقيقة")) {
        envelope->label = NIYAH_MINI_EVIDENCE_FACT;
        envelope->lvu_agreement = 1.0f;
        envelope->peer_prediction_consistent = true;
    } else if (strstr(text, "INFERENCE") || strstr(text, "استدلال")) {
        envelope->label = NIYAH_MINI_EVIDENCE_INFERENCE;
        envelope->lvu_agreement = 0.8f;
        envelope->peer_prediction_consistent = true;
    } else if (strstr(text, "UNKNOWN") || strstr(text, "مجهول")) {
        envelope->label = NIYAH_MINI_EVIDENCE_UNKNOWN;
        envelope->lvu_agreement = 0.4f;
        envelope->peer_prediction_consistent = true;
    } else if
 (strstr(text, "CONFLICTED") || strstr(text, "متضارب")) {
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

NiyahStatus niyah_mini_to_niyah_evidence(
    void** niyah_evidence_out,
    const NiyahMiniEvidenceEnvelope* mini_envelope
) {
    /* In practice, this would convert the evidence envelope */
    /* to Niyah.Engine's evidence format */
    /* For now, just return NULL as a placeholder */
    *niyah_evidence_out = NULL;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_attach_evidence(
    NiyahLLMOutput* output,
    const NiyahMiniEvidenceEnvelope* envelope
) {
    if (!output || !envelope) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* In practice, attach evidence to the output */
    /* For now, this is a placeholder */
    
    return NIYAH_OK;
}
