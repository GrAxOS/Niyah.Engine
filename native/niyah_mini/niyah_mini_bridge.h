#ifndef NIYAH_MINI_BRIDGE_H
#define NIYAH_MINI_BRIDGE_H

#include "../niyah.h"
#include "niyah_mini_model.h"
#include "niyah_mini_vocab.h"

/* ==========================================================================
 * NiyahMini Bridge - Integration with Niyah.Engine
 * 
 * This bridge allows NiyahMini to work with the existing Niyah.Engine
 * infrastructure while maintaining the critical contracts:
 * - NIYAH_ERR_NO_WEIGHTS when no weights loaded
 * - text = NULL when no output should be generated
 * - Evidence-aware output formatting
 * 
 * NO modification to the critical contracts of Niyah.Engine.
 * ========================================================================== */

/* Special token IDs */
#define NIYAH_MINI_PAD_TOKEN_ID      0
#define NIYAH_MINI_BOS_TOKEN_ID      1
#define NIYAH_MINI_EOS_TOKEN_ID      2
#define NIYAH_MINI_UNK_TOKEN_ID      3

/* NiyahMini model wrapped to match NiyahModel interface */
typedef struct {
    NiyahMiniModel mini_model;
    NiyahMiniConfig config;
    bool weights_loaded;
} NiyahMiniWrappedModel;

/* Initialize wrapped model */
NIYAH_API NiyahStatus niyah_mini_wrapped_init(
    NiyahMiniWrappedModel* wrapped,
    const NiyahMiniConfig* config
);

/* Load wrapped model from files */
NIYAH_API NiyahStatus niyah_mini_wrapped_load(
    NiyahMiniWrappedModel* wrapped,
    const char* config_path,
    const char* weights_path
);

/* Free wrapped model */
NIYAH_API void niyah_mini_wrapped_free(NiyahMiniWrappedModel* wrapped);

/* Convert wrapped model to NiyahModel for use with existing API */
NIYAH_API NiyahStatus niyah_mini_to_niyah_model(
    NiyahModel* niyah_model,
    const NiyahMiniWrappedModel* wrapped
);

/* Generate text using wrapped model (matches niyah_llm_generate contract) */
NIYAH_API NiyahLLMOutput niyah_mini_wrapped_generate(
    const NiyahMiniWrappedModel* wrapped,
    const NiyahTokenizer* tokenizer,
    const char* prompt,
    int32_t max_tokens
);

/* ==========================================================================
 * Evidence-Aware Generation
 * ========================================================================== */

/* Evidence label types */
typedef enum {
    NIYAH_MINI_EVIDENCE_FACT = 0,
    NIYAH_MINI_EVIDENCE_INFERENCE = 1,
    NIYAH_MINI_EVIDENCE_UNKNOWN = 2,
    NIYAH_MINI_EVIDENCE_CONFLICTED = 3
} NiyahMiniEvidenceLabel;

/* Evidence envelope for NiyahMini output */
typedef struct {
    NiyahMiniEvidenceLabel label;
    char* answer;
    char** source_ids;
    int32_t n_source_ids;
    char** limitations;
    int32_t n_limitations;
    char** verification_steps;
    int32_t n_verification_steps;
    float lvu_agreement;
    bool peer_prediction_consistent;
} NiyahMiniEvidenceEnvelope;

/* Initialize evidence envelope */
NIYAH_API NiyahStatus niyah_mini_evidence_envelope_init(
    NiyahMiniEvidenceEnvelope* envelope
);

/* Free evidence envelope */
NIYAH_API void niyah_mini_evidence_envelope_free(
    NiyahMiniEvidenceEnvelope* envelope
);

/* Format output with evidence labels */
NIYAH_API NiyahStatus niyah_mini_format_evidence_output(
    NiyahMiniEvidenceEnvelope* envelope,
    const char* text,
    const char** source_ids,
    int32_t n_source_ids
);

/* ==========================================================================
 * Integration with Niyah.Engine's Evidence System
 * ========================================================================== */

/* Convert NiyahMini evidence to Niyah evidence */
NIYAH_API NiyahStatus niyah_mini_to_niyah_evidence(
    void** niyah_evidence_out,
    const NiyahMiniEvidenceEnvelope* mini_envelope
);

/* Attach evidence to NiyahLLMOutput */
NIYAH_API NiyahStatus niyah_mini_attach_evidence(
    NiyahLLMOutput* output,
    const NiyahMiniEvidenceEnvelope* envelope
);

#endif /* NIYAH_MINI_BRIDGE_H */
