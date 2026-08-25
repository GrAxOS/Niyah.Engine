#ifndef NIYAH_EVIDENCE_ENVELOPE_H
#define NIYAH_EVIDENCE_ENVELOPE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Single definition of the evidence export macro. evidence_graph.h and
 * evidence_reasoner.h include this header, so the guard keeps the macro from
 * being redefined in a translation unit that pulls in more than one of them.
 * Define NIYAH_BRIDGE_EXPORTS when compiling the library itself.
 */
#ifndef NIYAH_EVIDENCE_API
    #ifdef _WIN32
        #ifdef NIYAH_BRIDGE_EXPORTS
            #define NIYAH_EVIDENCE_API __declspec(dllexport)
        #else
            #define NIYAH_EVIDENCE_API __declspec(dllimport)
        #endif
    #else
        #define NIYAH_EVIDENCE_API __attribute__((visibility("default")))
    #endif
#endif

typedef enum {
    NIYAH_EVIDENCE_ENVELOPE_OK = 0,
    NIYAH_EVIDENCE_ENVELOPE_ERROR = 1,
    NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY = 2,
    NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS = 3
} NiyahEvidenceEnvelopeStatus;

typedef struct NiyahEvidenceEnvelope {
    char source_id[256];
    char evidence_type[64];
    uint8_t content_hash[64];
    size_t content_hash_size;
    uint64_t created_at;
    uint64_t verified_at;
    bool verified;
    float confidence;
} NiyahEvidenceEnvelope;

/* ============================================================================
 * Envelope lifecycle
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_create(
    NiyahEvidenceEnvelope **out,
    const char *source_id,
    const char *evidence_type,
    const uint8_t *content_hash,
    size_t content_hash_size);

NIYAH_EVIDENCE_API void niyah_evidence_envelope_destroy(
    NiyahEvidenceEnvelope *env);

/* ============================================================================
 * Verification
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_verify(
    NiyahEvidenceEnvelope *env,
    float confidence);

/* ============================================================================
 * Serialization (for audit log)
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_serialize(
    const NiyahEvidenceEnvelope *env,
    char *buffer,
    size_t buffer_size,
    size_t *out_size);

#endif /* NIYAH_EVIDENCE_ENVELOPE_H */
