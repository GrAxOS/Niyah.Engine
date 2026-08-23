#include "evidence_envelope.h"
#include "niyah_core.h"

#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Evidence envelope lifecycle
 * ============================================================================ */

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_create(
    NiyahEvidenceEnvelope **out,
    const char *source_id,
    const char *evidence_type,
    const uint8_t *content_hash,
    size_t content_hash_size)
{
    if (!out || !source_id || !evidence_type || !content_hash) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    NiyahEvidenceEnvelope *env = calloc(1, sizeof(NiyahEvidenceEnvelope));
    if (!env) {
        return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;
    }

    /* Copy source ID */
    size_t source_id_len = strlen(source_id);
    if (source_id_len >= sizeof(env->source_id)) {
        free(env);
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    memcpy(env->source_id, source_id, source_id_len + 1);

    /* Copy evidence type */
    size_t evidence_type_len = strlen(evidence_type);
    if (evidence_type_len >= sizeof(env->evidence_type)) {
        free(env);
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    memcpy(env->evidence_type, evidence_type, evidence_type_len + 1);

    /* Copy content hash */
    if (content_hash_size > sizeof(env->content_hash)) {
        free(env);
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    memcpy(env->content_hash, content_hash, content_hash_size);
    env->content_hash_size = content_hash_size;

    env->created_at = niyah_core_timestamp_now();
    env->verified = false;
    env->confidence = 0.0f;

    *out = env;
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

void niyah_evidence_envelope_destroy(NiyahEvidenceEnvelope *env) {
    if (!env) {
        return;
    }
    free(env);
}

/* ============================================================================
 * Verification
 * ============================================================================ */

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_verify(
    NiyahEvidenceEnvelope *env,
    float confidence)
{
    if (!env) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    env->verified = true;
    env->confidence = confidence;
    env->verified_at = niyah_core_timestamp_now();

    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

/* ============================================================================
 * Serialization (for audit log)
 * ============================================================================ */

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_serialize(
    const NiyahEvidenceEnvelope *env,
    char *buffer,
    size_t buffer_size,
    size_t *out_size)
{
    if (!env || !buffer || !out_size) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    int written = snprintf(
        buffer,
        buffer_size,
        "{\"source_id\":\"%s\",\"evidence_type\":\"%s\",\"content_hash\":\"",
        env->source_id,
        env->evidence_type
    );

    if (written < 0 || (size_t)written >= buffer_size) {
        return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;
    }

    /* Append content hash as hex */
    size_t offset = (size_t)written;
    for (size_t i = 0; i < env->content_hash_size; ++i) {
        int hex_written = snprintf(
            buffer + offset,
            buffer_size - offset,
            "%02x",
            env->content_hash[i]
        );
        if (hex_written < 0 || (size_t)hex_written >= buffer_size - offset) {
            return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;
        }
        offset += (size_t)hex_written;
    }

    /* Append metadata */
    int meta_written = snprintf(
        buffer + offset,
        buffer_size - offset,
        "\",\"created_at\":%llu,\"verified\":%s,\"confidence\":%.4f}",
        (unsigned long long)env->created_at,
        env->verified ? "true" : "false",
        env->confidence
    );

    if (meta_written < 0 || (size_t)meta_written >= buffer_size - offset) {
        return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;
    }

    *out_size = offset + (size_t)meta_written;
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}
