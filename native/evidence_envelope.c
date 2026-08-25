#include "evidence_envelope.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_create(
        NiyahEvidenceEnvelope** out,
        const char*   source_id,
        const char*   evidence_type,
        const uint8_t* content_hash,
        size_t        content_hash_size) {

    if (!out || !source_id || !evidence_type) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;

    *out = (NiyahEvidenceEnvelope*)calloc(1, sizeof(NiyahEvidenceEnvelope));
    if (!*out) return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;

    strncpy((*out)->source_id,     source_id,     sizeof((*out)->source_id) - 1);
    strncpy((*out)->evidence_type, evidence_type, sizeof((*out)->evidence_type) - 1);

    size_t copy_len = content_hash_size < sizeof((*out)->content_hash)
                      ? content_hash_size : sizeof((*out)->content_hash);
    if (content_hash && copy_len > 0) {
        memcpy((*out)->content_hash, content_hash, copy_len);
        (*out)->content_hash_size = copy_len;
    }

    (*out)->created_at = (uint64_t)time(NULL);
    (*out)->verified   = false;
    (*out)->confidence = 0.0f;

    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

void niyah_evidence_envelope_destroy(NiyahEvidenceEnvelope* env) {
    free(env);
}

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_verify(
        NiyahEvidenceEnvelope* env, float confidence) {
    if (!env) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    if (confidence < 0.0f || confidence > 1.0f) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;

    env->verified    = true;
    env->confidence  = confidence;
    env->verified_at = (uint64_t)time(NULL);
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_serialize(
        const NiyahEvidenceEnvelope* env,
        char* buffer, size_t buffer_size, size_t* out_size) {
    if (!env || !buffer || buffer_size == 0) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;

    int n = snprintf(buffer, buffer_size,
        "{\"source_id\":\"%s\",\"evidence_type\":\"%s\","
        "\"verified\":%s,\"confidence\":%.4f,"
        "\"created_at\":%llu,\"verified_at\":%llu}",
        env->source_id, env->evidence_type,
        env->verified ? "true" : "false",
        env->confidence,
        (unsigned long long)env->created_at,
        (unsigned long long)env->verified_at);

    if (n < 0 || (size_t)n >= buffer_size) return NIYAH_EVIDENCE_ENVELOPE_ERROR;
    if (out_size) *out_size = (size_t)n;
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}
