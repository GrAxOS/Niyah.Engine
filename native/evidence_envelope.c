#include "evidence_envelope.h"
<<<<<<< HEAD
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
=======

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Was: `// Evidence envelope stubs`. */

static uint64_t now_unix(void)
{
    const time_t t = time(NULL);
    return (t == (time_t)-1) ? 0u : (uint64_t)t;
}

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_create(
    NiyahEvidenceEnvelope **out,
    const char *source_id,
    const char *evidence_type,
    const uint8_t *content_hash,
    size_t content_hash_size)
{
    if (!out || !source_id || !evidence_type) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    *out = NULL;

    /*
     * Reject oversized identifiers instead of truncating them. Silently
     * shortening a source id would make two different sources indistinguishable
     * in the audit log, which defeats the purpose of an evidence envelope.
     */
    if (strlen(source_id) >= 256u || strlen(evidence_type) >= 64u) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    if (content_hash_size > 64u) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    if (content_hash_size > 0u && !content_hash) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    NiyahEvidenceEnvelope *env =
        (NiyahEvidenceEnvelope *)calloc(1, sizeof(NiyahEvidenceEnvelope));
    if (!env) {
        return NIYAH_EVIDENCE_ENVELOPE_OUT_OF_MEMORY;
    }

    memcpy(env->source_id, source_id, strlen(source_id));
    memcpy(env->evidence_type, evidence_type, strlen(evidence_type));

    if (content_hash_size > 0u) {
        memcpy(env->content_hash, content_hash, content_hash_size);
    }
    env->content_hash_size = content_hash_size;

    env->created_at = now_unix();
    env->verified_at = 0u;
    env->verified = false;
    env->confidence = 0.0f;

    *out = env;
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

void niyah_evidence_envelope_destroy(NiyahEvidenceEnvelope *env)
{
>>>>>>> origin/main
    free(env);
}

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_verify(
<<<<<<< HEAD
        NiyahEvidenceEnvelope* env, float confidence) {
    if (!env) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    if (confidence < 0.0f || confidence > 1.0f) return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;

    env->verified    = true;
    env->confidence  = confidence;
    env->verified_at = (uint64_t)time(NULL);
=======
    NiyahEvidenceEnvelope *env,
    float confidence)
{
    if (!env) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    /* Confidence is a probability. Anything outside [0,1] is a caller bug. */
    if (!(confidence >= 0.0f) || !(confidence <= 1.0f)) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    env->verified = true;
    env->verified_at = now_unix();
    env->confidence = confidence;

>>>>>>> origin/main
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}

NiyahEvidenceEnvelopeStatus niyah_evidence_envelope_serialize(
<<<<<<< HEAD
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
=======
    const NiyahEvidenceEnvelope *env,
    char *buffer,
    size_t buffer_size,
    size_t *out_size)
{
    if (!env) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }
    if (!buffer && buffer_size != 0u) {
        return NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS;
    }

    static const char hex[] = "0123456789abcdef";
    char hash_hex[129];
    size_t n = env->content_hash_size;
    if (n > 64u) {
        n = 64u;
    }
    for (size_t i = 0; i < n; ++i) {
        hash_hex[i * 2u]      = hex[(env->content_hash[i] >> 4) & 0x0F];
        hash_hex[i * 2u + 1u] = hex[env->content_hash[i] & 0x0F];
    }
    hash_hex[n * 2u] = '\0';

    const int needed = snprintf(
        buffer, buffer_size,
        "{\"source_id\":\"%s\",\"evidence_type\":\"%s\","
        "\"content_hash\":\"%s\",\"content_hash_size\":%zu,"
        "\"created_at\":%" PRIu64 ",\"verified_at\":%" PRIu64 ","
        "\"verified\":%s,\"confidence\":%.6f}",
        env->source_id,
        env->evidence_type,
        hash_hex,
        env->content_hash_size,
        env->created_at,
        env->verified_at,
        env->verified ? "true" : "false",
        (double)env->confidence);

    if (needed < 0) {
        return NIYAH_EVIDENCE_ENVELOPE_ERROR;
    }

    /* Report the required length either way so callers can size and retry. */
    if (out_size) {
        *out_size = (size_t)needed;
    }

    if ((size_t)needed >= buffer_size) {
        return NIYAH_EVIDENCE_ENVELOPE_ERROR;
    }

>>>>>>> origin/main
    return NIYAH_EVIDENCE_ENVELOPE_OK;
}
