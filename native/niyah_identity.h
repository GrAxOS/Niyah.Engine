#ifndef NIYAH_IDENTITY_H
#define NIYAH_IDENTITY_H

#include "niyah.h"
#include "niyah_sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Measured self-knowledge.
 *
 * Ask any language model what it is and the answer is *generated*: it comes
 * out of training data or a system prompt, travels through the sampler, and
 * is therefore a claim rather than a fact. It can be stale, wrong, or simply
 * forged by whoever wrote the prompt. This is why a model will cheerfully
 * recite any origin story it is handed.
 *
 * Nothing in this file passes through the model. Every field is either
 * compiled in, read from the live config, measured from the bytes in memory,
 * or explicitly marked NIYAH_UNKNOWN.
 *
 * Two design rules:
 *
 *   1. The engine and the weights are different artefacts and are reported
 *      separately. Niyah is the engine. The weights it loads may well be
 *      someone else's, and saying otherwise is precisely the dishonesty this
 *      module exists to prevent.
 *
 *   2. UNKNOWN is a real answer. niyah_identity_is() returns Kleene truth
 *      (see niyah_core.c). Without a manifest it returns NIYAH_UNKNOWN
 *      instead of guessing.
 */

#define NIYAH_IDENTITY_NAME_MAX     128
#define NIYAH_IDENTITY_ORIGIN_MAX   256
#define NIYAH_IDENTITY_LICENSE_MAX   64

typedef struct {
    /* --- Engine: compiled in, always known --------------------------- */
    char engine_name[32];
    char engine_version[32];
    char engine_build_date[32];

    /* --- Weights: measured from the running process ------------------ */
    bool     weights_loaded;
    size_t   weights_bytes;
    uint64_t parameter_count;
    bool     weights_hashed;      /* hashing is opt-in; it is not free */
    char     weights_sha256[NIYAH_SHA256_HEX_BYTES];

    /* --- Shape: read from the live config, not from a description ---- */
    NiyahModelConfig config;

    /* --- Provenance: from a manifest file, or UNKNOWN ---------------- */
    NiyahTruth provenance_known;
    char       model_name[NIYAH_IDENTITY_NAME_MAX];
    char       origin[NIYAH_IDENTITY_ORIGIN_MAX];
    char       license[NIYAH_IDENTITY_LICENSE_MAX];
    char       declared_weights_sha256[NIYAH_SHA256_HEX_BYTES];
    char       corpus_manifest_sha256[NIYAH_SHA256_HEX_BYTES];

    /*
     * Did the measured weights hash match what the manifest declared?
     * NIYAH_UNKNOWN until both are available. NIYAH_FALSE means the weights
     * were replaced after the manifest was written.
     */
    NiyahTruth weights_match_manifest;
} NiyahIdentity;

/* Zero the struct and fill in the compiled-in engine fields. */
NIYAH_API void niyah_identity_init(NiyahIdentity* identity);

/*
 * Read shape and size from a live NiyahLLM. Cheap: no hashing.
 * llm may be NULL, which records "no weights loaded" honestly.
 */
NIYAH_API NiyahStatus niyah_identity_capture(NiyahIdentity* identity,
                                             const NiyahLLM* llm);

/*
 * SHA-256 over the actual weight bytes in memory. Separate from capture()
 * because it costs roughly a second per gigabyte.
 */
NIYAH_API NiyahStatus niyah_identity_hash_weights(NiyahIdentity* identity,
                                                  const NiyahLLM* llm);

/*
 * Load provenance from a flat JSON manifest. Recognised keys, all optional:
 *
 *   model_name, origin, license, weights_sha256, corpus_manifest_sha256
 *
 * If the manifest declares weights_sha256 and the weights have already been
 * hashed, the two are compared and weights_match_manifest is set.
 */
NIYAH_API NiyahStatus niyah_identity_load_manifest(NiyahIdentity* identity,
                                                   const char* manifest_path);

/*
 * Answer a claim about what this is, as Kleene truth.
 *
 *   NIYAH_TRUE     the claim is supported by verified provenance
 *   NIYAH_FALSE    the claim contradicts verified provenance
 *   NIYAH_UNKNOWN  there is not enough evidence to say
 *
 * Matching is a case-insensitive substring test against the engine name and
 * the manifest model name. Provenance that failed its hash check yields
 * UNKNOWN for every model claim, never TRUE and never FALSE.
 */
NIYAH_API NiyahTruth niyah_identity_is(const NiyahIdentity* identity,
                                       const char* claim);

/*
 * True when a prompt is asking the system about itself, in English or Arabic.
 * Callers should route these to niyah_identity_report() instead of the model.
 */
NIYAH_API bool niyah_identity_is_self_query(const char* prompt);

/* Deterministic human-readable report. Identical input gives identical bytes. */
NIYAH_API NiyahStatus niyah_identity_report(const NiyahIdentity* identity,
                                            char* buffer,
                                            size_t buffer_size,
                                            size_t* out_size);

/* Same content as flat JSON, for run manifests and audit logs. */
NIYAH_API NiyahStatus niyah_identity_report_json(const NiyahIdentity* identity,
                                                 char* buffer,
                                                 size_t buffer_size,
                                                 size_t* out_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIYAH_IDENTITY_H */
