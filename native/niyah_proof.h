#ifndef NIYAH_PROOF_H
#define NIYAH_PROOF_H

#include "niyah.h"
#include "niyah_sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_PROOF_V1_HEADER "NIYAH-PROOF-V1"

typedef struct {
    uint8_t digest[NIYAH_SHA256_BYTES];
    uint8_t prompt_hash[NIYAH_SHA256_BYTES];
    uint8_t output_hash[NIYAH_SHA256_BYTES];
    uint8_t rules_hash[NIYAH_SHA256_BYTES];
} NiyahProofV1;

/*
 * Build a deterministic receipt without retaining prompt/output/rule text.
 * The proof digest is domain-separated and binds the three component hashes:
 *
 *   SHA256("NIYAH-PROOF-V1" || 0x00 ||
 *          prompt_hash || output_hash || rules_hash)
 */
NIYAH_API NiyahStatus niyah_proof_v1_generate(
    const void* prompt,
    size_t prompt_size,
    const void* output,
    size_t output_size,
    const void* rules,
    size_t rules_size,
    NiyahProofV1* out);

/* Text receipt containing only hashes, never the raw inputs. */
NIYAH_API NiyahStatus niyah_proof_v1_serialize(
    const NiyahProofV1* proof,
    char* buffer,
    size_t buffer_size,
    size_t* out_size);

NIYAH_API NiyahStatus niyah_proof_v1_save(
    const char* path,
    const NiyahProofV1* proof);

/* Recompute the receipt from supplied inputs and compare with a saved proof. */
NIYAH_API NiyahStatus niyah_proof_v1_verify_file(
    const char* path,
    const void* prompt,
    size_t prompt_size,
    const void* output,
    size_t output_size,
    const void* rules,
    size_t rules_size,
    bool* matches);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIYAH_PROOF_H */
