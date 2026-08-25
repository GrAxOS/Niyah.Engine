#ifndef NIYAH_SHA256_H
#define NIYAH_SHA256_H

#include "niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * FIPS 180-4 SHA-256.
 *
 * The project had no hash function at all, while evidence_envelope.h declares
 * a content_hash[64] field that nothing was able to populate. Provenance,
 * tamper detection and reproducible run manifests all need this, so it lives
 * in the core library rather than in one subsystem.
 */

#define NIYAH_SHA256_BYTES      32
#define NIYAH_SHA256_HEX_BYTES  65   /* 64 hex digits + NUL */
#define NIYAH_SHA256_BLOCK      64

typedef struct {
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t  block[NIYAH_SHA256_BLOCK];
    size_t   block_used;
} NiyahSha256;

NIYAH_API void niyah_sha256_init(NiyahSha256* ctx);
NIYAH_API void niyah_sha256_update(NiyahSha256* ctx,
                                   const void* data,
                                   size_t length);
NIYAH_API void niyah_sha256_final(NiyahSha256* ctx,
                                  uint8_t out[NIYAH_SHA256_BYTES]);

/* One-shot over a buffer already in memory. */
NIYAH_API void niyah_sha256_buffer(const void* data,
                                   size_t length,
                                   uint8_t out[NIYAH_SHA256_BYTES]);

/* Streams the file in chunks; safe for multi-gigabyte weight blobs. */
NIYAH_API NiyahStatus niyah_sha256_file(const char* path,
                                        uint8_t out[NIYAH_SHA256_BYTES]);

/* Lowercase hex, always NUL terminated. */
NIYAH_API void niyah_sha256_to_hex(const uint8_t digest[NIYAH_SHA256_BYTES],
                                   char out[NIYAH_SHA256_HEX_BYTES]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIYAH_SHA256_H */
