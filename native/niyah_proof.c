#include "niyah_proof.h"

#include <stdio.h>
#include <string.h>

static bool valid_buffer(const void* data, size_t size)
{
    return size == 0u || data != NULL;
}

static void hash_component(const void* data, size_t size,
                           uint8_t out[NIYAH_SHA256_BYTES])
{
    static const uint8_t empty = 0u;
    niyah_sha256_buffer(size == 0u ? &empty : data, size, out);
}

static bool digest_equal(const uint8_t a[NIYAH_SHA256_BYTES],
                         const uint8_t b[NIYAH_SHA256_BYTES])
{
    uint8_t diff = 0u;
    for (size_t i = 0u; i < NIYAH_SHA256_BYTES; ++i) {
        diff |= (uint8_t)(a[i] ^ b[i]);
    }
    return diff == 0u;
}

NiyahStatus niyah_proof_v1_generate(
    const void* prompt,
    size_t prompt_size,
    const void* output,
    size_t output_size,
    const void* rules,
    size_t rules_size,
    NiyahProofV1* out)
{
    if (!out ||
        !valid_buffer(prompt, prompt_size) ||
        !valid_buffer(output, output_size) ||
        !valid_buffer(rules, rules_size)) {
        return NIYAH_ERR_INVALID_ARG;
    }

    hash_component(prompt, prompt_size, out->prompt_hash);
    hash_component(output, output_size, out->output_hash);
    hash_component(rules, rules_size, out->rules_hash);

    NiyahSha256 ctx;
    static const char domain[] = NIYAH_PROOF_V1_HEADER;
    static const uint8_t separator = 0u;

    niyah_sha256_init(&ctx);
    niyah_sha256_update(&ctx, domain, sizeof(domain) - 1u);
    niyah_sha256_update(&ctx, &separator, 1u);
    niyah_sha256_update(&ctx, out->prompt_hash, NIYAH_SHA256_BYTES);
    niyah_sha256_update(&ctx, out->output_hash, NIYAH_SHA256_BYTES);
    niyah_sha256_update(&ctx, out->rules_hash, NIYAH_SHA256_BYTES);
    niyah_sha256_final(&ctx, out->digest);

    return NIYAH_OK;
}

NiyahStatus niyah_proof_v1_serialize(
    const NiyahProofV1* proof,
    char* buffer,
    size_t buffer_size,
    size_t* out_size)
{
    if (!proof || (!buffer && buffer_size != 0u)) {
        return NIYAH_ERR_INVALID_ARG;
    }

    char digest_hex[NIYAH_SHA256_HEX_BYTES];
    char prompt_hex[NIYAH_SHA256_HEX_BYTES];
    char output_hex[NIYAH_SHA256_HEX_BYTES];
    char rules_hex[NIYAH_SHA256_HEX_BYTES];

    niyah_sha256_to_hex(proof->digest, digest_hex);
    niyah_sha256_to_hex(proof->prompt_hash, prompt_hex);
    niyah_sha256_to_hex(proof->output_hash, output_hex);
    niyah_sha256_to_hex(proof->rules_hash, rules_hex);

    const int needed = snprintf(
        buffer,
        buffer_size,
        NIYAH_PROOF_V1_HEADER "\n"
        "hash: %s\n"
        "prompt_hash: %s\n"
        "output_hash: %s\n"
        "rules_hash: %s\n",
        digest_hex,
        prompt_hex,
        output_hex,
        rules_hex);

    if (needed < 0) {
        return NIYAH_ERR_IO;
    }

    if (out_size) {
        *out_size = (size_t)needed;
    }

    if (!buffer || (size_t)needed >= buffer_size) {
        return NIYAH_ERR_OVERFLOW;
    }

    return NIYAH_OK;
}

NiyahStatus niyah_proof_v1_save(const char* path, const NiyahProofV1* proof)
{
    if (!path || !proof) {
        return NIYAH_ERR_INVALID_ARG;
    }

    char receipt[384];
    size_t receipt_size = 0u;
    NiyahStatus status = niyah_proof_v1_serialize(
        proof, receipt, sizeof(receipt), &receipt_size);
    if (status != NIYAH_OK) {
        return status;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        return NIYAH_ERR_IO;
    }

    const bool wrote = fwrite(receipt, 1u, receipt_size, f) == receipt_size;
    const bool closed = fclose(f) == 0;
    return wrote && closed ? NIYAH_OK : NIYAH_ERR_IO;
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool parse_hex_digest(const char* text,
                             uint8_t out[NIYAH_SHA256_BYTES])
{
    if (!text || strlen(text) != 64u) {
        return false;
    }

    for (size_t i = 0u; i < NIYAH_SHA256_BYTES; ++i) {
        const int hi = hex_nibble(text[i * 2u]);
        const int lo = hex_nibble(text[i * 2u + 1u]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static void trim_newline(char* line)
{
    const size_t n = strlen(line);
    size_t end = n;
    while (end > 0u && (line[end - 1u] == '\n' || line[end - 1u] == '\r')) {
        line[--end] = '\0';
    }
}

static bool read_hash_line(FILE* f, const char* label,
                           uint8_t out[NIYAH_SHA256_BYTES])
{
    char line[96];
    if (!fgets(line, sizeof(line), f)) {
        return false;
    }
    trim_newline(line);

    const size_t label_len = strlen(label);
    if (strncmp(line, label, label_len) != 0) {
        return false;
    }
    return parse_hex_digest(line + label_len, out);
}

static NiyahStatus load_proof(const char* path, NiyahProofV1* proof)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }

    char header[64];
    bool ok = fgets(header, sizeof(header), f) != NULL;
    if (ok) {
        trim_newline(header);
        ok = strcmp(header, NIYAH_PROOF_V1_HEADER) == 0;
    }
    if (ok) ok = read_hash_line(f, "hash: ", proof->digest);
    if (ok) ok = read_hash_line(f, "prompt_hash: ", proof->prompt_hash);
    if (ok) ok = read_hash_line(f, "output_hash: ", proof->output_hash);
    if (ok) ok = read_hash_line(f, "rules_hash: ", proof->rules_hash);

    if (ok) {
        char trailing[2];
        ok = fgets(trailing, sizeof(trailing), f) == NULL;
    }

    if (fclose(f) != 0) {
        return NIYAH_ERR_IO;
    }
    return ok ? NIYAH_OK : NIYAH_ERR_IO;
}

NiyahStatus niyah_proof_v1_verify_file(
    const char* path,
    const void* prompt,
    size_t prompt_size,
    const void* output,
    size_t output_size,
    const void* rules,
    size_t rules_size,
    bool* matches)
{
    if (!path || !matches ||
        !valid_buffer(prompt, prompt_size) ||
        !valid_buffer(output, output_size) ||
        !valid_buffer(rules, rules_size)) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *matches = false;

    NiyahProofV1 stored;
    NiyahStatus status = load_proof(path, &stored);
    if (status != NIYAH_OK) {
        return status;
    }

    NiyahProofV1 actual;
    status = niyah_proof_v1_generate(
        prompt, prompt_size,
        output, output_size,
        rules, rules_size,
        &actual);
    if (status != NIYAH_OK) {
        return status;
    }

    *matches = digest_equal(stored.digest, actual.digest) &&
               digest_equal(stored.prompt_hash, actual.prompt_hash) &&
               digest_equal(stored.output_hash, actual.output_hash) &&
               digest_equal(stored.rules_hash, actual.rules_hash);
    return NIYAH_OK;
}
