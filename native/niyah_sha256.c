#include "niyah_sha256.h"

#include <stdio.h>
#include <string.h>

/* FIPS 180-4, section 4.2.2 */
static const uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

#define ROTR(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x)     (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define BSIG1(x)     (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SSIG0(x)     (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SSIG1(x)     (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_compress(uint32_t state[8], const uint8_t block[64])
{
    uint32_t w[64];

    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i * 4 + 0] << 24)
             | ((uint32_t)block[i * 4 + 1] << 16)
             | ((uint32_t)block[i * 4 + 2] <<  8)
             | ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        w[i] = SSIG1(w[i - 2]) + w[i - 7] + SSIG0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + BSIG1(e) + CH(e, f, g)
                          + kRoundConstants[i] + w[i];
        const uint32_t t2 = BSIG0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void niyah_sha256_init(NiyahSha256* ctx)
{
    if (!ctx) {
        return;
    }
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
    ctx->bit_length = 0;
    ctx->block_used = 0;
    memset(ctx->block, 0, sizeof(ctx->block));
}

void niyah_sha256_update(NiyahSha256* ctx, const void* data, size_t length)
{
    if (!ctx || (!data && length > 0)) {
        return;
    }

    const uint8_t* p = (const uint8_t*)data;
    ctx->bit_length += (uint64_t)length * 8u;

    /* Top up a partial block first. */
    if (ctx->block_used > 0) {
        const size_t need = NIYAH_SHA256_BLOCK - ctx->block_used;
        const size_t take = (length < need) ? length : need;

        memcpy(ctx->block + ctx->block_used, p, take);
        ctx->block_used += take;
        p += take;
        length -= take;

        if (ctx->block_used == NIYAH_SHA256_BLOCK) {
            sha256_compress(ctx->state, ctx->block);
            ctx->block_used = 0;
        }
    }

    while (length >= NIYAH_SHA256_BLOCK) {
        sha256_compress(ctx->state, p);
        p += NIYAH_SHA256_BLOCK;
        length -= NIYAH_SHA256_BLOCK;
    }

    if (length > 0) {
        memcpy(ctx->block, p, length);
        ctx->block_used = length;
    }
}

void niyah_sha256_final(NiyahSha256* ctx, uint8_t out[NIYAH_SHA256_BYTES])
{
    if (!ctx || !out) {
        return;
    }

    const uint64_t bits = ctx->bit_length;

    /* 0x80, then zeros, then the 64-bit big-endian length. */
    ctx->block[ctx->block_used++] = 0x80u;

    if (ctx->block_used > NIYAH_SHA256_BLOCK - 8) {
        memset(ctx->block + ctx->block_used, 0,
               NIYAH_SHA256_BLOCK - ctx->block_used);
        sha256_compress(ctx->state, ctx->block);
        ctx->block_used = 0;
    }

    memset(ctx->block + ctx->block_used, 0,
           (NIYAH_SHA256_BLOCK - 8) - ctx->block_used);

    for (int i = 0; i < 8; ++i) {
        ctx->block[NIYAH_SHA256_BLOCK - 1 - i] =
            (uint8_t)((bits >> (8 * i)) & 0xffu);
    }
    sha256_compress(ctx->state, ctx->block);

    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (uint8_t)((ctx->state[i] >> 24) & 0xffu);
        out[i * 4 + 1] = (uint8_t)((ctx->state[i] >> 16) & 0xffu);
        out[i * 4 + 2] = (uint8_t)((ctx->state[i] >>  8) & 0xffu);
        out[i * 4 + 3] = (uint8_t)((ctx->state[i]      ) & 0xffu);
    }

    /* Do not leave the tail of the message sitting in the context. */
    memset(ctx->block, 0, sizeof(ctx->block));
    ctx->block_used = 0;
}

void niyah_sha256_buffer(const void* data, size_t length,
                         uint8_t out[NIYAH_SHA256_BYTES])
{
    NiyahSha256 ctx;
    niyah_sha256_init(&ctx);
    niyah_sha256_update(&ctx, data, length);
    niyah_sha256_final(&ctx, out);
}

NiyahStatus niyah_sha256_file(const char* path, uint8_t out[NIYAH_SHA256_BYTES])
{
    if (!path || !out) {
        return NIYAH_ERR_INVALID_ARG;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }

    NiyahSha256 ctx;
    niyah_sha256_init(&ctx);

    /* Chunked so a multi-gigabyte weight blob never has to be resident. */
    static const size_t kChunk = 65536u;
    uint8_t buffer[65536];

    for (;;) {
        const size_t got = fread(buffer, 1, kChunk, f);
        if (got > 0) {
            niyah_sha256_update(&ctx, buffer, got);
        }
        if (got < kChunk) {
            break;
        }
    }

    const int failed = ferror(f);
    fclose(f);

    if (failed) {
        return NIYAH_ERR_IO;
    }

    niyah_sha256_final(&ctx, out);
    return NIYAH_OK;
}

void niyah_sha256_to_hex(const uint8_t digest[NIYAH_SHA256_BYTES],
                         char out[NIYAH_SHA256_HEX_BYTES])
{
    static const char kHex[] = "0123456789abcdef";

    if (!out) {
        return;
    }
    if (!digest) {
        out[0] = '\0';
        return;
    }

    for (int i = 0; i < NIYAH_SHA256_BYTES; ++i) {
        out[i * 2 + 0] = kHex[(digest[i] >> 4) & 0x0fu];
        out[i * 2 + 1] = kHex[digest[i] & 0x0fu];
    }
    out[NIYAH_SHA256_BYTES * 2] = '\0';
}
