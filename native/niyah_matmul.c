#include "niyah.h"

#include <string.h>

/*
 * Was: `// MatMul stubs / TODO: Implement when types are available`.
 *
 * Blocked row-major GEMM. The i-k-j loop order streams `b` and `out` along
 * contiguous rows, which is what row-major storage wants; tiling keeps the
 * working set inside L1/L2 for the shapes this engine actually uses.
 */

#define NIYAH_MM_TILE_M 32
#define NIYAH_MM_TILE_K 64
#define NIYAH_MM_TILE_N 128

void niyah_matmul(float* out,
                  const float* a,
                  const float* b,
                  int32_t m, int32_t k, int32_t n)
{
    if (!out || !a || !b || m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    memset(out, 0, (size_t)m * (size_t)n * sizeof(float));

    for (int32_t i0 = 0; i0 < m; i0 += NIYAH_MM_TILE_M) {
        const int32_t i_max = (i0 + NIYAH_MM_TILE_M < m) ? i0 + NIYAH_MM_TILE_M : m;

        for (int32_t k0 = 0; k0 < k; k0 += NIYAH_MM_TILE_K) {
            const int32_t k_max = (k0 + NIYAH_MM_TILE_K < k) ? k0 + NIYAH_MM_TILE_K : k;

            for (int32_t j0 = 0; j0 < n; j0 += NIYAH_MM_TILE_N) {
                const int32_t j_max = (j0 + NIYAH_MM_TILE_N < n) ? j0 + NIYAH_MM_TILE_N : n;

                for (int32_t i = i0; i < i_max; ++i) {
                    float* NIYAH_RESTRICT out_row = out + (size_t)i * (size_t)n;
                    const float* a_row = a + (size_t)i * (size_t)k;

                    for (int32_t p = k0; p < k_max; ++p) {
                        const float av = a_row[p];
                        if (av == 0.0f) {
                            continue;
                        }
                        const float* NIYAH_RESTRICT b_row = b + (size_t)p * (size_t)n;
                        for (int32_t j = j0; j < j_max; ++j) {
                            out_row[j] += av * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void niyah_matmul_bt(float* out,
                     const float* a,
                     const float* b,
                     int32_t m, int32_t k, int32_t n)
{
    if (!out || !a || !b || m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    for (int32_t i = 0; i < m; ++i) {
        const float* a_row = a + (size_t)i * (size_t)k;
        float* out_row = out + (size_t)i * (size_t)n;

        for (int32_t j = 0; j < n; ++j) {
            const float* b_row = b + (size_t)j * (size_t)k;

            /* Four-way unrolled dot product with independent accumulators so
             * the FP pipeline is not serialised on a single dependency chain. */
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int32_t p = 0;
            for (; p + 3 < k; p += 4) {
                s0 += a_row[p]     * b_row[p];
                s1 += a_row[p + 1] * b_row[p + 1];
                s2 += a_row[p + 2] * b_row[p + 2];
                s3 += a_row[p + 3] * b_row[p + 3];
            }
            for (; p < k; ++p) {
                s0 += a_row[p] * b_row[p];
            }
            out_row[j] = (s0 + s1) + (s2 + s3);
        }
    }
}

void niyah_matvec(float* out,
                  const float* w,
                  const float* x,
                  int32_t n_out, int32_t n_in)
{
    if (!out || !w || !x || n_out <= 0 || n_in <= 0) {
        return;
    }

    for (int32_t i = 0; i < n_out; ++i) {
        const float* row = w + (size_t)i * (size_t)n_in;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int32_t p = 0;
        for (; p + 3 < n_in; p += 4) {
            s0 += row[p]     * x[p];
            s1 += row[p + 1] * x[p + 1];
            s2 += row[p + 2] * x[p + 2];
            s3 += row[p + 3] * x[p + 3];
        }
        for (; p < n_in; ++p) {
            s0 += row[p] * x[p];
        }
        out[i] = (s0 + s1) + (s2 + s3);
    }
}

void niyah_add_inplace(float* dst, const float* src, int32_t n)
{
    if (!dst || !src || n <= 0) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void niyah_scale_inplace(float* dst, float scale, int32_t n)
{
    if (!dst || n <= 0) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        dst[i] *= scale;
    }
}
