#include "niyah.h"
#include <stddef.h>

/*
 * out[m x n] = a[m x k] * b[k x n]
 * Row-major, cache-friendly inner loop.
 */
void niyah_matmul(float* out, const float* a, const float* b,
                  int32_t m, int32_t k, int32_t n) {
    if (!out || !a || !b || m <= 0 || k <= 0 || n <= 0) return;

    for (int32_t i = 0; i < m; i++) {
        const float* row_a = a + i * k;
        float*       row_o = out + i * n;

        for (int32_t j = 0; j < n; j++) row_o[j] = 0.0f;

        for (int32_t p = 0; p < k; p++) {
            float aip = row_a[p];
            const float* row_b = b + p * n;
            for (int32_t j = 0; j < n; j++)
                row_o[j] += aip * row_b[j];
        }
    }
}
