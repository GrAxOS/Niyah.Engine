#include "niyah_matmul.h"
#include "niyah_core.h"

#include <string.h>

bool niyah_matmul_f32(const float *a, const float *b, float *c,
                       size_t m, size_t k, size_t n) {
    size_t mk = 0u, kn = 0u, mn = 0u;

    if (a == NULL || b == NULL || c == NULL) {
        return false;
    }
    if (m == 0u || k == 0u || n == 0u) {
        return false;
    }
    if (!niyah_mul_size(m, k, &mk) || !niyah_mul_size(k, n, &kn) ||
        !niyah_mul_size(m, n, &mn)) {
        return false;
    }

    memset(c, 0, mn * sizeof(float));

    for (size_t i = 0u; i < m; ++i) {
        const float *a_row = a + i * k;
        float *c_row = c + i * n;

        for (size_t p = 0u; p < k; ++p) {
            const float a_val = a_row[p];
            const float *b_row = b + p * n;

            for (size_t j = 0u; j < n; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }

    return true;
}

bool niyah_matmul_f32_bt(const float *a, const float *b_transposed, float *c,
                          size_t m, size_t k, size_t n) {
    size_t mk = 0u, nk = 0u, mn = 0u;

    if (a == NULL || b_transposed == NULL || c == NULL) {
        return false;
    }
    if (m == 0u || k == 0u || n == 0u) {
        return false;
    }
    if (!niyah_mul_size(m, k, &mk) || !niyah_mul_size(n, k, &nk) ||
        !niyah_mul_size(m, n, &mn)) {
        return false;
    }

    for (size_t i = 0u; i < m; ++i) {
        const float *a_row = a + i * k;
        float *c_row = c + i * n;

        for (size_t j = 0u; j < n; ++j) {
            const float *w_row = b_transposed + j * k;
            float acc = 0.0f;

            for (size_t p = 0u; p < k; ++p) {
                acc += a_row[p] * w_row[p];
            }

            c_row[j] = acc;
        }
    }

    return true;
}
