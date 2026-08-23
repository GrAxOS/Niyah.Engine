#include "niyah_swiglu.h"
#include "niyah_matmul.h"
#include "niyah_core.h"

#include <math.h>

bool niyah_swiglu_f32(const float *x, const float *w_gate,
                       const float *w_up, const float *w_down,
                       float *gate_scratch, float *up_scratch, float *out,
                       size_t rows, size_t dim, size_t hidden_dim) {
    size_t rows_dim = 0u;
    size_t rows_hidden = 0u;

    if (x == NULL || w_gate == NULL || w_up == NULL || w_down == NULL ||
        gate_scratch == NULL || up_scratch == NULL || out == NULL) {
        return false;
    }
    if (rows == 0u || dim == 0u || hidden_dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(rows, dim, &rows_dim) ||
        !niyah_mul_size(rows, hidden_dim, &rows_hidden)) {
        return false;
    }

    if (!niyah_matmul_f32_bt(x, w_gate, gate_scratch, rows, dim,
                              hidden_dim)) {
        return false;
    }
    if (!niyah_matmul_f32_bt(x, w_up, up_scratch, rows, dim, hidden_dim)) {
        return false;
    }

    for (size_t i = 0u; i < rows_hidden; ++i) {
        const float v = gate_scratch[i];
        const float silu = v / (1.0f + expf(-v));
        gate_scratch[i] = silu * up_scratch[i];
    }

    return niyah_matmul_f32_bt(gate_scratch, w_down, out, rows, hidden_dim,
                                dim);
}
