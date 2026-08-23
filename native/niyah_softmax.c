#include "niyah_softmax.h"
#include "niyah_core.h"

#include <math.h>

bool niyah_softmax_f32(float *x, size_t rows, size_t dim) {
    size_t total = 0u;

    if (x == NULL) {
        return false;
    }
    if (rows == 0u || dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(rows, dim, &total)) {
        return false;
    }

    for (size_t r = 0u; r < rows; ++r) {
        float *row = x + r * dim;

        float row_max = row[0];
        for (size_t i = 1u; i < dim; ++i) {
            if (row[i] > row_max) {
                row_max = row[i];
            }
        }

        double sum = 0.0;
        for (size_t i = 0u; i < dim; ++i) {
            const double e = exp((double)row[i] - (double)row_max);
            row[i] = (float)e;
            sum += e;
        }

        const double inv_sum = 1.0 / sum;
        for (size_t i = 0u; i < dim; ++i) {
            row[i] = (float)((double)row[i] * inv_sum);
        }
    }

    return true;
}
