#include "niyah_rmsnorm.h"
#include "niyah_core.h"

#include <math.h>

bool niyah_rmsnorm_f32(const float *x, const float *weight, float *out,
                        size_t rows, size_t dim, float eps) {
    size_t total = 0u;

    if (x == NULL || weight == NULL || out == NULL) {
        return false;
    }
    if (rows == 0u || dim == 0u) {
        return false;
    }
    if (!niyah_mul_size(rows, dim, &total)) {
        return false;
    }

    for (size_t r = 0u; r < rows; ++r) {
        const float *x_row = x + r * dim;
        float *out_row = out + r * dim;

        double sum_sq = 0.0;
        for (size_t i = 0u; i < dim; ++i) {
            const double v = (double)x_row[i];
            sum_sq += v * v;
        }

        const double mean_sq = sum_sq / (double)dim;
        const float inv_rms = (float)(1.0 / sqrt(mean_sq + (double)eps));

        for (size_t i = 0u; i < dim; ++i) {
            out_row[i] = x_row[i] * inv_rms * weight[i];
        }
    }

    return true;
}
