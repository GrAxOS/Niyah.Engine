#include "niyah_rope.h"
#include "niyah_core.h"

#include <math.h>

bool niyah_rope_f32(float *x, size_t rows, size_t dim, size_t pos_offset,
                     float theta_base) {
    size_t total = 0u;

    if (x == NULL) {
        return false;
    }
    if (rows == 0u || dim == 0u || (dim % 2u) != 0u) {
        return false;
    }
    if (!niyah_mul_size(rows, dim, &total)) {
        return false;
    }

    const size_t half = dim / 2u;

    for (size_t r = 0u; r < rows; ++r) {
        float *row = x + r * dim;
        const double position = (double)(pos_offset + r);

        for (size_t i = 0u; i < half; ++i) {
            const double exponent = -2.0 * (double)i / (double)dim;
            const double theta = pow((double)theta_base, exponent);
            const double angle = position * theta;
            const double cos_a = cos(angle);
            const double sin_a = sin(angle);

            const float x0 = row[2u * i];
            const float x1 = row[2u * i + 1u];

            row[2u * i]      = (float)((double)x0 * cos_a - (double)x1 * sin_a);
            row[2u * i + 1u] = (float)((double)x0 * sin_a + (double)x1 * cos_a);
        }
    }

    return true;
}
