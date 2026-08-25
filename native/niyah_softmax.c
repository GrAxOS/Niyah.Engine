#include "niyah.h"
#include <math.h>
#include <stddef.h>

void niyah_softmax(float* x, int32_t n) {
    if (!x || n <= 0) return;

    /* Numerically stable: subtract max before exp */
    float max_val = x[0];
    for (int32_t i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    if (sum > 0.0f)
        for (int32_t i = 0; i < n; i++) x[i] /= sum;
}
