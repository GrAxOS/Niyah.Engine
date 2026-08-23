#include "niyah_model.h"
#include <stddef.h>
#include <stdint.h>

static int add_size(size_t a, size_t b, size_t *out)
{
    return niyah_add_size(a, b, out) ? 0 : -1;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    return niyah_mul_size(a, b, out) ? 0 : -1;
}

int niyah_model_layout_make(uint32_t vocab_size,
                            uint32_t embed_dim,
                            uint32_t n_layers,
                            uint32_t n_kv_heads,
                            uint32_t ffn_dim,
                            NiyahModelLayout *out)
{
    size_t total = 0;
    size_t term = 0;
    size_t layer_params = 0;
    size_t kv_projection = 0;

    if (!out || vocab_size == 0 || embed_dim == 0 || n_layers == 0 ||
        n_kv_heads == 0 || ffn_dim == 0) {
        return -1;
    }

    if (mul_size((size_t)vocab_size, (size_t)embed_dim, &term) != 0 ||
        add_size(total, term, &total) != 0) {
        return -1;
    }

    if (mul_size((size_t)n_kv_heads, (size_t)embed_dim, &kv_projection) != 0) {
        return -1;
    }

    /* WQ + WO + WG + WU + WD + WK + WV + two RMS vectors. */
    layer_params = 0;
    if (mul_size((size_t)embed_dim, (size_t)embed_dim, &term) != 0 ||
        add_size(layer_params, term, &layer_params) != 0 ||
        add_size(layer_params, term, &layer_params) != 0 ||
        mul_size((size_t)ffn_dim, (size_t)embed_dim, &term) != 0 ||
        add_size(layer_params, term, &layer_params) != 0 ||
        add_size(layer_params, term, &layer_params) != 0 ||
        mul_size((size_t)embed_dim, (size_t)ffn_dim, &term) != 0 ||
        add_size(layer_params, term, &layer_params) != 0 ||
        add_size(layer_params, kv_projection, &layer_params) != 0 ||
        add_size(layer_params, kv_projection, &layer_params) != 0 ||
        add_size(layer_params, (size_t)embed_dim, &layer_params) != 0 ||
        add_size(layer_params, (size_t)embed_dim, &layer_params) != 0) {
        return -1;
    }

    if (mul_size(layer_params, (size_t)n_layers, &term) != 0 ||
        add_size(total, term, &total) != 0 ||
        add_size(total, (size_t)embed_dim, &total) != 0 ||
        mul_size((size_t)vocab_size, (size_t)embed_dim, &term) != 0 ||
        add_size(total, term, &total) != 0) {
        return -1;
    }

    if (mul_size(total, sizeof(float), &term) != 0) {
        return -1;
    }

    out->token_count = vocab_size;
    out->layer_count = n_layers;
    out->parameter_floats = total;
    out->bytes = term;
    return 0;
}
