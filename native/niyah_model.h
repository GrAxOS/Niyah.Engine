#ifndef NIYAH_MODEL_H
#define NIYAH_MODEL_H

#include "niyah_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t token_count;
    size_t layer_count;
    size_t parameter_floats;
    size_t bytes;
} NiyahModelLayout;

int niyah_model_layout_make(uint32_t vocab_size,
                            uint32_t embed_dim,
                            uint32_t n_layers,
                            uint32_t n_kv_heads,
                            uint32_t ffn_dim,
                            NiyahModelLayout *out);

#ifdef __cplusplus
}
#endif

#endif
