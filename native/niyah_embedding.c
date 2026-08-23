#include "niyah_embedding.h"
#include "niyah_matmul.h"
#include "niyah_core.h"

bool niyah_embedding_lookup_f32(const float *table, size_t vocab_size,
                                 size_t embedding_dim,
                                 const uint32_t *token_ids,
                                 size_t num_tokens, float *out) {
    size_t vocab_stride = 0u;
    size_t out_elems = 0u;

    if (table == NULL || token_ids == NULL || out == NULL) {
        return false;
    }
    if (vocab_size == 0u || embedding_dim == 0u || num_tokens == 0u) {
        return false;
    }
    if (!niyah_mul_size(vocab_size, embedding_dim, &vocab_stride) ||
        !niyah_mul_size(num_tokens, embedding_dim, &out_elems)) {
        return false;
    }

    for (size_t i = 0u; i < num_tokens; ++i) {
        if ((size_t)token_ids[i] >= vocab_size) {
            return false;
        }
        const float *row = table + (size_t)token_ids[i] * embedding_dim;
        float *dst = out + i * embedding_dim;
        for (size_t j = 0u; j < embedding_dim; ++j) {
            dst[j] = row[j];
        }
    }

    return true;
}

bool niyah_lm_head_f32(const float *hidden, const float *w_head,
                        float *logits, size_t rows, size_t embedding_dim,
                        size_t vocab_size) {
    return niyah_matmul_f32_bt(hidden, w_head, logits, rows, embedding_dim,
                                vocab_size);
}
