#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Simple bag-of-tokens embedding + cosine similarity ─────────────────── */

/*
 * Compute a mean-pooled embedding from token IDs.
 * Each token maps to a row in the embedding table (if provided),
 * otherwise uses a simple hash projection into `dim` dimensions.
 */
static float hash_feature(int32_t token_id, int32_t dim_idx, int32_t dim) {
    /* Deterministic pseudo-random projection via LCG hash */
    unsigned int h = (unsigned int)(token_id * 2654435761u + dim_idx * 40503u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32767.5f) - 1.0f; /* [-1, 1] */
}

NiyahEmbedding* niyah_embedding_compute(
        const int32_t* tokens, int32_t n_tokens,
        int32_t dim, const char* doc_id) {

    if (!tokens || n_tokens <= 0 || dim <= 0) return NULL;

    NiyahEmbedding* emb = (NiyahEmbedding*)calloc(1, sizeof(NiyahEmbedding));
    if (!emb) return NULL;

    emb->vector = (float*)calloc((size_t)dim, sizeof(float));
    if (!emb->vector) { free(emb); return NULL; }

    emb->dim    = dim;
    emb->doc_id = doc_id ? _strdup(doc_id) : NULL;

    /* Mean pool */
    for (int32_t t = 0; t < n_tokens; t++) {
        for (int32_t d = 0; d < dim; d++)
            emb->vector[d] += hash_feature(tokens[t], d, dim);
    }
    for (int32_t d = 0; d < dim; d++) emb->vector[d] /= (float)n_tokens;

    /* L2 normalise */
    float norm = 0.0f;
    for (int32_t d = 0; d < dim; d++) norm += emb->vector[d] * emb->vector[d];
    norm = sqrtf(norm);
    if (norm > 1e-9f)
        for (int32_t d = 0; d < dim; d++) emb->vector[d] /= norm;

    return emb;
}

float niyah_embedding_cosine(const NiyahEmbedding* a, const NiyahEmbedding* b) {
    if (!a || !b || !a->vector || !b->vector || a->dim != b->dim) return 0.0f;
    float dot = 0.0f;
    for (int32_t d = 0; d < a->dim; d++) dot += a->vector[d] * b->vector[d];
    return dot; /* vectors are already L2-normalised */
}

void niyah_embedding_free(NiyahEmbedding* emb) {
    if (!emb) return;
    free(emb->vector);
    free(emb->doc_id);
    free(emb);
}
