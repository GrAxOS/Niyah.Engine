#include "niyah.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Was: `// Embedding stubs`. */

NiyahStatus niyah_embedding_init(NiyahEmbedding* embedding,
                                 int32_t dim,
                                 const char* doc_id)
{
    if (!embedding || dim <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(embedding, 0, sizeof(*embedding));

    embedding->vector = (float*)calloc((size_t)dim, sizeof(float));
    if (!embedding->vector) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    embedding->dim = dim;

    if (doc_id) {
        const size_t n = strlen(doc_id) + 1u;
        embedding->doc_id = (char*)malloc(n);
        if (!embedding->doc_id) {
            free(embedding->vector);
            embedding->vector = NULL;
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        memcpy(embedding->doc_id, doc_id, n);
    }

    return NIYAH_OK;
}

void niyah_embedding_free(NiyahEmbedding* embedding)
{
    if (!embedding) {
        return;
    }
    free(embedding->vector);
    free(embedding->doc_id);
    memset(embedding, 0, sizeof(*embedding));
}

float niyah_dot(const float* a, const float* b, int32_t n)
{
    if (!a || !b || n <= 0) {
        return 0.0f;
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

float niyah_l2_norm(const float* v, int32_t n)
{
    if (!v || n <= 0) {
        return 0.0f;
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += (double)v[i] * (double)v[i];
    }
    return (float)sqrt(sum);
}

void niyah_normalize(float* v, int32_t n)
{
    if (!v || n <= 0) {
        return;
    }
    const float norm = niyah_l2_norm(v, n);
    if (norm <= 0.0f) {
        return; /* zero vector has no direction; leave it alone */
    }
    const float inv = 1.0f / norm;
    for (int32_t i = 0; i < n; ++i) {
        v[i] *= inv;
    }
}

float niyah_cosine_similarity(const float* a, const float* b, int32_t n)
{
    if (!a || !b || n <= 0) {
        return 0.0f;
    }

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }

    const double denom = sqrt(na) * sqrt(nb);
    if (denom <= 0.0) {
        return 0.0f;
    }
    return (float)(dot / denom);
}

int32_t niyah_embedding_top_k(const NiyahEmbedding* query,
                              const NiyahEmbedding* corpus,
                              int32_t corpus_size,
                              int32_t k,
                              int32_t* out_indices,
                              float* out_scores)
{
    if (!query || !query->vector || !corpus || corpus_size <= 0 ||
        k <= 0 || !out_indices) {
        return 0;
    }

    if (k > corpus_size) {
        k = corpus_size;
    }

    float* scores = (float*)malloc((size_t)corpus_size * sizeof(float));
    bool* taken = (bool*)calloc((size_t)corpus_size, sizeof(bool));
    if (!scores || !taken) {
        free(scores);
        free(taken);
        return 0;
    }

    for (int32_t i = 0; i < corpus_size; ++i) {
        const int32_t dim = corpus[i].dim < query->dim
            ? corpus[i].dim : query->dim;
        scores[i] = corpus[i].vector
            ? niyah_cosine_similarity(query->vector, corpus[i].vector, dim)
            : -2.0f;
    }

    int32_t written = 0;
    for (int32_t slot = 0; slot < k; ++slot) {
        int32_t best = -1;
        float best_score = -2.0f;

        for (int32_t i = 0; i < corpus_size; ++i) {
            if (!taken[i] && scores[i] > best_score) {
                best_score = scores[i];
                best = i;
            }
        }

        if (best < 0) {
            break;
        }

        taken[best] = true;
        out_indices[written] = best;
        if (out_scores) {
            out_scores[written] = best_score;
        }
        ++written;
    }

    free(scores);
    free(taken);
    return written;
}
