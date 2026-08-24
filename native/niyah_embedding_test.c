#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

int main(void)
{
    /* dot / l2_norm. */
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    assert(CLOSE(niyah_dot(a, b, 3), 32.0f));       /* 4 + 10 + 18 */
    assert(CLOSE(niyah_l2_norm(a, 3), sqrtf(14.0f)));

    /* Cosine similarity: identical vectors -> 1. */
    assert(CLOSE(niyah_cosine_similarity(a, a, 3), 1.0f));

    /* Orthogonal -> 0. */
    const float x[2] = {1.0f, 0.0f};
    const float y[2] = {0.0f, 1.0f};
    assert(CLOSE(niyah_cosine_similarity(x, y, 2), 0.0f));

    /* Opposite -> -1. */
    const float neg[2] = {-1.0f, 0.0f};
    assert(CLOSE(niyah_cosine_similarity(x, neg, 2), -1.0f));

    /* Scale invariance. */
    const float scaled[3] = {2.0f, 4.0f, 6.0f};
    assert(CLOSE(niyah_cosine_similarity(a, scaled, 3), 1.0f));

    /* A zero vector has no direction; must return 0, not NaN. */
    const float zero[3] = {0.0f, 0.0f, 0.0f};
    const float cs = niyah_cosine_similarity(a, zero, 3);
    assert(isfinite(cs));
    assert(CLOSE(cs, 0.0f));

    /* normalize gives unit length and leaves an all-zero vector alone. */
    float v[3] = {3.0f, 0.0f, 4.0f};
    niyah_normalize(v, 3);
    assert(CLOSE(niyah_l2_norm(v, 3), 1.0f));
    assert(CLOSE(v[0], 0.6f));
    assert(CLOSE(v[2], 0.8f));

    float z[3] = {0.0f, 0.0f, 0.0f};
    niyah_normalize(z, 3);
    for (int i = 0; i < 3; ++i) {
        assert(isfinite(z[i]));
    }

    /* init / free. */
    NiyahEmbedding emb;
    memset(&emb, 0, sizeof(emb));
    assert(niyah_embedding_init(&emb, 4, "doc-1") == NIYAH_OK);
    assert(emb.dim == 4);
    assert(emb.vector != NULL);
    assert(emb.doc_id != NULL && strcmp(emb.doc_id, "doc-1") == 0);
    for (int i = 0; i < 4; ++i) {
        assert(emb.vector[i] == 0.0f);
    }
    assert(niyah_embedding_init(&emb, 0, "bad") == NIYAH_ERR_INVALID_ARG);
    niyah_embedding_free(&emb);
    assert(emb.vector == NULL);

    /* top_k must rank by descending similarity. */
    NiyahEmbedding query;
    NiyahEmbedding corpus[3];
    memset(&query, 0, sizeof(query));
    memset(corpus, 0, sizeof(corpus));

    assert(niyah_embedding_init(&query, 2, "q") == NIYAH_OK);
    query.vector[0] = 1.0f;
    query.vector[1] = 0.0f;

    assert(niyah_embedding_init(&corpus[0], 2, "orthogonal") == NIYAH_OK);
    corpus[0].vector[0] = 0.0f;
    corpus[0].vector[1] = 1.0f;              /* cos = 0.0 */

    assert(niyah_embedding_init(&corpus[1], 2, "exact") == NIYAH_OK);
    corpus[1].vector[0] = 1.0f;
    corpus[1].vector[1] = 0.0f;              /* cos = 1.0 */

    assert(niyah_embedding_init(&corpus[2], 2, "diagonal") == NIYAH_OK);
    corpus[2].vector[0] = 1.0f;
    corpus[2].vector[1] = 1.0f;              /* cos = 0.7071 */

    int32_t idx[3] = {-1, -1, -1};
    float scores[3] = {0};
    const int32_t found =
        niyah_embedding_top_k(&query, corpus, 3, 3, idx, scores);
    assert(found == 3);

    /* Ranked: exact (1), diagonal (2), orthogonal (0). */
    assert(idx[0] == 1);
    assert(idx[1] == 2);
    assert(idx[2] == 0);

    assert(CLOSE(scores[0], 1.0f));
    assert(CLOSE(scores[1], 0.70711f));
    assert(CLOSE(scores[2], 0.0f));

    /* Scores are non-increasing. */
    assert(scores[0] >= scores[1] && scores[1] >= scores[2]);

    /* k larger than the corpus is clamped; k smaller truncates. */
    assert(niyah_embedding_top_k(&query, corpus, 3, 99, idx, scores) == 3);
    assert(niyah_embedding_top_k(&query, corpus, 3, 1, idx, scores) == 1);
    assert(idx[0] == 1);

    /* Degenerate inputs. */
    assert(niyah_embedding_top_k(NULL, corpus, 3, 1, idx, scores) == 0);
    assert(niyah_dot(NULL, b, 3) == 0.0f);
    assert(niyah_l2_norm(NULL, 3) == 0.0f);

    niyah_embedding_free(&query);
    for (int i = 0; i < 3; ++i) {
        niyah_embedding_free(&corpus[i]);
    }

    return 0;
}
