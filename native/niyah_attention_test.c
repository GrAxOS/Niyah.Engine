#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "niyah.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

static void test_causal_masking(void)
{
    /*
     * batch=1, seq=2, dim=2, n_head=1 so head_dim=2.
     *   q = [[1,0],[0,1]]
     *   k = [[1,0],[0,1]]
     *   v = [[1,2],[3,4]]
     *
     * Token 0 may only attend to token 0, so out[0] must equal v[0] exactly.
     * This is the property that proves the causal mask is real: without it,
     * token 0 would mix in v[1] and drift toward [2,3].
     */
    float q[4] = {1, 0, 0, 1};
    float k[4] = {1, 0, 0, 1};
    float v[4] = {1, 2, 3, 4};
    float out[4] = {0};

    NiyahMultiHeadAttentionState state;
    state.q = q;
    state.k = k;
    state.v = v;
    state.out = out;
    state.batch = 1;
    state.seq = 2;
    state.dim = 2;
    state.n_head = 1;

    niyah_multihead_attention_forward(&state);

    assert(CLOSE(out[0], 1.0f));
    assert(CLOSE(out[1], 2.0f));

    /* Token 1 attends to both, weighting token 1 higher (its score is
     * 1/sqrt(2) versus 0), so the result sits between v[0] and v[1] but
     * closer to v[1]. */
    assert(out[2] > 1.0f && out[2] < 3.0f);
    assert(out[3] > 2.0f && out[3] < 4.0f);
    assert(out[2] > 2.0f);

    /* Attention output is a convex combination of v rows, so with identical v
     * rows the output must equal that row everywhere. */
    float v_same[4] = {7, 9, 7, 9};
    state.v = v_same;
    niyah_multihead_attention_forward(&state);
    for (int t = 0; t < 2; ++t) {
        assert(CLOSE(out[t * 2], 7.0f));
        assert(CLOSE(out[t * 2 + 1], 9.0f));
    }
}

static void test_projection_free_path(void)
{
    /* niyah_attention_forward uses x as Q, K and V via the qkv scratch. */
    const int seq = 3, dim = 4;
    float x[12];
    for (int i = 0; i < 12; ++i) {
        x[i] = (float)(i % 5) - 2.0f;
    }

    float qkv[3 * 12];
    float out[12] = {0};

    NiyahAttentionState state;
    state.qkv = qkv;
    state.out = NULL;
    state.batch = 1;
    state.seq = seq;
    state.dim = dim;
    state.n_head = 2;

    niyah_attention_forward(&state, x, out);

    for (int i = 0; i < 12; ++i) {
        assert(isfinite(out[i]));
    }

    /* The first token attends only to itself, so it is returned unchanged. */
    for (int d = 0; d < dim; ++d) {
        assert(CLOSE(out[d], x[d]));
    }
}

static void test_kv_cache_and_decode(void)
{
    NiyahKVCache cache;
    memset(&cache, 0, sizeof(cache));

    /* 1 layer, 2 kv heads, head_dim 2, up to 4 positions. */
    assert(niyah_kv_cache_init(&cache, 1, 2, 2, 4) == NIYAH_OK);
    assert(cache.length == 0);
    assert(cache.k != NULL && cache.v != NULL);

    /* Invalid geometry must be rejected. */
    NiyahKVCache bad;
    assert(niyah_kv_cache_init(&bad, 0, 2, 2, 4) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_kv_cache_init(NULL, 1, 2, 2, 4) == NIYAH_ERR_INVALID_ARG);

    /* Write position 0 for both kv heads: k = v = [1,1]. */
    const int head_dim = 2;
    for (int h = 0; h < 2; ++h) {
        const size_t slot = (size_t)h * 4u * (size_t)head_dim;
        cache.k[slot] = 1.0f;
        cache.k[slot + 1] = 1.0f;
        cache.v[slot] = 1.0f;
        cache.v[slot + 1] = 1.0f;
    }

    float q[4] = {1, 0, 0, 1};   /* 2 heads x head_dim 2 */
    float out[4] = {0};
    float scores[4] = {0};

    /* n_head must be a multiple of n_kv_head; 2 and 2 means no GQA sharing. */
    assert(niyah_attention_decode(out, q, &cache, 0, 2, 0, scores) == NIYAH_OK);

    /* Only one cached position, so softmax gives weight 1 and out == v. */
    for (int i = 0; i < 4; ++i) {
        assert(CLOSE(out[i], 1.0f));
    }
    assert(cache.length == 1);

    /* Out-of-range layer / position must be refused. */
    assert(niyah_attention_decode(out, q, &cache, 5, 2, 0, scores)
           == NIYAH_ERR_INVALID_ARG);
    assert(niyah_attention_decode(out, q, &cache, 0, 2, 99, scores)
           == NIYAH_ERR_INVALID_ARG);
    /* n_head not divisible by n_kv_head. */
    assert(niyah_attention_decode(out, q, &cache, 0, 3, 0, scores)
           == NIYAH_ERR_INVALID_ARG);

    niyah_kv_cache_reset(&cache);
    assert(cache.length == 0);

    niyah_kv_cache_free(&cache);
    assert(cache.k == NULL && cache.v == NULL);
}

int main(void)
{
    test_causal_masking();
    test_projection_free_path();
    test_kv_cache_and_decode();

    /* Degenerate inputs. */
    niyah_multihead_attention_forward(NULL);
    niyah_attention_forward(NULL, NULL, NULL);

    return 0;
}
