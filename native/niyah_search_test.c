#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>

#include "niyah.h"

int main(void)
{
    NiyahSearchResponse r;
    memset(&r, 0, sizeof(r));

    assert(niyah_search_response_init(&r, 4) == NIYAH_OK);
    assert(r.capacity >= 4);
    assert(r.n_results == 0);
    assert(r.results != NULL);

    assert(niyah_search_response_push(&r, "Low", "snippet lo",
                                      "https://a", 0.1f) == NIYAH_OK);
    assert(niyah_search_response_push(&r, "High", "snippet hi",
                                      "https://b", 0.9f) == NIYAH_OK);
    assert(niyah_search_response_push(&r, "Mid", "snippet mid",
                                      "https://c", 0.5f) == NIYAH_OK);
    assert(r.n_results == 3);

    /* Strings are copied, not aliased: mutating the caller's buffer afterwards
     * must not change the stored result. */
    char title[16];
    strcpy(title, "Temp");
    assert(niyah_search_response_push(&r, title, "s", "https://d", 0.7f)
           == NIYAH_OK);
    strcpy(title, "MUTATED");
    bool found_temp = false;
    for (int32_t i = 0; i < r.n_results; ++i) {
        if (strcmp(r.results[i].title, "Temp") == 0) {
            found_temp = true;
        }
    }
    assert(found_temp);

    /* Sort is descending by score. */
    niyah_search_response_sort(&r);
    assert(r.n_results == 4);
    for (int32_t i = 1; i < r.n_results; ++i) {
        assert(r.results[i - 1].score >= r.results[i].score);
    }
    assert(strcmp(r.results[0].title, "High") == 0);
    assert(fabsf(r.results[0].score - 0.9f) < 1e-6f);
    assert(strcmp(r.results[r.n_results - 1].title, "Low") == 0);

    /* Pushing past capacity grows rather than corrupting memory. */
    for (int i = 0; i < 32; ++i) {
        assert(niyah_search_response_push(&r, "Bulk", "s", "https://x",
                                          (float)i * 0.01f) == NIYAH_OK);
    }
    assert(r.n_results == 36);
    niyah_search_response_sort(&r);
    for (int32_t i = 1; i < r.n_results; ++i) {
        assert(r.results[i - 1].score >= r.results[i].score);
    }

    /* Sorting an empty or single-element response is a no-op, not a crash. */
    NiyahSearchResponse empty;
    memset(&empty, 0, sizeof(empty));
    assert(niyah_search_response_init(&empty, 2) == NIYAH_OK);
    niyah_search_response_sort(&empty);
    assert(empty.n_results == 0);
    niyah_search_response_free(&empty);

    /* Degenerate inputs. */
    assert(niyah_search_response_init(NULL, 4) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_search_response_push(NULL, "t", "s", "u", 1.0f)
           == NIYAH_ERR_INVALID_ARG);
    niyah_search_response_sort(NULL);
    niyah_search_response_free(NULL);

    niyah_search_response_free(&r);
    assert(r.results == NULL);
    assert(r.n_results == 0);

    return 0;
}
