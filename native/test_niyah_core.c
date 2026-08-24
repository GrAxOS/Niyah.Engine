#undef NDEBUG
#include <assert.h>
#include <string.h>

#include "niyah.h"

int main(void)
{
    /* Version: NIYAH_VERSION used to be referenced without being defined. */
    assert(niyah_version() != NULL);
    assert(strcmp(niyah_version(), "0.2.0") == 0);
    assert(niyah_version_major() == 0);
    assert(niyah_version_minor() == 2);
    assert(niyah_version_patch() == 0);

    /* Truth labels. */
    assert(strcmp(niyah_truth_to_string(NIYAH_TRUE), "true") == 0);
    assert(strcmp(niyah_truth_to_string(NIYAH_FALSE), "false") == 0);
    assert(strcmp(niyah_truth_to_string(NIYAH_UNKNOWN), "unknown") == 0);

    assert(niyah_truth_from_string("true") == NIYAH_TRUE);
    assert(niyah_truth_from_string("false") == NIYAH_FALSE);
    assert(niyah_truth_from_string("banana") == NIYAH_UNKNOWN);
    assert(niyah_truth_from_string(NULL) == NIYAH_UNKNOWN);

    /* Kleene NOT. */
    assert(niyah_truth_not(NIYAH_TRUE) == NIYAH_FALSE);
    assert(niyah_truth_not(NIYAH_FALSE) == NIYAH_TRUE);
    assert(niyah_truth_not(NIYAH_UNKNOWN) == NIYAH_UNKNOWN);

    /* Kleene AND: FALSE dominates, so UNKNOWN AND FALSE is decidable. */
    assert(niyah_truth_and(NIYAH_TRUE, NIYAH_TRUE) == NIYAH_TRUE);
    assert(niyah_truth_and(NIYAH_TRUE, NIYAH_FALSE) == NIYAH_FALSE);
    assert(niyah_truth_and(NIYAH_UNKNOWN, NIYAH_FALSE) == NIYAH_FALSE);
    assert(niyah_truth_and(NIYAH_UNKNOWN, NIYAH_TRUE) == NIYAH_UNKNOWN);
    assert(niyah_truth_and(NIYAH_UNKNOWN, NIYAH_UNKNOWN) == NIYAH_UNKNOWN);

    /* Kleene OR: TRUE dominates. */
    assert(niyah_truth_or(NIYAH_FALSE, NIYAH_FALSE) == NIYAH_FALSE);
    assert(niyah_truth_or(NIYAH_UNKNOWN, NIYAH_TRUE) == NIYAH_TRUE);
    assert(niyah_truth_or(NIYAH_UNKNOWN, NIYAH_FALSE) == NIYAH_UNKNOWN);

    /* Implication is defined as NOT a OR b. */
    assert(niyah_truth_implies(NIYAH_TRUE, NIYAH_FALSE) == NIYAH_FALSE);
    assert(niyah_truth_implies(NIYAH_FALSE, NIYAH_UNKNOWN) == NIYAH_TRUE);

    /* Status strings. */
    assert(strcmp(niyah_status_to_string(NIYAH_OK), "ok") == 0);
    assert(strcmp(niyah_status_to_string(NIYAH_ERR_NO_WEIGHTS),
                  "no weights loaded") == 0);

    return 0;
}
