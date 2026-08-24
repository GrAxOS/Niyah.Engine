#include "niyah.h"

#include <string.h>

/*
 * These were previously defined twice: once as `static inline` in niyah.h and
 * once here without `static`, and this file referenced an undefined
 * NIYAH_VERSION macro. Both are fixed: the header only declares, this file
 * defines, and NIYAH_VERSION is built from the component macros.
 */

const char* niyah_version(void)
{
    return NIYAH_VERSION;
}

int32_t niyah_version_major(void) { return NIYAH_VERSION_MAJOR; }
int32_t niyah_version_minor(void) { return NIYAH_VERSION_MINOR; }
int32_t niyah_version_patch(void) { return NIYAH_VERSION_PATCH; }

const char* niyah_truth_to_string(NiyahTruth truth)
{
    switch (truth) {
        case NIYAH_FALSE:   return "false";
        case NIYAH_TRUE:    return "true";
        case NIYAH_UNKNOWN: return "unknown";
        default:            return "invalid";
    }
}

NiyahTruth niyah_truth_from_string(const char* text)
{
    if (!text) {
        return NIYAH_UNKNOWN;
    }
    if (strcmp(text, "true") == 0 || strcmp(text, "1") == 0) {
        return NIYAH_TRUE;
    }
    if (strcmp(text, "false") == 0 || strcmp(text, "0") == 0) {
        return NIYAH_FALSE;
    }
    return NIYAH_UNKNOWN;
}

/* Kleene strong three-valued logic. UNKNOWN propagates unless the other
 * operand already decides the result. */

NiyahTruth niyah_truth_not(NiyahTruth a)
{
    switch (a) {
        case NIYAH_TRUE:  return NIYAH_FALSE;
        case NIYAH_FALSE: return NIYAH_TRUE;
        default:          return NIYAH_UNKNOWN;
    }
}

NiyahTruth niyah_truth_and(NiyahTruth a, NiyahTruth b)
{
    if (a == NIYAH_FALSE || b == NIYAH_FALSE) {
        return NIYAH_FALSE;
    }
    if (a == NIYAH_TRUE && b == NIYAH_TRUE) {
        return NIYAH_TRUE;
    }
    return NIYAH_UNKNOWN;
}

NiyahTruth niyah_truth_or(NiyahTruth a, NiyahTruth b)
{
    if (a == NIYAH_TRUE || b == NIYAH_TRUE) {
        return NIYAH_TRUE;
    }
    if (a == NIYAH_FALSE && b == NIYAH_FALSE) {
        return NIYAH_FALSE;
    }
    return NIYAH_UNKNOWN;
}

NiyahTruth niyah_truth_implies(NiyahTruth a, NiyahTruth b)
{
    return niyah_truth_or(niyah_truth_not(a), b);
}

const char* niyah_status_to_string(NiyahStatus status)
{
    switch (status) {
        case NIYAH_OK:                return "ok";
        case NIYAH_ERR_INVALID_ARG:   return "invalid argument";
        case NIYAH_ERR_OUT_OF_MEMORY: return "out of memory";
        case NIYAH_ERR_IO:            return "i/o error";
        case NIYAH_ERR_UNSUPPORTED:   return "unsupported";
        case NIYAH_ERR_NO_WEIGHTS:    return "no weights loaded";
        case NIYAH_ERR_SHAPE:         return "shape mismatch";
        case NIYAH_ERR_NOT_FOUND:     return "not found";
        case NIYAH_ERR_OVERFLOW:      return "arithmetic overflow";
        default:                      return "unknown status";
    }
}
