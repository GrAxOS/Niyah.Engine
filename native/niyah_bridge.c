#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Bridge API - stub implementations
// All actual types and functions are defined in niyah.h

const char* niyah_get_version(void) {
    return niyah_version();
}

const char* niyah_get_truth_string(NiyahTruth truth) {
    return niyah_truth_to_string(truth);
}

// TODO: Add actual bridge functions when niyah.h defines NiyahContext, etc.
