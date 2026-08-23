#include "niyah.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* niyah_version(void) {
    return NIYAH_VERSION;
}

const char* niyah_truth_to_string(NiyahTruth truth) {
    switch (truth) {
        case NIYAH_FALSE: return "false";
        case NIYAH_TRUE: return "true";
        case NIYAH_UNKNOWN: return "unknown";
        default: return "invalid";
    }
}
