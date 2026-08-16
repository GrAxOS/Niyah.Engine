#include "niyah_truth.h"

const char *niyah_truth_class_name(NiyahTruthClass classification) {
    switch (classification) {
        case NIYAH_TRUTH_FACT: return "FACT";
        case NIYAH_TRUTH_INFERENCE: return "INFERENCE";
        case NIYAH_TRUTH_UNCERTAIN: return "UNCERTAIN";
        case NIYAH_TRUTH_UNKNOWN: return "UNKNOWN";
        case NIYAH_TRUTH_CONFLICTED: return "CONFLICTED";
        default: return "UNKNOWN";
    }
}
