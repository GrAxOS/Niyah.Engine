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

// يعد الحمير: FACT بمصدر واحد = حمار
NiyahTruthClass niyah_truth_classify(const NiyahEvidenceStatus *s) {
    if (!s) return NIYAH_TRUTH_UNKNOWN;
    if (s->source_count == 0) return NIYAH_TRUTH_UNKNOWN;
    if (s->conflict_count > 0) {
        return (s->conflict_count >= s->source_count) ? NIYAH_TRUTH_CONFLICTED : NIYAH_TRUTH_UNCERTAIN;
    }
    if (s->source_count >= 2) return NIYAH_TRUTH_FACT;
    return NIYAH_TRUTH_INFERENCE;
}

int niyah_truth_is_donkey(const NiyahEvidenceStatus *s) {
    if (!s) return 1;
    if (s->source_count <= 1) return 1; // GPT/Grok/Gemini pattern
    return 0;
}
