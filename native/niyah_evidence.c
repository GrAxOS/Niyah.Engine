#include "niyah_evidence.h"

const char *niyah_evidence_kind_name(NiyahEvidenceKind kind) {
    switch (kind) {
        case NIYAH_EVIDENCE_FACT: return "fact";
        case NIYAH_EVIDENCE_INFERENCE: return "inference";
        case NIYAH_EVIDENCE_UNCERTAIN: return "uncertain";
        case NIYAH_EVIDENCE_UNKNOWN: return "unknown";
        case NIYAH_EVIDENCE_CONFLICTED: return "conflicted";
        default: return "unknown";
    }
}

bool niyah_evidence_valid(const NiyahEvidence *evidence) {
    if (evidence == NULL || evidence->claim == NULL || evidence->claim[0] == '\0') {
        return false;
    }

    if (evidence->kind == NIYAH_EVIDENCE_FACT &&
        (evidence->source == NULL || evidence->source[0] == '\0')) {
        return false;
    }

    return evidence->kind <= NIYAH_EVIDENCE_CONFLICTED;
}
