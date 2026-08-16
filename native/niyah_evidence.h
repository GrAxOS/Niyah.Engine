#ifndef NIYAH_EVIDENCE_H
#define NIYAH_EVIDENCE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_EVIDENCE_FACT = 0,
    NIYAH_EVIDENCE_INFERENCE = 1,
    NIYAH_EVIDENCE_UNCERTAIN = 2,
    NIYAH_EVIDENCE_UNKNOWN = 3,
    NIYAH_EVIDENCE_CONFLICTED = 4
} NiyahEvidenceKind;

typedef struct {
    NiyahEvidenceKind kind;
    const char *source;
    const char *claim;
} NiyahEvidence;

const char *niyah_evidence_kind_name(NiyahEvidenceKind kind);
bool niyah_evidence_valid(const NiyahEvidence *evidence);

#ifdef __cplusplus
}
#endif

#endif
