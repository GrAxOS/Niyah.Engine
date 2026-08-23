#ifndef NIYAH_EVIDENCE_REASONER_H
#define NIYAH_EVIDENCE_REASONER_H

#include "evidence_graph.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
    #ifdef NIYAH_BRIDGE_EXPORTS
        #define NIYAH_EVIDENCE_API __declspec(dllexport)
    #else
        #define NIYAH_EVIDENCE_API __declspec(dllimport)
    #endif
#else
    #define NIYAH_EVIDENCE_API __attribute__((visibility("default")))
#endif

typedef enum {
    NIYAH_EVIDENCE_REASONER_OK = 0,
    NIYAH_EVIDENCE_REASONER_ERROR = 1,
    NIYAH_EVIDENCE_REASONER_OUT_OF_MEMORY = 2,
    NIYAH_EVIDENCE_REASONER_INVALID_ARGS = 3
} NiyahEvidenceReasonerStatus;

typedef enum {
    NIYAH_EVIDENCE_VERDICT_SUPPORTED = 0,
    NIYAH_EVIDENCE_VERDICT_LIKELY = 1,
    NIYAH_EVIDENCE_VERDICT_UNCERTAIN = 2,
    NIYAH_EVIDENCE_VERDICT_UNLIKELY = 3
} NiyahEvidenceVerdictType;

typedef struct NiyahEvidenceVerdict {
    NiyahEvidenceVerdictType verdict_type;
    float net_confidence;
    size_t supporting_count;
    size_t contradicting_count;
    uint64_t evaluated_at;
} NiyahEvidenceVerdict;

typedef struct NiyahEvidenceComparison {
    NiyahEvidenceVerdict verdict_a;
    NiyahEvidenceVerdict verdict_b;
    bool are_contradictory;
    size_t stronger_claim; /* SIZE_MAX if tie */
    uint64_t compared_at;
} NiyahEvidenceComparison;

typedef struct NiyahEvidenceReasoner {
    NiyahEvidenceGraph *graph;
    uint64_t created_at;
} NiyahEvidenceReasoner;

/* ============================================================================
 * Reasoner lifecycle
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceReasonerStatus niyah_evidence_reasoner_create(
    NiyahEvidenceReasoner **out,
    NiyahEvidenceGraph *graph);

NIYAH_EVIDENCE_API void niyah_evidence_reasoner_destroy(
    NiyahEvidenceReasoner *reasoner);

/* ============================================================================
 * Reasoning: Evaluate a claim
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceReasonerStatus niyah_evidence_reasoner_evaluate_claim(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_node_index,
    NiyahEvidenceVerdict *verdict);

/* ============================================================================
 * Reasoning: Compare two claims
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceReasonerStatus niyah_evidence_reasoner_compare_claims(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_a_index,
    size_t claim_b_index,
    NiyahEvidenceComparison *comparison);

/* ============================================================================
 * Serialization (for audit)
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceReasonerStatus niyah_evidence_reasoner_serialize_verdict(
    const NiyahEvidenceVerdict *verdict,
    char *buffer,
    size_t buffer_size,
    size_t *out_size);

#endif /* NIYAH_EVIDENCE_REASONER_H */
