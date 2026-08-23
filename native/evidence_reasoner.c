#include "evidence_reasoner.h"
#include "niyah_core.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ============================================================================
 * Evidence reasoner lifecycle
 * ============================================================================ */

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_create(
    NiyahEvidenceReasoner **out,
    NiyahEvidenceGraph *graph)
{
    if (!out || !graph) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    NiyahEvidenceReasoner *reasoner = calloc(1, sizeof(NiyahEvidenceReasoner));
    if (!reasoner) {
        return NIYAH_EVIDENCE_REASONER_OUT_OF_MEMORY;
    }

    reasoner->graph = graph;
    reasoner->created_at = niyah_core_timestamp_now();

    *out = reasoner;
    return NIYAH_EVIDENCE_REASONER_OK;
}

void niyah_evidence_reasoner_destroy(NiyahEvidenceReasoner *reasoner) {
    if (!reasoner) {
        return;
    }
    free(reasoner);
}

/* ============================================================================
 * Reasoning: Evaluate a claim
 * ============================================================================ */

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_evaluate_claim(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_node_index,
    NiyahEvidenceVerdict *verdict)
{
    if (!reasoner || !verdict) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    if (claim_node_index >= reasoner->graph->node_count) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    memset(verdict, 0, sizeof(NiyahEvidenceVerdict));

    /* Get base confidence */
    float net_confidence = 0.0f;
    NiyahEvidenceGraphStatus status = niyah_evidence_graph_aggregate_confidence(
        reasoner->graph,
        claim_node_index,
        &net_confidence
    );

    if (status != NIYAH_EVIDENCE_GRAPH_OK) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }

    verdict->net_confidence = net_confidence;

    /* Find supporting evidence */
    size_t supporting[64];
    size_t supporting_count = 0;
    niyah_evidence_graph_find_supporting(
        reasoner->graph,
        claim_node_index,
        supporting,
        &supporting_count,
        64
    );
    verdict->supporting_count = supporting_count;

    /* Find contradicting evidence */
    size_t contradicting[64];
    size_t contradicting_count = 0;
    niyah_evidence_graph_find_contradicting(
        reasoner->graph,
        claim_node_index,
        contradicting,
        &contradicting_count,
        64
    );
    verdict->contradicting_count = contradicting_count;

    /* Determine verdict */
    if (net_confidence >= 0.8f && contradicting_count == 0) {
        verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_SUPPORTED;
    } else if (net_confidence >= 0.5f && contradicting_count <= supporting_count) {
        verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_LIKELY;
    } else if (net_confidence >= 0.3f) {
        verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_UNCERTAIN;
    } else {
        verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_UNLIKELY;
    }

    verdict->evaluated_at = niyah_core_timestamp_now();

    return NIYAH_EVIDENCE_REASONER_OK;
}

/* ============================================================================
 * Reasoning: Compare two claims
 * ============================================================================ */

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_compare_claims(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_a_index,
    size_t claim_b_index,
    NiyahEvidenceComparison *comparison)
{
    if (!reasoner || !comparison) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    if (claim_a_index >= reasoner->graph->node_count ||
        claim_b_index >= reasoner->graph->node_count) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    memset(comparison, 0, sizeof(NiyahEvidenceComparison));

    /* Evaluate both claims */
    NiyahEvidenceVerdict verdict_a, verdict_b;
    NiyahEvidenceReasonerStatus status_a = niyah_evidence_reasoner_evaluate_claim(
        reasoner, claim_a_index, &verdict_a
    );
    NiyahEvidenceReasonerStatus status_b = niyah_evidence_reasoner_evaluate_claim(
        reasoner, claim_b_index, &verdict_b
    );

    if (status_a != NIYAH_EVIDENCE_REASONER_OK ||
        status_b != NIYAH_EVIDENCE_REASONER_OK) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }

    comparison->verdict_a = verdict_a;
    comparison->verdict_b = verdict_b;

    /* Check if claims contradict each other */
    /* Look for an edge between them */
    bool found_contradiction = false;
    for (size_t i = 0; i < reasoner->graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &reasoner->graph->edges[i];
        if ((edge->from_node == claim_a_index && edge->to_node == claim_b_index) ||
            (edge->from_node == claim_b_index && edge->to_node == claim_a_index)) {
            if (edge->edge_type == NIYAH_EVIDENCE_EDGE_CONTRADICTS) {
                found_contradiction = true;
                break;
            }
        }
    }

    comparison->are_contradictory = found_contradiction;

    /* Determine which claim is stronger */
    if (verdict_a.net_confidence > verdict_b.net_confidence) {
        comparison->stronger_claim = claim_a_index;
    } else if (verdict_b.net_confidence > verdict_a.net_confidence) {
        comparison->stronger_claim = claim_b_index;
    } else {
        comparison->stronger_claim = SIZE_MAX; /* Tie */
    }

    comparison->compared_at = niyah_core_timestamp_now();

    return NIYAH_EVIDENCE_REASONER_OK;
}

/* ============================================================================
 * Serialization (for audit)
 * ============================================================================ */

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_serialize_verdict(
    const NiyahEvidenceVerdict *verdict,
    char *buffer,
    size_t buffer_size,
    size_t *out_size)
{
    if (!verdict || !buffer || !out_size) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    const char *verdict_str = "unknown";
    switch (verdict->verdict_type) {
        case NIYAH_EVIDENCE_VERDICT_SUPPORTED: verdict_str = "supported"; break;
        case NIYAH_EVIDENCE_VERDICT_LIKELY: verdict_str = "likely"; break;
        case NIYAH_EVIDENCE_VERDICT_UNCERTAIN: verdict_str = "uncertain"; break;
        case NIYAH_EVIDENCE_VERDICT_UNLIKELY: verdict_str = "unlikely"; break;
    }

    int written = snprintf(
        buffer,
        buffer_size,
        "{\"verdict_type\":\"%s\",\"net_confidence\":%.4f,\"supporting_count\":%zu,\"contradicting_count\":%zu,\"evaluated_at\":%llu}",
        verdict_str,
        verdict->net_confidence,
        verdict->supporting_count,
        verdict->contradicting_count,
        (unsigned long long)verdict->evaluated_at
    );

    if (written < 0 || (size_t)written >= buffer_size) {
        return NIYAH_EVIDENCE_REASONER_OUT_OF_MEMORY;
    }

    *out_size = (size_t)written;
    return NIYAH_EVIDENCE_REASONER_OK;
}
