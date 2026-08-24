#include "evidence_reasoner.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Was: `// Evidence reasoner stubs`. */

#define NIYAH_REASONER_SCAN_MAX 256u

static uint64_t now_unix(void)
{
    const time_t t = time(NULL);
    return (t == (time_t)-1) ? 0u : (uint64_t)t;
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_create(
    NiyahEvidenceReasoner **out,
    NiyahEvidenceGraph *graph)
{
    if (!out || !graph) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    *out = NULL;

    NiyahEvidenceReasoner *reasoner =
        (NiyahEvidenceReasoner *)calloc(1, sizeof(NiyahEvidenceReasoner));
    if (!reasoner) {
        return NIYAH_EVIDENCE_REASONER_OUT_OF_MEMORY;
    }

    reasoner->graph = graph;   /* borrowed */
    reasoner->created_at = now_unix();

    *out = reasoner;
    return NIYAH_EVIDENCE_REASONER_OK;
}

void niyah_evidence_reasoner_destroy(NiyahEvidenceReasoner *reasoner)
{
    /* The graph is owned by the caller; only the reasoner is released. */
    free(reasoner);
}

static NiyahEvidenceVerdictType classify(float net_confidence,
                                        size_t supporting,
                                        size_t contradicting)
{
    /* No evidence at all is not the same as balanced evidence. */
    if (supporting == 0u && contradicting == 0u) {
        return NIYAH_EVIDENCE_VERDICT_UNCERTAIN;
    }
    if (net_confidence >= 0.60f) {
        return NIYAH_EVIDENCE_VERDICT_SUPPORTED;
    }
    if (net_confidence >= 0.20f) {
        return NIYAH_EVIDENCE_VERDICT_LIKELY;
    }
    if (net_confidence > -0.20f) {
        return NIYAH_EVIDENCE_VERDICT_UNCERTAIN;
    }
    return NIYAH_EVIDENCE_VERDICT_UNLIKELY;
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_evaluate_claim(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_node_index,
    NiyahEvidenceVerdict *verdict)
{
    if (!reasoner || !reasoner->graph || !verdict) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }
    if (claim_node_index >= reasoner->graph->node_count) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    memset(verdict, 0, sizeof(*verdict));

    size_t supporting = 0u;
    size_t contradicting = 0u;

    /* Pass NULL to count without materialising the id list. */
    if (niyah_evidence_graph_find_supporting(
            reasoner->graph, claim_node_index, NULL, &supporting, 0u)
        != NIYAH_EVIDENCE_GRAPH_OK) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }
    if (niyah_evidence_graph_find_contradicting(
            reasoner->graph, claim_node_index, NULL, &contradicting, 0u)
        != NIYAH_EVIDENCE_GRAPH_OK) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }

    float net = 0.0f;
    if (niyah_evidence_graph_aggregate_confidence(
            reasoner->graph, claim_node_index, &net)
        != NIYAH_EVIDENCE_GRAPH_OK) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }

    verdict->net_confidence = net;
    verdict->supporting_count = supporting;
    verdict->contradicting_count = contradicting;
    verdict->verdict_type = classify(net, supporting, contradicting);
    verdict->evaluated_at = now_unix();

    return NIYAH_EVIDENCE_REASONER_OK;
}

static bool has_contradiction_between(const NiyahEvidenceGraph *graph,
                                     size_t a,
                                     size_t b)
{
    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &graph->edges[i];
        if (edge->edge_type != NIYAH_EVIDENCE_EDGE_CONTRADICTS) {
            continue;
        }
        if ((edge->from_node == a && edge->to_node == b) ||
            (edge->from_node == b && edge->to_node == a)) {
            return true;
        }
    }
    return false;
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_compare_claims(
    NiyahEvidenceReasoner *reasoner,
    size_t claim_a_index,
    size_t claim_b_index,
    NiyahEvidenceComparison *comparison)
{
    if (!reasoner || !reasoner->graph || !comparison) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    memset(comparison, 0, sizeof(*comparison));

    NiyahEvidenceReasonerStatus status = niyah_evidence_reasoner_evaluate_claim(
        reasoner, claim_a_index, &comparison->verdict_a);
    if (status != NIYAH_EVIDENCE_REASONER_OK) {
        return status;
    }
    status = niyah_evidence_reasoner_evaluate_claim(
        reasoner, claim_b_index, &comparison->verdict_b);
    if (status != NIYAH_EVIDENCE_REASONER_OK) {
        return status;
    }

    /* Contradictory if the graph says so explicitly, or if the two verdicts
     * land on opposite ends of the scale. */
    const bool explicit_conflict =
        has_contradiction_between(reasoner->graph, claim_a_index, claim_b_index);
    const bool verdict_conflict =
        (comparison->verdict_a.verdict_type == NIYAH_EVIDENCE_VERDICT_SUPPORTED &&
         comparison->verdict_b.verdict_type == NIYAH_EVIDENCE_VERDICT_UNLIKELY) ||
        (comparison->verdict_b.verdict_type == NIYAH_EVIDENCE_VERDICT_SUPPORTED &&
         comparison->verdict_a.verdict_type == NIYAH_EVIDENCE_VERDICT_UNLIKELY);

    comparison->are_contradictory = explicit_conflict || verdict_conflict;

    const float a = comparison->verdict_a.net_confidence;
    const float b = comparison->verdict_b.net_confidence;

    if (a > b) {
        comparison->stronger_claim = claim_a_index;
    } else if (b > a) {
        comparison->stronger_claim = claim_b_index;
    } else {
        comparison->stronger_claim = SIZE_MAX;   /* documented tie sentinel */
    }

    comparison->compared_at = now_unix();
    return NIYAH_EVIDENCE_REASONER_OK;
}

static const char *verdict_name(NiyahEvidenceVerdictType type)
{
    switch (type) {
        case NIYAH_EVIDENCE_VERDICT_SUPPORTED: return "supported";
        case NIYAH_EVIDENCE_VERDICT_LIKELY:    return "likely";
        case NIYAH_EVIDENCE_VERDICT_UNCERTAIN: return "uncertain";
        case NIYAH_EVIDENCE_VERDICT_UNLIKELY:  return "unlikely";
        default:                               return "unknown";
    }
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_serialize_verdict(
    const NiyahEvidenceVerdict *verdict,
    char *buffer,
    size_t buffer_size,
    size_t *out_size)
{
    if (!verdict) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }
    if (!buffer && buffer_size != 0u) {
        return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;
    }

    const int needed = snprintf(
        buffer, buffer_size,
        "{\"verdict\":\"%s\",\"net_confidence\":%.6f,"
        "\"supporting_count\":%zu,\"contradicting_count\":%zu,"
        "\"evaluated_at\":%" PRIu64 "}",
        verdict_name(verdict->verdict_type),
        (double)verdict->net_confidence,
        verdict->supporting_count,
        verdict->contradicting_count,
        verdict->evaluated_at);

    if (needed < 0) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }
    if (out_size) {
        *out_size = (size_t)needed;
    }
    if ((size_t)needed >= buffer_size) {
        return NIYAH_EVIDENCE_REASONER_ERROR;
    }

    return NIYAH_EVIDENCE_REASONER_OK;
}
