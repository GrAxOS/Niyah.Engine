#include "evidence_reasoner.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_create(
        NiyahEvidenceReasoner** out,
        NiyahEvidenceGraph*     graph) {
    if (!out || !graph) return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;

    *out = (NiyahEvidenceReasoner*)calloc(1, sizeof(NiyahEvidenceReasoner));
    if (!*out) return NIYAH_EVIDENCE_REASONER_OUT_OF_MEMORY;

    (*out)->graph      = graph;
    (*out)->created_at = (uint64_t)time(NULL);
    return NIYAH_EVIDENCE_REASONER_OK;
}

void niyah_evidence_reasoner_destroy(NiyahEvidenceReasoner* reasoner) {
    free(reasoner);
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_evaluate_claim(
        NiyahEvidenceReasoner* reasoner,
        size_t claim_node_index,
        NiyahEvidenceVerdict*  verdict) {
    if (!reasoner || !verdict) return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;

    float net = 0.0f;
    niyah_evidence_graph_aggregate_confidence(
        reasoner->graph, claim_node_index, &net);

    size_t sup_buf[256], con_buf[256];
    size_t sup_count = 0, con_count = 0;
    niyah_evidence_graph_find_supporting(
        reasoner->graph, claim_node_index, sup_buf, &sup_count, 256);
    niyah_evidence_graph_find_contradicting(
        reasoner->graph, claim_node_index, con_buf, &con_count, 256);

    verdict->net_confidence      = net;
    verdict->supporting_count    = sup_count;
    verdict->contradicting_count = con_count;
    verdict->evaluated_at        = (uint64_t)time(NULL);

    if      (net >= 0.7f) verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_SUPPORTED;
    else if (net >= 0.3f) verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_LIKELY;
    else if (net >= -0.3f)verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_UNCERTAIN;
    else                  verdict->verdict_type = NIYAH_EVIDENCE_VERDICT_UNLIKELY;

    return NIYAH_EVIDENCE_REASONER_OK;
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_compare_claims(
        NiyahEvidenceReasoner*   reasoner,
        size_t                   claim_a_index,
        size_t                   claim_b_index,
        NiyahEvidenceComparison* comparison) {
    if (!reasoner || !comparison) return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;

    NiyahEvidenceReasonerStatus sa =
        niyah_evidence_reasoner_evaluate_claim(reasoner, claim_a_index, &comparison->verdict_a);
    NiyahEvidenceReasonerStatus sb =
        niyah_evidence_reasoner_evaluate_claim(reasoner, claim_b_index, &comparison->verdict_b);
    if (sa != NIYAH_EVIDENCE_REASONER_OK || sb != NIYAH_EVIDENCE_REASONER_OK)
        return NIYAH_EVIDENCE_REASONER_ERROR;

    float net_a = comparison->verdict_a.net_confidence;
    float net_b = comparison->verdict_b.net_confidence;

    comparison->are_contradictory = (net_a > 0.0f && net_b < 0.0f) ||
                                    (net_a < 0.0f && net_b > 0.0f);
    comparison->compared_at = (uint64_t)time(NULL);

    if      (net_a > net_b + 0.05f) comparison->stronger_claim = 0;
    else if (net_b > net_a + 0.05f) comparison->stronger_claim = 1;
    else                             comparison->stronger_claim = SIZE_MAX;

    return NIYAH_EVIDENCE_REASONER_OK;
}

NiyahEvidenceReasonerStatus niyah_evidence_reasoner_serialize_verdict(
        const NiyahEvidenceVerdict* verdict,
        char* buffer, size_t buffer_size, size_t* out_size) {
    if (!verdict || !buffer || buffer_size == 0) return NIYAH_EVIDENCE_REASONER_INVALID_ARGS;

    static const char* verdict_names[] = {
        "SUPPORTED", "LIKELY", "UNCERTAIN", "UNLIKELY"
    };
    const char* vname = (verdict->verdict_type <= NIYAH_EVIDENCE_VERDICT_UNLIKELY)
                        ? verdict_names[verdict->verdict_type] : "UNKNOWN";

    int n = snprintf(buffer, buffer_size,
        "{\"verdict\":\"%s\",\"net_confidence\":%.4f,"
        "\"supporting\":%zu,\"contradicting\":%zu,\"evaluated_at\":%llu}",
        vname, verdict->net_confidence,
        verdict->supporting_count, verdict->contradicting_count,
        (unsigned long long)verdict->evaluated_at);

    if (n < 0 || (size_t)n >= buffer_size) return NIYAH_EVIDENCE_REASONER_ERROR;
    if (out_size) *out_size = (size_t)n;
    return NIYAH_EVIDENCE_REASONER_OK;
}
