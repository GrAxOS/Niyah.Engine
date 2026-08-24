#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "evidence_envelope.h"
#include "evidence_graph.h"
#include "evidence_reasoner.h"

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

static void test_envelope(void)
{
    const uint8_t hash[4] = {0xDE, 0xAD, 0xBE, 0xEF};

    NiyahEvidenceEnvelope* env = NULL;
    assert(niyah_evidence_envelope_create(&env, "src-1", "citation",
                                          hash, sizeof(hash))
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(env != NULL);
    assert(strcmp(env->source_id, "src-1") == 0);
    assert(strcmp(env->evidence_type, "citation") == 0);
    assert(env->content_hash_size == 4);
    assert(memcmp(env->content_hash, hash, 4) == 0);

    /* A fresh envelope is unverified with zero confidence: evidence is not
     * trusted until something verifies it. */
    assert(env->verified == false);
    assert(env->confidence == 0.0f);
    assert(env->verified_at == 0);
    assert(env->created_at > 0);

    /* Verification records a timestamp and a bounded confidence. */
    assert(niyah_evidence_envelope_verify(env, 0.9f)
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(env->verified == true);
    assert(CLOSE(env->confidence, 0.9f));
    assert(env->verified_at > 0);

    /* Confidence is a probability; out-of-range values are rejected. */
    assert(niyah_evidence_envelope_verify(env, 1.5f)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);
    assert(niyah_evidence_envelope_verify(env, -0.1f)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);
    assert(CLOSE(env->confidence, 0.9f));   /* unchanged */

    /* Serialization: NULL buffer with size 0 is a size query. */
    size_t needed = 0;
    assert(niyah_evidence_envelope_serialize(env, NULL, 0, &needed)
           != NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(needed > 0);

    char buf[512];
    size_t written = 0;
    assert(niyah_evidence_envelope_serialize(env, buf, sizeof(buf), &written)
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(written == needed);
    assert(strstr(buf, "\"source_id\":\"src-1\"") != NULL);
    assert(strstr(buf, "\"evidence_type\":\"citation\"") != NULL);
    assert(strstr(buf, "deadbeef") != NULL);
    assert(strstr(buf, "\"verified\":true") != NULL);

    /* Truncation is reported, not silently accepted. */
    char tiny[8];
    assert(niyah_evidence_envelope_serialize(env, tiny, sizeof(tiny), NULL)
           != NIYAH_EVIDENCE_ENVELOPE_OK);

    niyah_evidence_envelope_destroy(env);

    /*
     * An over-long source id is rejected rather than truncated. Truncating
     * would silently alias two distinct provenance records onto one id.
     */
    char long_id[400];
    memset(long_id, 'x', sizeof(long_id) - 1u);
    long_id[sizeof(long_id) - 1u] = '\0';

    NiyahEvidenceEnvelope* rejected = NULL;
    assert(niyah_evidence_envelope_create(&rejected, long_id, "citation",
                                          NULL, 0)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);
    assert(rejected == NULL);

    /* An oversized hash is rejected too. */
    uint8_t big_hash[128] = {0};
    assert(niyah_evidence_envelope_create(&rejected, "s", "t",
                                          big_hash, sizeof(big_hash))
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);

    /* NULL required arguments. */
    assert(niyah_evidence_envelope_create(NULL, "s", "t", NULL, 0)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);
    assert(niyah_evidence_envelope_create(&rejected, NULL, "t", NULL, 0)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);

    /* A hash size with no hash pointer is contradictory. */
    assert(niyah_evidence_envelope_create(&rejected, "s", "t", NULL, 8)
           == NIYAH_EVIDENCE_ENVELOPE_INVALID_ARGS);

    niyah_evidence_envelope_destroy(NULL);
}

static void test_graph_and_reasoner(void)
{
    NiyahEvidenceGraph* graph = NULL;
    assert(niyah_evidence_graph_create(&graph, 8, 16)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(graph != NULL);
    assert(graph->node_count == 0);

    /* Two pieces of verified evidence and one bare claim. */
    NiyahEvidenceEnvelope* strong = NULL;
    NiyahEvidenceEnvelope* weak = NULL;
    assert(niyah_evidence_envelope_create(&strong, "peer-review", "paper",
                                          NULL, 0)
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(niyah_evidence_envelope_create(&weak, "blog-post", "opinion",
                                          NULL, 0)
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(niyah_evidence_envelope_verify(strong, 0.9f)
           == NIYAH_EVIDENCE_ENVELOPE_OK);
    assert(niyah_evidence_envelope_verify(weak, 0.8f)
           == NIYAH_EVIDENCE_ENVELOPE_OK);

    size_t n_strong = 99, n_weak = 99, n_claim = 99;
    assert(niyah_evidence_graph_add_node(graph, "strong", strong, &n_strong)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_graph_add_node(graph, "weak", weak, &n_weak)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_graph_add_node(graph, "claim", NULL, &n_claim)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(n_strong == 0 && n_weak == 1 && n_claim == 2);
    assert(graph->node_count == 3);

    /* Duplicate node ids are rejected so ids stay unique keys. */
    assert(niyah_evidence_graph_add_node(graph, "claim", NULL, NULL)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);

    /* strong SUPPORTS claim, weight 1.0 */
    assert(niyah_evidence_graph_add_edge(graph, n_strong, n_claim,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);

    /* A claim cannot support itself. */
    assert(niyah_evidence_graph_add_edge(graph, n_claim, n_claim,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);

    /* Weights are bounded, and endpoints must exist. */
    assert(niyah_evidence_graph_add_edge(graph, n_strong, n_claim,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        5.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);
    assert(niyah_evidence_graph_add_edge(graph, 99, n_claim,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);

    /* Aggregate confidence = 1.0 * 0.9 = 0.9 */
    float net = 0.0f;
    assert(niyah_evidence_graph_aggregate_confidence(graph, n_claim, &net)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(CLOSE(net, 0.9f));

    size_t supporting[8];
    size_t count = 0;
    assert(niyah_evidence_graph_find_supporting(graph, n_claim, supporting,
                                                &count, 8)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(count == 1);
    assert(supporting[0] == n_strong);

    /* A NULL output array counts without materialising ids. */
    size_t count_only = 0;
    assert(niyah_evidence_graph_find_supporting(graph, n_claim, NULL,
                                                &count_only, 0)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(count_only == 1);

    /* Reasoner verdict: 0.9 clears the SUPPORTED threshold. */
    NiyahEvidenceReasoner* reasoner = NULL;
    assert(niyah_evidence_reasoner_create(&reasoner, graph)
           == NIYAH_EVIDENCE_REASONER_OK);

    NiyahEvidenceVerdict verdict;
    assert(niyah_evidence_reasoner_evaluate_claim(reasoner, n_claim, &verdict)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(verdict.verdict_type == NIYAH_EVIDENCE_VERDICT_SUPPORTED);
    assert(verdict.supporting_count == 1);
    assert(verdict.contradicting_count == 0);
    assert(CLOSE(verdict.net_confidence, 0.9f));

    /* Now add contradicting evidence: net = 0.9 - 0.8 = 0.1 -> UNCERTAIN. */
    assert(niyah_evidence_graph_add_edge(graph, n_weak, n_claim,
                                        NIYAH_EVIDENCE_EDGE_CONTRADICTS,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);

    assert(niyah_evidence_graph_aggregate_confidence(graph, n_claim, &net)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(CLOSE(net, 0.1f));

    assert(niyah_evidence_reasoner_evaluate_claim(reasoner, n_claim, &verdict)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(verdict.verdict_type == NIYAH_EVIDENCE_VERDICT_UNCERTAIN);
    assert(verdict.supporting_count == 1);
    assert(verdict.contradicting_count == 1);

    size_t against[8];
    size_t against_count = 0;
    assert(niyah_evidence_graph_find_contradicting(graph, n_claim, against,
                                                   &against_count, 8)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(against_count == 1);
    assert(against[0] == n_weak);

    /*
     * IMPLIES and REFERENCES are structural, not evidential: they must not
     * move the confidence needle or show up in the support counts.
     */
    const float before = verdict.net_confidence;
    assert(niyah_evidence_graph_add_edge(graph, n_weak, n_claim,
                                        NIYAH_EVIDENCE_EDGE_REFERENCES,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_reasoner_evaluate_claim(reasoner, n_claim, &verdict)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(CLOSE(verdict.net_confidence, before));
    assert(verdict.supporting_count == 1);
    assert(verdict.contradicting_count == 1);

    /* An isolated claim with no evidence is UNCERTAIN, not SUPPORTED. */
    size_t isolated = 0;
    assert(niyah_evidence_graph_add_node(graph, "isolated", NULL, &isolated)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_reasoner_evaluate_claim(reasoner, isolated, &verdict)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(verdict.verdict_type == NIYAH_EVIDENCE_VERDICT_UNCERTAIN);
    assert(verdict.supporting_count == 0);
    assert(verdict.contradicting_count == 0);
    assert(CLOSE(verdict.net_confidence, 0.0f));

    /* Comparison: the well-supported claim beats the isolated one. */
    NiyahEvidenceComparison cmp;
    assert(niyah_evidence_reasoner_compare_claims(reasoner, n_claim,
                                                  isolated, &cmp)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(cmp.stronger_claim == n_claim);
    assert(cmp.compared_at > 0);

    /* Comparing a claim with itself is a genuine tie -> SIZE_MAX sentinel. */
    assert(niyah_evidence_reasoner_compare_claims(reasoner, isolated,
                                                  isolated, &cmp)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(cmp.stronger_claim == SIZE_MAX);

    /* An explicit CONTRADICTS edge is reported as contradictory. */
    assert(niyah_evidence_graph_add_edge(graph, n_strong, n_weak,
                                        NIYAH_EVIDENCE_EDGE_CONTRADICTS,
                                        1.0f, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_reasoner_compare_claims(reasoner, n_strong,
                                                  n_weak, &cmp)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(cmp.are_contradictory == true);

    /* Verdict serialization. */
    char buf[256];
    size_t written = 0;
    assert(niyah_evidence_reasoner_serialize_verdict(&verdict, buf,
                                                     sizeof(buf), &written)
           == NIYAH_EVIDENCE_REASONER_OK);
    assert(written > 0);
    assert(strstr(buf, "\"verdict\"") != NULL);
    assert(strstr(buf, "\"net_confidence\"") != NULL);

    /* Out-of-range node indices. */
    assert(niyah_evidence_reasoner_evaluate_claim(reasoner, 999, &verdict)
           == NIYAH_EVIDENCE_REASONER_INVALID_ARGS);
    assert(niyah_evidence_graph_aggregate_confidence(graph, 999, &net)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);

    /* A reasoner needs a graph. */
    NiyahEvidenceReasoner* orphan = NULL;
    assert(niyah_evidence_reasoner_create(&orphan, NULL)
           == NIYAH_EVIDENCE_REASONER_INVALID_ARGS);

    niyah_evidence_reasoner_destroy(reasoner);
    niyah_evidence_reasoner_destroy(NULL);

    /* The graph borrows envelopes, so destroying it must leave them valid. */
    niyah_evidence_graph_destroy(graph);
    assert(strong->verified == true);
    assert(CLOSE(strong->confidence, 0.9f));

    niyah_evidence_envelope_destroy(strong);
    niyah_evidence_envelope_destroy(weak);

    /* Capacity validation. */
    NiyahEvidenceGraph* zero = NULL;
    assert(niyah_evidence_graph_create(&zero, 0, 4)
           == NIYAH_EVIDENCE_GRAPH_INVALID_ARGS);
    niyah_evidence_graph_destroy(NULL);
}

static void test_graph_capacity(void)
{
    /* Exactly one node and one edge of headroom. */
    NiyahEvidenceGraph* graph = NULL;
    assert(niyah_evidence_graph_create(&graph, 2, 1)
           == NIYAH_EVIDENCE_GRAPH_OK);

    assert(niyah_evidence_graph_add_node(graph, "a", NULL, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);
    assert(niyah_evidence_graph_add_node(graph, "b", NULL, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);

    /* Third node exceeds max_nodes. */
    assert(niyah_evidence_graph_add_node(graph, "c", NULL, NULL)
           == NIYAH_EVIDENCE_GRAPH_ERROR);
    assert(graph->node_count == 2);

    assert(niyah_evidence_graph_add_edge(graph, 0, 1,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        0.5f, NULL)
           == NIYAH_EVIDENCE_GRAPH_OK);

    /* Second edge exceeds max_edges. */
    assert(niyah_evidence_graph_add_edge(graph, 1, 0,
                                        NIYAH_EVIDENCE_EDGE_SUPPORTS,
                                        0.5f, NULL)
           == NIYAH_EVIDENCE_GRAPH_ERROR);
    assert(graph->edge_count == 1);

    niyah_evidence_graph_destroy(graph);
}

int main(void)
{
    test_envelope();
    test_graph_and_reasoner();
    test_graph_capacity();
    return 0;
}
