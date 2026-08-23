#include "evidence_graph.h"
#include "niyah_core.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ============================================================================
 * Evidence graph lifecycle
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_create(
    NiyahEvidenceGraph **out,
    size_t max_nodes,
    size_t max_edges)
{
    if (!out || max_nodes == 0 || max_edges == 0) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    NiyahEvidenceGraph *graph = calloc(1, sizeof(NiyahEvidenceGraph));
    if (!graph) {
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    graph->nodes = calloc(max_nodes, sizeof(NiyahEvidenceNode));
    if (!graph->nodes) {
        free(graph);
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    graph->edges = calloc(max_edges, sizeof(NiyahEvidenceEdge));
    if (!graph->edges) {
        free(graph->nodes);
        free(graph);
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    graph->max_nodes = max_nodes;
    graph->max_edges = max_edges;
    graph->node_count = 0;
    graph->edge_count = 0;

    *out = graph;
    return NIYAH_EVIDENCE_GRAPH_OK;
}

void niyah_evidence_graph_destroy(NiyahEvidenceGraph *graph) {
    if (!graph) {
        return;
    }

    /* Free node envelopes */
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].envelope) {
            niyah_evidence_envelope_destroy(graph->nodes[i].envelope);
        }
    }

    free(graph->edges);
    free(graph->nodes);
    free(graph);
}

/* ============================================================================
 * Node management
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_add_node(
    NiyahEvidenceGraph *graph,
    const char *node_id,
    NiyahEvidenceEnvelope *envelope,
    size_t *out_node_index)
{
    if (!graph || !node_id || !envelope || !out_node_index) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    if (graph->node_count >= graph->max_nodes) {
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    /* Check for duplicate node_id */
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (strcmp(graph->nodes[i].node_id, node_id) == 0) {
            return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
        }
    }

    NiyahEvidenceNode *node = &graph->nodes[graph->node_count];
    memset(node, 0, sizeof(NiyahEvidenceNode));

    size_t node_id_len = strlen(node_id);
    if (node_id_len >= sizeof(node->node_id)) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    memcpy(node->node_id, node_id, node_id_len + 1);

    node->envelope = envelope;
    node->created_at = niyah_core_timestamp_now();

    *out_node_index = graph->node_count;
    graph->node_count++;

    return NIYAH_EVIDENCE_GRAPH_OK;
}

/* ============================================================================
 * Edge management
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_add_edge(
    NiyahEvidenceGraph *graph,
    size_t from_node,
    size_t to_node,
    NiyahEvidenceEdgeType edge_type,
    float weight,
    size_t *out_edge_index)
{
    if (!graph || !out_edge_index) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    if (from_node >= graph->node_count || to_node >= graph->node_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    if (graph->edge_count >= graph->max_edges) {
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    NiyahEvidenceEdge *edge = &graph->edges[graph->edge_count];
    edge->from_node = from_node;
    edge->to_node = to_node;
    edge->edge_type = edge_type;
    edge->weight = weight;
    edge->created_at = niyah_core_timestamp_now();

    *out_edge_index = graph->edge_count;
    graph->edge_count++;

    return NIYAH_EVIDENCE_GRAPH_OK;
}

/* ============================================================================
 * Query: Find supporting evidence
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_find_supporting(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *supporting_nodes,
    size_t *supporting_count,
    size_t max_supporting)
{
    if (!graph || !supporting_nodes || !supporting_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    *supporting_count = 0;

    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &graph->edges[i];

        if (edge->to_node == node_index &&
            (edge->edge_type == NIYAH_EVIDENCE_EDGE_SUPPORTS ||
             edge->edge_type == NIYAH_EVIDENCE_EDGE_IMPLIES)) {
            
            if (*supporting_count >= max_supporting) {
                return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
            }

            supporting_nodes[*supporting_count] = edge->from_node;
            (*supporting_count)++;
        }
    }

    return NIYAH_EVIDENCE_GRAPH_OK;
}

/* ============================================================================
 * Query: Find contradicting evidence
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_find_contradicting(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *contradicting_nodes,
    size_t *contradicting_count,
    size_t max_contradicting)
{
    if (!graph || !contradicting_nodes || !contradicting_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    *contradicting_count = 0;

    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &graph->edges[i];

        if (edge->to_node == node_index &&
            edge->edge_type == NIYAH_EVIDENCE_EDGE_CONTRADICTS) {
            
            if (*contradicting_count >= max_contradicting) {
                return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
            }

            contradicting_nodes[*contradicting_count] = edge->from_node;
            (*contradicting_count)++;
        }
    }

    return NIYAH_EVIDENCE_GRAPH_OK;
}

/* ============================================================================
 * Confidence aggregation
 * ============================================================================ */

NiyahEvidenceGraphStatus niyah_evidence_graph_aggregate_confidence(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    float *net_confidence)
{
    if (!graph || !net_confidence) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    if (node_index >= graph->node_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    const NiyahEvidenceNode *node = &graph->nodes[node_index];
    if (!node->envelope || !node->envelope->verified) {
        *net_confidence = 0.0f;
        return NIYAH_EVIDENCE_GRAPH_OK;
    }

    float base_confidence = node->envelope->confidence;

    /* Find supporting evidence */
    size_t supporting[64];
    size_t supporting_count = 0;
    niyah_evidence_graph_find_supporting(
        graph, node_index, supporting, &supporting_count, 64
    );

    /* Find contradicting evidence */
    size_t contradicting[64];
    size_t contradicting_count = 0;
    niyah_evidence_graph_find_contradicting(
        graph, node_index, contradicting, &contradicting_count, 64
    );

    /* Simple aggregation: base + (supporting * 0.1) - (contradicting * 0.2) */
    float net = base_confidence;
    net += (float)supporting_count * 0.1f;
    net -= (float)contradicting_count * 0.2f;

    /* Clamp to [0, 1] */
    if (net < 0.0f) net = 0.0f;
    if (net > 1.0f) net = 1.0f;

    *net_confidence = net;
    return NIYAH_EVIDENCE_GRAPH_OK;
}
