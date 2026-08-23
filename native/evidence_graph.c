#include "evidence_graph.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

NiyahEvidenceGraphStatus niyah_evidence_graph_create(
        NiyahEvidenceGraph** out,
        size_t max_nodes,
        size_t max_edges) {
    if (!out || max_nodes == 0) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;

    *out = (NiyahEvidenceGraph*)calloc(1, sizeof(NiyahEvidenceGraph));
    if (!*out) return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;

    (*out)->nodes = (NiyahEvidenceNode*)calloc(max_nodes, sizeof(NiyahEvidenceNode));
    (*out)->edges = (NiyahEvidenceEdge*)calloc(max_edges, sizeof(NiyahEvidenceEdge));
    if (!(*out)->nodes || !(*out)->edges) {
        free((*out)->nodes); free((*out)->edges); free(*out);
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    (*out)->max_nodes  = max_nodes;
    (*out)->max_edges  = max_edges;
    (*out)->node_count = 0;
    (*out)->edge_count = 0;
    return NIYAH_EVIDENCE_GRAPH_OK;
}

void niyah_evidence_graph_destroy(NiyahEvidenceGraph* graph) {
    if (!graph) return;
    free(graph->nodes);
    free(graph->edges);
    free(graph);
}

NiyahEvidenceGraphStatus niyah_evidence_graph_add_node(
        NiyahEvidenceGraph*  graph,
        const char*          node_id,
        NiyahEvidenceEnvelope* envelope,
        size_t*              out_node_index) {
    if (!graph || !node_id) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    if (graph->node_count >= graph->max_nodes) return NIYAH_EVIDENCE_GRAPH_ERROR;

    NiyahEvidenceNode* n = &graph->nodes[graph->node_count];
    strncpy(n->node_id, node_id, sizeof(n->node_id) - 1);
    n->envelope   = envelope;
    n->created_at = (uint64_t)time(NULL);

    if (out_node_index) *out_node_index = graph->node_count;
    graph->node_count++;
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_add_edge(
        NiyahEvidenceGraph*  graph,
        size_t from_node, size_t to_node,
        NiyahEvidenceEdgeType edge_type,
        float weight,
        size_t* out_edge_index) {
    if (!graph) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    if (from_node >= graph->node_count || to_node >= graph->node_count)
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    if (graph->edge_count >= graph->max_edges) return NIYAH_EVIDENCE_GRAPH_ERROR;

    NiyahEvidenceEdge* e = &graph->edges[graph->edge_count];
    e->from_node  = from_node;
    e->to_node    = to_node;
    e->edge_type  = edge_type;
    e->weight     = weight;
    e->created_at = (uint64_t)time(NULL);

    if (out_edge_index) *out_edge_index = graph->edge_count;
    graph->edge_count++;
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_find_supporting(
        const NiyahEvidenceGraph* graph,
        size_t node_index,
        size_t* supporting_nodes,
        size_t* supporting_count,
        size_t max_supporting) {
    if (!graph || !supporting_nodes || !supporting_count) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;

    *supporting_count = 0;
    for (size_t i = 0; i < graph->edge_count && *supporting_count < max_supporting; i++) {
        if (graph->edges[i].to_node == node_index &&
            graph->edges[i].edge_type == NIYAH_EVIDENCE_EDGE_SUPPORTS)
            supporting_nodes[(*supporting_count)++] = graph->edges[i].from_node;
    }
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_find_contradicting(
        const NiyahEvidenceGraph* graph,
        size_t node_index,
        size_t* contradicting_nodes,
        size_t* contradicting_count,
        size_t max_contradicting) {
    if (!graph || !contradicting_nodes || !contradicting_count) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;

    *contradicting_count = 0;
    for (size_t i = 0; i < graph->edge_count && *contradicting_count < max_contradicting; i++) {
        if (graph->edges[i].to_node == node_index &&
            graph->edges[i].edge_type == NIYAH_EVIDENCE_EDGE_CONTRADICTS)
            contradicting_nodes[(*contradicting_count)++] = graph->edges[i].from_node;
    }
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_aggregate_confidence(
        const NiyahEvidenceGraph* graph,
        size_t node_index,
        float* net_confidence) {
    if (!graph || !net_confidence) return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;

    float pos = 0.0f, neg = 0.0f;
    for (size_t i = 0; i < graph->edge_count; i++) {
        if (graph->edges[i].to_node != node_index) continue;
        if (graph->edges[i].edge_type == NIYAH_EVIDENCE_EDGE_SUPPORTS)
            pos += graph->edges[i].weight;
        else if (graph->edges[i].edge_type == NIYAH_EVIDENCE_EDGE_CONTRADICTS)
            neg += graph->edges[i].weight;
    }
    *net_confidence = pos - neg;
    return NIYAH_EVIDENCE_GRAPH_OK;
}
