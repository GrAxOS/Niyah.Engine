#include "evidence_graph.h"
<<<<<<< HEAD
=======

>>>>>>> origin/main
#include <stdlib.h>
#include <string.h>
#include <time.h>

<<<<<<< HEAD
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
=======
/*
 * Was: `// Evidence graph stubs`.
 *
 * Ownership: the graph owns its node and edge arrays but NOT the
 * NiyahEvidenceEnvelope objects the nodes point at. Envelopes may be shared
 * between graphs, so destroying a graph must not invalidate them.
 */

static uint64_t now_unix(void)
{
    const time_t t = time(NULL);
    return (t == (time_t)-1) ? 0u : (uint64_t)t;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_create(
    NiyahEvidenceGraph **out,
    size_t max_nodes,
    size_t max_edges)
{
    if (!out || max_nodes == 0u || max_edges == 0u) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    *out = NULL;

    NiyahEvidenceGraph *graph =
        (NiyahEvidenceGraph *)calloc(1, sizeof(NiyahEvidenceGraph));
    if (!graph) {
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    graph->nodes = (NiyahEvidenceNode *)calloc(max_nodes,
                                               sizeof(NiyahEvidenceNode));
    graph->edges = (NiyahEvidenceEdge *)calloc(max_edges,
                                               sizeof(NiyahEvidenceEdge));
    if (!graph->nodes || !graph->edges) {
        free(graph->nodes);
        free(graph->edges);
        free(graph);
        return NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY;
    }

    graph->max_nodes = max_nodes;
    graph->max_edges = max_edges;
    graph->node_count = 0u;
    graph->edge_count = 0u;

    *out = graph;
    return NIYAH_EVIDENCE_GRAPH_OK;
}

void niyah_evidence_graph_destroy(NiyahEvidenceGraph *graph)
{
    if (!graph) {
        return;
    }
>>>>>>> origin/main
    free(graph->nodes);
    free(graph->edges);
    free(graph);
}

NiyahEvidenceGraphStatus niyah_evidence_graph_add_node(
<<<<<<< HEAD
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
=======
    NiyahEvidenceGraph *graph,
    const char *node_id,
    NiyahEvidenceEnvelope *envelope,
    size_t *out_node_index)
{
    if (!graph || !node_id) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (strlen(node_id) >= sizeof(graph->nodes[0].node_id)) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (graph->node_count >= graph->max_nodes) {
        return NIYAH_EVIDENCE_GRAPH_ERROR;
    }

    /* Node ids identify claims; duplicates would make queries ambiguous. */
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (strcmp(graph->nodes[i].node_id, node_id) == 0) {
            return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
        }
    }

    NiyahEvidenceNode *node = &graph->nodes[graph->node_count];
    memset(node, 0, sizeof(*node));
    memcpy(node->node_id, node_id, strlen(node_id));
    node->envelope = envelope;   /* borrowed, not owned */
    node->created_at = now_unix();

    if (out_node_index) {
        *out_node_index = graph->node_count;
    }
    ++graph->node_count;

>>>>>>> origin/main
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_add_edge(
<<<<<<< HEAD
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
=======
    NiyahEvidenceGraph *graph,
    size_t from_node,
    size_t to_node,
    NiyahEvidenceEdgeType edge_type,
    float weight,
    size_t *out_edge_index)
{
    if (!graph) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (from_node >= graph->node_count || to_node >= graph->node_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    /* A claim supporting or contradicting itself is not meaningful evidence. */
    if (from_node == to_node) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (edge_type != NIYAH_EVIDENCE_EDGE_SUPPORTS &&
        edge_type != NIYAH_EVIDENCE_EDGE_CONTRADICTS &&
        edge_type != NIYAH_EVIDENCE_EDGE_IMPLIES &&
        edge_type != NIYAH_EVIDENCE_EDGE_REFERENCES) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (!(weight >= 0.0f) || !(weight <= 1.0f)) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (graph->edge_count >= graph->max_edges) {
        return NIYAH_EVIDENCE_GRAPH_ERROR;
    }

    NiyahEvidenceEdge *edge = &graph->edges[graph->edge_count];
    edge->from_node = from_node;
    edge->to_node = to_node;
    edge->edge_type = edge_type;
    edge->weight = weight;
    edge->created_at = now_unix();

    if (out_edge_index) {
        *out_edge_index = graph->edge_count;
    }
    ++graph->edge_count;

    return NIYAH_EVIDENCE_GRAPH_OK;
}

static NiyahEvidenceGraphStatus collect_incoming(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    NiyahEvidenceEdgeType want,
    size_t *nodes,
    size_t *count,
    size_t max_nodes)
{
    if (!graph || !count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }
    if (node_index >= graph->node_count) {
        return NIYAH_EVIDENCE_GRAPH_INVALID_ARGS;
    }

    *count = 0u;

    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &graph->edges[i];
        if (edge->to_node != node_index || edge->edge_type != want) {
            continue;
        }
        if (nodes && *count < max_nodes) {
            nodes[*count] = edge->from_node;
        } else if (nodes) {
            /* Caller's buffer is full; stop rather than overrun it. */
            return NIYAH_EVIDENCE_GRAPH_ERROR;
        }
        ++(*count);
    }

>>>>>>> origin/main
    return NIYAH_EVIDENCE_GRAPH_OK;
}

NiyahEvidenceGraphStatus niyah_evidence_graph_find_supporting(
<<<<<<< HEAD
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
=======
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *supporting_nodes,
    size_t *supporting_count,
    size_t max_supporting)
{
    return collect_incoming(graph, node_index,
                            NIYAH_EVIDENCE_EDGE_SUPPORTS,
                            supporting_nodes, supporting_count,
                            max_supporting);
}

NiyahEvidenceGraphStatus niyah_evidence_graph_find_contradicting(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *contradicting_nodes,
    size_t *contradicting_count,
    size_t max_contradicting)
{
    return collect_incoming(graph, node_index,
                            NIYAH_EVIDENCE_EDGE_CONTRADICTS,
                            contradicting_nodes, contradicting_count,
                            max_contradicting);
}

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

    /*
     * Net confidence = weighted support - weighted contradiction, where each
     * contribution is the edge weight times the source envelope's own
     * confidence. An unverified envelope contributes its stored confidence,
     * which is 0 until niyah_evidence_envelope_verify is called -- so
     * unverified evidence cannot move the needle.
     */
    double support = 0.0;
    double against = 0.0;

    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NiyahEvidenceEdge *edge = &graph->edges[i];
        if (edge->to_node != node_index) {
            continue;
        }

        const NiyahEvidenceNode *src = &graph->nodes[edge->from_node];
        const double conf = src->envelope
            ? (double)src->envelope->confidence : 0.0;
        const double contribution = (double)edge->weight * conf;

        if (edge->edge_type == NIYAH_EVIDENCE_EDGE_SUPPORTS) {
            support += contribution;
        } else if (edge->edge_type == NIYAH_EVIDENCE_EDGE_CONTRADICTS) {
            against += contribution;
        }
    }

    double net = support - against;
    if (net > 1.0) {
        net = 1.0;
    } else if (net < -1.0) {
        net = -1.0;
    }

    *net_confidence = (float)net;
>>>>>>> origin/main
    return NIYAH_EVIDENCE_GRAPH_OK;
}
