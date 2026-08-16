#include "niyah_graph.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int grow_nodes(NiyahGraph *g) {
    size_t next = g->node_capacity ? g->node_capacity * 2u : 32u;
    if (next < g->node_capacity || next > SIZE_MAX / sizeof(*g->nodes)) return 0;
    NiyahGraphNode *p = (NiyahGraphNode *)realloc(g->nodes, next * sizeof(*p));
    if (!p) return 0;
    g->nodes = p;
    g->node_capacity = next;
    return 1;
}

static int grow_edges(NiyahGraph *g) {
    size_t next = g->edge_capacity ? g->edge_capacity * 2u : 64u;
    if (next < g->edge_capacity || next > SIZE_MAX / sizeof(*g->edges)) return 0;
    NiyahGraphEdge *p = (NiyahGraphEdge *)realloc(g->edges, next * sizeof(*p));
    if (!p) return 0;
    g->edges = p;
    g->edge_capacity = next;
    return 1;
}

void niyah_graph_init(NiyahGraph *graph) {
    if (!graph) return;
    memset(graph, 0, sizeof(*graph));
}

void niyah_graph_free(NiyahGraph *graph) {
    if (!graph) return;
    free(graph->nodes);
    free(graph->edges);
    memset(graph, 0, sizeof(*graph));
}

int niyah_graph_add_node(NiyahGraph *graph, NiyahGraphNode node) {
    if (!graph || node.id == 0) return 0;
    if (niyah_graph_find_node(graph, node.id)) return 0;
    if (graph->node_count == graph->node_capacity && !grow_nodes(graph)) return 0;
    graph->nodes[graph->node_count++] = node;
    return 1;
}

int niyah_graph_add_edge(NiyahGraph *graph, NiyahGraphEdge edge) {
    if (!graph || edge.id == 0 || edge.from_id == 0 || edge.to_id == 0) return 0;
    if (!niyah_graph_find_node(graph, edge.from_id) || !niyah_graph_find_node(graph, edge.to_id)) return 0;
    if (niyah_graph_find_edge(graph, edge.id)) return 0;
    if (graph->edge_count == graph->edge_capacity && !grow_edges(graph)) return 0;
    graph->edges[graph->edge_count++] = edge;
    return 1;
}

const NiyahGraphNode *niyah_graph_find_node(const NiyahGraph *graph, uint64_t id) {
    if (!graph || id == 0) return NULL;
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].id == id) return &graph->nodes[i];
    }
    return NULL;
}

const NiyahGraphEdge *niyah_graph_find_edge(const NiyahGraph *graph, uint64_t id) {
    if (!graph || id == 0) return NULL;
    for (size_t i = 0; i < graph->edge_count; ++i) {
        if (graph->edges[i].id == id) return &graph->edges[i];
    }
    return NULL;
}
