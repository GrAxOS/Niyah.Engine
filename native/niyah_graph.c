#include "niyah.h"

#include <stdlib.h>
#include <string.h>

/* Was: `// Graph stubs`. */

static char* dup_str(const char* s)
{
    if (!s) {
        return NULL;
    }
    const size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) {
        memcpy(out, s, n);
    }
    return out;
}

NiyahGraph* niyah_graph_create(void)
{
    return (NiyahGraph*)calloc(1, sizeof(NiyahGraph));
}

void niyah_graph_destroy(NiyahGraph* graph)
{
    if (!graph) {
        return;
    }

    for (int32_t i = 0; i < graph->n_edges; ++i) {
        if (graph->edges[i]) {
            free(graph->edges[i]->relation);
            free(graph->edges[i]);
        }
    }
    free(graph->edges);

    for (int32_t i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i]) {
            free(graph->nodes[i]->id);
            free(graph->nodes[i]->label);
            free(graph->nodes[i]->edges); /* edge objects owned by graph->edges */
            free(graph->nodes[i]);
        }
    }
    free(graph->nodes);

    free(graph);
}

NiyahGraphNode* niyah_graph_find_node(const NiyahGraph* graph, const char* id)
{
    if (!graph || !id) {
        return NULL;
    }
    for (int32_t i = 0; i < graph->n_nodes; ++i) {
        NiyahGraphNode* node = graph->nodes[i];
        if (node && node->id && strcmp(node->id, id) == 0) {
            return node;
        }
    }
    return NULL;
}

NiyahGraphNode* niyah_graph_add_node(NiyahGraph* graph,
                                     const char* id,
                                     const char* label,
                                     void* data)
{
    if (!graph || !id) {
        return NULL;
    }

    NiyahGraphNode* existing = niyah_graph_find_node(graph, id);
    if (existing) {
        return existing; /* ids are unique: adding twice is a lookup */
    }

    if (graph->n_nodes == graph->node_capacity) {
        const int32_t next = graph->node_capacity ? graph->node_capacity * 2 : 16;
        NiyahGraphNode** grown = (NiyahGraphNode**)realloc(
            graph->nodes, (size_t)next * sizeof(NiyahGraphNode*));
        if (!grown) {
            return NULL;
        }
        graph->nodes = grown;
        graph->node_capacity = next;
    }

    NiyahGraphNode* node = (NiyahGraphNode*)calloc(1, sizeof(NiyahGraphNode));
    if (!node) {
        return NULL;
    }

    node->id = dup_str(id);
    if (!node->id) {
        free(node);
        return NULL;
    }
    node->label = label ? dup_str(label) : NULL;
    node->data = data;

    graph->nodes[graph->n_nodes++] = node;
    return node;
}

static int node_append_edge(NiyahGraphNode* node, NiyahGraphEdge* edge)
{
    if (node->n_edges == node->edge_capacity) {
        const int32_t next = node->edge_capacity ? node->edge_capacity * 2 : 4;
        NiyahGraphEdge** grown = (NiyahGraphEdge**)realloc(
            node->edges, (size_t)next * sizeof(NiyahGraphEdge*));
        if (!grown) {
            return -1;
        }
        node->edges = grown;
        node->edge_capacity = next;
    }
    node->edges[node->n_edges++] = edge;
    return 0;
}

NiyahGraphEdge* niyah_graph_add_edge(NiyahGraph* graph,
                                     const char* from_id,
                                     const char* to_id,
                                     const char* relation,
                                     float weight)
{
    if (!graph || !from_id || !to_id) {
        return NULL;
    }

    NiyahGraphNode* from = niyah_graph_add_node(graph, from_id, NULL, NULL);
    NiyahGraphNode* to = niyah_graph_add_node(graph, to_id, NULL, NULL);
    if (!from || !to) {
        return NULL;
    }

    if (graph->n_edges == graph->edge_capacity) {
        const int32_t next = graph->edge_capacity ? graph->edge_capacity * 2 : 16;
        NiyahGraphEdge** grown = (NiyahGraphEdge**)realloc(
            graph->edges, (size_t)next * sizeof(NiyahGraphEdge*));
        if (!grown) {
            return NULL;
        }
        graph->edges = grown;
        graph->edge_capacity = next;
    }

    NiyahGraphEdge* edge = (NiyahGraphEdge*)calloc(1, sizeof(NiyahGraphEdge));
    if (!edge) {
        return NULL;
    }

    edge->from = from;
    edge->to = to;
    edge->relation = relation ? dup_str(relation) : NULL;
    edge->weight = weight;

    if (node_append_edge(from, edge) != 0) {
        free(edge->relation);
        free(edge);
        return NULL;
    }

    graph->edges[graph->n_edges++] = edge;
    return edge;
}

int32_t niyah_graph_neighbors(const NiyahGraph* graph,
                              const char* id,
                              NiyahGraphNode** out,
                              int32_t max_out)
{
    if (!graph || !id || !out || max_out <= 0) {
        return 0;
    }

    NiyahGraphNode* node = niyah_graph_find_node(graph, id);
    if (!node) {
        return 0;
    }

    int32_t written = 0;
    for (int32_t i = 0; i < node->n_edges && written < max_out; ++i) {
        if (node->edges[i] && node->edges[i]->to) {
            out[written++] = node->edges[i]->to;
        }
    }
    return written;
}
