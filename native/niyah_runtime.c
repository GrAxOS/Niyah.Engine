#include "niyah_runtime.h"

#include <string.h>

static bool valid_kind(uint32_t kind) {
    return kind <= NIYAH_GRAPH_SYSTEM;
}

bool niyah_runtime_graph_init(NiyahRuntimeGraph *graph,
                              void *buffer,
                              size_t buffer_size,
                              uint32_t node_capacity,
                              uint32_t edge_capacity) {
    if (!graph || !buffer || node_capacity == 0 || edge_capacity == 0)
        return false;

    niyah_pool_init(&graph->pool, buffer, buffer_size);
    graph->nodes = (NiyahRuntimeNode *)niyah_pool_alloc(
        &graph->pool,
        (size_t)node_capacity * sizeof(*graph->nodes),
        NIYAH_ALIGN_DEFAULT);
    graph->edges = (NiyahRuntimeEdge *)niyah_pool_alloc(
        &graph->pool,
        (size_t)edge_capacity * sizeof(*graph->edges),
        NIYAH_ALIGN_DEFAULT);

    if (!graph->nodes || !graph->edges) {
        memset(graph, 0, sizeof(*graph));
        return false;
    }

    graph->node_count = 0;
    graph->node_capacity = node_capacity;
    graph->edge_count = 0;
    graph->edge_capacity = edge_capacity;
    graph->next_id = 1;
    return true;
}

NiyahRuntimeNode *niyah_runtime_add_node(NiyahRuntimeGraph *graph,
                                         uint32_t kind,
                                         const char *label,
                                         uint64_t parent_id,
                                         uint64_t now_unix_ms) {
    if (!graph || graph->node_count >= graph->node_capacity || !valid_kind(kind))
        return NULL;

    NiyahRuntimeNode *node = &graph->nodes[graph->node_count++];
    memset(node, 0, sizeof(*node));
    node->id = graph->next_id++;
    node->parent_id = parent_id;
    node->created_unix_ms = now_unix_ms;
    node->kind = kind;
    node->status = 0;

    if (label) {
        (void)strncpy(node->label, label, sizeof(node->label) - 1);
        node->label[sizeof(node->label) - 1] = '\0';
    }
    return node;
}

NiyahRuntimeEdge *niyah_runtime_add_edge(NiyahRuntimeGraph *graph,
                                         uint64_t from_id,
                                         uint64_t to_id,
                                         uint32_t kind,
                                         float weight) {
    if (!graph || graph->edge_count >= graph->edge_capacity || from_id == 0 || to_id == 0)
        return NULL;
    if (from_id == to_id || !valid_kind(kind))
        return NULL;
    if (!niyah_runtime_find_node(graph, from_id) || !niyah_runtime_find_node(graph, to_id))
        return NULL;

    NiyahRuntimeEdge *edge = &graph->edges[graph->edge_count++];
    edge->id = graph->next_id++;
    edge->from_id = from_id;
    edge->to_id = to_id;
    edge->kind = kind;
    edge->status = 0;
    edge->weight = weight;
    return edge;
}

const NiyahRuntimeNode *niyah_runtime_find_node(const NiyahRuntimeGraph *graph,
                                                uint64_t id) {
    if (!graph || id == 0)
        return NULL;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].id == id)
            return &graph->nodes[i];
    }
    return NULL;
}

void niyah_runtime_clear(NiyahRuntimeGraph *graph) {
    if (!graph)
        return;
    graph->node_count = 0;
    graph->edge_count = 0;
    graph->next_id = 1;
}
