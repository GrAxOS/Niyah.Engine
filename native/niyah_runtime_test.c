#include "niyah_runtime.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    uint8_t buffer[16384];
    NiyahRuntimeGraph graph;

    assert(niyah_runtime_graph_init(&graph, buffer, sizeof(buffer), 32, 64));

    NiyahRuntimeNode *query = niyah_runtime_add_node(
        &graph, NIYAH_GRAPH_NODE_SYSTEM, "query", 0, 1);
    NiyahRuntimeNode *source = niyah_runtime_add_node(
        &graph, NIYAH_GRAPH_NODE_SOURCE, "source", query->id, 2);

    assert(query != NULL);
    assert(source != NULL);
    assert(graph.node_count == 2);

    NiyahRuntimeEdge *edge = niyah_runtime_add_edge(
        &graph, query->id, source->id, NIYAH_GRAPH_EDGE_RELATES, 0.75f);
    assert(edge != NULL);
    assert(graph.edge_count == 1);
    assert(niyah_runtime_find_node(&graph, source->id) == source);

    assert(niyah_runtime_add_edge(
        &graph, query->id, 99999, NIYAH_GRAPH_EDGE_RELATES, 1.0f) == NULL);
    assert(niyah_runtime_add_edge(
        &graph, query->id, query->id, NIYAH_GRAPH_EDGE_RELATES, 1.0f) == NULL);

    niyah_runtime_clear(&graph);
    assert(graph.node_count == 0);
    assert(graph.edge_count == 0);

    return 0;
}
