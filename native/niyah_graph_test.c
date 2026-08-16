#include "niyah_graph.h"

#include <assert.h>
#include <string.h>

int main(void) {
    NiyahGraph g;
    niyah_graph_init(&g);

    NiyahGraphNode source = {0};
    source.id = 1;
    source.kind = NIYAH_GRAPH_NODE_SOURCE;
    strcpy(source.label, "Source");
    strcpy(source.source_ref, "https://example.invalid/source");

    NiyahGraphNode claim = {0};
    claim.id = 2;
    claim.kind = NIYAH_GRAPH_NODE_CLAIM;
    strcpy(claim.label, "Claim");

    assert(niyah_graph_add_node(&g, source));
    assert(niyah_graph_add_node(&g, claim));
    assert(!niyah_graph_add_node(&g, source));

    NiyahGraphEdge edge = {0};
    edge.id = 10;
    edge.from_id = 1;
    edge.to_id = 2;
    edge.kind = NIYAH_GRAPH_EDGE_SUPPORTS;
    edge.weight = 1.0;

    assert(niyah_graph_add_edge(&g, edge));
    assert(!niyah_graph_add_edge(&g, edge));
    assert(niyah_graph_find_node(&g, 1) != NULL);
    assert(niyah_graph_find_edge(&g, 10) != NULL);

    NiyahGraphEdge bad = { .id = 11, .from_id = 1, .to_id = 999 };
    assert(!niyah_graph_add_edge(&g, bad));

    niyah_graph_free(&g);
    assert(g.nodes == NULL && g.edges == NULL);
    return 0;
}
