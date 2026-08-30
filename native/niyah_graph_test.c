#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "niyah.h"

int main(void)
{
    NiyahGraph* g = niyah_graph_create();
    assert(g != NULL);
    assert(g->n_nodes == 0);
    assert(g->n_edges == 0);

    NiyahGraphNode* a = niyah_graph_add_node(g, "a", "Alpha", NULL);
    assert(a != NULL);
    assert(strcmp(a->id, "a") == 0);
    assert(strcmp(a->label, "Alpha") == 0);
    assert(g->n_nodes == 1);

    NiyahGraphNode* b = niyah_graph_add_node(g, "b", "Beta", NULL);
    assert(b != NULL && b != a);
    assert(g->n_nodes == 2);

    /* Adding the same id twice returns the existing node instead of creating
     * a duplicate, so ids stay unique keys. */
    NiyahGraphNode* again = niyah_graph_add_node(g, "a", "Alpha again", NULL);
    assert(again == a);
    assert(g->n_nodes == 2);

    /* Lookup. */
    assert(niyah_graph_find_node(g, "a") == a);
    assert(niyah_graph_find_node(g, "b") == b);
    assert(niyah_graph_find_node(g, "missing") == NULL);
    assert(niyah_graph_find_node(g, NULL) == NULL);

    /* Edges. The old header made this impossible to populate: node.edges was
     * `struct NiyahGraphEdge**` while NiyahGraphEdge typedef'd an anonymous
     * struct, so the two types never matched. */
    NiyahGraphEdge* e = niyah_graph_add_edge(g, "a", "b", "supports", 0.75f);
    assert(e != NULL);
    assert(e->from == a);
    assert(e->to == b);
    assert(strcmp(e->relation, "supports") == 0);
    assert(fabsf(e->weight - 0.75f) < 1e-6f);
    assert(g->n_edges == 1);

    /* The edge is reachable from the source node's adjacency list. */
    assert(a->n_edges == 1);
    assert(a->edges != NULL);
    assert(a->edges[0] == e);

    /* Referencing an unknown id auto-creates the endpoint. */
    NiyahGraphEdge* e2 = niyah_graph_add_edge(g, "a", "c", "references", 0.5f);
    assert(e2 != NULL);
    assert(g->n_nodes == 3);
    assert(niyah_graph_find_node(g, "c") != NULL);

    /* Neighbors. */
    NiyahGraphNode* out[8];
    int32_t n = niyah_graph_neighbors(g, "a", out, 8);
    assert(n == 2);

    bool saw_b = false, saw_c = false;
    for (int32_t i = 0; i < n; ++i) {
        if (strcmp(out[i]->id, "b") == 0) saw_b = true;
        if (strcmp(out[i]->id, "c") == 0) saw_c = true;
    }
    assert(saw_b && saw_c);

    /* max_out is honoured. */
    assert(niyah_graph_neighbors(g, "a", out, 1) == 1);

    /* A node with no outgoing edges has no neighbors. */
    assert(niyah_graph_neighbors(g, "b", out, 8) == 0);
    assert(niyah_graph_neighbors(g, "missing", out, 8) == 0);

    /* Growth past the initial capacity. */
    for (int i = 0; i < 64; ++i) {
        char id[32];
        snprintf(id, sizeof(id), "n%d", i);
        assert(niyah_graph_add_node(g, id, id, NULL) != NULL);
    }
    assert(g->n_nodes == 3 + 64);

    /* Degenerate inputs. */
    assert(niyah_graph_add_node(NULL, "x", "X", NULL) == NULL);
    assert(niyah_graph_add_node(g, NULL, "X", NULL) == NULL);
    assert(niyah_graph_add_edge(g, NULL, "b", "r", 1.0f) == NULL);

    niyah_graph_destroy(g);
    niyah_graph_destroy(NULL);

    return 0;
}
