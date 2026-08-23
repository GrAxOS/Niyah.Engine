#include "niyah.h"
#include <stdlib.h>
#include <string.h>

/* ── Knowledge graph ───────────────────────────────────────────────────── */

NiyahGraph* niyah_graph_create(void) {
    NiyahGraph* g = (NiyahGraph*)calloc(1, sizeof(NiyahGraph));
    return g;
}

void niyah_graph_destroy(NiyahGraph* g) {
    if (!g) return;
    for (int32_t i = 0; i < g->n_nodes; i++) {
        NiyahGraphNode* n = g->nodes[i];
        if (!n) continue;
        free(n->id);
        free(n->label);
        free(n->edges);
        free(n);
    }
    for (int32_t i = 0; i < g->n_edges; i++) {
        NiyahGraphEdge* e = g->edges[i];
        if (!e) continue;
        free(e->relation);
        free(e);
    }
    free(g->nodes);
    free(g->edges);
    free(g);
}

NiyahGraphNode* niyah_graph_add_node(NiyahGraph* g,
                                      const char* id,
                                      const char* label,
                                      void* data) {
    if (!g || !id) return NULL;

    NiyahGraphNode** tmp = (NiyahGraphNode**)realloc(
        g->nodes, (size_t)(g->n_nodes + 1) * sizeof(NiyahGraphNode*));
    if (!tmp) return NULL;
    g->nodes = tmp;

    NiyahGraphNode* node = (NiyahGraphNode*)calloc(1, sizeof(NiyahGraphNode));
    if (!node) return NULL;
    node->id    = _strdup(id);
    node->label = label ? _strdup(label) : NULL;
    node->data  = data;

    g->nodes[g->n_nodes++] = node;
    return node;
}

NiyahGraphEdge* niyah_graph_add_edge(NiyahGraph* g,
                                      NiyahGraphNode* from,
                                      NiyahGraphNode* to,
                                      const char* relation,
                                      float weight) {
    if (!g || !from || !to) return NULL;

    NiyahGraphEdge** tmp = (NiyahGraphEdge**)realloc(
        g->edges, (size_t)(g->n_edges + 1) * sizeof(NiyahGraphEdge*));
    if (!tmp) return NULL;
    g->edges = tmp;

    NiyahGraphEdge* edge = (NiyahGraphEdge*)calloc(1, sizeof(NiyahGraphEdge));
    if (!edge) return NULL;
    edge->from     = from;
    edge->to       = to;
    edge->relation = relation ? _strdup(relation) : NULL;
    edge->weight   = weight;

    g->edges[g->n_edges++] = edge;

    /* Attach edge to source node */
    void** ne = (void**)realloc(
        from->edges, (size_t)(from->n_edges + 1) * sizeof(void*));
    if (ne) {
        from->edges = (struct NiyahGraphEdge**)ne;
        from->edges[from->n_edges++] = edge;
    }

    return edge;
}

NiyahGraphNode* niyah_graph_find_node(const NiyahGraph* g, const char* id) {
    if (!g || !id) return NULL;
    for (int32_t i = 0; i < g->n_nodes; i++)
        if (g->nodes[i] && strcmp(g->nodes[i]->id, id) == 0)
            return g->nodes[i];
    return NULL;
}
