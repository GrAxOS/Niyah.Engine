#ifndef NIYAH_GRAPH_H
#define NIYAH_GRAPH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_GRAPH_NODE_SOURCE = 1,
    NIYAH_GRAPH_NODE_DOCUMENT = 2,
    NIYAH_GRAPH_NODE_CLAIM = 3,
    NIYAH_GRAPH_NODE_PERSON = 4,
    NIYAH_GRAPH_NODE_ORGANIZATION = 5,
    NIYAH_GRAPH_NODE_TECHNOLOGY = 6,
    NIYAH_GRAPH_NODE_CODE = 7,
    NIYAH_GRAPH_NODE_SYSTEM = 8,
    NIYAH_GRAPH_NODE_UNKNOWN = 255
} NiyahGraphNodeKind;

typedef enum {
    NIYAH_GRAPH_EDGE_SUPPORTS = 1,
    NIYAH_GRAPH_EDGE_CONTRADICTS = 2,
    NIYAH_GRAPH_EDGE_RELATES = 3,
    NIYAH_GRAPH_EDGE_DERIVED_FROM = 4,
    NIYAH_GRAPH_EDGE_IMPLEMENTED_BY = 5,
    NIYAH_GRAPH_EDGE_DEPENDS_ON = 6
} NiyahGraphEdgeKind;

typedef struct {
    uint64_t id;
    NiyahGraphNodeKind kind;
    char label[192];
    char source_ref[512];
    uint8_t content_hash[32];
} NiyahGraphNode;

typedef struct {
    uint64_t id;
    uint64_t from_id;
    uint64_t to_id;
    NiyahGraphEdgeKind kind;
    double weight;
} NiyahGraphEdge;

typedef struct {
    NiyahGraphNode *nodes;
    size_t node_count;
    size_t node_capacity;
    NiyahGraphEdge *edges;
    size_t edge_count;
    size_t edge_capacity;
} NiyahGraph;

void niyah_graph_init(NiyahGraph *graph);
void niyah_graph_free(NiyahGraph *graph);
int niyah_graph_add_node(NiyahGraph *graph, NiyahGraphNode node);
int niyah_graph_add_edge(NiyahGraph *graph, NiyahGraphEdge edge);
const NiyahGraphNode *niyah_graph_find_node(const NiyahGraph *graph, uint64_t id);
const NiyahGraphEdge *niyah_graph_find_edge(const NiyahGraph *graph, uint64_t id);

#ifdef __cplusplus
}
#endif
#endif
