#ifndef NIYAH_RUNTIME_H
#define NIYAH_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "niyah_core.h"
#include "niyah_evidence.h"
#include "niyah_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t id;
    uint64_t parent_id;
    uint64_t created_unix_ms;
    uint32_t kind;
    uint32_t status;
    char label[128];
} NiyahRuntimeNode;

typedef struct {
    uint64_t id;
    uint64_t from_id;
    uint64_t to_id;
    uint32_t kind;
    uint32_t status;
    float weight;
} NiyahRuntimeEdge;

typedef struct {
    NiyahPool pool;
    NiyahRuntimeNode *nodes;
    NiyahRuntimeEdge *edges;
    uint32_t node_count;
    uint32_t node_capacity;
    uint32_t edge_count;
    uint32_t edge_capacity;
    uint64_t next_id;
} NiyahRuntimeGraph;

bool niyah_runtime_graph_init(NiyahRuntimeGraph *graph,
                              void *buffer,
                              size_t buffer_size,
                              uint32_t node_capacity,
                              uint32_t edge_capacity);

NiyahRuntimeNode *niyah_runtime_add_node(NiyahRuntimeGraph *graph,
                                         uint32_t kind,
                                         const char *label,
                                         uint64_t parent_id,
                                         uint64_t now_unix_ms);

NiyahRuntimeEdge *niyah_runtime_add_edge(NiyahRuntimeGraph *graph,
                                         uint64_t from_id,
                                         uint64_t to_id,
                                         uint32_t kind,
                                         float weight);

const NiyahRuntimeNode *niyah_runtime_find_node(const NiyahRuntimeGraph *graph,
                                                uint64_t id);

void niyah_runtime_clear(NiyahRuntimeGraph *graph);

#ifdef __cplusplus
}
#endif

#endif
