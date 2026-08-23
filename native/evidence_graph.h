#ifndef NIYAH_EVIDENCE_GRAPH_H
#define NIYAH_EVIDENCE_GRAPH_H

#include "evidence_envelope.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
    #ifdef NIYAH_BRIDGE_EXPORTS
        #define NIYAH_EVIDENCE_API __declspec(dllexport)
    #else
        #define NIYAH_EVIDENCE_API __declspec(dllimport)
    #endif
#else
    #define NIYAH_EVIDENCE_API __attribute__((visibility("default")))
#endif

typedef enum {
    NIYAH_EVIDENCE_GRAPH_OK = 0,
    NIYAH_EVIDENCE_GRAPH_ERROR = 1,
    NIYAH_EVIDENCE_GRAPH_OUT_OF_MEMORY = 2,
    NIYAH_EVIDENCE_GRAPH_INVALID_ARGS = 3
} NiyahEvidenceGraphStatus;

typedef enum {
    NIYAH_EVIDENCE_EDGE_SUPPORTS = 0,
    NIYAH_EVIDENCE_EDGE_CONTRADICTS = 1,
    NIYAH_EVIDENCE_EDGE_IMPLIES = 2,
    NIYAH_EVIDENCE_EDGE_REFERENCES = 3
} NiyahEvidenceEdgeType;

typedef struct NiyahEvidenceNode {
    char node_id[256];
    NiyahEvidenceEnvelope *envelope;
    uint64_t created_at;
} NiyahEvidenceNode;

typedef struct NiyahEvidenceEdge {
    size_t from_node;
    size_t to_node;
    NiyahEvidenceEdgeType edge_type;
    float weight;
    uint64_t created_at;
} NiyahEvidenceEdge;

typedef struct NiyahEvidenceGraph {
    NiyahEvidenceNode *nodes;
    NiyahEvidenceEdge *edges;
    size_t max_nodes;
    size_t max_edges;
    size_t node_count;
    size_t edge_count;
} NiyahEvidenceGraph;

/* ============================================================================
 * Graph lifecycle
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_create(
    NiyahEvidenceGraph **out,
    size_t max_nodes,
    size_t max_edges);

NIYAH_EVIDENCE_API void niyah_evidence_graph_destroy(
    NiyahEvidenceGraph *graph);

/* ============================================================================
 * Node management
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_add_node(
    NiyahEvidenceGraph *graph,
    const char *node_id,
    NiyahEvidenceEnvelope *envelope,
    size_t *out_node_index);

/* ============================================================================
 * Edge management
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_add_edge(
    NiyahEvidenceGraph *graph,
    size_t from_node,
    size_t to_node,
    NiyahEvidenceEdgeType edge_type,
    float weight,
    size_t *out_edge_index);

/* ============================================================================
 * Query: Find supporting evidence
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_find_supporting(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *supporting_nodes,
    size_t *supporting_count,
    size_t max_supporting);

/* ============================================================================
 * Query: Find contradicting evidence
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_find_contradicting(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    size_t *contradicting_nodes,
    size_t *contradicting_count,
    size_t max_contradicting);

/* ============================================================================
 * Confidence aggregation
 * ============================================================================ */

NIYAH_EVIDENCE_API NiyahEvidenceGraphStatus niyah_evidence_graph_aggregate_confidence(
    const NiyahEvidenceGraph *graph,
    size_t node_index,
    float *net_confidence);

#endif /* NIYAH_EVIDENCE_GRAPH_H */
