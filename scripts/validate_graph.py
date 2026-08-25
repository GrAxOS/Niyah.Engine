#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_NODE_TYPES = {
    "domain", "topic", "concept", "practice", "technology", "language",
    "standard", "protocol", "algorithm", "architecture", "component",
    "constraint", "requirement", "risk", "decision", "evidence", "document",
}

ALLOWED_EDGE_TYPES = {
    "contains", "part_of", "related_to", "implements", "enables", "depends_on",
    "requires", "constrains", "satisfies", "supports", "derived_from", "causes",
    "mitigates", "conflicts_with", "contradicts", "supersedes", "validated_by",
    "evidenced_by", "uses", "validates",
}


def validate_graph(graph: dict) -> tuple[bool, list[str]]:
    """Validate graph structure and return (is_valid, errors)."""
    errors = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    if not nodes:
        errors.append("Graph has no nodes")
    
    node_ids = {n.get("id") for n in nodes}
    
    for i, node in enumerate(nodes):
        if not node.get("id"):
            errors.append(f"Node {i} missing id")
        if not node.get("type"):
            errors.append(f"Node {node.get('id', i)} missing type")
        elif node["type"] not in ALLOWED_NODE_TYPES:
            errors.append(f"Node {node.get('id')} has unknown type: {node['type']}")
        if not node.get("label"):
            errors.append(f"Node {node.get('id', i)} missing label")
    
    for i, edge in enumerate(edges):
        if not edge.get("id"):
            errors.append(f"Edge {i} missing id")
        if not edge.get("source"):
            errors.append(f"Edge {edge.get('id', i)} missing source")
        elif edge["source"] not in node_ids:
            errors.append(f"Edge {edge.get('id')} source {edge['source']} not in nodes")
        if not edge.get("target"):
            errors.append(f"Edge {edge.get('id', i)} missing target")
        elif edge["target"] not in node_ids:
            errors.append(f"Edge {edge.get('id')} target {edge['target']} not in nodes")
        if not edge.get("type"):
            errors.append(f"Edge {edge.get('id', i)} missing type")
        elif edge["type"] not in ALLOWED_EDGE_TYPES:
            errors.append(f"Edge {edge.get('id')} has unknown type: {edge['type']}")
    
    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate graph structure")
    parser.add_argument("--graph", default=str(ROOT / "data" / "canonical_graph_v2.json"))
    args = parser.parse_args()
    
    try:
        graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
        is_valid, errors = validate_graph(graph)
        
        if is_valid:
            print("✓ Graph is valid")
            return 0
        else:
            print(f"✗ Graph has {len(errors)} errors:", file=sys.stderr)
            for error in errors[:20]:
                print(f"  - {error}", file=sys.stderr)
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
