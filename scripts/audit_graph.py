#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
REQUIRED_EVIDENCE_EDGES = {"contradicts", "conflicts_with", "supersedes", "causes", "mitigates"}


def sha256_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def audit(graph: dict) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    evidence = graph.get("evidence", [])
    node_ids = [n.get("id") for n in nodes]
    node_id_set = set(node_ids)
    evidence_ids = {e.get("id") for e in evidence}

    duplicate_ids = sorted(k for k, v in Counter(node_ids).items() if k and v > 1)
    unknown_node_types = sorted({n.get("type") for n in nodes if n.get("type") not in ALLOWED_NODE_TYPES})
    unknown_edge_types = sorted({e.get("type") for e in edges if e.get("type") not in ALLOWED_EDGE_TYPES})
    orphan_edges = [e.get("id") for e in edges if e.get("source") not in node_id_set or e.get("target") not in node_id_set]
    unsupported_asserted_nodes = [
        n.get("id") for n in nodes
        if n.get("status") == "asserted" and not any(ev in evidence_ids for ev in n.get("provenance", []))
    ]
    missing_required_evidence = [
        e.get("id") for e in edges
        if e.get("type") in REQUIRED_EVIDENCE_EDGES
        and not any(ev in evidence_ids for ev in (e.get("provenance", []) + e.get("evidence", [])))
    ]

    valid_shape_nodes = all(bool(n.get("id")) and bool(n.get("type")) and bool(n.get("label")) for n in nodes)
    valid_shape_edges = all(bool(e.get("id")) and bool(e.get("source")) and bool(e.get("target")) and bool(e.get("type")) for e in edges)
    provenance_objects = len(nodes) + len(edges)
    provenance_bound = sum(bool(n.get("provenance")) for n in nodes) + sum(bool(e.get("provenance")) for e in edges)
    provenance_coverage = provenance_bound / provenance_objects if provenance_objects else 1.0
    completeness_score = (
        sum(bool(n.get("id")) and bool(n.get("type")) and bool(n.get("label")) for n in nodes) / len(nodes)
        if nodes else 1.0
    )
    validated = sum(1 for e in edges if e.get("status") == "validated")
    conflicting = sum(1 for e in edges if e.get("status") == "validated" and e.get("type") in {"contradicts", "conflicts_with"})
    contradiction_index = conflicting / validated if validated else 0.0

    schema_compliance = bool(valid_shape_nodes and valid_shape_edges and not unknown_node_types and not unknown_edge_types)
    referential_integrity = not orphan_edges
    reasons = []
    if not schema_compliance: reasons.append("schema_compliance")
    if not referential_integrity: reasons.append("referential_integrity")
    if duplicate_ids: reasons.append("duplicate_ids")
    if unsupported_asserted_nodes: reasons.append("unsupported_asserted_nodes")
    if missing_required_evidence: reasons.append("required_evidence_missing")
    status = "PASSED" if not reasons and completeness_score == 1.0 else "FAIL"

    return {
        "schema_version": "2.0.0",
        "nodes": len(nodes),
        "edges": len(edges),
        "evidence": len(evidence),
        "completeness_score": completeness_score,
        "schema_compliance": schema_compliance,
        "referential_integrity": referential_integrity,
        "duplicate_id_count": len(duplicate_ids),
        "orphan_edge_count": len(orphan_edges),
        "provenance_coverage": provenance_coverage,
        "contradiction_index": contradiction_index,
        "unknown_node_types": unknown_node_types,
        "unknown_edge_types": unknown_edge_types,
        "unsupported_asserted_nodes": unsupported_asserted_nodes,
        "required_evidence_missing": missing_required_evidence,
        "content_digest": sha256_json(graph),
        "status": status,
        "fail_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=str(ROOT / "data" / "canonical_graph_v2.json"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    report = audit(graph)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
