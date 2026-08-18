#!/usr/bin/env python3
"""Deterministic SKG audit: duplicates, dangling edges, provenance, status misuse."""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "chunks"


def load_graphs() -> list[dict]:
    graphs = []
    for path in sorted(glob.glob(str(CHUNKS / "*.json"))):
        with open(path, encoding="utf-8") as fh:
            try:
                graphs.append((path, json.load(fh)))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"INVALID_JSON {path}: {exc}")
    return graphs


def content_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    graphs = load_graphs()
    nodes: list[dict] = []
    edges: list[dict] = []
    evidence_ids: set[str] = set()

    for path, doc in graphs:
        graph = doc.get("graph", {})
        for node in graph.get("nodes", []):
            node["_source_file"] = str(path.relative_to(ROOT))
            nodes.append(node)
        edges.extend(graph.get("edges", []))
        evidence_ids.update(ev.get("id") for ev in graph.get("evidence", []) if ev.get("id"))

    node_counts = Counter(node.get("id") for node in nodes)
    duplicate_nodes = sorted(k for k, v in node_counts.items() if k and v > 1)
    node_ids = set(node_counts)

    dangling = []
    for edge in edges:
        src = edge.get("source")
        dst = edge.get("target")
        if src not in node_ids or dst not in node_ids:
            dangling.append({"source": src, "target": dst, "type": edge.get("type")})

    unsupported_assertions = []
    for node in nodes:
        provenance = node.get("provenance", [])
        if node.get("status", "asserted") == "asserted":
            if not provenance or not any(ev in evidence_ids for ev in provenance):
                unsupported_assertions.append(node.get("id"))

    report = {
        "files": len(graphs),
        "nodes": len(nodes),
        "edges": len(edges),
        "duplicate_node_ids": duplicate_nodes,
        "dangling_edges": dangling,
        "unsupported_asserted_nodes": unsupported_assertions,
        "node_content_hashes": {node.get("id"): content_hash(node) for node in nodes if node.get("id")},
        "status": "PASS" if not duplicate_nodes and not dangling and not unsupported_assertions else "FAIL",
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
