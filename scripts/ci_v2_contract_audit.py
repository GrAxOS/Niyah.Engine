#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "chunks"
AUDITS = ROOT / "audits"
PUBLIC = ROOT / "public"

REQUIRED_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return {"__error__": str(e)}

def graph(obj):
    if isinstance(obj, dict) and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    if isinstance(obj, dict):
        return obj
    return {}

def main():
    AUDITS.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True)

    files = sorted(CHUNKS.glob("*.json"))
    parse_errors = []
    legacy_schema_chunks = []
    nodes = []
    edges = []
    evidence = []

    for f in files:
        obj = load_json(f)
        rel = str(f.relative_to(ROOT))

        if "__error__" in obj:
            parse_errors.append({"file": rel, "error": obj["__error__"]})
            continue

        schema = obj.get("schema", {})
        if schema.get("version") != "2.1.0":
            legacy_schema_chunks.append({
                "file": rel,
                "schema": schema.get("name"),
                "version": schema.get("version")
            })

        g = graph(obj)

        for n in g.get("nodes", []) or []:
            if isinstance(n, dict):
                n["_file"] = rel
                nodes.append(n)

        for e in g.get("edges", []) or []:
            if isinstance(e, dict):
                e["_file"] = rel
                edges.append(e)

        for ev in g.get("evidence", []) or []:
            if isinstance(ev, dict):
                ev["_file"] = rel
                evidence.append(ev)

    node_ids = [n.get("id") for n in nodes if n.get("id")]
    edge_ids = [e.get("id") for e in edges if e.get("id")]
    node_set = set(node_ids)

    duplicate_nodes = sorted([k for k,v in Counter(node_ids).items() if v > 1])
    duplicate_edges = sorted([k for k,v in Counter(edge_ids).items() if v > 1])

    id_errors = []
    for n in nodes:
        if not n.get("id") or not ID_RE.match(str(n.get("id"))):
            id_errors.append({"kind": "node", "id": n.get("id"), "file": n.get("_file")})
    for e in edges:
        if not e.get("id") or not ID_RE.match(str(e.get("id"))):
            id_errors.append({"kind": "edge", "id": e.get("id"), "file": e.get("_file")})

    dangling = []
    for e in edges:
        if e.get("source") not in node_set or e.get("target") not in node_set:
            dangling.append({
                "id": e.get("id"),
                "source": e.get("source"),
                "target": e.get("target"),
                "missing_source": e.get("source") not in node_set,
                "missing_target": e.get("target") not in node_set,
                "file": e.get("_file")
            })

    missing_pedagogy = []
    for n in nodes:
        p = ((n.get("properties") or {}).get("pedagogy") or {})
        missing = [x for x in REQUIRED_LEVELS if x not in p]
        if missing:
            missing_pedagogy.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "missing": missing,
                "file": n.get("_file")
            })

    supported_nodes = set()
    supported_edges = set()
    for ev in evidence:
        for x in ev.get("supports_nodes", []) or []:
            supported_nodes.add(x)
        for x in ev.get("supports_edges", []) or []:
            supported_edges.add(x)

    uncited_confirmed_nodes = []
    for n in nodes:
        props = n.get("properties") or {}
        if props.get("confidence") == "CONFIRMED" and n.get("id") not in supported_nodes:
            uncited_confirmed_nodes.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "name": props.get("name"),
                "file": n.get("_file")
            })

    summary = {
        "chunk_files": len(files),
        "legacy_schema_chunks": len(legacy_schema_chunks),
        "parse_errors": len(parse_errors),
        "node_occurrences": len(nodes),
        "unique_nodes": len(set(node_ids)),
        "edge_occurrences": len(edges),
        "evidence_occurrences": len(evidence),
        "id_errors": len(id_errors),
        "duplicate_node_ids": len(duplicate_nodes),
        "duplicate_edge_ids": len(duplicate_edges),
        "dangling_edges": len(dangling),
        "missing_pedagogy": len(missing_pedagogy),
        "uncited_confirmed_nodes": len(uncited_confirmed_nodes)
    }

    audit = {
        "summary": summary,
        "issues": {
            "parse_errors": parse_errors,
            "legacy_schema_chunks": legacy_schema_chunks,
            "id_errors": id_errors,
            "duplicate_node_ids": duplicate_nodes,
            "duplicate_edge_ids": duplicate_edges,
            "dangling_edges": dangling,
            "missing_pedagogy": missing_pedagogy,
            "uncited_confirmed_nodes": uncited_confirmed_nodes
        }
    }

    (AUDITS / "v2_contract_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (AUDITS / "v2_contract_audit.md").write_text(
        "# V2.1 Knowledge Graph Contract Audit\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8"
    )

    failures = (
        summary["parse_errors"] +
        summary["legacy_schema_chunks"] +
        summary["id_errors"] +
        summary["duplicate_edge_ids"] +
        summary["dangling_edges"]
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
