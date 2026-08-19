#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "chunks"
AUDITS = ROOT / "audits"
NORMALIZED = ROOT / "normalized"

LEVELS = ["L0","L1","L2","L3","L4","L5"]

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return {"__parse_error__": str(e)}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def graph_of(obj):
    if isinstance(obj, dict) and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj or "evidence" in obj):
        return obj
    return {}

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")

def main():
    AUDITS.mkdir(exist_ok=True)
    NORMALIZED.mkdir(exist_ok=True)

    files = sorted(CHUNKS.glob("*.json")) if CHUNKS.exists() else []
    chunks = []
    nodes = []
    edges = []
    evidence = []
    schemas = []
    errors = []

    for p in files:
        obj = read_json(p)
        rec = {
            "file": str(p.relative_to(ROOT)),
            "sha256": sha(p),
            "size": p.stat().st_size
        }

        if "__parse_error__" in obj:
            rec["parse_error"] = obj["__parse_error__"]
            errors.append(rec)
            chunks.append(rec)
            continue

        schema = obj.get("schema", {}) if isinstance(obj, dict) else {}
        g = graph_of(obj)

        rec["schema_name"] = schema.get("name")
        rec["schema_version"] = schema.get("version")
        rec["topic"] = schema.get("topic")
        rec["domain"] = schema.get("domain")
        rec["node_count"] = len(g.get("nodes", []) or [])
        rec["edge_count"] = len(g.get("edges", []) or [])
        rec["evidence_count"] = len(g.get("evidence", []) or [])

        chunks.append(rec)
        schemas.append({"file": rec["file"], **schema})

        for n in g.get("nodes", []) or []:
            if isinstance(n, dict):
                n["_file"] = rec["file"]
                nodes.append(n)

        for e in g.get("edges", []) or []:
            if isinstance(e, dict):
                e["_file"] = rec["file"]
                edges.append(e)

        for ev in g.get("evidence", []) or []:
            if isinstance(ev, dict):
                ev["_file"] = rec["file"]
                evidence.append(ev)

    node_ids = [n.get("id") for n in nodes if n.get("id")]
    edge_ids = [e.get("id") for e in edges if e.get("id")]
    node_id_set = set(node_ids)

    dup_nodes = sorted([k for k,v in Counter(node_ids).items() if v > 1])
    dup_edges = sorted([k for k,v in Counter(edge_ids).items() if v > 1])

    dangling = []
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s not in node_id_set or t not in node_id_set:
            dangling.append({
                "id": e.get("id"),
                "source": s,
                "target": t,
                "missing_source": s not in node_id_set,
                "missing_target": t not in node_id_set,
                "file": e.get("_file")
            })

    missing_pedagogy = []
    for n in nodes:
        p = (n.get("properties") or {}).get("pedagogy") or {}
        miss = [x for x in LEVELS if x not in p]
        if miss:
            missing_pedagogy.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "missing": miss,
                "file": n.get("_file")
            })

    evidence_nodes = set()
    evidence_edges = set()
    for ev in evidence:
        for x in ev.get("supports_nodes", []) or []:
            evidence_nodes.add(x)
        for x in ev.get("supports_edges", []) or []:
            evidence_edges.add(x)

    uncited_confirmed_nodes = []
    for n in nodes:
        props = n.get("properties") or {}
        if props.get("confidence") == "CONFIRMED" and n.get("id") not in evidence_nodes:
            uncited_confirmed_nodes.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "name": props.get("name"),
                "file": n.get("_file")
            })

    schema_versions = Counter((s.get("name"), s.get("version")) for s in schemas)
    topics = Counter(s.get("topic") for s in schemas if s.get("topic"))
    domains = Counter(s.get("domain") for s in schemas if s.get("domain"))
    node_types = Counter(n.get("type") for n in nodes if n.get("type"))
    edge_relations = Counter(e.get("relation") for e in edges if e.get("relation"))

    audit = {
        "repo": "khawrizm-sovereign-graph",
        "root": str(ROOT),
        "summary": {
            "chunk_files": len(files),
            "parsed_chunks": len(chunks) - len(errors),
            "parse_errors": len(errors),
            "nodes": len(nodes),
            "edges": len(edges),
            "evidence": len(evidence),
            "duplicate_node_ids": len(dup_nodes),
            "duplicate_edge_ids": len(dup_edges),
            "dangling_edges": len(dangling),
            "missing_pedagogy": len(missing_pedagogy),
            "uncited_confirmed_nodes": len(uncited_confirmed_nodes)
        },
        "schema_versions": [{"name": k[0], "version": k[1], "count": v} for k,v in schema_versions.items()],
        "topics": [{"topic": k, "count": v} for k,v in topics.most_common()],
        "domains": [{"domain": k, "count": v} for k,v in domains.most_common()],
        "node_types": [{"type": k, "count": v} for k,v in node_types.most_common()],
        "edge_relations": [{"relation": k, "count": v} for k,v in edge_relations.most_common()],
        "issues": {
            "parse_errors": errors,
            "duplicate_node_ids": dup_nodes,
            "duplicate_edge_ids": dup_edges,
            "dangling_edges": dangling,
            "missing_pedagogy": missing_pedagogy,
            "uncited_confirmed_nodes": uncited_confirmed_nodes
        }
    }

    write_jsonl(NORMALIZED / "chunks.jsonl", chunks)
    write_jsonl(NORMALIZED / "nodes.jsonl", nodes)
    write_jsonl(NORMALIZED / "edges.jsonl", edges)
    write_jsonl(NORMALIZED / "evidence.jsonl", evidence)
    write_jsonl(NORMALIZED / "schemas.jsonl", schemas)

    (AUDITS / "knowledge_graph_repo_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n"
    )

    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print("audit=audits/knowledge_graph_repo_audit.json")
    print("normalized=normalized/*.jsonl")

    if errors or dangling:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
