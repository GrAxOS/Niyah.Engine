#!/usr/bin/env python3
import json
import re
import html
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd()
CHUNKS = ROOT / "chunks"
SCHEMA = ROOT / "schema" / "canonical_knowledge_graph_v2.1.0.json"
AUDITS = ROOT / "audits"
PUBLIC = ROOT / "public"

AUDITS.mkdir(exist_ok=True)
PUBLIC.mkdir(exist_ok=True)

NODE_RE = re.compile(r"^n_[a-f0-9]{64}$")
EDGE_RE = re.compile(r"^e_[a-f0-9]{64}$")
EV_RE = re.compile(r"^ev_[a-f0-9]{64}$")

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def clean(x):
    if isinstance(x, str):
        return html.unescape(x)
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

def graph_of(obj):
    if isinstance(obj, dict) and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj or "evidence" in obj):
        return obj
    return {}

parse_errors = []
schema_errors = []
id_errors = []
duplicate_errors = []
dangling_edges = []
legacy_chunks = []
chunks = []
nodes = []
edges = []
evidence = []

schema_exists = SCHEMA.exists()
schema_version_required = "2.1.0"

for p in sorted(CHUNKS.glob("*.json")):
    rel = str(p.relative_to(ROOT)).replace("\\", "/")
    try:
        obj = clean(load_json(p))
    except Exception as e:
        parse_errors.append({"file": rel, "error": str(e)})
        continue

    s = obj.get("schema", {}) if isinstance(obj, dict) else {}
    g = graph_of(obj)

    version = s.get("version")
    if version != schema_version_required:
        legacy_chunks.append({"file": rel, "schema_version": version})

    chunks.append({
        "file": rel,
        "schema_version": version,
        "nodes": len(g.get("nodes", []) or []),
        "edges": len(g.get("edges", []) or []),
        "evidence": len(g.get("evidence", []) or []),
    })

    for n in g.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id", ""))
        nodes.append({"id": nid, "file": rel, "type": n.get("type")})
        if not NODE_RE.match(nid):
            id_errors.append({"kind": "node", "id": nid, "file": rel, "reason": "not_v2_1_canonical_id"})

    for e in g.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id", ""))
        src = str(e.get("source", ""))
        tgt = str(e.get("target", ""))
        edges.append({"id": eid, "source": src, "target": tgt, "file": rel})
        if eid and not EDGE_RE.match(eid):
            id_errors.append({"kind": "edge", "id": eid, "file": rel, "reason": "not_v2_1_canonical_id"})
        if not eid:
            id_errors.append({"kind": "edge", "id": None, "file": rel, "reason": "missing_edge_id"})

    for ev in g.get("evidence", []) or []:
        if not isinstance(ev, dict):
            continue
        evid = str(ev.get("id", ""))
        evidence.append({"id": evid, "file": rel})
        if not EV_RE.match(evid):
            id_errors.append({"kind": "evidence", "id": evid, "file": rel, "reason": "not_v2_1_canonical_id"})

node_ids = [x["id"] for x in nodes if x["id"]]
edge_ids = [x["id"] for x in edges if x["id"]]
ev_ids = [x["id"] for x in evidence if x["id"]]

node_set = set(node_ids)

for item_id, count in Counter(node_ids).items():
    if count > 1:
        duplicate_errors.append({"kind": "node", "id": item_id, "count": count})

for item_id, count in Counter(edge_ids).items():
    if count > 1:
        duplicate_errors.append({"kind": "edge", "id": item_id, "count": count})

for e in edges:
    ms = e["source"] not in node_set
    mt = e["target"] not in node_set
    if ms or mt:
        dangling_edges.append({
            "id": e["id"],
            "source": e["source"],
            "target": e["target"],
            "missing_source": ms,
            "missing_target": mt,
            "file": e["file"],
        })

report = {
    "contract": "canonical_knowledge_graph_v2.1.0",
    "schema_file_exists": schema_exists,
    "summary": {
        "chunk_files": len(chunks),
        "legacy_schema_chunks": len(legacy_chunks),
        "parse_errors": len(parse_errors),
        "node_occurrences": len(nodes),
        "unique_nodes": len(set(node_ids)),
        "edge_occurrences": len(edges),
        "evidence_occurrences": len(evidence),
        "id_errors": len(id_errors),
        "duplicate_errors": len(duplicate_errors),
        "dangling_edges": len(dangling_edges),
    },
    "legacy_chunks": legacy_chunks,
    "parse_errors": parse_errors,
    "id_errors": id_errors,
    "duplicate_errors": duplicate_errors,
    "dangling_edges": dangling_edges,
}

(AUDITS / "v2_contract_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

md = ["# KHZ Graph v2.1 Contract Audit", "", "## Summary"]
for k, v in report["summary"].items():
    md.append(f"- {k}: {v}")

md += ["", "## Decision"]
if report["summary"]["legacy_schema_chunks"] or report["summary"]["id_errors"] or report["summary"]["duplicate_errors"] or report["summary"]["dangling_edges"]:
    md.append("FAIL: current chunks are not canonical v2.1 compliant.")
else:
    md.append("PASS: current chunks satisfy v2.1 gate.")

md += ["", "## First legacy chunks"]
for x in legacy_chunks[:50]:
    md.append(f"- {x['file']} schema_version={x['schema_version']}")

md += ["", "## First dangling edges"]
for x in dangling_edges[:50]:
    md.append(f"- {x['file']} source={x['source']} target={x['target']} missing_source={x['missing_source']} missing_target={x['missing_target']}")

md += ["", "## First duplicate IDs"]
for x in duplicate_errors[:80]:
    md.append(f"- {x['kind']} {x['id']} count={x['count']}")

md += ["", "## First ID errors"]
for x in id_errors[:80]:
    md.append(f"- {x['kind']} {x['id']} file={x['file']} reason={x['reason']}")

(AUDITS / "v2_contract_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

tests = [
    ("schema_file_exists", schema_exists, str(schema_exists)),
    ("no_parse_errors", len(parse_errors) == 0, str(len(parse_errors))),
    ("all_chunks_v2_1", len(legacy_chunks) == 0, str(len(legacy_chunks))),
    ("canonical_ids", len(id_errors) == 0, str(len(id_errors))),
    ("no_duplicate_ids", len(duplicate_errors) == 0, str(len(duplicate_errors))),
    ("no_dangling_edges", len(dangling_edges) == 0, str(len(dangling_edges))),
]

suite = ET.Element("testsuite", name="khz_graph_v2_contract", tests=str(len(tests)), failures=str(sum(1 for _, ok, _ in tests if not ok)))
for name, ok, value in tests:
    case = ET.SubElement(suite, "testcase", name=name)
    if not ok:
        failure = ET.SubElement(case, "failure", message=f"{name}={value}")
        failure.text = f"{name}={value}"

ET.ElementTree(suite).write(AUDITS / "v2_contract_junit.xml", encoding="utf-8", xml_declaration=True)

safe = "\n".join(md).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
(PUBLIC / "index.html").write_text(f"<html><body><h1>KHZ Graph v2.1 Contract Audit</h1><pre>{safe}</pre></body></html>", encoding="utf-8")

print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

if any(not ok for _, ok, _ in tests):
    raise SystemExit(1)
