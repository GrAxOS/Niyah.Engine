#!/usr/bin/env python3
import json
import re
import csv
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd()
AUDITS = ROOT / "audits"
NORMALIZED = ROOT / "normalized"
PUBLIC = ROOT / "public"

AUDITS.mkdir(exist_ok=True)
NORMALIZED.mkdir(exist_ok=True)
PUBLIC.mkdir(exist_ok=True)

IGNORE_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
SCAN_TOPS = ["chunks", "sources/90_legacy/legacy_graph_sources", "data", "knowledge", "schema", "manifests", "evidence", "tests", "docs"]

NODE_V21 = re.compile(r"^n_[a-f0-9]{64}$")
EDGE_V21 = re.compile(r"^e_[a-f0-9]{64}$")
EV_V21 = re.compile(r"^ev_[a-f0-9]{64}$")
SNAKE_FILE = re.compile(r"^[a-z0-9][a-z0-9_./-]*\.json$")

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def sha256_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def graph_of(obj):
    if isinstance(obj, dict) and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj or "evidence" in obj):
        return obj
    return {}

def top_dir(path):
    parts = rel(path).split("/")
    return parts[0] if parts else ""

def is_ignored(path):
    return any(part in IGNORE_DIRS for part in path.parts)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

json_files = []
for p in ROOT.rglob("*.json"):
    if is_ignored(p):
        continue
    json_files.append(p)

json_files = sorted(json_files)

file_rows = []
parse_errors = []
graph_files = []
nodes = []
edges = []
evidence = []
schema_versions = []
empty_graph_files = []
naming_issues = []
id_issues = []

for p in json_files:
    r = rel(p)
    td = top_dir(p)
    row = {
        "file": r,
        "top_dir": td,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
        "parse_ok": False,
        "schema_name": None,
        "schema_version": None,
        "has_graph": False,
        "node_count": 0,
        "edge_count": 0,
        "evidence_count": 0,
    }

    if not SNAKE_FILE.match(r.lower()):
        naming_issues.append({"kind": "file_name", "file": r, "reason": "non_canonical_file_name"})

    try:
        obj = load_json(p)
        row["parse_ok"] = True
    except Exception as e:
        parse_errors.append({"file": r, "error": str(e)})
        file_rows.append(row)
        continue

    schema = obj.get("schema", {}) if isinstance(obj, dict) else {}
    if isinstance(schema, dict):
        row["schema_name"] = schema.get("name")
        row["schema_version"] = schema.get("version")
        if schema.get("name") or schema.get("version"):
            schema_versions.append({
                "file": r,
                "name": schema.get("name"),
                "version": schema.get("version"),
            })

    g = graph_of(obj)
    if g:
        row["has_graph"] = True
        graph_files.append(r)

        g_nodes = g.get("nodes", []) or []
        g_edges = g.get("edges", []) or []
        g_evidence = g.get("evidence", []) or []

        row["node_count"] = len(g_nodes)
        row["edge_count"] = len(g_edges)
        row["evidence_count"] = len(g_evidence)

        if len(g_nodes) == 0 and len(g_edges) == 0 and len(g_evidence) == 0:
            empty_graph_files.append(r)

        for n in g_nodes:
            if isinstance(n, dict):
                nid = str(n.get("id", ""))
                nodes.append({
                    "id": nid,
                    "type": n.get("type"),
                    "label": n.get("label") or (n.get("properties") or {}).get("name"),
                    "file": r,
                    "schema_version": row["schema_version"],
                })
                if not nid:
                    id_issues.append({"kind": "node", "id": nid, "file": r, "reason": "missing_node_id"})
                elif row["schema_version"] == "2.1.0" and not NODE_V21.match(nid):
                    id_issues.append({"kind": "node", "id": nid, "file": r, "reason": "v2_1_id_not_canonical"})

        for e in g_edges:
            if isinstance(e, dict):
                eid = str(e.get("id", ""))
                src = str(e.get("source", ""))
                tgt = str(e.get("target", ""))
                etype = e.get("type") or e.get("relation")
                edges.append({
                    "id": eid,
                    "source": src,
                    "target": tgt,
                    "type": etype,
                    "file": r,
                    "schema_version": row["schema_version"],
                })
                if not eid:
                    id_issues.append({"kind": "edge", "id": eid, "file": r, "reason": "missing_edge_id"})
                elif row["schema_version"] == "2.1.0" and not EDGE_V21.match(eid):
                    id_issues.append({"kind": "edge", "id": eid, "file": r, "reason": "v2_1_id_not_canonical"})

        for ev in g_evidence:
            if isinstance(ev, dict):
                evid = str(ev.get("id", ""))
                evidence.append({
                    "id": evid,
                    "type": ev.get("type"),
                    "file": r,
                    "schema_version": row["schema_version"],
                    "supports_nodes": ev.get("supports_nodes", []),
                    "supports_edges": ev.get("supports_edges", []),
                })
                if not evid:
                    id_issues.append({"kind": "evidence", "id": evid, "file": r, "reason": "missing_evidence_id"})
                elif row["schema_version"] == "2.1.0" and not EV_V21.match(evid):
                    id_issues.append({"kind": "evidence", "id": evid, "file": r, "reason": "v2_1_id_not_canonical"})

    file_rows.append(row)

node_ids = [x["id"] for x in nodes if x["id"]]
edge_ids = [x["id"] for x in edges if x["id"]]
evidence_ids = [x["id"] for x in evidence if x["id"]]

node_set = set(node_ids)
edge_set = set(edge_ids)

duplicate_nodes = [
    {"id": k, "count": v, "files": sorted(set(x["file"] for x in nodes if x["id"] == k))}
    for k, v in Counter(node_ids).items()
    if v > 1
]

duplicate_edges = [
    {"id": k, "count": v, "files": sorted(set(x["file"] for x in edges if x["id"] == k))}
    for k, v in Counter(edge_ids).items()
    if v > 1
]

duplicate_evidence = [
    {"id": k, "count": v, "files": sorted(set(x["file"] for x in evidence if x["id"] == k))}
    for k, v in Counter(evidence_ids).items()
    if v > 1
]

dangling_edges = []
for e in edges:
    ms = e["source"] not in node_set
    mt = e["target"] not in node_set
    if ms or mt:
        dangling_edges.append({
            "id": e["id"],
            "source": e["source"],
            "target": e["target"],
            "type": e["type"],
            "missing_source": ms,
            "missing_target": mt,
            "file": e["file"],
        })

orphan_evidence_refs = []
for ev in evidence:
    for n in ev.get("supports_nodes") or []:
        if n not in node_set:
            orphan_evidence_refs.append({"evidence": ev["id"], "missing_node": n, "file": ev["file"]})
    for e in ev.get("supports_edges") or []:
        if e not in edge_set:
            orphan_evidence_refs.append({"evidence": ev["id"], "missing_edge": e, "file": ev["file"]})

schema_counter = Counter((s.get("name"), s.get("version")) for s in schema_versions)
top_counter = Counter(row["top_dir"] for row in file_rows)
version_counter = Counter(row["schema_version"] for row in file_rows if row["schema_version"])

summary = {
    "json_files_total": len(json_files),
    "json_parse_errors": len(parse_errors),
    "graph_files": len(graph_files),
    "empty_graph_files": len(empty_graph_files),
    "node_occurrences": len(nodes),
    "unique_node_ids": len(node_set),
    "edge_occurrences": len(edges),
    "unique_edge_ids": len(edge_set),
    "evidence_occurrences": len(evidence),
    "unique_evidence_ids": len(set(evidence_ids)),
    "duplicate_node_ids": len(duplicate_nodes),
    "duplicate_edge_ids": len(duplicate_edges),
    "duplicate_evidence_ids": len(duplicate_evidence),
    "dangling_edges": len(dangling_edges),
    "orphan_evidence_refs": len(orphan_evidence_refs),
    "id_issues": len(id_issues),
    "naming_issues": len(naming_issues),
}

audit = {
    "summary": summary,
    "top_level_json_files": [{"top_dir": k, "count": v} for k, v in sorted(top_counter.items())],
    "schema_versions": [{"name": k[0], "version": k[1], "count": v} for k, v in schema_counter.items()],
    "schema_version_counts": [{"version": k, "count": v} for k, v in version_counter.items()],
    "issues": {
        "parse_errors": parse_errors,
        "empty_graph_files": empty_graph_files,
        "duplicate_nodes": duplicate_nodes,
        "duplicate_edges": duplicate_edges,
        "duplicate_evidence": duplicate_evidence,
        "dangling_edges": dangling_edges,
        "orphan_evidence_refs": orphan_evidence_refs,
        "id_issues": id_issues,
        "naming_issues": naming_issues,
    },
}

write_json(AUDITS / "full_repo_knowledge_audit.json", audit)

with (AUDITS / "full_repo_file_inventory.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()) if file_rows else ["file"])
    w.writeheader()
    for row in file_rows:
        w.writerow(row)

md = ["# Full Repository Knowledge Audit", "", "## Summary"]
for k, v in summary.items():
    md.append(f"- {k}: {v}")

md += ["", "## JSON files by top directory"]
for x in audit["top_level_json_files"]:
    md.append(f"- {x['top_dir']}: {x['count']}")

md += ["", "## Schema versions"]
for x in audit["schema_versions"]:
    md.append(f"- name={x['name']} version={x['version']} count={x['count']}")

md += ["", "## First 100 dangling edges"]
for x in dangling_edges[:100]:
    md.append(f"- {x['file']} source={x['source']} target={x['target']} missing_source={x['missing_source']} missing_target={x['missing_target']}")

md += ["", "## First 150 duplicate nodes"]
for x in duplicate_nodes[:150]:
    md.append(f"- {x['id']}: count={x['count']} files={', '.join(x['files'][:8])}")

md += ["", "## First 100 empty graph files"]
for x in empty_graph_files[:100]:
    md.append(f"- {x}")

md += ["", "## Recommendation"]
md.append("Keep chunks/ and legacy_graph_sources/ as legacy source material. Build a separate canonical v2.1 generated artifact under data/generated/ or knowledge/20_canonical/ instead of rewriting legacy chunks directly.")

(AUDITS / "full_repo_knowledge_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("audit=audits/full_repo_knowledge_audit.json")
print("report=audits/full_repo_knowledge_audit.md")
print("inventory=audits/full_repo_file_inventory.csv")



