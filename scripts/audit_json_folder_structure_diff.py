#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd()
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

SCAN = ["sources/90_legacy/legacy_graph_sources", "chunks", "data", "knowledge", "schema", "manifests", "normalized"]
IGNORE = {".git", "__pycache__"}

def rel(p): return str(p.relative_to(ROOT)).replace("\\", "/")
def load(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def keys(x): return tuple(sorted(x.keys())) if isinstance(x, dict) else tuple()
def graph_of(x):
    if isinstance(x, dict) and isinstance(x.get("graph"), dict): return x["graph"]
    if isinstance(x, dict) and any(k in x for k in ["nodes","edges","evidence"]): return x
    return {}

out = {}
for top in SCAN:
    base = ROOT / top
    if not base.exists(): continue
    files = sorted(base.rglob("*.json"))
    d = {
        "json_files": len(files),
        "parse_errors": [],
        "top_level_shapes": Counter(),
        "graph_shapes": Counter(),
        "schema_versions": Counter(),
        "node_shapes": Counter(),
        "edge_shapes": Counter(),
        "evidence_shapes": Counter(),
        "empty_graph_files": [],
        "node_count": 0,
        "edge_count": 0,
        "evidence_count": 0
    }
    for p in files:
        if any(part in IGNORE for part in p.parts): continue
        try:
            x = load(p)
        except Exception as e:
            d["parse_errors"].append({"file": rel(p), "error": str(e)})
            continue
        d["top_level_shapes"][keys(x)] += 1
        s = x.get("schema", {}) if isinstance(x, dict) else {}
        if isinstance(s, dict):
            d["schema_versions"][(s.get("name"), s.get("version"))] += 1
        g = graph_of(x)
        if g:
            d["graph_shapes"][keys(g)] += 1
            ns = g.get("nodes", []) or []
            es = g.get("edges", []) or []
            evs = g.get("evidence", []) or []
            d["node_count"] += len(ns)
            d["edge_count"] += len(es)
            d["evidence_count"] += len(evs)
            if not ns and not es and not evs:
                d["empty_graph_files"].append(rel(p))
            for n in ns:
                if isinstance(n, dict): d["node_shapes"][keys(n)] += 1
            for e in es:
                if isinstance(e, dict): d["edge_shapes"][keys(e)] += 1
            for ev in evs:
                if isinstance(ev, dict): d["evidence_shapes"][keys(ev)] += 1
    out[top] = {
        "json_files": d["json_files"],
        "parse_error_count": len(d["parse_errors"]),
        "node_count": d["node_count"],
        "edge_count": d["edge_count"],
        "evidence_count": d["evidence_count"],
        "empty_graph_count": len(d["empty_graph_files"]),
        "schema_versions": [{"name": k[0], "version": k[1], "count": v} for k,v in d["schema_versions"].most_common()],
        "top_level_shapes": [{"keys": list(k), "count": v} for k,v in d["top_level_shapes"].most_common()],
        "graph_shapes": [{"keys": list(k), "count": v} for k,v in d["graph_shapes"].most_common()],
        "node_shapes": [{"keys": list(k), "count": v} for k,v in d["node_shapes"].most_common(10)],
        "edge_shapes": [{"keys": list(k), "count": v} for k,v in d["edge_shapes"].most_common(10)],
        "evidence_shapes": [{"keys": list(k), "count": v} for k,v in d["evidence_shapes"].most_common(10)],
        "empty_graph_files_first_30": d["empty_graph_files"][:30],
        "parse_errors": d["parse_errors"][:30]
    }

(AUDITS / "json_folder_structure_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

for top, d in out.items():
    print("##", top)
    print("json_files=", d["json_files"], "parse_errors=", d["parse_error_count"], "nodes=", d["node_count"], "edges=", d["edge_count"], "evidence=", d["evidence_count"], "empty_graph=", d["empty_graph_count"])
    print("schema_versions=", d["schema_versions"][:5])
    print("top_shapes=", d["top_level_shapes"][:3])
    print("graph_shapes=", d["graph_shapes"][:3])
    print()
print("audit=audits/json_folder_structure_diff.json")



