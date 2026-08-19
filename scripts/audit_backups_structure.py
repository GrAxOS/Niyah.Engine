#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd()
legacy_graph_sources = ROOT / "sources" / "90_legacy" / "legacy_graph_sources"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

def rel(p): return str(p.relative_to(ROOT)).replace("\\", "/")
def load(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def keys(x): return tuple(sorted(x.keys())) if isinstance(x, dict) else tuple()
def graph_of(x):
    if isinstance(x, dict) and isinstance(x.get("graph"), dict): return x["graph"]
    if isinstance(x, dict) and any(k in x for k in ["nodes","edges","evidence"]): return x
    return {}

out = {}
for folder in sorted([p for p in legacy_graph_sources.iterdir() if p.is_dir()]):
    d = {
        "json_files": 0,
        "parse_errors": [],
        "schema_versions": Counter(),
        "top_shapes": Counter(),
        "graph_shapes": Counter(),
        "node_count": 0,
        "edge_count": 0,
        "evidence_count": 0,
        "empty_graph_files": []
    }

    for p in sorted(folder.rglob("*.json")):
        d["json_files"] += 1
        try:
            x = load(p)
        except Exception as e:
            d["parse_errors"].append({"file": rel(p), "error": str(e)})
            continue

        d["top_shapes"][keys(x)] += 1

        s = x.get("schema", {}) if isinstance(x, dict) else {}
        if isinstance(s, dict):
            d["schema_versions"][(s.get("name"), s.get("version"))] += 1

        g = graph_of(x)
        if g:
            d["graph_shapes"][keys(g)] += 1
            ns = g.get("nodes", []) or []
            es = g.get("edges", []) or []
            ev = g.get("evidence", []) or []
            d["node_count"] += len(ns)
            d["edge_count"] += len(es)
            d["evidence_count"] += len(ev)
            if not ns and not es and not ev:
                d["empty_graph_files"].append(rel(p))

    out[folder.name] = {
        "json_files": d["json_files"],
        "parse_error_count": len(d["parse_errors"]),
        "node_count": d["node_count"],
        "edge_count": d["edge_count"],
        "evidence_count": d["evidence_count"],
        "empty_graph_count": len(d["empty_graph_files"]),
        "schema_versions": [{"name": k[0], "version": k[1], "count": v} for k,v in d["schema_versions"].most_common()],
        "top_shapes": [{"keys": list(k), "count": v} for k,v in d["top_shapes"].most_common()],
        "graph_shapes": [{"keys": list(k), "count": v} for k,v in d["graph_shapes"].most_common()],
        "empty_graph_files_first_20": d["empty_graph_files"][:20],
        "parse_errors": d["parse_errors"]
    }

manifest = legacy_graph_sources / "manifest_backup_100.json"
if manifest.exists():
    try:
        out["_manifest_backup_100"] = load(manifest)
    except Exception as e:
        out["_manifest_backup_100"] = {"error": str(e)}

(AUDITS / "legacy_graph_sources_structure_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

for name, d in out.items():
    print("##", name)
    if isinstance(d, dict) and "json_files" in d:
        print("json_files=", d["json_files"], "parse_errors=", d["parse_error_count"], "nodes=", d["node_count"], "edges=", d["edge_count"], "evidence=", d["evidence_count"], "empty=", d["empty_graph_count"])
        print("schema_versions=", d["schema_versions"])
        print("top_shapes=", d["top_shapes"][:3])
        print("graph_shapes=", d["graph_shapes"][:3])
    else:
        print(d)
    print()

print("audit=audits/legacy_graph_sources_structure_audit.json")



