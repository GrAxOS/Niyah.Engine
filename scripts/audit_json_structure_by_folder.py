#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.cwd()
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

IGNORE = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def ignored(p):
    return any(x in IGNORE for x in p.parts)

def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def graph_of(obj):
    if isinstance(obj, dict) and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj or "evidence" in obj):
        return obj
    return {}

def keys_of(x):
    if isinstance(x, dict):
        return sorted(x.keys())
    return []

folders = defaultdict(lambda: {
    "files": 0,
    "parse_errors": 0,
    "top_level_keysets": Counter(),
    "schema_versions": Counter(),
    "graph_keysets": Counter(),
    "node_keysets": Counter(),
    "edge_keysets": Counter(),
    "evidence_keysets": Counter(),
    "examples": defaultdict(list),
})

for p in sorted(ROOT.rglob("*.json")):
    if ignored(p):
        continue

    folder = rel(p).split("/")[0]
    f = folders[folder]
    f["files"] += 1

    try:
        obj = load(p)
    except Exception as e:
        f["parse_errors"] += 1
        f["examples"]["parse_errors"].append({"file": rel(p), "error": str(e)})
        continue

    top_keys = tuple(keys_of(obj))
    f["top_level_keysets"][top_keys] += 1
    if len(f["examples"][str(top_keys)]) < 3:
        f["examples"][str(top_keys)].append(rel(p))

    schema = obj.get("schema", {}) if isinstance(obj, dict) else {}
    if isinstance(schema, dict):
        f["schema_versions"][(schema.get("name"), schema.get("version"))] += 1

    g = graph_of(obj)
    if g:
        graph_keys = tuple(keys_of(g))
        f["graph_keysets"][graph_keys] += 1

        for n in g.get("nodes", []) or []:
            if isinstance(n, dict):
                f["node_keysets"][tuple(keys_of(n))] += 1

        for e in g.get("edges", []) or []:
            if isinstance(e, dict):
                f["edge_keysets"][tuple(keys_of(e))] += 1

        for ev in g.get("evidence", []) or []:
            if isinstance(ev, dict):
                f["evidence_keysets"][tuple(keys_of(ev))] += 1

out = {}
for folder, data in folders.items():
    out[folder] = {
        "files": data["files"],
        "parse_errors": data["parse_errors"],
        "schema_versions": [
            {"name": k[0], "version": k[1], "count": v}
            for k, v in data["schema_versions"].most_common()
        ],
        "top_level_keysets": [
            {"keys": list(k), "count": v}
            for k, v in data["top_level_keysets"].most_common()
        ],
        "graph_keysets": [
            {"keys": list(k), "count": v}
            for k, v in data["graph_keysets"].most_common()
        ],
        "node_keysets": [
            {"keys": list(k), "count": v}
            for k, v in data["node_keysets"].most_common(20)
        ],
        "edge_keysets": [
            {"keys": list(k), "count": v}
            for k, v in data["edge_keysets"].most_common(20)
        ],
        "evidence_keysets": [
            {"keys": list(k), "count": v}
            for k, v in data["evidence_keysets"].most_common(20)
        ],
    }

(AUDITS / "json_structure_by_folder.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
)

md = ["# JSON Structure By Folder", ""]
for folder, data in out.items():
    md.append(f"## {folder}")
    md.append(f"- files: {data['files']}")
    md.append(f"- parse_errors: {data['parse_errors']}")
    md.append("")
    md.append("### schema_versions")
    for x in data["schema_versions"]:
        md.append(f"- name={x['name']} version={x['version']} count={x['count']}")
    md.append("")
    md.append("### top_level_keysets")
    for x in data["top_level_keysets"]:
        md.append(f"- count={x['count']} keys={x['keys']}")
    md.append("")
    md.append("### graph_keysets")
    for x in data["graph_keysets"]:
        md.append(f"- count={x['count']} keys={x['keys']}")
    md.append("")

(AUDITS / "json_structure_by_folder.md").write_text(
    "\n".join(md) + "\n",
    encoding="utf-8"
)

print(json.dumps({
    "folders": sorted(out.keys()),
    "report": "audits/json_structure_by_folder.md",
    "json": "audits/json_structure_by_folder.json"
}, ensure_ascii=False, indent=2))
