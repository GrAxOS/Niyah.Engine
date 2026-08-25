#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
SOURCES = ROOT / "sources"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

DIRS = [
    "sources/00_inbox",
    "sources/10_raw_graph",
    "sources/20_open_data",
    "sources/30_vendor_docs",
    "sources/40_verified_sources",
    "sources/50_datasets",
    "sources/90_legacy",
    "sources/99_rejected"
]

def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)

def git_mv(src, dst):
    s = ROOT / src
    d = ROOT / dst
    if s.exists() and not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "mv", src, dst])

for d in DIRS:
    p = ROOT / d
    p.mkdir(parents=True, exist_ok=True)
    keep = p / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")

git_mv("legacy_graph_sources", "sources/90_legacy/legacy_graph_sources")

dup = ROOT / "sources" / "90_legacy" / "legacy_graph_sources"
if dup.exists():
    import shutil
    shutil.rmtree(dup)

(SOURCES / "README.md").write_text("""# Sources

## Purpose

This directory stores raw source material only.

## Layout

- 00_inbox: new unclassified source drops
- 10_raw_graph: raw graph-shaped JSON sources
- 20_open_data: open datasets
- 30_vendor_docs: vendor or official documentation captures
- 40_verified_sources: sources approved for lesson extraction
- 50_datasets: structured datasets
- 90_legacy: legacy preserved material
- 99_rejected: rejected or invalid sources

## Rule

sources/ is not knowledge.
knowledge/10_taxonomy is taxonomy.
knowledge/20_lessons is verified atomic knowledge.
knowledge/30_canonical is canonical export.
""", encoding="utf-8", newline="\n")

policy = {
    "name": "khz_source_routing_policy",
    "version": "0.1.0",
    "routing": {
        "graph_with_nodes_edges": "sources/10_raw_graph",
        "vendor_or_official_doc": "sources/30_vendor_docs",
        "verified_atomic_source": "sources/40_verified_sources",
        "dataset": "sources/50_datasets",
        "legacy": "sources/90_legacy",
        "unknown": "sources/00_inbox",
        "invalid": "sources/99_rejected"
    },
    "rules": [
        "Do not place raw sources under knowledge/",
        "Do not place legacy_graph_sources under knowledge/",
        "Only VERIFIED_ATOMIC_LESSON files belong under knowledge/20_lessons",
        "Only canonical exports belong under knowledge/30_canonical"
    ]
}

(SOURCES / "source_routing_policy.json").write_text(
    json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
)

report = {
    "created_dirs": DIRS,
    "moved": [
        {
            "from": "legacy_graph_sources",
            "to": "sources/90_legacy/legacy_graph_sources",
            "exists_now": (ROOT / "sources/90_legacy/legacy_graph_sources").exists()
        }
    ],
    "removed_duplicate_knowledge_backup": not dup.exists(),
    "policy": "sources/source_routing_policy.json"
}

(AUDITS / "sources_layout_audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
)

print(json.dumps(report, ensure_ascii=False, indent=2))


