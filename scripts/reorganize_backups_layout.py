#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
legacy_graph_sources = ROOT / "sources" / "90_legacy" / "legacy_graph_sources"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

MOVES = [
    ("legacy_graph_sources/chunks_50", "legacy_graph_sources/10_legacy_chunks_50_mixed"),
    ("legacy_graph_sources/it_50", "legacy_graph_sources/20_reserved_it_50_empty"),
    ("legacy_graph_sources/manifest_backup_100.json", "legacy_graph_sources/00_manifest_backup_100.json")
]

def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)

def git_mv(src, dst):
    s = ROOT / src
    d = ROOT / dst
    if s.exists() and not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "mv", src, dst])

for src, dst in MOVES:
    git_mv(src, dst)

readme = legacy_graph_sources / "README.md"
readme.write_text("""# legacy_graph_sources

## Purpose

This directory stores legacy and reserved source material only.

## Layout

- 00_manifest_backup_100.json: backup manifest for the old 100 chunk plan
- 10_legacy_chunks_50_mixed: legacy 50 chunk backup, mixed real and empty graph records
- 20_reserved_it_50_empty: reserved IT expansion backup, currently empty graph records

## Rule

legacy_graph_sources are not canonical knowledge.
legacy_graph_sources are not validated lessons.
legacy_graph_sources are preserved for audit and recovery only.
""", encoding="utf-8", newline="\n")

layout = {
    "name": "khz_legacy_graph_sources_layout",
    "version": "0.1.0",
    "policy": "legacy_audit_only_not_canonical",
    "directories": {
        "00_manifest_backup_100.json": "legacy backup manifest",
        "10_legacy_chunks_50_mixed": "mixed legacy backup data",
        "20_reserved_it_50_empty": "reserved empty IT expansion backup"
    }
}

(legacy_graph_sources / "layout_manifest.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps(layout, ensure_ascii=False, indent=2))



