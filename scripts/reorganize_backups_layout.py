#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
BACKUPS = ROOT / "knowledge" / "90_legacy" / "source_backups"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

MOVES = [
    ("backups/chunks_50", "backups/10_legacy_chunks_50_mixed"),
    ("backups/it_50", "backups/20_reserved_it_50_empty"),
    ("backups/manifest_backup_100.json", "backups/00_manifest_backup_100.json")
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

readme = BACKUPS / "README.md"
readme.write_text("""# Backups

## Purpose

This directory stores legacy and reserved source material only.

## Layout

- 00_manifest_backup_100.json: backup manifest for the old 100 chunk plan
- 10_legacy_chunks_50_mixed: legacy 50 chunk backup, mixed real and empty graph records
- 20_reserved_it_50_empty: reserved IT expansion backup, currently empty graph records

## Rule

Backups are not canonical knowledge.
Backups are not validated lessons.
Backups are preserved for audit and recovery only.
""", encoding="utf-8", newline="\n")

layout = {
    "name": "khz_backups_layout",
    "version": "0.1.0",
    "policy": "legacy_audit_only_not_canonical",
    "directories": {
        "00_manifest_backup_100.json": "legacy backup manifest",
        "10_legacy_chunks_50_mixed": "mixed legacy backup data",
        "20_reserved_it_50_empty": "reserved empty IT expansion backup"
    }
}

(BACKUPS / "layout_manifest.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps(layout, ensure_ascii=False, indent=2))

