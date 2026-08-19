#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "knowledge" / "10_taxonomy"
AUDITS = ROOT / "audits"

CHECK_PREFIXES = [
    "cloud/azure",
    "cloud/aws",
    "cloud/gcp",
    "cloud/huawei_cloud",
    "operating_systems/linux/ubuntu",
    "operating_systems/linux/kali",
    "human_languages/arabic",
    "human_languages/english",
    "science/physics"
]

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def leaf_dirs(prefix):
    start = BASE / prefix
    if not start.exists():
        return []
    out = []
    for p in sorted(start.rglob("*")):
        if not p.is_dir():
            continue
        if p.name in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            continue
        has_levels = all((p / lvl).exists() for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"])
        children = [c for c in p.iterdir() if c.is_dir() and c.name not in {"L0", "L1", "L2", "L3", "L4", "L5"}]
        if has_levels and not children:
            out.append(p)
    return out

issues = []
checked = 0

for prefix in CHECK_PREFIXES:
    for leaf in leaf_dirs(prefix):
        checked += 1
        idx = leaf / "index.json"
        if not idx.exists():
            issues.append({"path": rel(leaf), "error": "missing_index_json"})
            continue
        try:
            obj = json.loads(idx.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append({"path": rel(idx), "error": "invalid_json", "detail": str(e)})
            continue
        sources = obj.get("sources", [])
        if not sources:
            issues.append({"path": rel(idx), "error": "missing_sources"})
            continue
        for s in sources:
            if not s.get("url") or not s.get("title"):
                issues.append({"path": rel(idx), "error": "source_missing_title_or_url"})

out = {
    "checked_leaf_dirs": checked,
    "issue_count": len(issues),
    "issues": issues
}

AUDITS.mkdir(exist_ok=True)
(AUDITS / "source_backed_taxonomy_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(out, ensure_ascii=False, indent=2))

if issues:
    sys.exit(1)
