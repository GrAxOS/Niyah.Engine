#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
REG = ROOT / "knowledge" / "registry" / "learning_taxonomy.json"
BASE = ROOT / "knowledge" / "domains"
AUDITS = ROOT / "audits"
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]

def slug(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def walk(obj, prefix):
    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths += walk(v, prefix + [slug(k)])
    elif isinstance(obj, list):
        for item in obj:
            paths.append(prefix + [slug(item)])
    else:
        paths.append(prefix + [slug(obj)])
    return paths

data = json.loads(REG.read_text(encoding="utf-8"))
paths = walk(data["domains"], [])
missing = []

for parts in paths:
    p = BASE.joinpath(*parts)
    if not p.exists():
        missing.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    for level in LEVELS:
        lp = p / level
        if not lp.exists():
            missing.append(str(lp.relative_to(ROOT)).replace("\\", "/"))

AUDITS.mkdir(exist_ok=True)
out = {
    "checked_leaf_paths": len(paths),
    "missing_count": len(missing),
    "missing": missing
}
(AUDITS / "learning_taxonomy_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))

if missing:
    sys.exit(1)
