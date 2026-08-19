#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.cwd()
REG = ROOT / "knowledge" / "00_registry" / "learning_taxonomy.json"
BASE = ROOT / "knowledge" / "10_taxonomy"
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

def main():
    data = json.loads(REG.read_text(encoding="utf-8"))
    paths = walk(data["domains"], [])

    created = []
    for parts in paths:
        domain_path = BASE.joinpath(*parts)
        domain_path.mkdir(parents=True, exist_ok=True)
        keep = domain_path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

        for level in LEVELS:
            level_path = domain_path / level
            level_path.mkdir(exist_ok=True)
            meta = level_path / "README.md"
            if not meta.exists():
                meta.write_text(
                    f"# {' / '.join(parts)} / {level}\n\nStatus: RESERVED\n\nPurpose: curated knowledge expansion.\n",
                    encoding="utf-8"
                )
        created.append(str(domain_path.relative_to(ROOT)).replace("\\", "/"))

    AUDITS.mkdir(exist_ok=True)
    report = {
        "taxonomy": str(REG.relative_to(ROOT)).replace("\\", "/"),
        "domain_leaf_count": len(created),
        "level_count_per_leaf": len(LEVELS),
        "created_or_verified": created
    }
    (AUDITS / "learning_taxonomy_layout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (AUDITS / "learning_taxonomy_layout.md").write_text(
        "# Learning Taxonomy Layout\n\n"
        + f"- domain_leaf_count: {len(created)}\n"
        + f"- levels_per_leaf: {len(LEVELS)}\n\n"
        + "\n".join(f"- {x}" for x in created)
        + "\n",
        encoding="utf-8"
    )
    print(json.dumps({"domain_leaf_count": len(created), "audit": "audits/learning_taxonomy_layout.md"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
