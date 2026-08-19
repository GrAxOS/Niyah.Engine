#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()

MOVES = [
    ("knowledge/00_registry", "knowledge/00_registry"),
    ("knowledge/10_domains", "knowledge/10_domains"),
    ("knowledge/20_canonical", "knowledge/20_canonical"),
    ("knowledge/30_staging", "knowledge/30_staging"),
    ("knowledge/40_aliases", "knowledge/40_aliases"),
    ("knowledge/50_rejected", "knowledge/50_rejected"),
    ("knowledge/60_reserved", "knowledge/60_reserved"),
]

REPLACEMENTS = {
    "knowledge/10_domains": "knowledge/10_domains",
    "knowledge\\\\domains": "knowledge\\\\10_domains",
    '"knowledge" / "10_domains"': '"knowledge" / "10_domains"',
    "'knowledge' / '10_domains'": "'knowledge' / '10_domains'",
    "knowledge/00_registry": "knowledge/00_registry",
    "knowledge\\\\registry": "knowledge\\\\00_registry",
    '"knowledge" / "00_registry"': '"knowledge" / "00_registry"',
    "'knowledge' / '00_registry'": "'knowledge' / '00_registry'",
    "knowledge/20_canonical": "knowledge/20_canonical",
    "knowledge/30_staging": "knowledge/30_staging",
    "knowledge/40_aliases": "knowledge/40_aliases",
    "knowledge/50_rejected": "knowledge/50_rejected",
    "knowledge/60_reserved": "knowledge/60_reserved",
}

def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)

def exists_git_path(p):
    return (ROOT / p).exists()

def git_mv(src, dst):
    if exists_git_path(src) and not exists_git_path(dst):
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        run(["git", "mv", src, dst])
    elif exists_git_path(src) and exists_git_path(dst):
        for child in sorted((ROOT / src).iterdir()):
            target = ROOT / dst / child.name
            if target.exists():
                continue
            run(["git", "mv", str(child.relative_to(ROOT)), str(target.relative_to(ROOT))])
        try:
            (ROOT / src).rmdir()
        except OSError:
            pass

for src, dst in MOVES:
    git_mv(src, dst)

for d in [
    "knowledge/00_registry",
    "knowledge/10_domains",
    "knowledge/20_canonical",
    "knowledge/30_staging",
    "knowledge/40_aliases",
    "knowledge/50_rejected",
    "knowledge/60_reserved",
    "knowledge/90_legacy",
]:
    p = ROOT / d
    p.mkdir(parents=True, exist_ok=True)
    keep = p / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")

for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "knowledge").rglob("*.json")) + list((ROOT / "knowledge").rglob("*.md")):
    s = p.read_text(encoding="utf-8")
    old = s
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    if s != old:
        p.write_text(s, encoding="utf-8", newline="\n")

for idx in (ROOT / "knowledge").rglob("index.json"):
    try:
        obj = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        continue

    changed = False

    if isinstance(obj, dict):
        path = obj.get("path")
        if isinstance(path, str):
            new_path = path
            for a, b in REPLACEMENTS.items():
                new_path = new_path.replace(a.replace("\\\\", "/"), b.replace("\\\\", "/"))
            if new_path != path:
                obj["path"] = new_path
                changed = True

        expected = str(idx.parent.relative_to(ROOT)).replace("\\", "/")
        if obj.get("path") != expected:
            obj["path"] = expected
            changed = True

    if changed:
        idx.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

readme = ROOT / "knowledge" / "README.md"
readme.write_text("""# KHZ Knowledge Tree

## Directory contract

- 00_registry: taxonomy, naming policy, validation registry
- 10_domains: staged domain curriculum and leaf knowledge
- 20_canonical: verified export targets only
- 30_staging: candidate material before verification
- 40_aliases: ID alias maps and renamed concepts
- 50_rejected: rejected or invalid material
- 60_reserved: future expansion placeholders
- 90_legacy: preserved legacy material only

## Rule

Do not place canonical knowledge directly under 10_domains unless the leaf has:
- status = VERIFIED_ATOMIC_LESSON
- atomic_facts[]
- source_title
- source_url
- canonical_policy.allowed_as_canonical = true
""", encoding="utf-8", newline="\n")

manifest = {
    "name": "khz_knowledge_tree_layout",
    "version": "0.1.0",
    "root": "knowledge",
    "directories": {
        "00_registry": "taxonomy and naming policy",
        "10_domains": "domain curriculum and leaf knowledge",
        "20_canonical": "verified canonical exports",
        "30_staging": "candidate material",
        "40_aliases": "alias and rename maps",
        "50_rejected": "rejected material",
        "60_reserved": "future placeholders",
        "90_legacy": "legacy material"
    }
}

(ROOT / "knowledge" / "00_registry" / "tree_layout.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
