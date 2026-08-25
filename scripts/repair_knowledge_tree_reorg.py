#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
OLD = ROOT / "knowledge" / "10_domains"
NEW = ROOT / "knowledge" / "10_domains"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

moved = []
patched_scripts = []
patched_indexes = []

NEW.mkdir(parents=True, exist_ok=True)

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if OLD.exists():
    for src in sorted(OLD.rglob("*")):
        rel = src.relative_to(OLD)
        dst = NEW / rel

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and src.name == "index.json":
            try:
                s = load_json(src)
                d = load_json(dst)
                if s.get("status") == "VERIFIED_ATOMIC_LESSON" and d.get("status") != "VERIFIED_ATOMIC_LESSON":
                    shutil.copy2(src, dst)
                    moved.append({"from": str(src), "to": str(dst), "mode": "overwrite_with_verified"})
            except Exception:
                pass
        elif not dst.exists():
            shutil.copy2(src, dst)
            moved.append({"from": str(src), "to": str(dst), "mode": "copy"})

    shutil.rmtree(OLD)

for p in sorted((ROOT / "scripts").glob("*.py")):
    txt = p.read_text(encoding="utf-8")
    new = txt.replace("knowledge/10_domains", "knowledge/10_domains")
    new = new.replace('"knowledge" / "10_domains"', '"knowledge" / "10_domains"')
    new = new.replace("'knowledge' / '10_domains'", "'knowledge' / '10_domains'")
    if new != txt:
        p.write_text(new, encoding="utf-8")
        patched_scripts.append(str(p.relative_to(ROOT)).replace("\\", "/"))

for p in sorted(NEW.rglob("index.json")):
    try:
        obj = load_json(p)
    except Exception:
        continue

    changed = False

    if isinstance(obj.get("path"), str) and obj["path"].startswith("knowledge/10_domains/"):
        obj["path"] = obj["path"].replace("knowledge/10_domains/", "knowledge/10_domains/", 1)
        changed = True

    for f in obj.get("atomic_facts", []) or []:
        if isinstance(f, dict):
            pass

    if changed:
        write_json(p, obj)
        patched_indexes.append(str(p.relative_to(ROOT)).replace("\\", "/"))

report = {
    "old_base": str(OLD.relative_to(ROOT)).replace("\\", "/"),
    "new_base": str(NEW.relative_to(ROOT)).replace("\\", "/"),
    "merged_files": len(moved),
    "patched_scripts": patched_scripts,
    "patched_indexes": patched_indexes
}

write_json(AUDITS / "knowledge_tree_reorg_repair.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
