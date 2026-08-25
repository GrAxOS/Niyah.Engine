#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.cwd()
K = ROOT / "knowledge"

OLD = K / "10_domains"
TAX = K / "10_taxonomy"
LESSONS = K / "20_lessons"

REG = K / "00_registry"
CANON = K / "30_canonical"
STAGING = K / "40_staging"
ALIASES = K / "50_aliases"
REJECTED = K / "60_rejected"
RESERVED = K / "70_reserved"
LEGACY = K / "90_legacy"

AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

for d in [REG, TAX, LESSONS, CANON, STAGING, ALIASES, REJECTED, RESERVED, LEGACY]:
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitkeep").write_text("", encoding="utf-8")

if OLD.exists() and not TAX.exists():
    shutil.move(str(OLD), str(TAX))

if OLD.exists() and TAX.exists():
    for p in sorted(OLD.rglob("*")):
        rel = p.relative_to(OLD)
        dst = TAX / rel
        if p.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
    shutil.rmtree(OLD)

verified = []
downgraded_taxonomy = []

for idx in sorted(TAX.rglob("index.json")):
    try:
        obj = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        continue

    if obj.get("status") != "VERIFIED_ATOMIC_LESSON":
        if isinstance(obj.get("path"), str):
            obj["path"] = obj["path"].replace("knowledge/10_domains/", "knowledge/10_taxonomy/").replace("knowledge/domains/", "knowledge/10_taxonomy/")
            idx.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        continue

    rel_leaf = idx.parent.relative_to(TAX)
    dst_leaf = LESSONS / rel_leaf
    dst_leaf.parent.mkdir(parents=True, exist_ok=True)

    if dst_leaf.exists():
        shutil.rmtree(dst_leaf)
    shutil.copytree(idx.parent, dst_leaf)

    lesson_idx = dst_leaf / "index.json"
    lesson = json.loads(lesson_idx.read_text(encoding="utf-8"))
    lesson["path"] = "knowledge/20_lessons/" + str(rel_leaf).replace("\\", "/")
    lesson["updated_utc"] = datetime.now(timezone.utc).isoformat()
    lesson_idx.write_text(json.dumps(lesson, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    obj["status"] = "SOURCE_BACKED_TAXONOMY"
    obj["payload_kind"] = "taxonomy_reference_not_lesson"
    obj["atomic_facts"] = []
    obj["path"] = "knowledge/10_taxonomy/" + str(rel_leaf).replace("\\", "/")
    obj["knowledge_policy"] = {
        "is_actual_knowledge": False,
        "is_curriculum_seed": True,
        "is_source_backed_category": True,
        "is_verified_atomic_lesson": False,
        "requires_leaf_specific_sources_before_lesson": True
    }
    obj["canonical_policy"] = {
        "allowed_as_canonical": False,
        "reason": "Taxonomy copy only. Verified lesson lives under knowledge/20_lessons."
    }
    idx.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verified.append(str(lesson_idx.relative_to(ROOT)).replace("\\", "/"))
    downgraded_taxonomy.append(str(idx.relative_to(ROOT)).replace("\\", "/"))

patches = {
    "scripts/build_learning_taxonomy_layout.py": [
        ('knowledge" / "10_domains"', 'knowledge" / "10_taxonomy"'),
        ('knowledge" / "domains"', 'knowledge" / "10_taxonomy"'),
        ('"knowledge/10_domains"', '"knowledge/10_taxonomy"'),
        ('"knowledge/domains"', '"knowledge/10_taxonomy"')
    ],
    "scripts/validate_learning_taxonomy_layout.py": [
        ('knowledge" / "10_domains"', 'knowledge" / "10_taxonomy"'),
        ('knowledge" / "domains"', 'knowledge" / "10_taxonomy"'),
        ('"knowledge/10_domains"', '"knowledge/10_taxonomy"'),
        ('"knowledge/domains"', '"knowledge/10_taxonomy"')
    ],
    "scripts/seed_source_backed_taxonomy.py": [
        ('knowledge" / "10_domains"', 'knowledge" / "10_taxonomy"'),
        ('knowledge" / "domains"', 'knowledge" / "10_taxonomy"'),
        ('"knowledge/10_domains"', '"knowledge/10_taxonomy"'),
        ('"knowledge/domains"', '"knowledge/10_taxonomy"')
    ],
    "scripts/validate_source_backed_taxonomy.py": [
        ('knowledge" / "10_domains"', 'knowledge" / "10_taxonomy"'),
        ('knowledge" / "domains"', 'knowledge" / "10_taxonomy"'),
        ('"knowledge/10_domains"', '"knowledge/10_taxonomy"'),
        ('"knowledge/domains"', '"knowledge/10_taxonomy"')
    ],
    "scripts/audit_knowledge_payload_quality.py": [
        ('knowledge" / "10_domains"', 'knowledge" / "20_lessons"'),
        ('knowledge" / "domains"', 'knowledge" / "20_lessons"'),
        ('"knowledge/10_domains"', '"knowledge/20_lessons"'),
        ('"knowledge/domains"', '"knowledge/20_lessons"')
    ],
    "scripts/list_verified_lessons.py": [
        ('Path("knowledge/10_domains")', 'Path("knowledge/20_lessons")'),
        ('Path("knowledge/domains")', 'Path("knowledge/20_lessons")'),
        ('"knowledge/10_domains"', '"knowledge/20_lessons"'),
        ('"knowledge/domains"', '"knowledge/20_lessons"')
    ],
    "scripts/seed_verified_lesson_batch.py": [
        ('knowledge/10_domains/', 'knowledge/20_lessons/'),
        ('knowledge/domains/', 'knowledge/20_lessons/')
    ],
    "scripts/promote_verified_lesson.py": [
        ('knowledge/10_domains/', 'knowledge/20_lessons/'),
        ('knowledge/domains/', 'knowledge/20_lessons/')
    ]
}

patched = []
for file, reps in patches.items():
    p = ROOT / file
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8")
    n = s
    for a, b in reps:
        n = n.replace(a, b)
    if n != s:
        p.write_text(n, encoding="utf-8")
        patched.append(file)

readme = K / "README.md"
readme.write_text("""# KHZ Knowledge Layout

- 00_registry: registries and tree layout metadata
- 10_taxonomy: taxonomy only, not verified knowledge
- 20_lessons: verified atomic lessons only
- 30_canonical: canonical graph exports
- 40_staging: candidate material
- 50_aliases: alias and rename maps
- 60_rejected: rejected or invalid material
- 70_reserved: reserved future topics
- 90_legacy: preserved legacy material

Rule:
A lesson is real knowledge only when status = VERIFIED_ATOMIC_LESSON and atomic_facts contain source_title and source_url.
""", encoding="utf-8")

report = {
    "taxonomy_root": "knowledge/10_taxonomy",
    "lesson_root": "knowledge/20_lessons",
    "verified_lessons_moved": verified,
    "taxonomy_indexes_downgraded": downgraded_taxonomy,
    "patched_scripts": patched
}

(AUDITS / "knowledge_folder_rename_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
