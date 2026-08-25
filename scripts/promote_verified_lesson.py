#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.cwd()

def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

if len(sys.argv) != 2:
    fail("usage: python scripts/promote_verified_lesson.py path/to/lesson.json")

src = Path(sys.argv[1])
if not src.exists():
    fail(f"missing input: {src}")

lesson = json.loads(src.read_text(encoding="utf-8"))

required = ["path", "id", "provider_or_subject", "atomic_facts", "sources", "learning_levels"]
for k in required:
    if k not in lesson:
        fail(f"missing field: {k}")

if not lesson["atomic_facts"]:
    fail("atomic_facts is empty")

for f in lesson["atomic_facts"]:
    for k in ["id", "claim", "source_title", "source_url", "confidence"]:
        if k not in f or f[k] in ("", None):
            fail(f"bad atomic fact missing {k}: {f}")

for s in lesson["sources"]:
    for k in ["title", "url"]:
        if k not in s or s[k] in ("", None):
            fail(f"bad source missing {k}: {s}")

target_dir = ROOT / lesson["path"]
target_dir.mkdir(parents=True, exist_ok=True)

for lvl in ["L0","L1","L2","L3","L4","L5"]:
    (target_dir / lvl).mkdir(exist_ok=True)
    readme = target_dir / lvl / "README.md"
    if not readme.exists():
        readme.write_text(f"# {lesson['path']} / {lvl}\n\nStatus: VERIFIED_ATOMIC_LESSON\n", encoding="utf-8")

lesson["schema"] = {"name": "khz_verified_atomic_lesson", "version": "0.1.0"}
lesson["status"] = "VERIFIED_ATOMIC_LESSON"
lesson["payload_kind"] = "source_backed_atomic_knowledge"
lesson["updated_utc"] = datetime.now(timezone.utc).isoformat()
lesson["knowledge_policy"] = {
    "is_actual_knowledge": True,
    "is_curriculum_seed": True,
    "is_source_backed_category": True,
    "is_verified_atomic_lesson": True,
    "requires_leaf_specific_sources_before_lesson": False
}
lesson["canonical_policy"] = {
    "allowed_as_canonical": True,
    "reason": "Contains source-backed atomic facts with explicit source title and URL."
}

out = target_dir / "index.json"
out.write_text(json.dumps(lesson, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
