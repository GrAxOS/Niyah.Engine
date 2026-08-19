#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "knowledge" / "domains"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

changed = []

for idx in sorted(BASE.rglob("index.json")):
    obj = json.loads(idx.read_text(encoding="utf-8"))

    status = obj.get("status")
    if status == "SOURCE_BACKED_TAXONOMY":
        obj["payload_kind"] = "taxonomy_reference_not_lesson"
        obj["knowledge_policy"] = {
            "is_actual_knowledge": False,
            "is_curriculum_seed": True,
            "is_source_backed_category": True,
            "is_verified_atomic_lesson": False,
            "requires_leaf_specific_sources_before_lesson": True
        }
        obj["canonical_policy"] = {
            "allowed_as_canonical": False,
            "reason": "This file is a source-backed taxonomy seed only. It has no verified atomic facts."
        }
        if "atomic_facts" not in obj:
            obj["atomic_facts"] = []

        idx.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        changed.append(str(idx.relative_to(ROOT)).replace("\\", "/"))

report = {
    "updated_index_files": len(changed),
    "policy": "SOURCE_BACKED_TAXONOMY files are explicitly marked non-canonical and not full knowledge.",
    "files": changed
}

(AUDITS / "taxonomy_noncanonical_downgrade.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print(json.dumps({
    "updated_index_files": len(changed),
    "audit": "audits/taxonomy_noncanonical_downgrade.json"
}, ensure_ascii=False, indent=2))
