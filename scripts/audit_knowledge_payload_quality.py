#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
BASE = ROOT / "knowledge" / "20_lessons"
AUDITS = ROOT / "audits"
AUDITS.mkdir(exist_ok=True)

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

indexes = sorted(BASE.rglob("index.json"))

summary = Counter()
issues = []
rows = []

for idx in indexes:
    try:
        obj = load_json(idx)
    except Exception as e:
        summary["invalid_json"] += 1
        issues.append({"file": rel(idx), "issue": "invalid_json", "error": str(e)})
        continue

    status = obj.get("status")
    canonical = ((obj.get("canonical_policy") or {}).get("allowed_as_canonical"))
    sources = obj.get("sources") or []
    claims = obj.get("claims") or []
    atomic_facts = obj.get("atomic_facts") or []

    has_source_url = any(isinstance(s, dict) and s.get("url") and s.get("title") for s in sources)
    has_atomic_facts = bool(atomic_facts)
    atomic_facts_with_sources = 0

    for fact in atomic_facts:
        if isinstance(fact, dict) and fact.get("claim") and fact.get("source_url") and fact.get("source_title"):
            atomic_facts_with_sources += 1

    if not has_source_url:
        issues.append({"file": rel(idx), "issue": "missing_source_url"})

    if status == "SOURCE_BACKED_TAXONOMY":
        summary["source_backed_taxonomy"] += 1

    if status == "VERIFIED_ATOMIC_LESSON":
        summary["verified_atomic_lesson"] += 1
        if not has_atomic_facts:
            issues.append({"file": rel(idx), "issue": "verified_lesson_without_atomic_facts"})
        if atomic_facts_with_sources != len(atomic_facts):
            issues.append({"file": rel(idx), "issue": "atomic_fact_missing_source"})

    if canonical is True:
        summary["canonical_allowed"] += 1
        if status != "VERIFIED_ATOMIC_LESSON":
            issues.append({"file": rel(idx), "issue": "canonical_allowed_but_not_verified_lesson"})
        if not has_atomic_facts:
            issues.append({"file": rel(idx), "issue": "canonical_allowed_without_atomic_facts"})

    if not has_atomic_facts:
        summary["taxonomy_only_not_knowledge"] += 1

    rows.append({
        "file": rel(idx),
        "status": status,
        "canonical_allowed": canonical,
        "source_count": len(sources),
        "claim_count": len(claims),
        "atomic_fact_count": len(atomic_facts),
        "atomic_facts_with_sources": atomic_facts_with_sources
    })

report = {
    "summary": dict(summary),
    "index_files": len(indexes),
    "issue_count": len(issues),
    "issues": issues,
    "rows": rows
}

(AUDITS / "knowledge_payload_quality_audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

md = ["# Knowledge Payload Quality Audit", ""]
md.append(f"- index_files: {len(indexes)}")
for k, v in sorted(summary.items()):
    md.append(f"- {k}: {v}")
md.append(f"- issue_count: {len(issues)}")
md.append("")
md.append("## Policy")
md.append("- SOURCE_BACKED_TAXONOMY is not knowledge.")
md.append("- VERIFIED_ATOMIC_LESSON requires atomic_facts with source_title and source_url.")
md.append("- CANONICAL_GRAPH requires VERIFIED_ATOMIC_LESSON.")
md.append("")
md.append("## Issues")
for issue in issues[:300]:
    md.append(f"- {issue}")

(AUDITS / "knowledge_payload_quality_audit.md").write_text(
    "\n".join(md) + "\n",
    encoding="utf-8"
)

print(json.dumps({
    "index_files": len(indexes),
    "summary": dict(summary),
    "issue_count": len(issues),
    "audit": "audits/knowledge_payload_quality_audit.md"
}, ensure_ascii=False, indent=2))
