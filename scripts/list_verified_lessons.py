#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path("knowledge/20_lessons")
verified = []

for p in sorted(BASE.rglob("index.json")):
    x = json.loads(p.read_text(encoding="utf-8"))
    if x.get("status") == "VERIFIED_ATOMIC_LESSON":
        verified.append({
            "file": str(p).replace("\\", "/"),
            "id": x.get("id"),
            "facts": len(x.get("atomic_facts", [])),
            "sources": len(x.get("sources", [])),
            "canonical": bool((x.get("canonical_policy") or {}).get("allowed_as_canonical"))
        })

print(json.dumps({
    "verified_count": len(verified),
    "verified": verified
}, ensure_ascii=False, indent=2))
