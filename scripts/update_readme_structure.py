from pathlib import Path

p = Path("README.md")
s = p.read_text(encoding="utf-8")

start = s.find("Repository structure")
end = s.find("Security boundary")

new = """Repository structure
schema/
  canonical_knowledge_graph_v2.1.0.json
  sovereign_knowledge_graph_v2.0.0.json
  sovereign_knowledge_graph_v1.0.0.json

chunks/
  legacy source material; preserved for auditability

knowledge/
  00_registry/
    registries and tree layout metadata
  10_taxonomy/
    taxonomy only; not verified knowledge
  20_lessons/
    verified atomic lessons only
  30_canonical/
    canonical graph exports
  40_staging/
    candidate material before validation
  50_aliases/
    alias and rename maps
  60_rejected/
    rejected or invalid material
  70_reserved/
    reserved future topics
  90_legacy/
    preserved legacy material

audits/
  validation outputs and quality gates

normalized/
  generated JSONL projections

scripts/
  audit, validation, taxonomy, and lesson promotion tools

"""

if start == -1 or end == -1:
    raise SystemExit("README markers not found")

p.write_text(s[:start] + new + s[end:], encoding="utf-8")
