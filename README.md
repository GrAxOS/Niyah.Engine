# Khawrizm Sovereign Graph

Evidence-first local knowledge graph for PostgreSQL/Supabase deployments.

## Canonical model

- PostgreSQL relational core: `nodes`, `edges`, `evidence`
- JSONB only for variable properties
- JSON Schema for interchange validation
- pgvector/HNSW for semantic retrieval
- PostgreSQL FTS/GIN for lexical retrieval
- Recursive CTEs for bounded graph traversal
- Provenance as first-class evidence relations
- Candidate → validate graph mutation pipeline
- Versioned temporal state and append-only audit records

## Knowledge policy

A record is not an asserted fact merely because it appears in an LLM prompt, generated JSON, design note, benchmark description, or repository comment.

Assertions require evidence.

Unverified claims remain `candidate` or `inferred` with explicit provenance and derivation metadata.

See `docs/knowledge-policy-v2.md` and `docs/migration-v1-to-v2.md`.

## Repository structure
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

Security boundary

"Zero telemetry" and "air-gapped" are deployment properties. The graph stores verification evidence; it does not make those claims true by declaration.

## Retrieval boundary

HNSW is an approximate vector index, not a semantic graph. Exact nearest-neighbor evaluation remains the validation baseline for recall measurements.

## Data boundary

Exported JSON is a projection of canonical database state. PostgreSQL is the transactional source of truth.
