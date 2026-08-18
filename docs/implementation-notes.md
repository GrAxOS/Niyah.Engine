# Implementation Notes

## Canonical storage

PostgreSQL is the transactional source of truth.

- `skg.nodes`: stable semantic entities
- `skg.edges`: directed typed relations
- `skg.evidence`: immutable source references and hashes
- `skg.node_evidence`: evidence bindings for nodes
- `skg.edge_evidence`: evidence bindings for edges
- `skg_audit.events`: append-only mutation history

JSON is the portable interchange format. JSONB is used only for flexible properties and metadata. Known, frequently filtered fields remain typed SQL columns.

## Retrieval

Retrieval is intentionally separate from graph truth.

1. lexical candidate generation
2. semantic candidate generation
3. graph-neighbor expansion
4. rank fusion
5. relation/type validation
6. evidence validation
7. candidate/validated state transition

A nearest-neighbor result never creates an asserted relation by itself.

## Embeddings

Embedding model, vector dimension, metric and version form a single contract. A model-space change requires reindexing or a separate vector contract. The canonical schema therefore does not hard-code a universal dimension.

## HNSW

HNSW is an approximate nearest-neighbor retrieval index, not a domain graph relation. It is a physical search structure over vectors and must remain outside semantic edge semantics.

## Local mode

The SQL migration is plain PostgreSQL plus `vector`. No Supabase-specific runtime API is required for the core graph.

Supabase can add API/auth/RLS around the same PostgreSQL schema. A self-hosted PostgreSQL deployment can use the same core tables and queries. Neon can use the same PostgreSQL/pgvector relational design where the required extension/version is available.

## Security

For exposed Supabase tables, enable RLS and grant least privilege. Vector-search functions should apply access filters inside the database function when filters affect candidate selection; applying filters only after vector ranking can reduce result quality under selective filtering.

## Determinism

Stable outputs require explicit normalization, stable IDs, deterministic sort keys, versioned algorithms, and fixed configuration. Model assistance may propose structured information but cannot bypass schema, evidence, or integrity checks.
