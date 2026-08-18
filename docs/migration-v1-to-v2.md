# Migration v1 -> v2

## Critical corrections

The v1 graph contained schema-level policy objects and implementation claims mixed into semantic graph nodes. It also treated prompt-derived claims as fully verified facts and used free-form IDs that prevented reliable global integrity checks. The current repository keeps legacy material for audit history and introduces a strict v2 projection.

## Reclassification

- prompt-derived architecture/system claims -> `candidate` or `inferred` unless implementation evidence exists
- `Ring-0`, `zero telemetry`, `air-gapped`, `ZK`, `Raft`, and performance claims -> capability/deployment evidence, not canonical semantic facts without implementation evidence
- HNSW -> vector index capability, not graph-domain relationship
- FTS -> retrieval capability
- JSONB -> storage representation
- JSON Schema -> validation contract
- RLS -> security mechanism
- recursive CTE -> query mechanism
- entity-resolution algorithms -> reasoning layer
- evidence -> immutable first-class records

## ID policy

v1 identifiers such as `n_sle_v5` remain historical identifiers only. New canonical identifiers use content-derived SHA-256 IDs after versioned canonicalization.

## Merge policy

Do not destructively rewrite or delete historical chunks. Build a v2 canonical projection and preserve v1 source hashes/manifests so the transformation remains reproducible.

## Validation gates

PASS requires:

- zero duplicate canonical node IDs
- zero dangling edges
- every asserted node with evidence
- every validated edge with evidence
- schema validation success
- deterministic ordering
- explicit status for candidate/inferred content
