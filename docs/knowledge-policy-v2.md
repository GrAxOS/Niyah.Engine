# Knowledge Policy v2

## Evidence classes

`verified_external_source` requires a stable source locator and a recorded content hash.

`local_repository` records facts directly established by repository contents.

`local_database` records facts established by database inspection or query results.

`user_assertion` records supplied claims without silently promoting them to externally verified facts.

`derived_computation` records deterministic results generated from stored evidence and versioned algorithms.

## Assertion states

- `asserted`: evidence-backed fact.
- `candidate`: unresolved proposed fact or relation.
- `inferred`: deterministic derived result with explicit derivation metadata.
- `deprecated`: historically valid but no longer current.
- `rejected`: evaluated and rejected.

## Non-fact separation

The following MUST NOT be stored as canonical domain facts unless implementation evidence exists:

- branding claims
- unsupported security claims
- "zero telemetry" claims without network/runtime verification
- Ring-0 claims without kernel/runtime evidence
- cryptographic proof claims without proof artifacts
- deterministic claims for probabilistic/model-generated behavior
- performance claims without benchmark evidence
- compliance claims without the applicable requirement and verification evidence

## Graph invariants

1. Every edge source and target must exist.
2. Every asserted node and validated edge must have provenance.
3. Candidate edges must remain distinguishable from validated edges.
4. Merges must preserve all source evidence.
5. Hash equality is integrity evidence, not semantic identity proof.
6. Semantic similarity is candidate-generation evidence, not merge authorization by itself.
7. JSONB is an extensibility boundary, not a replacement for relational keys and foreign keys.
8. Search indexes are derived state, never authoritative graph state.
9. Vector embeddings are model-versioned derived artifacts.
10. Temporal state must distinguish valid time from transaction time.
11. Contradictions are retained and resolved explicitly; they are not silently deleted.
12. Recursive graph expansion must have finite depth and row bounds.
13. Database constraints enforce critical invariants independently of application code.
14. Schema migrations are version-controlled and reproducible.
15. Exported JSON is a projection of canonical database state.

## Retrieval policy

Use lexical search for exact terminology, identifiers, errors, and code symbols.

Use semantic search for meaning-level recall.

Use hybrid retrieval when both exact terminology and semantic recall matter.

Use graph traversal after retrieval to recover relational context.

Use Reciprocal Rank Fusion or another explicitly versioned fusion method rather than silently mixing incomparable score ranges.

## Vector policy

All vectors used in the same similarity space must use the same embedding model/version and dimensionality.

Approximate HNSW retrieval must be benchmarked against exact nearest-neighbor search on representative queries.

Selective metadata filters must be applied inside the database search function so the planner can account for them.

## Security policy

For exposed Supabase tables, enable RLS and grant the minimum required privileges.

Never expose service-role or secret keys to clients.

Keep security-definer helper functions outside exposed API schemas.

Index columns used by RLS predicates when query plans justify it.

Zero-telemetry deployment is an operational property requiring independent network/runtime verification; the graph cannot prove it merely by storing a boolean.

## Local-first profile

Canonical storage: PostgreSQL.

Graph model: normalized nodes and edges.

Semi-structured attributes: JSONB.

Validation: JSON Schema + PostgreSQL constraints.

Semantic search: pgvector.

Approximate index: HNSW.

Lexical search: PostgreSQL FTS + GIN.

Graph traversal: recursive CTEs.

Provenance: immutable evidence rows + node/edge evidence joins.

Audit: append-only mutation records.

Optional capabilities such as RDF/OWL, SHACL, Cypher, GNNs, PageRank, VF2, ZK proofs, distributed consensus, or graph partitioning belong in capability-specific modules and must not be confused with the canonical storage model.
