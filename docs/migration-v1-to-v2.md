# v1 -> v2 Migration

## Canonical authority

`schema/sovereign_knowledge_graph_v2.0.0.json` is the canonical interchange contract.
PostgreSQL is the indexed, constrained projection of that JSON source of truth.

## 1. Preserve v1

Do not delete legacy chunks. Keep the original files and hashes as historical source material.

Legacy claims including `Sovereign Logic Engine v5`, `RING-0`, `zero-telemetry`, `air-gapped`, and `confidence: 1.0` are not promoted to asserted facts solely because they occur in configuration/prompt material.

## 2. Reclassify legacy claims

Classify prompt/config claims as candidate or inferred until implementation evidence exists. Keep the original claim text in evidence metadata with source class `local_repository` or `user_assertion` as appropriate.

## 3. Generate canonical IDs

Node ID:

`n_` + SHA-256(`normalize(label)` + `|` + `normalize(type)` + `|` + `normalize(scope)`)

Edge ID:

`e_` + SHA-256(`source` + `|` + `edge_type` + `|` + `target`)

Normalization is versioned and must be identical during ingestion and re-ingestion.

## 4. Import evidence first

Insert evidence rows first, then nodes, then edges. Bind node/edge provenance to existing evidence records. Reject asserted nodes whose provenance cannot be resolved.

## 5. Node import

Import `id`, `type`, `label`, `status`, `scope`, description and canonical properties. Map predictable scope fields to relational `domain` where available. Keep variable payloads in JSONB.

## 6. Edge import

All newly created edges start as `candidate`. A candidate can be promoted only after endpoint integrity, allowed edge type, type compatibility, cycle/mutual-exclusion checks and evidence requirements pass.

Evidence is mandatory for:

- `contradicts`
- `conflicts_with`
- `supersedes`
- `causes`
- `mitigates`

## 7. Retrieval projection

Embeddings, FTS vectors, and indexes are derived PostgreSQL runtime state. They are never evidence and never create semantic truth by themselves.

One embedding model/dimension pair is allowed per embedding space. A model or dimension change creates a new embedding space or requires complete reindexing.

## 8. PostgreSQL deployment

Run:

```bash
docker compose up -d
```

The local PostgreSQL network is configured with no external network attachment. Apply migrations from `/sql` in lexical order.

## 9. Verification

```bash
python scripts/audit_graph.py --graph data/canonical_graph_v2.json
python scripts/train_embeddings.py --model-name <local-model> --dimensions <model-dimension>
pytest -q tests/test_validation.py
```

A corpus is `PASSED` only when duplicate IDs, dangling edges, unsupported asserted nodes, unknown types, missing required evidence and schema failures are all absent.
