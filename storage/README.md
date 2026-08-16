# Storage

Niyah.Engine keeps the storage contract local-first and backend-neutral.

## Backends

- SQLite is the default embedded local database.
- PostgreSQL is an optional backend for a local PostgreSQL installation or an institution-managed PostgreSQL service.
- Neon can be used only when the operator explicitly selects a Neon/PostgreSQL backend. It is not required by the engine and is not a hidden network dependency.

The logical model is shared across backends: sessions, sources, documents, chunks, claims, evidence, graph nodes, graph edges, fetch records, and keyword metadata.

## Local invariants

- No cloud database is contacted by default.
- Credentials are not stored in source files.
- Schema changes are explicit and versioned.
- Writes that span related records use transactions.
- Foreign-key relationships are enforced.
- Content and claim hashes are retained for provenance.
- Query values are parameterized.
- Network retrieval metadata is stored separately from document content.
- Evidence classification is explicit: `FACT`, `INFERENCE`, `UNCERTAIN`, `UNKNOWN`, `CONFLICTED`.

## Retrieval indexes

SQLite uses FTS5 for local full-text retrieval. PostgreSQL uses `tsvector`/`tsquery` and a GIN index for the equivalent lexical path. These indexes are retrieval structures; they do not assert that a source is correct. SQLite FTS5 supports phrase, prefix, boolean, NEAR, and column-filter queries, and its built-in `bm25()` function provides relevance ranking. PostgreSQL full-text search supports normalized lexemes, positional information, `tsquery` operators, and indexed `tsvector` documents.

A future semantic/vector backend may be added without replacing the lexical path. PostgreSQL deployments may additionally use pgvector when explicitly enabled; exact nearest-neighbor search and approximate HNSW/IVFFlat are separate choices and must not silently change evidence semantics.

## Migration files

- `schema/001_initial.sql` — core local state.
- `schema/002_rag.sql` — documents, chunks, FTS5, keywords, fetch records, and claims.
- `postgres/001_rag.sql` — equivalent optional PostgreSQL structures.
