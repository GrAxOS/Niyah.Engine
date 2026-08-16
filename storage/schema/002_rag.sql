PRAGMA foreign_keys = ON;

-- Local RAG storage. SQLite is the canonical embedded store.
-- FTS5 provides exact/phrase/prefix retrieval; BM25 is applied by the query layer.

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    canonical_uri TEXT NOT NULL,
    title TEXT,
    media_type TEXT NOT NULL DEFAULT 'text/plain',
    language TEXT,
    content_sha256 TEXT NOT NULL UNIQUE,
    content_bytes INTEGER NOT NULL CHECK (content_bytes >= 0),
    retrieved_at TEXT NOT NULL,
    published_at TEXT,
    modified_at TEXT,
    etag TEXT,
    last_modified TEXT,
    parser_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','indexed','rejected','failed'))
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(canonical_uri);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    heading TEXT,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL UNIQUE,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    UNIQUE(document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_ordinal ON document_chunks(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON document_chunks(text_sha256);

CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    heading,
    text,
    keywords,
    tokenize = 'unicode61 remove_diacritics 1'
);

CREATE TABLE IF NOT EXISTS chunk_keywords (
    chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0),
    PRIMARY KEY (chunk_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_chunk_keywords_keyword ON chunk_keywords(keyword);

CREATE TABLE IF NOT EXISTS source_fetches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    requested_uri TEXT NOT NULL,
    effective_uri TEXT,
    http_status INTEGER,
    content_type TEXT,
    content_length INTEGER,
    fetched_at TEXT NOT NULL,
    duration_ms INTEGER,
    error_code TEXT,
    robots_result TEXT CHECK (robots_result IN ('allow','deny','unavailable','unknown'))
);

CREATE INDEX IF NOT EXISTS idx_source_fetches_source_time ON source_fetches(source_id, fetched_at DESC);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    claim_sha256 TEXT NOT NULL UNIQUE,
    classification TEXT NOT NULL CHECK (classification IN ('FACT','INFERENCE','UNCERTAIN','UNKNOWN','CONFLICTED')),
    extractor_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claims_classification ON claims(classification);

CREATE TABLE IF NOT EXISTS claim_keywords (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0),
    PRIMARY KEY (claim_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_claim_keywords_keyword ON claim_keywords(keyword);

-- Version marker for the application migration runner.
INSERT OR IGNORE INTO schema_version(version) VALUES (2);
