-- Optional PostgreSQL backend for the same logical model used by local SQLite.
-- Keep PostgreSQL as a storage backend; it is not required for local operation.

CREATE TABLE IF NOT EXISTS sources (
    id text PRIMARY KEY,
    canonical_uri text NOT NULL,
    title text,
    media_type text,
    language text,
    content_sha256 text NOT NULL UNIQUE,
    retrieved_at timestamptz,
    source_kind text NOT NULL CHECK (source_kind IN ('local','web','user','system','unknown')),
    trust_tier smallint NOT NULL DEFAULT 3 CHECK (trust_tier BETWEEN 0 AND 4)
);

CREATE TABLE IF NOT EXISTS documents (
    id text PRIMARY KEY,
    source_id text NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    canonical_uri text NOT NULL,
    title text,
    media_type text NOT NULL DEFAULT 'text/plain',
    language text,
    content_sha256 text NOT NULL UNIQUE,
    content_bytes bigint NOT NULL CHECK (content_bytes >= 0),
    retrieved_at timestamptz NOT NULL,
    published_at timestamptz,
    modified_at timestamptz,
    etag text,
    last_modified text,
    parser_version text NOT NULL,
    status text NOT NULL CHECK (status IN ('pending','indexed','rejected','failed'))
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(canonical_uri);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    start_offset bigint NOT NULL CHECK (start_offset >= 0),
    end_offset bigint NOT NULL CHECK (end_offset >= start_offset),
    heading text,
    text text NOT NULL,
    text_sha256 text NOT NULL UNIQUE,
    token_count integer NOT NULL CHECK (token_count >= 0),
    UNIQUE(document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_ordinal ON document_chunks(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON document_chunks(text_sha256);

-- PostgreSQL full-text retrieval. Query code may populate/search this column.
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector;
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON document_chunks USING GIN(search_vector);

CREATE TABLE IF NOT EXISTS chunk_keywords (
    chunk_id text NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    keyword text NOT NULL,
    weight real NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0),
    PRIMARY KEY (chunk_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_chunk_keywords_keyword ON chunk_keywords(keyword);

CREATE TABLE IF NOT EXISTS source_fetches (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id text NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    requested_uri text NOT NULL,
    effective_uri text,
    http_status integer,
    content_type text,
    content_length bigint,
    fetched_at timestamptz NOT NULL,
    duration_ms bigint,
    error_code text,
    robots_result text CHECK (robots_result IN ('allow','deny','unavailable','unknown'))
);

CREATE INDEX IF NOT EXISTS idx_source_fetches_source_time ON source_fetches(source_id, fetched_at DESC);

CREATE TABLE IF NOT EXISTS claims (
    id text PRIMARY KEY,
    chunk_id text NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    claim_text text NOT NULL,
    claim_sha256 text NOT NULL UNIQUE,
    classification text NOT NULL CHECK (classification IN ('FACT','INFERENCE','UNCERTAIN','UNKNOWN','CONFLICTED')),
    extractor_version text NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claims_classification ON claims(classification);
