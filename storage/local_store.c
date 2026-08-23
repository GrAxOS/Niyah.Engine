#include "local_store.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

struct NiyahStore {
    sqlite3 *db;
};

static NiyahStoreStatus map_sqlite(int rc)
{
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return NIYAH_STORE_OK;
    if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) return NIYAH_STORE_BUSY;
    if (rc == SQLITE_CONSTRAINT ||
        rc == SQLITE_CONSTRAINT_CHECK ||
        rc == SQLITE_CONSTRAINT_FOREIGNKEY ||
        rc == SQLITE_CONSTRAINT_PRIMARYKEY ||
        rc == SQLITE_CONSTRAINT_UNIQUE) return NIYAH_STORE_SCHEMA;
    return NIYAH_STORE_IO;
}

NiyahStoreStatus niyah_store_open(const char *path, NiyahStore **out_store)
{
    if (!path || path[0] == '\0' || !out_store) return NIYAH_STORE_INVALID;
    *out_store = NULL;

    NiyahStore *store = (NiyahStore *)calloc(1u, sizeof(*store));
    if (!store) return NIYAH_STORE_IO;

    const int rc = sqlite3_open_v2(
        path,
        &store->db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        NULL);

    if (rc != SQLITE_OK) {
        if (store->db) (void)sqlite3_close(store->db);
        free(store);
        return map_sqlite(rc);
    }

    if (sqlite3_busy_timeout(store->db, 2500) != SQLITE_OK) {
        niyah_store_close(store);
        return NIYAH_STORE_IO;
    }

    const int pragma_rc = sqlite3_exec(
        store->db,
        "PRAGMA foreign_keys=ON;"
        "PRAGMA journal_mode=WAL;"
        "PRAGMA synchronous=NORMAL;"
        "PRAGMA busy_timeout=2500;",
        NULL,
        NULL,
        NULL);

    if (pragma_rc != SQLITE_OK) {
        niyah_store_close(store);
        return map_sqlite(pragma_rc);
    }

    *out_store = store;
    return NIYAH_STORE_OK;
}

void niyah_store_close(NiyahStore *store)
{
    if (!store) return;
    if (store->db) (void)sqlite3_close(store->db);
    free(store);
}

NiyahStoreStatus niyah_store_init_schema(NiyahStore *store)
{
    if (!store || !store->db) return NIYAH_STORE_INVALID;

    static const char sql[] =
        "BEGIN IMMEDIATE;"
        "PRAGMA foreign_keys=ON;"
        "CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
        "CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, language TEXT NOT NULL, title TEXT);"
        "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user','assistant','system')), content TEXT NOT NULL, created_at TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, canonical_uri TEXT NOT NULL, title TEXT, media_type TEXT, language TEXT, content_sha256 TEXT NOT NULL, retrieved_at TEXT, source_kind TEXT NOT NULL CHECK(source_kind IN ('local','web','user','system','unknown')));"
        "CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, claim TEXT NOT NULL, classification TEXT NOT NULL CHECK(classification IN ('FACT','INFERENCE','UNCERTAIN','UNKNOWN','CONFLICTED')), evidence_sha256 TEXT NOT NULL, created_at TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS graph_nodes(id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}');"
        "CREATE TABLE IF NOT EXISTS graph_edges(id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_id TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE, target_node_id TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE, relation TEXT NOT NULL, evidence_id TEXT REFERENCES evidence(id) ON DELETE SET NULL);"
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);"
        "CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_sha256);"
        "CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id);"
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id);"
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id);"
        "CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, canonical_uri TEXT NOT NULL, title TEXT, media_type TEXT NOT NULL DEFAULT 'text/plain', language TEXT, content_sha256 TEXT NOT NULL UNIQUE, content_bytes INTEGER NOT NULL CHECK(content_bytes >= 0), retrieved_at TEXT NOT NULL, published_at TEXT, modified_at TEXT, etag TEXT, last_modified TEXT, parser_version TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','indexed','rejected','failed')));"
        "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);"
        "CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(canonical_uri);"
        "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);"
        "CREATE TABLE IF NOT EXISTS document_chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL CHECK(ordinal >= 0), start_offset INTEGER NOT NULL CHECK(start_offset >= 0), end_offset INTEGER NOT NULL CHECK(end_offset >= start_offset), heading TEXT, text TEXT NOT NULL, text_sha256 TEXT NOT NULL UNIQUE, token_count INTEGER NOT NULL CHECK(token_count >= 0), UNIQUE(document_id, ordinal));"
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_ordinal ON document_chunks(document_id, ordinal);"
        "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON document_chunks(text_sha256);"
        "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(chunk_id UNINDEXED, title, heading, text, keywords, tokenize='unicode61 remove_diacritics 1');"
        "CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE, claim_text TEXT NOT NULL, claim_sha256 TEXT NOT NULL UNIQUE, classification TEXT NOT NULL CHECK(classification IN ('FACT','INFERENCE','UNCERTAIN','UNKNOWN','CONFLICTED')), extractor_version TEXT NOT NULL, created_at TEXT NOT NULL);"
        "CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);"
        "CREATE INDEX IF NOT EXISTS idx_claims_classification ON claims(classification);"
        "INSERT OR IGNORE INTO schema_version(version) VALUES(2);"
        "COMMIT;";

    const int rc = sqlite3_exec(store->db, sql, NULL, NULL, NULL);
    if (rc == SQLITE_OK) return NIYAH_STORE_OK;
    (void)sqlite3_exec(store->db, "ROLLBACK;", NULL, NULL, NULL);
    return map_sqlite(rc);
}

static NiyahStoreStatus prepare_and_bind_text(
    sqlite3 *db,
    const char *sql,
    const char *const *values,
    size_t count,
    sqlite3_stmt **out_stmt)
{
    if (!db || !sql || !values || !out_stmt || count == 0u) return NIYAH_STORE_INVALID;
    *out_stmt = NULL;

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return map_sqlite(rc);

    for (size_t i = 0u; i < count; ++i) {
        rc = values[i]
            ? sqlite3_bind_text(stmt, (int)(i + 1u), values[i], -1, SQLITE_TRANSIENT)
            : sqlite3_bind_null(stmt, (int)(i + 1u));
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            return map_sqlite(rc);
        }
    }

    *out_stmt = stmt;
    return NIYAH_STORE_OK;
}

NiyahStoreStatus niyah_store_insert_source(
    NiyahStore *store,
    const char *id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    const char *source_kind)
{
    if (!store || !store->db || !id || !canonical_uri || !content_sha256 || !source_kind)
        return NIYAH_STORE_INVALID;

    static const char sql[] =
        "INSERT INTO sources(id,canonical_uri,title,media_type,language,content_sha256,source_kind) VALUES(?,?,?,?,?,?,?);";
    const char *values[] = {id, canonical_uri, title, media_type, language, content_sha256, source_kind};
    sqlite3_stmt *stmt = NULL;
    NiyahStoreStatus status = prepare_and_bind_text(store->db, sql, values, 7u, &stmt);
    if (status != NIYAH_STORE_OK) return status;
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return map_sqlite(rc);
}

NiyahStoreStatus niyah_store_insert_document(
    NiyahStore *store,
    const char *id,
    const char *source_id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    long long content_bytes,
    const char *retrieved_at,
    const char *parser_version,
    const char *status_value)
{
    if (!store || !store->db || !id || !source_id || !canonical_uri ||
        !content_sha256 || content_bytes < 0 || !retrieved_at || !parser_version || !status_value)
        return NIYAH_STORE_INVALID;

    static const char sql[] =
        "INSERT INTO documents(id,source_id,canonical_uri,title,media_type,language,content_sha256,content_bytes,retrieved_at,parser_version,status) VALUES(?,?,?,?,?,?,?,?,?,?,?);";
    const char *values[] = {
        id, source_id, canonical_uri, title, media_type, language,
        content_sha256, retrieved_at, parser_version, status_value
    };

    sqlite3_stmt *stmt = NULL;
    NiyahStoreStatus status = prepare_and_bind_text(store->db, sql, values, 10u, &stmt);
    if (status != NIYAH_STORE_OK) return status;

    int rc = sqlite3_bind_int64(stmt, 8, content_bytes);
    if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return map_sqlite(rc);
}

NiyahStoreStatus niyah_store_insert_chunk(
    NiyahStore *store,
    const char *id,
    const char *document_id,
    long long ordinal,
    long long start_offset,
    long long end_offset,
    const char *heading,
    const char *text,
    const char *text_sha256,
    long long token_count)
{
    if (!store || !store->db || !id || !document_id || ordinal < 0 ||
        start_offset < 0 || end_offset < start_offset || !text || !text_sha256 || token_count < 0)
        return NIYAH_STORE_INVALID;

    static const char sql[] =
        "INSERT INTO document_chunks(id,document_id,ordinal,start_offset,end_offset,heading,text,text_sha256,token_count) VALUES(?,?,?,?,?,?,?,?,?);";
    const char *values[] = {id, document_id, heading, text, text_sha256};
    sqlite3_stmt *stmt = NULL;
    NiyahStoreStatus status = prepare_and_bind_text(store->db, sql, values, 5u, &stmt);
    if (status != NIYAH_STORE_OK) return status;

    int rc = sqlite3_bind_int64(stmt, 3, ordinal);
    if (rc == SQLITE_OK) rc = sqlite3_bind_int64(stmt, 4, start_offset);
    if (rc == SQLITE_OK) rc = sqlite3_bind_int64(stmt, 5, end_offset);
    if (rc == SQLITE_OK) rc = sqlite3_bind_int64(stmt, 9, token_count);
    if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return map_sqlite(rc);
}

NiyahStoreStatus niyah_store_insert_claim(
    NiyahStore *store,
    const char *id,
    const char *chunk_id,
    const char *claim_text,
    const char *claim_sha256,
    const char *classification,
    const char *extractor_version,
    const char *created_at)
{
    if (!store || !store->db || !id || !chunk_id || !claim_text || !claim_sha256 ||
        !classification || !extractor_version || !created_at)
        return NIYAH_STORE_INVALID;

    static const char sql[] =
        "INSERT INTO claims(id,chunk_id,claim_text,claim_sha256,classification,extractor_version,created_at) VALUES(?,?,?,?,?,?,?);";
    const char *values[] = {id, chunk_id, claim_text, claim_sha256, classification, extractor_version, created_at};
    sqlite3_stmt *stmt = NULL;
    NiyahStoreStatus status = prepare_and_bind_text(store->db, sql, values, 7u, &stmt);
    if (status != NIYAH_STORE_OK) return status;
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return map_sqlite(rc);
}
