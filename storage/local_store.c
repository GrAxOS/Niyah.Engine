#include "local_store.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

struct NiyahStore {
    sqlite3 *db;
};

static NiyahStoreStatus map_sqlite(int rc) {
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return NIYAH_STORE_OK;
    if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) return NIYAH_STORE_BUSY;
    return NIYAH_STORE_IO;
}

NiyahStoreStatus niyah_store_open(const char *path, NiyahStore **out_store) {
    if (!path || !out_store) return NIYAH_STORE_INVALID;
    *out_store = NULL;

    NiyahStore *store = (NiyahStore *)calloc(1, sizeof(*store));
    if (!store) return NIYAH_STORE_IO;

    int rc = sqlite3_open_v2(path, &store->db,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                             NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(store->db);
        free(store);
        return map_sqlite(rc);
    }

    (void)sqlite3_busy_timeout(store->db, 2500);
    if (sqlite3_exec(store->db, "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA busy_timeout=2500;", NULL, NULL, NULL) != SQLITE_OK) {
        niyah_store_close(store);
        return NIYAH_STORE_IO;
    }

    *out_store = store;
    return NIYAH_STORE_OK;
}

void niyah_store_close(NiyahStore *store) {
    if (!store) return;
    if (store->db) sqlite3_close(store->db);
    free(store);
}

NiyahStoreStatus niyah_store_init_schema(NiyahStore *store) {
    if (!store || !store->db) return NIYAH_STORE_INVALID;

    const char *sql =
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
        "INSERT OR IGNORE INTO schema_version(version) VALUES(1);";

    int rc = sqlite3_exec(store->db, sql, NULL, NULL, NULL);
    return rc == SQLITE_OK ? NIYAH_STORE_OK : NIYAH_STORE_SCHEMA;
}

NiyahStoreStatus niyah_store_insert_source(
    NiyahStore *store,
    const char *id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    const char *source_kind) {

    if (!store || !store->db || !id || !canonical_uri || !content_sha256 || !source_kind) {
        return NIYAH_STORE_INVALID;
    }

    static const char *sql =
        "INSERT INTO sources(id, canonical_uri, title, media_type, language, content_sha256, source_kind) "
        "VALUES(?,?,?,?,?,?,?);";

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(store->db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return map_sqlite(rc);

    sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, canonical_uri, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, title, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, media_type, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, language, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, content_sha256, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, source_kind, -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return map_sqlite(rc);
}
