# Local storage

The engine stores local state explicitly. SQLite is the default embedded database because it requires no server and keeps the local install self-contained. PostgreSQL is treated as an optional local/enterprise backend, never as a hidden dependency.

Rules:
- no cloud database by default
- no telemetry table
- no credentials in source
- schema migrations are explicit and versioned
- evidence records keep source hashes and provenance
- SQL access uses parameters, not string-built queries
