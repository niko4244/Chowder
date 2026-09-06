from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable


CURRENT_SCHEMA_VERSION = 4
# SQLite application_id is a 32-bit marker stored in the database header.
# 0x43484F57 == ASCII "CHOW".
CHOWDER_APPLICATION_ID = 0x43484F57

_LEGACY_CHOWDER_ANCHORS = frozenset(
    {
        "experiments",
        "recursive_repair_sessions",
    }
)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _migration_1_baseline(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS chowder_schema_history (
               version INTEGER PRIMARY KEY,
               name TEXT NOT NULL
           )"""
    )


def _migration_2_execution_incidents(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS execution_incidents (
               incident_id TEXT PRIMARY KEY,
               experiment_id TEXT NOT NULL,
               run_id TEXT NOT NULL,
               executor_name TEXT NOT NULL,
               fingerprint_sha256 TEXT NOT NULL,
               signature_kind TEXT NOT NULL,
               gpu_hours_spent REAL NOT NULL,
               capture_json TEXT NOT NULL,
               analysis_json TEXT NOT NULL
           )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_incidents_experiment ON execution_incidents(experiment_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_incidents_fingerprint ON execution_incidents(fingerprint_sha256)"
    )


def _migration_4_teacher_signals(connection: sqlite3.Connection) -> None:
    """Append-only ledger of teacher signals (Teacher Fabric Slice B).

    One row per *distinct* stored signal, keyed by the store's content
    address (digest over request_digest + payload content hash). This is
    the evidence record, not the cache: payload bytes live in the
    TeacherSignalStore's content-addressed files, and this table records
    what was stored, immutably (rows go in once via `_insert_immutable`;
    cache eviction never rewrites or removes them).
    """
    connection.execute(
        """CREATE TABLE IF NOT EXISTS teacher_signals (
               entry_key TEXT PRIMARY KEY,
               artifact_digest TEXT NOT NULL,
               request_digest TEXT NOT NULL,
               payload_file_sha256 TEXT NOT NULL,
               signal_kind TEXT NOT NULL,
               teacher_id TEXT NOT NULL,
               model_revision TEXT NOT NULL,
               tokenizer_identity_sha256 TEXT,
               signal_id TEXT NOT NULL,
               stored_at TEXT NOT NULL,
               metadata_json TEXT NOT NULL
           )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_teacher_signals_request ON teacher_signals(request_digest)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_teacher_signals_teacher ON teacher_signals(teacher_id)"
    )


def _migration_3_recursive_recovery_claims(connection: sqlite3.Connection) -> None:
    """Fence recursive-repair recovery so only one controller may resume."""

    # The session table is component-owned and may not exist in registries that
    # never used recursive repair. Creating the claims table conditionally avoids
    # manufacturing the whole recursive schema in unrelated runs; the trace store
    # creates the same table after it creates its session table on first use.
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "recursive_repair_sessions" not in tables:
        return
    connection.execute(
        """CREATE TABLE IF NOT EXISTS recursive_repair_recovery_claims (
               session_id TEXT PRIMARY KEY,
               claim_token TEXT NOT NULL UNIQUE,
               claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY(session_id) REFERENCES recursive_repair_sessions(session_id)
           )"""
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline-version-marker", _migration_1_baseline),
    Migration(2, "execution-incidents", _migration_2_execution_incidents),
    Migration(3, "recursive-recovery-claims", _migration_3_recursive_recovery_claims),
    Migration(4, "teacher-signals", _migration_4_teacher_signals),
)


def schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def application_id(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA application_id").fetchone()
    return int(row[0]) if row is not None else 0


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _looks_like_legacy_chowder(connection: sqlite3.Connection) -> bool:
    """Conservatively recognize a pre-application-id Chowder database."""

    return bool(_table_names(connection) & _LEGACY_CHOWDER_ANCHORS)


def _ensure_history_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS chowder_schema_history (
               version INTEGER PRIMARY KEY,
               name TEXT NOT NULL
           )"""
    )


def _backfill_history(connection: sqlite3.Connection, current: int) -> None:
    """Repair missing migration metadata for supported historical versions."""

    for migration in MIGRATIONS:
        if migration.version > current:
            break
        connection.execute(
            "INSERT OR IGNORE INTO chowder_schema_history (version, name) VALUES (?, ?)",
            (migration.version, migration.name),
        )


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Apply forward-only, ownership-checked transactional SQLite migrations."""

    current = schema_version(connection)
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this Chowder build "
            f"({CURRENT_SCHEMA_VERSION})"
        )

    app_id = application_id(connection)
    if app_id not in (0, CHOWDER_APPLICATION_ID):
        raise RuntimeError(
            f"database application_id {app_id} does not belong to Chowder"
        )

    tables = _table_names(connection)
    if app_id == 0 and tables and not _looks_like_legacy_chowder(connection):
        raise RuntimeError(
            "SQLite database is not recognizable as a Chowder registry/database; "
            "refusing to adopt it"
        )

    with connection:
        connection.execute(f"PRAGMA application_id={CHOWDER_APPLICATION_ID}")
        _ensure_history_table(connection)
        _backfill_history(connection, current)

    for migration in MIGRATIONS:
        if migration.version <= current:
            continue
        with connection:
            migration.apply(connection)
            _ensure_history_table(connection)
            connection.execute(
                "INSERT OR IGNORE INTO chowder_schema_history (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
            connection.execute(f"PRAGMA user_version={migration.version}")
        current = migration.version
    return current


def connect_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection
