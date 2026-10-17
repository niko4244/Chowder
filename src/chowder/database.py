from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable


CURRENT_SCHEMA_VERSION = 6
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


def _migration_4_goal_lifecycle(connection: sqlite3.Connection) -> None:
    """Persist frozen objective identity, assessments, and terminal decisions."""

    connection.execute(
        """CREATE TABLE IF NOT EXISTS goal_objectives (
               objective_version TEXT PRIMARY KEY,
               identity_json TEXT NOT NULL,
               goal_json TEXT NOT NULL,
               created_at TEXT NOT NULL
           )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS goal_assessments (
               assessment_id TEXT PRIMARY KEY,
               objective_version TEXT NOT NULL,
               artifact_identity TEXT NOT NULL,
               status TEXT NOT NULL,
               assessment_json TEXT NOT NULL,
               recorded_at TEXT NOT NULL,
               FOREIGN KEY(objective_version) REFERENCES goal_objectives(objective_version)
           )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS goal_terminal_events (
               event_id INTEGER PRIMARY KEY AUTOINCREMENT,
               objective_version TEXT NOT NULL,
               terminal_state TEXT NOT NULL,
               artifact_identity TEXT NOT NULL,
               recorded_at TEXT NOT NULL,
               FOREIGN KEY(objective_version) REFERENCES goal_objectives(objective_version)
           )"""
    )


def _migration_5_goal_protocol_migrations(connection: sqlite3.Connection) -> None:
    """Persist explicit human-approved legacy protocol-contract migrations."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS goal_objective_migrations (
               migration_id TEXT PRIMARY KEY,
               source_objective_version TEXT NOT NULL,
               target_objective_version TEXT NOT NULL UNIQUE,
               source_identity_json TEXT NOT NULL,
               target_identity_json TEXT NOT NULL,
               protocol_contract_digest TEXT NOT NULL,
               approval_json TEXT NOT NULL,
               provenance_json TEXT NOT NULL,
               recorded_at TEXT NOT NULL,
               FOREIGN KEY(source_objective_version) REFERENCES goal_objectives(objective_version),
               FOREIGN KEY(target_objective_version) REFERENCES goal_objectives(objective_version)
           )"""
    )


def _migration_6_timestamp_bound_migration_hashes(connection: sqlite3.Connection) -> None:
    """Version migration hashes so legacy records remain verifiable."""
    connection.execute(
        "ALTER TABLE goal_objective_migrations "
        "ADD COLUMN migration_hash_version INTEGER NOT NULL DEFAULT 1"
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline-version-marker", _migration_1_baseline),
    Migration(2, "execution-incidents", _migration_2_execution_incidents),
    Migration(3, "recursive-recovery-claims", _migration_3_recursive_recovery_claims),
    Migration(4, "goal-lifecycle", _migration_4_goal_lifecycle),
    Migration(5, "goal-protocol-contract-migrations", _migration_5_goal_protocol_migrations),
    Migration(6, "timestamp-bound-migration-hashes", _migration_6_timestamp_bound_migration_hashes),
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
