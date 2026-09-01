from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable


CURRENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _migration_1_baseline(connection: sqlite3.Connection) -> None:
    # Version 1 marks the pre-migration Chowder schema. Existing databases may
    # already contain any subset of those tables; ownership remains with the
    # component schemas that created them. This migration deliberately does not
    # rewrite legacy tables.
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


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline-version-marker", _migration_1_baseline),
    Migration(2, "execution-incidents", _migration_2_execution_incidents),
)


def schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Apply forward-only, transactional SQLite schema migrations."""

    current = schema_version(connection)
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this Chowder build "
            f"({CURRENT_SCHEMA_VERSION})"
        )

    for migration in MIGRATIONS:
        if migration.version <= current:
            continue
        with connection:
            migration.apply(connection)
            connection.execute(
                "INSERT OR IGNORE INTO chowder_schema_history (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
            connection.execute(f"PRAGMA user_version={migration.version}")
        current = migration.version
    return current


def connect_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    apply_migrations(connection)
    return connection
