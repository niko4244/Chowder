import sqlite3

import pytest

from chowder.database import (
    CHOWDER_APPLICATION_ID,
    CURRENT_SCHEMA_VERSION,
    application_id,
    schema_version,
)
from chowder.recursive_trace import RecursiveRepairTraceStore, SCHEMA as TRACE_SCHEMA
from chowder.registry import RunRegistry


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def test_trace_store_new_database_uses_central_chowder_identity(tmp_path):
    path = tmp_path / "trace-first.sqlite"
    with RecursiveRepairTraceStore(path) as store:
        assert application_id(store._conn) == CHOWDER_APPLICATION_ID
        assert schema_version(store._conn) == CURRENT_SCHEMA_VERSION
        tables = _tables(store._conn)
        assert "recursive_repair_sessions" in tables
        assert "execution_incidents" in tables
        assert "chowder_schema_history" in tables


def test_registry_can_open_database_created_trace_first(tmp_path):
    path = tmp_path / "trace-first.sqlite"
    with RecursiveRepairTraceStore(path):
        pass
    with RunRegistry(path) as registry:
        assert application_id(registry._conn) == CHOWDER_APPLICATION_ID
        tables = _tables(registry._conn)
        assert "recursive_repair_sessions" in tables
        assert "experiments" in tables


def test_trace_store_can_open_database_created_registry_first(tmp_path):
    path = tmp_path / "registry-first.sqlite"
    with RunRegistry(path):
        pass
    with RecursiveRepairTraceStore(path) as store:
        assert application_id(store._conn) == CHOWDER_APPLICATION_ID
        tables = _tables(store._conn)
        assert "experiments" in tables
        assert "recursive_repair_sessions" in tables


def test_legacy_trace_only_database_is_recognized_and_migrated(tmp_path):
    path = tmp_path / "legacy-trace.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(TRACE_SCHEMA)
    connection.commit()
    assert application_id(connection) == 0
    assert schema_version(connection) == 0
    connection.close()

    with RecursiveRepairTraceStore(path) as store:
        assert application_id(store._conn) == CHOWDER_APPLICATION_ID
        assert schema_version(store._conn) == CURRENT_SCHEMA_VERSION
        assert "recursive_repair_sessions" in _tables(store._conn)


def test_unrelated_unversioned_database_is_refused_by_both_components(tmp_path):
    registry_path = tmp_path / "unrelated-registry.sqlite"
    connection = sqlite3.connect(registry_path)
    connection.execute("CREATE TABLE customer_data (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="not recognizable as a Chowder database"):
        RunRegistry(registry_path)

    trace_path = tmp_path / "unrelated-trace.sqlite"
    connection = sqlite3.connect(trace_path)
    connection.execute("CREATE TABLE customer_data (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="not recognizable as a Chowder database"):
        RecursiveRepairTraceStore(trace_path)

    for path in (registry_path, trace_path):
        check = sqlite3.connect(path)
        try:
            assert application_id(check) == 0
            assert schema_version(check) == 0
            assert _tables(check) == {"customer_data"}
        finally:
            check.close()
