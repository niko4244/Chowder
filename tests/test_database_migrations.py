import sqlite3

import pytest

from chowder.database import (
    CHOWDER_APPLICATION_ID,
    CURRENT_SCHEMA_VERSION,
    application_id,
    schema_version,
)
from chowder.executor_investigator import analyze_execution_failure
from chowder.execution_failure import ExecutionFailure, ExecutionStage
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.investigation import RemediationRegistry
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.registry import RegistryInvariantError, RunRegistry
from chowder.resources import ResourceUsage


def _experiment():
    return Experiment(
        "exp-1",
        None,
        Hypothesis("obs", "cause", "fix"),
        {},
        1.0,
    )


def _hardware():
    return HardwareProfile(16, 32, 100, 12, 40, 3)


def _create_legacy_experiments_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY,
            parent_id TEXT,
            estimated_gpu_hours REAL NOT NULL,
            hypothesis_json TEXT NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL
        )"""
    )


def test_legacy_database_is_forward_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    _create_legacy_experiments_table(connection)
    connection.execute(
        "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
        ("legacy", None, 1.0, "{}", "{}", "planned"),
    )
    connection.commit()
    connection.close()

    with RunRegistry(path) as registry:
        assert registry.has_experiment("legacy")
        assert schema_version(registry._conn) == CURRENT_SCHEMA_VERSION
        assert application_id(registry._conn) == CHOWDER_APPLICATION_ID
        tables = {
            row[0]
            for row in registry._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "execution_incidents" in tables
        assert "chowder_schema_history" in tables


def test_supported_version_without_history_table_is_repaired_and_migrated(tmp_path):
    path = tmp_path / "legacy-v1.sqlite"
    connection = sqlite3.connect(path)
    _create_legacy_experiments_table(connection)
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()

    with RunRegistry(path) as registry:
        assert schema_version(registry._conn) == CURRENT_SCHEMA_VERSION
        assert application_id(registry._conn) == CHOWDER_APPLICATION_ID
        history = list(
            registry._conn.execute(
                "SELECT version, name FROM chowder_schema_history ORDER BY version"
            )
        )
        assert history == [
            (1, "baseline-version-marker"),
            (2, "execution-incidents"),
        ]


def test_unrelated_versioned_database_is_not_adopted(tmp_path):
    path = tmp_path / "unrelated.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE unrelated_data (id INTEGER PRIMARY KEY)")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="not recognizable as a Chowder registry"):
        RunRegistry(path)

    check = sqlite3.connect(path)
    try:
        assert application_id(check) == 0
        assert schema_version(check) == 1
        tables = _table_names(check)
        assert "chowder_schema_history" not in tables
        assert "execution_incidents" not in tables
    finally:
        check.close()


def test_database_marked_for_another_application_is_refused(tmp_path):
    path = tmp_path / "foreign-app.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA application_id=123456")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="does not belong to Chowder"):
        RunRegistry(path)


def test_database_newer_than_runtime_is_refused_without_claiming_file(tmp_path):
    path = tmp_path / "future.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="newer than this Chowder build"):
        RunRegistry(path)

    check = sqlite3.connect(path)
    try:
        assert application_id(check) == 0
        assert schema_version(check) == CURRENT_SCHEMA_VERSION + 1
    finally:
        check.close()


def test_identical_training_run_replay_is_idempotent_but_different_content_is_rejected(tmp_path):
    path = tmp_path / "runs.sqlite"
    with RunRegistry(path) as registry:
        registry.record_experiment(_experiment())
        artifact = TrainingArtifact(
            "run-1",
            "exp-1",
            "adapter-a",
            0.5,
            telemetry={"loss": 1.0},
            evidence={"sha": "a"},
        )
        registry.record_training_artifact(artifact)
        registry.record_training_artifact(artifact)
        divergent = TrainingArtifact(
            "run-1",
            "exp-1",
            "adapter-b",
            0.5,
            telemetry={"loss": 1.0},
            evidence={"sha": "b"},
        )
        with pytest.raises(RegistryInvariantError, match="different content"):
            registry.record_training_artifact(divergent)
        assert [row.artifact_ref for row in registry.list_training_artifacts()] == ["adapter-a"]


def test_execution_incident_is_persisted_idempotently(tmp_path):
    path = tmp_path / "incident.sqlite"
    context = ExecutionContext(
        _hardware(),
        str(tmp_path),
        seed=1,
        resolved_config={"backend": {"runtime": {"active_accelerator_count": 2}}},
    )
    failure = ExecutionFailure(
        "worker crash",
        run_id="run-crash",
        experiment_id="exp-crash",
        executor_name="transformers-peft",
        stage=ExecutionStage.TRAIN,
        cause_type="RuntimeError",
        cause_message="CUBLAS_STATUS_EXECUTION_FAILED",
        resource_usage=ResourceUsage.from_wall_time(
            wall_seconds=10,
            active_accelerator_count=2,
            visible_accelerator_count=2,
        ),
    )
    analysis = analyze_execution_failure(
        failure,
        context=context,
        registry=RemediationRegistry(),
        gpu_hour_budget=0.25,
        investigation_id="inv-crash",
        occurred_at="2026-09-01T00:00:00+00:00",
    )
    with RunRegistry(path) as registry:
        registry.record_execution_incident(analysis)
        registry.record_execution_incident(analysis)
        rows = list(registry.list_execution_incidents())
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-crash"
        assert rows[0]["signature_kind"] == "cuda_execution_failed"
        assert rows[0]["gpu_hours_spent"] == pytest.approx(20 / 3600)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
