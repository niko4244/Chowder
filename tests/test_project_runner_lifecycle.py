from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from chowder.executors import EvaluationOutcome, TrainingArtifact
from chowder.goal_lifecycle import GoalLifecycleError, GoalTerminalState
from chowder.hardware import HardwareSnapshot
from chowder.models import Goal, MetricTarget
from chowder.project import project_from_mapping
from chowder.improvement.constitution import Constitution
from chowder.project_runner import (
    ProtocolContractMigrationApproval,
    _protocol_contract_digest,
    _project_benchmark_digest,
    migrate_legacy_protocol_contract,
    run_project,
)
from chowder.registry import RunRegistry

from test_project import _payload


PROTOCOL = "a" * 64


class _Trainer:
    name = "test-trainer"
    calls = 0

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        self.calls += 1
        return TrainingArtifact(
            "train-1",
            experiment.experiment_id,
            "/artifact",
            0.1,
            evidence={"artifact": "test"},
        )

    def cancel(self, run_id):
        pass


class _Evaluator:
    name = "test-evaluator"
    metric = 0.7

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            "eval-1",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": self.metric},
            0.1,
            {"protocol_sha256": PROTOCOL},
        )

    def cancel(self, run_id):
        pass


def _project(tmp_path: Path, *, baseline_quality: float, objective_version: str = "project"):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(
        '{"prompt":"hello","expected":"hello"}\n', encoding="utf-8"
    )
    payload = _payload(tmp_path)
    payload["objective_version"] = objective_version
    payload["baseline"]["metrics"]["quality"] = baseline_quality
    payload["baseline"]["evaluation_protocol_sha256"] = PROTOCOL
    return project_from_mapping(payload, source_dir=tmp_path)


def _patch_runner(monkeypatch, trainer, evaluator):
    monkeypatch.setattr("chowder.project_runner.TransformersPeftExecutor", lambda: trainer)
    monkeypatch.setattr("chowder.project_runner.TransformersTextEvaluator", lambda: evaluator)
    monkeypatch.setattr(
        "chowder.project_runner.detect_hardware",
        lambda path: HardwareSnapshot(
            platform="test",
            cpu_count=2,
            ram_gb=8.0,
            storage_total_gb=10.0,
            storage_free_gb=9.0,
            accelerators=(),
        ),
    )


def test_project_runner_reports_success_only_when_parent_goal_is_met(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)

    outcome = run_project(project)

    assert outcome.succeeded is True
    assert outcome.generation.goal_terminal_state == GoalTerminalState.STOP_GOALS_MET.value
    assert trainer.calls == 0


def test_project_runner_promotion_without_goal_completion_is_not_success(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.2)
    _patch_runner(monkeypatch, trainer, evaluator)

    outcome = run_project(project)

    assert outcome.promoted_experiment_id == project.experiment.experiment_id
    assert outcome.generation.goal_assessment is not None
    assert outcome.generation.goal_assessment.status.value == "UNMET"
    assert outcome.generation.goal_terminal_state is None
    assert outcome.succeeded is False
    assert trainer.calls == 1


def test_project_runner_refuses_changed_benchmark_on_resume(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True

    (tmp_path / "eval.jsonl").write_text(
        '{"prompt":"changed","expected":"hello"}\n', encoding="utf-8"
    )
    with pytest.raises(GoalLifecycleError, match="objective identity changed"):
        run_project(project)
    assert trainer.calls == 0


@pytest.mark.parametrize(
    "mutate",
    (
        lambda config, project: config["backend"].update(precision="bf16"),
        lambda config, project: config["backend"].update(quantization="4bit"),
        lambda config, project: config["evaluation"].update(device="auto"),
        lambda config, project: config["backend"].update(revision="revision-2"),
        lambda config, project: config.update(seed=8),
    ),
)
def test_project_runner_refuses_changed_protocol_configuration_on_resume(
    tmp_path, monkeypatch, mutate
):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True

    changed_config = deepcopy(project.config)
    changed = replace(project, config=changed_config)
    mutate(changed_config, changed)
    with pytest.raises(GoalLifecycleError, match="objective identity changed"):
        run_project(changed)
    assert trainer.calls == 0


def test_project_runner_refuses_changed_protocol_on_resume(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True

    changed_baseline = replace(
        project.baseline,
        evidence={"evaluation_protocol_sha256": "b" * 64},
    )
    changed = replace(project, baseline=changed_baseline)
    with pytest.raises(GoalLifecycleError, match="objective identity changed"):
        run_project(changed)
    assert trainer.calls == 0


def test_project_runner_refuses_missing_protocol_contract_on_resume(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True

    with RunRegistry(project.registry_path) as registry:
        stored = registry.get_goal_objective(project.objective_version)
        assert stored is not None
        legacy_goal = dict(stored["goal"])
        legacy_goal.pop("protocol_contract_digest")
        registry._conn.execute(
            "UPDATE goal_objectives SET goal_json = ? WHERE objective_version = ?",
            (json.dumps(legacy_goal, sort_keys=True, separators=(",", ":")), project.objective_version),
        )
        registry._conn.commit()

    with pytest.raises(GoalLifecycleError, match="migration required"):
        run_project(project)
    assert trainer.calls == 0


def _remove_protocol_contract(project):
    with RunRegistry(project.registry_path) as registry:
        stored = registry.get_goal_objective(project.objective_version)
        assert stored is not None
        legacy_goal = dict(stored["goal"])
        legacy_goal.pop("protocol_contract_digest", None)
        registry._conn.execute(
            "UPDATE goal_objectives SET goal_json = ? WHERE objective_version = ?",
            (
                json.dumps(legacy_goal, sort_keys=True, separators=(",", ":")),
                project.objective_version,
            ),
        )
        registry._conn.commit()


def test_legacy_protocol_migration_requires_approval(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)

    with pytest.raises(GoalLifecycleError, match="explicit human approval"):
        migrate_legacy_protocol_contract(
            project,
            new_objective_version="project-protocol-v2",
        )
    assert trainer.calls == 0


def test_approved_legacy_protocol_migration_is_append_only_and_resumable(
    tmp_path, monkeypatch
):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)

    migrated = migrate_legacy_protocol_contract(
        project,
        new_objective_version="project-protocol-v2",
        approval=ProtocolContractMigrationApproval(
            approval_id="approval-1",
            approver="operator@example.com",
            approved_at="2026-09-20T12:00:00Z",
            reason="Freeze the current evaluation/training contract",
        ),
    )

    assert migrated.objective_version == "project-protocol-v2"
    with RunRegistry(project.registry_path) as registry:
        assert registry.get_goal_objective("project") is not None
        target = registry.get_goal_objective("project-protocol-v2")
        assert target is not None
        assert target["goal"]["protocol_contract_digest"]
        migrations = registry.list_goal_objective_migrations()
        assert len(migrations) == 1
        assert migrations[0].source_objective_version == "project"
        assert migrations[0].target_objective_version == "project-protocol-v2"
        assert migrations[0].approval.approval_id == "approval-1"
        assert migrations[0].provenance.source_identity_digest

    resumed = run_project(migrated)
    assert resumed.succeeded is True
    assert trainer.calls == 0


def test_registry_rejects_incomplete_or_unrelated_migration_provenance(tmp_path, monkeypatch):
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, _Trainer(), _Evaluator())
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)

    with RunRegistry(project.registry_path) as registry:
        stored = registry.get_goal_objective(project.objective_version)
        assert stored is not None
        identity = stored["identity"]
        target_identity = Constitution().new_objective_identity(
            objective_version="project-protocol-v2",
            goal=project.goal,
            benchmark_digest=str(identity["benchmark_digest"]),
            evaluation_protocol_digest=str(identity["evaluation_protocol_digest"]),
        )
        target_payload = dict(stored["goal"])
        contract = _protocol_contract_digest(project)
        target_payload["protocol_contract_digest"] = contract
        valid = {
            "operation": "legacy_protocol_contract_migration",
            "source_objective_version": project.objective_version,
            "target_objective_version": target_identity.objective_version,
            "source_identity_digest": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "target_identity_digest": hashlib.sha256(
                json.dumps(
                    {
                        "objective_version": target_identity.objective_version,
                        "goal_digest": target_identity.goal_digest,
                        "benchmark_digest": target_identity.benchmark_digest,
                        "evaluation_protocol_digest": target_identity.evaluation_protocol_digest,
                        "constitution_digest": target_identity.constitution_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "protocol_contract_digest": contract,
            "registry_path": str(project.registry_path),
        }
        approval = {
            "approval_id": "approval-forged",
            "approver": "operator@example.com",
            "approved_at": "2026-09-20T12:00:00Z",
            "reason": "test",
        }
        forged_source = dict(valid)
        forged_source["source_identity_digest"] = "0" * 64
        with pytest.raises(ValueError, match="source identity"):
            registry.record_goal_objective_migration(
                source_objective_version=project.objective_version,
                target_identity=target_identity,
                target_goal_payload=target_payload,
                protocol_contract_digest=contract,
                approval=approval,
                provenance=forged_source,
            )
        assert registry.list_goal_objective_migrations() == ()

        incomplete = dict(valid)
        incomplete.pop("operation")
        with pytest.raises(ValueError, match="provenance"):
            registry.record_goal_objective_migration(
                source_objective_version=project.objective_version,
                target_identity=target_identity,
                target_goal_payload=target_payload,
                protocol_contract_digest=contract,
                approval=approval,
                provenance=incomplete,
            )
        malformed = dict(valid)
        malformed["protocol_contract_digest"] = "z" * 64
        with pytest.raises(ValueError, match="digest"):
            registry.record_goal_objective_migration(
                source_objective_version=project.objective_version,
                target_identity=target_identity,
                target_goal_payload=target_payload,
                protocol_contract_digest=contract,
                approval=approval,
                provenance=malformed,
            )
        unrelated = dict(valid)
        unrelated["target_objective_version"] = "unrelated"
        with pytest.raises(ValueError, match="target objective"):
            registry.record_goal_objective_migration(
                source_objective_version=project.objective_version,
                target_identity=target_identity,
                target_goal_payload=target_payload,
                protocol_contract_digest=contract,
                approval=approval,
                provenance=unrelated,
            )

        migration_id = registry.record_goal_objective_migration(
            source_objective_version=project.objective_version,
            target_identity=target_identity,
            target_goal_payload=target_payload,
            protocol_contract_digest=contract,
            approval=approval,
            provenance=valid,
        )
        migration = registry.list_goal_objective_migrations()[0]
        assert migration.migration_id == migration_id
        assert migration.approval.approval_id == "approval-forged"
        assert migration.target_identity.objective_version == "project-protocol-v2"


@pytest.mark.parametrize("column, payload", (
    ("approval_json", {"approval_id": "missing-fields"}),
    ("provenance_json", {"operation": "forged"}),
    ("approval_json", None),
    ("provenance_json", 123),
    ("source_identity_json", "not-json"),
))
def test_registry_migration_readback_fails_closed_on_corrupt_evidence(
    tmp_path, monkeypatch, column, payload
):
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, _Trainer(), _Evaluator())
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)
    migrate_legacy_protocol_contract(
        project,
        new_objective_version="project-protocol-v2",
        approval=ProtocolContractMigrationApproval(
            approval_id="approval-readback",
            approver="operator@example.com",
            approved_at="2026-09-20T12:00:00Z",
            reason="Freeze current contract",
        ),
    )
    with RunRegistry(project.registry_path) as registry:
        value = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if column in {"approval_json", "provenance_json"}
            else payload
        )
        registry._conn.execute(
            f"UPDATE goal_objective_migrations SET {column} = ?",
            (value,),
        )
        registry._conn.commit()
        with pytest.raises(ValueError, match="migration"):
            registry.list_goal_objective_migrations()



@pytest.mark.parametrize("column, value", (("migration_id", ""), ("recorded_at", "")))
def test_registry_migration_readback_rejects_malformed_scalar_json(
    tmp_path, monkeypatch, column, value
):
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, _Trainer(), _Evaluator())
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)
    migrate_legacy_protocol_contract(
        project,
        new_objective_version="project-protocol-v2",
        approval=ProtocolContractMigrationApproval(
            approval_id="approval-scalars",
            approver="operator@example.com",
            approved_at="2026-09-20T12:00:00Z",
            reason="Freeze current contract",
        ),
    )
    with RunRegistry(project.registry_path) as registry:
        registry._conn.execute(
            f"UPDATE goal_objective_migrations SET {column} = ?", (value,)
        )
        registry._conn.commit()
        with pytest.raises(ValueError, match="scalar"):
            registry.list_goal_objective_migrations()


@pytest.mark.parametrize("column", ("approval_json", "provenance_json", "recorded_at"))
def test_registry_migration_readback_rejects_valid_tampering(
    tmp_path, monkeypatch, column
):
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, _Trainer(), _Evaluator())
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)
    migrate_legacy_protocol_contract(
        project,
        new_objective_version="project-protocol-v2",
        approval=ProtocolContractMigrationApproval(
            approval_id="approval-tamper",
            approver="operator@example.com",
            approved_at="2026-09-20T12:00:00Z",
            reason="Freeze current contract",
        ),
    )
    with RunRegistry(project.registry_path) as registry:
        row = registry._conn.execute(
            f"SELECT {column} FROM goal_objective_migrations"
        ).fetchone()
        if column == "recorded_at":
            value = f"{row[0]}-tampered"
        else:
            payload = json.loads(row[0])
            if column == "approval_json":
                payload["reason"] = "different valid approval"
            else:
                payload["registry_path"] = "unrelated-registry.sqlite"
            value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        registry._conn.execute(
            f"UPDATE goal_objective_migrations SET {column} = ?",
            (value,),
        )
        registry._conn.commit()
        expected = "timestamp" if column == "recorded_at" else "migration"
        with pytest.raises(ValueError, match=expected):
            registry.list_goal_objective_migrations()


def test_migrated_protocol_contract_rejects_later_config_change(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    assert run_project(project).succeeded is True
    _remove_protocol_contract(project)
    migrated = migrate_legacy_protocol_contract(
        project,
        new_objective_version="project-protocol-v2",
        approval=ProtocolContractMigrationApproval(
            approval_id="approval-2",
            approver="operator@example.com",
            approved_at="2026-09-20T12:00:00Z",
            reason="Freeze current contract",
        ),
    )
    changed_config = deepcopy(migrated.config)
    changed_config["backend"]["precision"] = "bf16"
    changed = replace(migrated, config=changed_config)

    with pytest.raises(GoalLifecycleError, match="objective identity changed"):
        run_project(changed)


def test_project_runner_refuses_changed_goal_on_resume(tmp_path, monkeypatch):
    trainer = _Trainer()
    evaluator = _Evaluator()
    project = _project(tmp_path, baseline_quality=0.9)
    _patch_runner(monkeypatch, trainer, evaluator)
    first = run_project(project)
    assert first.succeeded is True

    changed_goal = Goal((MetricTarget("quality", minimum=0.95),), gpu_hour_budget=1.0)
    changed = replace(project, goal=changed_goal)
    with pytest.raises(GoalLifecycleError, match="objective identity changed"):
        run_project(changed)
    assert trainer.calls == 0
