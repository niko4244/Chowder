from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .combined_mechanism_experiment import CombinedMechanismExperiment
from .database import connect_database
from .executors import EvaluationOutcome, TrainingArtifact
from .goal_assessment import GoalAssessment
from .improvement.constitution import ObjectiveIdentity
from .failures import FailureRecord, FailureSourceRole, RepairPlan
from .models import Experiment, ExperimentResult, ExperimentStatus, Hypothesis
from .provenance import EvidenceManifest
from .run_events import RunEventPayload, event_experiment_id, event_payload, event_type_name


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    parent_id TEXT,
    estimated_gpu_hours REAL NOT NULL,
    hypothesis_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    gpu_hours REAL NOT NULL,
    telemetry_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    source_artifact_ref TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    gpu_hours REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT PRIMARY KEY,
    metrics_json TEXT NOT NULL,
    gpu_hours REAL NOT NULL,
    artifact_ref TEXT,
    evidence_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS failure_records (
    failure_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    evaluation_run_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    suite TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    source_role TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected TEXT NOT NULL,
    prediction TEXT NOT NULL,
    score REAL NOT NULL,
    failure_kind TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs(run_id)
);
CREATE TABLE IF NOT EXISTS repair_plans (
    plan_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    observation TEXT NOT NULL,
    suspected_cause TEXT NOT NULL,
    intervention TEXT NOT NULL,
    source_failure_ids_json TEXT NOT NULL,
    direct_training_allowed INTEGER NOT NULL,
    requires_independent_source INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
    experiment_id TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS combined_mechanism_experiments (
    experiment_key TEXT PRIMARY KEY,
    mechanisms_json TEXT NOT NULL,
    baseline_peak_vram_gb REAL NOT NULL,
    predicted_combined_peak_vram_gb REAL NOT NULL,
    actual_combined_peak_vram_gb REAL NOT NULL,
    prediction_error_gb REAL NOT NULL,
    baseline_wall_seconds REAL NOT NULL,
    combined_wall_seconds REAL NOT NULL,
    wall_time_penalty_ratio REAL NOT NULL,
    per_mechanism_predicted_savings_gb_json TEXT NOT NULL,
    telemetry_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teacher_signals (
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
);
"""

_EXPERIMENT_INSERT = """INSERT INTO experiments
   (experiment_id, parent_id, estimated_gpu_hours, hypothesis_json, config_json, status)
   VALUES (?, ?, ?, ?, ?, ?)"""


class RegistryInvariantError(ValueError):
    """Raised when durable scientific state would be overwritten or malformed."""


def _validate_recorded_at(value: object) -> str:
    """Require the canonical timezone-aware ``datetime.isoformat()`` form."""
    if not isinstance(value, str) or not value.strip():
        raise RegistryInvariantError("migration recorded_at must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RegistryInvariantError("migration recorded_at is not a canonical timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != value:
        raise RegistryInvariantError("migration recorded_at is not a canonical timestamp")
    return value


@dataclass(frozen=True)
class GoalObjectiveMigrationApproval:
    """Validated human approval attached to one objective migration."""

    approval_id: str
    approver: str
    approved_at: str
    reason: str

    def __post_init__(self) -> None:
        for name, value in (
            ("approval_id", self.approval_id),
            ("approver", self.approver),
            ("approved_at", self.approved_at),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RegistryInvariantError(f"migration approval {name} must be non-empty")

    @classmethod
    def from_dict(cls, value: object) -> "GoalObjectiveMigrationApproval":
        if not isinstance(value, dict) or set(value) != {
            "approval_id", "approver", "approved_at", "reason"
        }:
            raise RegistryInvariantError("migration approval fields are incomplete or unrelated")
        try:
            return cls(**value)
        except TypeError as exc:
            raise RegistryInvariantError("migration approval fields are malformed") from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "approval_id": self.approval_id,
            "approver": self.approver,
            "approved_at": self.approved_at,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GoalObjectiveMigrationProvenance:
    """Validated provenance binding one protocol-contract migration to its identities."""

    operation: str
    source_objective_version: str
    target_objective_version: str
    source_identity_digest: str
    target_identity_digest: str
    protocol_contract_digest: str
    registry_path: str

    @staticmethod
    def _require_digest(name: str, value: object) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise RegistryInvariantError(f"migration provenance {name} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise RegistryInvariantError(
                f"migration provenance {name} must be a SHA-256 digest"
            ) from exc
        return value

    def __post_init__(self) -> None:
        if self.operation != "legacy_protocol_contract_migration":
            raise RegistryInvariantError("migration provenance operation is invalid")
        for name, value in (
            ("source_objective_version", self.source_objective_version),
            ("target_objective_version", self.target_objective_version),
            ("registry_path", self.registry_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RegistryInvariantError(f"migration provenance {name} must be non-empty")
        if self.source_objective_version == self.target_objective_version:
            raise RegistryInvariantError("migration provenance source and target must differ")
        self._require_digest("source_identity_digest", self.source_identity_digest)
        self._require_digest("target_identity_digest", self.target_identity_digest)
        self._require_digest("protocol_contract_digest", self.protocol_contract_digest)

    @classmethod
    def from_dict(cls, value: object) -> "GoalObjectiveMigrationProvenance":
        if not isinstance(value, dict):
            raise RegistryInvariantError("migration provenance must be an object")
        required = {
            "operation",
            "source_objective_version",
            "target_objective_version",
            "source_identity_digest",
            "target_identity_digest",
            "protocol_contract_digest",
            "registry_path",
        }
        if set(value) != required:
            raise RegistryInvariantError("migration provenance fields are incomplete or unrelated")
        try:
            return cls(**value)
        except TypeError as exc:
            raise RegistryInvariantError("migration provenance fields are malformed") from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "operation": self.operation,
            "source_objective_version": self.source_objective_version,
            "target_objective_version": self.target_objective_version,
            "source_identity_digest": self.source_identity_digest,
            "target_identity_digest": self.target_identity_digest,
            "protocol_contract_digest": self.protocol_contract_digest,
            "registry_path": self.registry_path,
        }


@dataclass(frozen=True)
class GoalObjectiveMigration:
    """Validated, typed readback of one append-only objective migration."""

    migration_id: str
    source_objective_version: str
    target_objective_version: str
    source_identity: ObjectiveIdentity
    target_identity: ObjectiveIdentity
    protocol_contract_digest: str
    approval: GoalObjectiveMigrationApproval
    provenance: GoalObjectiveMigrationProvenance
    recorded_at: str
    migration_hash_version: int

    def __post_init__(self) -> None:
        for name, value in (
            ("migration_id", self.migration_id),
            ("source_objective_version", self.source_objective_version),
            ("target_objective_version", self.target_objective_version),
            ("protocol_contract_digest", self.protocol_contract_digest),
            ("recorded_at", self.recorded_at),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RegistryInvariantError(f"migration evidence {name} must be a non-empty string")
        _validate_recorded_at(self.recorded_at)
        if not isinstance(self.migration_hash_version, int) or isinstance(
            self.migration_hash_version, bool
        ) or self.migration_hash_version not in (1, 2):
            raise RegistryInvariantError("migration hash version is unsupported")
        if not isinstance(self.source_identity, ObjectiveIdentity) or not isinstance(
            self.target_identity, ObjectiveIdentity
        ):
            raise RegistryInvariantError("migration evidence identities are malformed")
        if not isinstance(self.approval, GoalObjectiveMigrationApproval) or not isinstance(
            self.provenance, GoalObjectiveMigrationProvenance
        ):
            raise RegistryInvariantError("migration evidence approval or provenance is malformed")
        if self.source_objective_version != self.source_identity.objective_version:
            raise RegistryInvariantError("migration source identity is inconsistent")
        if self.target_objective_version != self.target_identity.objective_version:
            raise RegistryInvariantError("migration target identity is inconsistent")
        if self.provenance.source_objective_version != self.source_objective_version:
            raise RegistryInvariantError("migration provenance source is inconsistent")
        if self.provenance.target_objective_version != self.target_objective_version:
            raise RegistryInvariantError("migration provenance target is inconsistent")
        if self.provenance.protocol_contract_digest != self.protocol_contract_digest:
            raise RegistryInvariantError("migration provenance contract is inconsistent")


class RunRegistry:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = connect_database(self.path)
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunRegistry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _insert_immutable(
        self,
        *,
        table: str,
        key_column: str,
        key: object,
        columns: Sequence[str],
        values: Sequence[object],
    ) -> None:
        """Insert once; identical replays are idempotent, divergent ones fail."""
        existing = self._conn.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE {key_column} = ?",
            (key,),
        ).fetchone()
        wanted = tuple(values)
        if existing is not None:
            if tuple(existing) == wanted:
                return
            raise RegistryInvariantError(
                f"immutable {table} record {key!r} already exists with different content"
            )
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            wanted,
        )

    @staticmethod
    def _experiment_row(experiment: Experiment) -> tuple[object, ...]:
        return (
            experiment.experiment_id,
            experiment.parent_id,
            experiment.estimated_gpu_hours,
            json.dumps(asdict(experiment.hypothesis), sort_keys=True),
            json.dumps(experiment.config_patch, sort_keys=True),
            experiment.status.value,
        )

    def _validate_experiment_batch(self, experiments: tuple[Experiment, ...]) -> None:
        existing_ids = {
            row[0] for row in self._conn.execute("SELECT experiment_id FROM experiments")
        }
        staged_ids = set(existing_ids)
        for experiment in experiments:
            experiment_id = experiment.experiment_id
            if experiment_id in staged_ids:
                raise RegistryInvariantError(
                    f"duplicate persisted experiment id: {experiment_id}"
                )
            parent_id = experiment.parent_id
            if parent_id is not None and parent_id not in staged_ids:
                raise RegistryInvariantError(f"unknown persisted parent: {parent_id}")
            staged_ids.add(experiment_id)

    def record_experiments(self, experiments: Iterable[Experiment]) -> None:
        rows = tuple(experiments)
        if not rows:
            return
        self._validate_experiment_batch(rows)
        values = tuple(self._experiment_row(experiment) for experiment in rows)
        with self._conn:
            self._conn.executemany(_EXPERIMENT_INSERT, values)

    def record_experiment(self, experiment: Experiment) -> None:
        self.record_experiments((experiment,))

    def has_experiment(self, experiment_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return row is not None

    def list_experiments(self) -> Iterable[Experiment]:
        """Read back every persisted Experiment, in the order they were
        recorded -- the join key candidate_selection.py needs to connect
        a historical result back to the config_patch that produced it
        (results/list_results() alone has no way to recover *what was
        tried*, only what it scored).

        Note: Experiment.tags is not part of the persisted schema (no
        tags_json column ever existed) and always round-trips as ()
        here -- a pre-existing limitation of this table, not something
        this method silently papers over.
        """
        rows = self._conn.execute(
            """SELECT experiment_id, parent_id, estimated_gpu_hours, hypothesis_json, config_json, status
               FROM experiments ORDER BY rowid"""
        )
        for experiment_id, parent_id, estimated_gpu_hours, hypothesis_json, config_json, status in rows:
            yield Experiment(
                experiment_id=experiment_id,
                parent_id=parent_id,
                hypothesis=Hypothesis(**json.loads(hypothesis_json)),
                config_patch=json.loads(config_json),
                estimated_gpu_hours=estimated_gpu_hours,
                status=ExperimentStatus(status),
            )

    def record_training_artifact(self, artifact: TrainingArtifact) -> None:
        columns = (
            "run_id", "experiment_id", "artifact_ref", "gpu_hours", "telemetry_json", "evidence_json"
        )
        values = (
            artifact.run_id,
            artifact.experiment_id,
            artifact.artifact_ref,
            artifact.gpu_hours,
            self._json(dict(artifact.telemetry)),
            self._json(dict(artifact.evidence)),
        )
        with self._conn:
            self._insert_immutable(
                table="training_runs", key_column="run_id", key=artifact.run_id,
                columns=columns, values=values,
            )

    def list_training_artifacts(self) -> Iterable[TrainingArtifact]:
        rows = self._conn.execute(
            """SELECT run_id, experiment_id, artifact_ref, gpu_hours, telemetry_json, evidence_json
               FROM training_runs ORDER BY rowid"""
        )
        for run_id, experiment_id, artifact_ref, gpu_hours, telemetry, evidence in rows:
            yield TrainingArtifact(
                run_id=run_id,
                experiment_id=experiment_id,
                artifact_ref=artifact_ref,
                gpu_hours=gpu_hours,
                telemetry=json.loads(telemetry),
                evidence=json.loads(evidence),
            )

    def record_combined_mechanism_experiment(self, experiment: CombinedMechanismExperiment) -> None:
        columns = (
            "experiment_key", "mechanisms_json", "baseline_peak_vram_gb",
            "predicted_combined_peak_vram_gb", "actual_combined_peak_vram_gb",
            "prediction_error_gb", "baseline_wall_seconds", "combined_wall_seconds",
            "wall_time_penalty_ratio", "per_mechanism_predicted_savings_gb_json", "telemetry_json",
        )
        values = (
            experiment.experiment_key,
            self._json(list(experiment.mechanisms)),
            experiment.baseline_peak_vram_gb,
            experiment.predicted_combined_peak_vram_gb,
            experiment.actual_combined_peak_vram_gb,
            experiment.prediction_error_gb,
            experiment.baseline_wall_seconds,
            experiment.combined_wall_seconds,
            experiment.wall_time_penalty_ratio,
            self._json(dict(experiment.per_mechanism_predicted_savings_gb)),
            self._json(
                {
                    "forward_seconds": experiment.forward_seconds,
                    "backward_seconds": experiment.backward_seconds,
                    "optimizer_seconds": experiment.optimizer_seconds,
                    "avg_gpu_utilization_percent": experiment.avg_gpu_utilization_percent,
                    "optimizer_state_bytes": experiment.optimizer_state_bytes,
                    "frozen_layer_streaming_bytes_transferred": (
                        experiment.frozen_layer_streaming_bytes_transferred
                    ),
                    "activation_offload_bytes_transferred": experiment.activation_offload_bytes_transferred,
                }
            ),
        )
        with self._conn:
            self._insert_immutable(
                table="combined_mechanism_experiments", key_column="experiment_key",
                key=experiment.experiment_key, columns=columns, values=values,
            )

    def list_combined_mechanism_experiments(self) -> Iterable[CombinedMechanismExperiment]:
        rows = self._conn.execute(
            """SELECT experiment_key, mechanisms_json, baseline_peak_vram_gb,
                      predicted_combined_peak_vram_gb, actual_combined_peak_vram_gb,
                      prediction_error_gb, baseline_wall_seconds, combined_wall_seconds,
                      wall_time_penalty_ratio, per_mechanism_predicted_savings_gb_json, telemetry_json
               FROM combined_mechanism_experiments ORDER BY rowid"""
        )
        for (
            experiment_key, mechanisms_json, baseline_peak_vram_gb,
            predicted_combined_peak_vram_gb, actual_combined_peak_vram_gb,
            prediction_error_gb, baseline_wall_seconds, combined_wall_seconds,
            wall_time_penalty_ratio, savings_json, telemetry_json,
        ) in rows:
            telemetry = json.loads(telemetry_json)
            yield CombinedMechanismExperiment(
                experiment_key=experiment_key,
                mechanisms=tuple(json.loads(mechanisms_json)),
                baseline_peak_vram_gb=baseline_peak_vram_gb,
                predicted_combined_peak_vram_gb=predicted_combined_peak_vram_gb,
                actual_combined_peak_vram_gb=actual_combined_peak_vram_gb,
                prediction_error_gb=prediction_error_gb,
                baseline_wall_seconds=baseline_wall_seconds,
                combined_wall_seconds=combined_wall_seconds,
                wall_time_penalty_ratio=wall_time_penalty_ratio,
                per_mechanism_predicted_savings_gb=json.loads(savings_json),
                forward_seconds=telemetry["forward_seconds"],
                backward_seconds=telemetry["backward_seconds"],
                optimizer_seconds=telemetry["optimizer_seconds"],
                avg_gpu_utilization_percent=telemetry["avg_gpu_utilization_percent"],
                optimizer_state_bytes=telemetry["optimizer_state_bytes"],
                frozen_layer_streaming_bytes_transferred=telemetry["frozen_layer_streaming_bytes_transferred"],
                activation_offload_bytes_transferred=telemetry["activation_offload_bytes_transferred"],
            )

    def record_evaluation_outcome(self, outcome: EvaluationOutcome) -> None:
        columns = (
            "run_id", "experiment_id", "source_artifact_ref", "metrics_json", "gpu_hours", "evidence_json"
        )
        values = (
            outcome.run_id,
            outcome.experiment_id,
            outcome.source_artifact_ref,
            self._json(dict(outcome.metrics)),
            outcome.gpu_hours,
            self._json(dict(outcome.evidence)),
        )
        with self._conn:
            self._insert_immutable(
                table="evaluation_runs", key_column="run_id", key=outcome.run_id,
                columns=columns, values=values,
            )

    def list_evaluation_outcomes(self) -> Iterable[EvaluationOutcome]:
        rows = self._conn.execute(
            """SELECT run_id, experiment_id, source_artifact_ref, metrics_json, gpu_hours, evidence_json
               FROM evaluation_runs ORDER BY rowid"""
        )
        for run_id, experiment_id, artifact_ref, metrics, gpu_hours, evidence in rows:
            yield EvaluationOutcome(
                run_id=run_id,
                experiment_id=experiment_id,
                source_artifact_ref=artifact_ref,
                metrics=json.loads(metrics),
                gpu_hours=gpu_hours,
                evidence=json.loads(evidence),
            )

    @staticmethod
    def _failure_row(failure: FailureRecord) -> tuple[object, ...]:
        return (
            failure.failure_id,
            failure.experiment_id,
            failure.evaluation_run_id,
            failure.evaluator,
            failure.suite,
            failure.row_index,
            failure.protocol_sha256,
            failure.artifact_sha256,
            failure.source_role.value,
            failure.prompt,
            failure.expected,
            failure.prediction,
            failure.score,
            failure.failure_kind,
            json.dumps(dict(failure.metadata), sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def record_failure(self, failure: FailureRecord) -> None:
        self.record_failures((failure,))

    def record_failures(self, failures: Iterable[FailureRecord]) -> None:
        rows = tuple(failures)
        if not rows:
            return
        columns = (
            "failure_id", "experiment_id", "evaluation_run_id", "evaluator", "suite", "row_index",
            "protocol_sha256", "artifact_sha256", "source_role", "prompt", "expected", "prediction",
            "score", "failure_kind", "metadata_json",
        )
        with self._conn:
            for failure in rows:
                values = self._failure_row(failure)
                self._insert_immutable(
                    table="failure_records", key_column="failure_id", key=failure.failure_id,
                    columns=columns, values=values,
                )

    def list_failures(self, *, experiment_id: str | None = None) -> Iterable[FailureRecord]:
        if experiment_id is None:
            rows = self._conn.execute(
                """SELECT failure_id, experiment_id, evaluation_run_id, evaluator, suite, row_index,
                          protocol_sha256, artifact_sha256, source_role, prompt, expected, prediction,
                          score, failure_kind, metadata_json
                   FROM failure_records ORDER BY rowid"""
            )
        else:
            rows = self._conn.execute(
                """SELECT failure_id, experiment_id, evaluation_run_id, evaluator, suite, row_index,
                          protocol_sha256, artifact_sha256, source_role, prompt, expected, prediction,
                          score, failure_kind, metadata_json
                   FROM failure_records WHERE experiment_id = ? ORDER BY rowid""",
                (experiment_id,),
            )
        for row in rows:
            (
                failure_id, exp_id, evaluation_run_id, evaluator, suite, row_index,
                protocol_sha256, artifact_sha256, source_role, prompt, expected,
                prediction, score, failure_kind, metadata_json,
            ) = row
            yield FailureRecord(
                failure_id=failure_id,
                experiment_id=exp_id,
                evaluation_run_id=evaluation_run_id,
                evaluator=evaluator,
                suite=suite,
                row_index=row_index,
                protocol_sha256=protocol_sha256,
                artifact_sha256=artifact_sha256,
                source_role=FailureSourceRole(source_role),
                prompt=prompt,
                expected=expected,
                prediction=prediction,
                score=score,
                failure_kind=failure_kind,
                metadata=json.loads(metadata_json),
            )

    def record_repair_plan(self, plan: RepairPlan) -> None:
        columns = (
            "plan_id", "cluster_id", "observation", "suspected_cause", "intervention",
            "source_failure_ids_json", "direct_training_allowed", "requires_independent_source",
        )
        values = (
            plan.plan_id,
            plan.cluster_id,
            plan.observation,
            plan.suspected_cause,
            plan.intervention,
            self._json(list(plan.source_failure_ids)),
            int(plan.direct_training_allowed),
            int(plan.requires_independent_source),
        )
        with self._conn:
            self._insert_immutable(
                table="repair_plans", key_column="plan_id", key=plan.plan_id,
                columns=columns, values=values,
            )

    def list_repair_plans(self) -> Iterable[RepairPlan]:
        rows = self._conn.execute(
            """SELECT plan_id, cluster_id, observation, suspected_cause, intervention,
                      source_failure_ids_json, direct_training_allowed, requires_independent_source
               FROM repair_plans ORDER BY rowid"""
        )
        for row in rows:
            (
                plan_id, cluster_id, observation, suspected_cause, intervention,
                source_failure_ids_json, direct_training_allowed, requires_independent_source,
            ) = row
            yield RepairPlan(
                plan_id=plan_id,
                cluster_id=cluster_id,
                observation=observation,
                suspected_cause=suspected_cause,
                intervention=intervention,
                source_failure_ids=tuple(json.loads(source_failure_ids_json)),
                direct_training_allowed=bool(direct_training_allowed),
                requires_independent_source=bool(requires_independent_source),
            )

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                (status, experiment_id),
            )
            if cursor.rowcount != 1:
                raise RegistryInvariantError(f"unknown persisted experiment id: {experiment_id}")

    def record_result(self, result: ExperimentResult) -> None:
        columns = ("experiment_id", "metrics_json", "gpu_hours", "artifact_ref", "evidence_json")
        values = (
            result.experiment_id,
            self._json(dict(result.metrics)),
            result.gpu_hours,
            result.artifact_ref,
            self._json(dict(result.evidence)),
        )
        with self._conn:
            self._insert_immutable(
                table="results", key_column="experiment_id", key=result.experiment_id,
                columns=columns, values=values,
            )

    def record_manifest(self, manifest: EvidenceManifest) -> str:
        digest = manifest.digest()
        columns = ("experiment_id", "digest", "manifest_json")
        values = (manifest.experiment_id, digest, manifest.canonical_json())
        with self._conn:
            self._insert_immutable(
                table="manifests", key_column="experiment_id", key=manifest.experiment_id,
                columns=columns, values=values,
            )
        return digest

    @staticmethod
    def _identity_payload(identity: ObjectiveIdentity) -> dict[str, str]:
        return {
            "objective_version": identity.objective_version,
            "goal_digest": identity.goal_digest,
            "benchmark_digest": identity.benchmark_digest,
            "evaluation_protocol_digest": identity.evaluation_protocol_digest,
            "constitution_digest": identity.constitution_digest,
        }

    @classmethod
    def _identity_digest(cls, identity_payload: object) -> str:
        if not isinstance(identity_payload, dict):
            raise RegistryInvariantError("persisted migration identity is invalid")
        return hashlib.sha256(cls._json(identity_payload).encode("utf-8")).hexdigest()

    @classmethod
    def _migration_id(
        cls,
        *,
        source_objective_version: str,
        target_identity_payload: dict[str, str],
        protocol_contract_digest: str,
        approval: GoalObjectiveMigrationApproval,
        provenance: GoalObjectiveMigrationProvenance,
        recorded_at: str | None,
        migration_hash_version: int,
    ) -> str:
        if migration_hash_version not in (1, 2):
            raise RegistryInvariantError("migration hash version is unsupported")
        payload = {
            "source": source_objective_version,
            "target": target_identity_payload,
            "contract": protocol_contract_digest,
            "approval": approval.to_dict(),
            "provenance": provenance.to_dict(),
        }
        if migration_hash_version == 2:
            _validate_recorded_at(recorded_at)
            payload["hash_version"] = 2
            payload["recorded_at"] = recorded_at
        return hashlib.sha256(cls._json(payload).encode("utf-8")).hexdigest()

    def record_goal_objective(
        self, identity: ObjectiveIdentity, goal_payload: dict[str, object]
    ) -> None:
        """Persist one frozen objective identity; divergent replays fail closed."""
        identity_json = self._json(self._identity_payload(identity))
        goal_json = self._json(goal_payload)
        columns = ("objective_version", "identity_json", "goal_json", "created_at")
        values = (
            identity.objective_version,
            identity_json,
            goal_json,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._conn:
            existing = self._conn.execute(
                "SELECT identity_json, goal_json FROM goal_objectives WHERE objective_version = ?",
                (identity.objective_version,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != (identity_json, goal_json):
                    raise RegistryInvariantError(
                        f"immutable goal objective {identity.objective_version!r} changed"
                    )
                return
            self._conn.execute(
                "INSERT INTO goal_objectives "
                "(objective_version, identity_json, goal_json, created_at) VALUES (?, ?, ?, ?)",
                values,
            )

    def get_goal_objective(self, objective_version: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT identity_json, goal_json FROM goal_objectives WHERE objective_version = ?",
            (objective_version,),
        ).fetchone()
        if row is None:
            return None
        return {"identity": json.loads(row[0]), "goal": json.loads(row[1])}

    def record_goal_objective_migration(
        self,
        *,
        source_objective_version: str,
        target_identity: ObjectiveIdentity,
        target_goal_payload: dict[str, object],
        protocol_contract_digest: str,
        approval: GoalObjectiveMigrationApproval | Mapping[str, str],
        provenance: GoalObjectiveMigrationProvenance | Mapping[str, object],
    ) -> str:
        """Atomically append a legacy-objective migration and its new objective."""
        source = self.get_goal_objective(source_objective_version)
        if source is None:
            raise RegistryInvariantError(
                f"cannot migrate unknown goal objective: {source_objective_version}"
            )
        if "protocol_contract_digest" in source["goal"]:
            raise RegistryInvariantError(
                "goal objective already has a protocol contract; migration is not applicable"
            )
        if target_identity.objective_version == source_objective_version:
            raise RegistryInvariantError("protocol migration requires a new objective version")
        if not protocol_contract_digest or len(protocol_contract_digest) != 64:
            raise RegistryInvariantError("protocol contract digest must be a SHA-256 digest")
        try:
            int(protocol_contract_digest, 16)
        except ValueError as exc:
            raise RegistryInvariantError("protocol contract digest must be a SHA-256 digest") from exc
        if (
            not isinstance(target_goal_payload, dict)
            or target_goal_payload.get("protocol_contract_digest") != protocol_contract_digest
        ):
            raise RegistryInvariantError(
                "target objective protocol-contract evidence is missing or inconsistent"
            )
        if isinstance(approval, GoalObjectiveMigrationApproval):
            validated_approval = approval
        else:
            validated_approval = GoalObjectiveMigrationApproval.from_dict(approval)
        if isinstance(provenance, GoalObjectiveMigrationProvenance):
            validated_provenance = provenance
        else:
            validated_provenance = GoalObjectiveMigrationProvenance.from_dict(provenance)
        source_identity = source["identity"]
        target_identity_payload = self._identity_payload(target_identity)
        if validated_provenance.source_objective_version != source_objective_version:
            raise RegistryInvariantError("migration provenance source objective is unrelated")
        if validated_provenance.target_objective_version != target_identity.objective_version:
            raise RegistryInvariantError("migration provenance target objective is inconsistent")
        if validated_provenance.protocol_contract_digest != protocol_contract_digest:
            raise RegistryInvariantError("migration provenance contract digest is inconsistent")
        if validated_provenance.registry_path != self.path:
            raise RegistryInvariantError("migration provenance registry is unrelated")
        if validated_provenance.source_identity_digest != self._identity_digest(source_identity):
            raise RegistryInvariantError("migration provenance source identity is inconsistent")
        if validated_provenance.target_identity_digest != self._identity_digest(target_identity_payload):
            raise RegistryInvariantError("migration provenance target identity is inconsistent")
        provenance_payload = validated_provenance.to_dict()
        recorded_at = datetime.now(timezone.utc).isoformat()
        migration_hash_version = 2
        migration_id = self._migration_id(
            source_objective_version=source_objective_version,
            target_identity_payload=target_identity_payload,
            protocol_contract_digest=protocol_contract_digest,
            approval=validated_approval,
            provenance=validated_provenance,
            recorded_at=recorded_at,
            migration_hash_version=migration_hash_version,
        )
        identity_json = self._json(self._identity_payload(target_identity))
        goal_json = self._json(target_goal_payload)
        with self._conn:
            existing = self._conn.execute(
                "SELECT 1 FROM goal_objectives WHERE objective_version = ?",
                (target_identity.objective_version,),
            ).fetchone()
            if existing is not None:
                raise RegistryInvariantError(
                    f"target objective version already exists: {target_identity.objective_version}"
                )
            self._conn.execute(
                "INSERT INTO goal_objectives "
                "(objective_version, identity_json, goal_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    target_identity.objective_version,
                    identity_json,
                    goal_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.execute(
                "INSERT INTO goal_objective_migrations "
                "(migration_id, source_objective_version, target_objective_version, "
                "source_identity_json, target_identity_json, protocol_contract_digest, "
                "approval_json, provenance_json, recorded_at, migration_hash_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    migration_id,
                    source_objective_version,
                    target_identity.objective_version,
                    self._json(source["identity"]),
                    identity_json,
                    protocol_contract_digest,
                    self._json(validated_approval.to_dict()),
                    self._json(provenance_payload),
                    recorded_at,
                    migration_hash_version,
                ),
            )
        return migration_id

    def list_goal_objective_migrations(self) -> tuple[GoalObjectiveMigration, ...]:
        rows = self._conn.execute(
            "SELECT migration_id, source_objective_version, target_objective_version, "
            "source_identity_json, target_identity_json, protocol_contract_digest, "
            "approval_json, provenance_json, recorded_at, migration_hash_version "
            "FROM goal_objective_migrations ORDER BY rowid"
        )
        migrations: list[GoalObjectiveMigration] = []
        for row in rows:
            try:
                (
                    migration_id,
                    source_version,
                    target_version,
                    source_identity_json,
                    target_identity_json,
                    contract_digest,
                    approval_json,
                    provenance_json,
                    recorded_at,
                    migration_hash_version,
                ) = row
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (
                        migration_id,
                        source_version,
                        target_version,
                        contract_digest,
                        recorded_at,
                        source_identity_json,
                        target_identity_json,
                        approval_json,
                        provenance_json,
                    )
                ) or not isinstance(migration_hash_version, int) or isinstance(
                    migration_hash_version, bool
                ):
                    raise RegistryInvariantError("persisted migration scalar field is malformed")
                source_identity = ObjectiveIdentity(**json.loads(source_identity_json))
                target_identity = ObjectiveIdentity(**json.loads(target_identity_json))
                approval = GoalObjectiveMigrationApproval.from_dict(json.loads(approval_json))
                provenance = GoalObjectiveMigrationProvenance.from_dict(json.loads(provenance_json))
                source = self.get_goal_objective(source_version)
                target = self.get_goal_objective(target_version)
                if source is None or target is None:
                    raise RegistryInvariantError("migration references an unknown objective")
                if source["identity"] != self._identity_payload(source_identity):
                    raise RegistryInvariantError("migration source identity does not match objective")
                if target["identity"] != self._identity_payload(target_identity):
                    raise RegistryInvariantError("migration target identity does not match objective")
                if contract_digest != provenance.protocol_contract_digest:
                    raise RegistryInvariantError("migration protocol contract is inconsistent")
                target_goal = target["goal"]
                if not isinstance(target_goal, dict) or target_goal.get("protocol_contract_digest") != contract_digest:
                    raise RegistryInvariantError("migration target contract evidence is inconsistent")
                expected_migration_id = self._migration_id(
                    source_objective_version=source_version,
                    target_identity_payload=self._identity_payload(target_identity),
                    protocol_contract_digest=contract_digest,
                    approval=approval,
                    provenance=provenance,
                    recorded_at=recorded_at,
                    migration_hash_version=migration_hash_version,
                )
                if migration_id != expected_migration_id:
                    raise RegistryInvariantError("migration content hash is inconsistent")
                migration = GoalObjectiveMigration(
                    migration_id=migration_id,
                    source_objective_version=source_version,
                    target_objective_version=target_version,
                    source_identity=source_identity,
                    target_identity=target_identity,
                    protocol_contract_digest=contract_digest,
                    approval=approval,
                    provenance=provenance,
                    recorded_at=recorded_at,
                    migration_hash_version=migration_hash_version,
                )
                if provenance.registry_path != self.path:
                    raise RegistryInvariantError("migration provenance registry is unrelated")
                if provenance.source_identity_digest != self._identity_digest(source["identity"]):
                    raise RegistryInvariantError("migration source provenance is inconsistent")
                if provenance.target_identity_digest != self._identity_digest(target["identity"]):
                    raise RegistryInvariantError("migration target provenance is inconsistent")
                migrations.append(migration)
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if isinstance(exc, RegistryInvariantError):
                    raise
                raise RegistryInvariantError("persisted migration evidence is malformed") from exc
        return tuple(migrations)

    def record_goal_assessment(
        self, objective_version: str, assessment: GoalAssessment
    ) -> str:
        """Append an immutable assessment and return its content key."""
        if assessment.goal_version != objective_version:
            raise RegistryInvariantError("assessment objective version does not match lifecycle")
        objective_row = self._conn.execute(
            "SELECT identity_json FROM goal_objectives WHERE objective_version = ?",
            (objective_version,),
        ).fetchone()
        if objective_row is None:
            raise RegistryInvariantError(
                f"cannot persist assessment for unknown goal objective: {objective_version}"
            )
        try:
            stored_identity = ObjectiveIdentity(**json.loads(objective_row[0]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RegistryInvariantError("persisted goal objective identity is invalid") from exc
        identity_fields = (
            ("goal", stored_identity.goal_digest, assessment.goal_digest),
            (
                "evaluation protocol",
                stored_identity.evaluation_protocol_digest,
                assessment.evaluation_protocol_digest,
            ),
            ("benchmark", stored_identity.benchmark_digest, assessment.benchmark_digest),
            (
                "constitution",
                stored_identity.constitution_digest,
                assessment.constitution_digest,
            ),
        )
        for label, expected, actual in identity_fields:
            if actual != expected:
                raise RegistryInvariantError(
                    f"assessment {label} digest does not match frozen objective"
                )
        payload = assessment.to_dict()
        assessment_json = self._json(payload)
        assessment_id = hashlib.sha256(assessment_json.encode("utf-8")).hexdigest()
        columns = (
            "assessment_id",
            "objective_version",
            "artifact_identity",
            "status",
            "assessment_json",
            "recorded_at",
        )
        values = (
            assessment_id,
            objective_version,
            assessment.artifact_identity,
            assessment.status.value,
            assessment_json,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._conn:
            self._insert_immutable(
                table="goal_assessments",
                key_column="assessment_id",
                key=assessment_id,
                columns=columns,
                values=values,
            )
        return assessment_id

    def list_goal_assessments(self, objective_version: str) -> tuple[GoalAssessment, ...]:
        rows = self._conn.execute(
            "SELECT assessment_json FROM goal_assessments "
            "WHERE objective_version = ? ORDER BY rowid",
            (objective_version,),
        )
        return tuple(GoalAssessment.from_dict(json.loads(row[0])) for row in rows)

    def latest_goal_assessment(self, objective_version: str) -> GoalAssessment | None:
        row = self._conn.execute(
            "SELECT assessment_json FROM goal_assessments "
            "WHERE objective_version = ? ORDER BY rowid DESC LIMIT 1",
            (objective_version,),
        ).fetchone()
        return None if row is None else GoalAssessment.from_dict(json.loads(row[0]))

    def record_goal_terminal(
        self, objective_version: str, terminal_state: object, artifact_identity: str
    ) -> None:
        state = getattr(terminal_state, "value", terminal_state)
        with self._conn:
            self._conn.execute(
                "INSERT INTO goal_terminal_events "
                "(objective_version, terminal_state, artifact_identity, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    objective_version,
                    str(state),
                    artifact_identity,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_goal_terminals(self, objective_version: str) -> tuple[dict[str, object], ...]:
        rows = self._conn.execute(
            "SELECT terminal_state, artifact_identity, recorded_at "
            "FROM goal_terminal_events WHERE objective_version = ? ORDER BY event_id",
            (objective_version,),
        )
        return tuple(
            {
                "terminal_state": row[0],
                "artifact_identity": row[1],
                "recorded_at": row[2],
            }
            for row in rows
        )

    def latest_goal_terminal(self, objective_version: str) -> str | None:
        row = self._conn.execute(
            "SELECT terminal_state FROM goal_terminal_events "
            "WHERE objective_version = ? ORDER BY event_id DESC LIMIT 1",
            (objective_version,),
        ).fetchone()
        return None if row is None else str(row[0])

    def record_execution_incident(self, analysis) -> None:
        """Persist structured executor crash evidence from Executor Investigator."""
        capture = analysis.capture
        fingerprint = analysis.fingerprint
        routed = analysis.routed
        if hasattr(routed, "investigation_id"):
            route_summary = {
                "type": "investigation",
                "id": routed.investigation_id,
                "status": routed.status.value,
            }
        else:
            route_summary = {
                "type": "known_remediation",
                "id": routed.remediation_id,
                "outcome": routed.outcome.value,
            }
        columns = (
            "incident_id", "experiment_id", "run_id", "executor_name", "fingerprint_sha256",
            "signature_kind", "gpu_hours_spent", "capture_json", "analysis_json",
        )
        values = (
            capture.incident_id,
            capture.experiment_id,
            capture.run_id or "unknown",
            capture.executor_name,
            fingerprint.fingerprint_sha256,
            fingerprint.signature_kind.value,
            capture.gpu_hours_spent,
            self._json(asdict(capture)),
            self._json(route_summary),
        )
        with self._conn:
            self._insert_immutable(
                table="execution_incidents", key_column="incident_id", key=capture.incident_id,
                columns=columns, values=values,
            )

    def list_execution_incidents(self) -> Iterable[dict[str, object]]:
        rows = self._conn.execute(
            """SELECT incident_id, experiment_id, run_id, executor_name,
                      fingerprint_sha256, signature_kind, gpu_hours_spent,
                      capture_json, analysis_json
               FROM execution_incidents ORDER BY rowid"""
        )
        for row in rows:
            yield {
                "incident_id": row[0],
                "experiment_id": row[1],
                "run_id": row[2],
                "executor_name": row[3],
                "fingerprint_sha256": row[4],
                "signature_kind": row[5],
                "gpu_hours_spent": row[6],
                "capture": json.loads(row[7]),
                "analysis": json.loads(row[8]),
            }

    def record_teacher_signal(
        self,
        *,
        entry_key: str,
        artifact_digest: str,
        request_digest: str,
        payload_file_sha256: str,
        signal_kind: str,
        teacher_id: str,
        model_revision: str,
        tokenizer_identity_sha256: str | None,
        signal_id: str,
        stored_at: str,
        metadata_json: str,
    ) -> None:
        """Append one teacher-signal ledger row (Teacher Fabric Slice B).

        Immutably keyed by the store content address: an identical replay
        is idempotent, a divergent one is a RegistryInvariantError -- the
        same discipline as every other evidence table.
        """
        columns = (
            "entry_key",
            "artifact_digest",
            "request_digest",
            "payload_file_sha256",
            "signal_kind",
            "teacher_id",
            "model_revision",
            "tokenizer_identity_sha256",
            "signal_id",
            "stored_at",
            "metadata_json",
        )
        values = (
            entry_key,
            artifact_digest,
            request_digest,
            payload_file_sha256,
            signal_kind,
            teacher_id,
            model_revision,
            tokenizer_identity_sha256,
            signal_id,
            stored_at,
            metadata_json,
        )
        with self._conn:
            self._insert_immutable(
                table="teacher_signals", key_column="entry_key", key=entry_key,
                columns=columns, values=values,
            )

    def list_teacher_signals(self) -> Iterable[dict[str, object]]:
        rows = self._conn.execute(
            """SELECT entry_key, artifact_digest, request_digest, payload_file_sha256,
                      signal_kind, teacher_id, model_revision, tokenizer_identity_sha256,
                      signal_id, stored_at, metadata_json
               FROM teacher_signals ORDER BY rowid"""
        )
        for row in rows:
            yield {
                "entry_key": row[0],
                "artifact_digest": row[1],
                "request_digest": row[2],
                "payload_file_sha256": row[3],
                "signal_kind": row[4],
                "teacher_id": row[5],
                "model_revision": row[6],
                "tokenizer_identity_sha256": row[7],
                "signal_id": row[8],
                "stored_at": row[9],
                "metadata": json.loads(row[10]),
            }

    def teacher_signal_row(self, entry_key: str) -> dict[str, object] | None:
        """The ledger row for *entry_key*, or None when never recorded."""
        row = self._conn.execute(
            """SELECT entry_key, artifact_digest, request_digest, payload_file_sha256,
                      signal_kind, teacher_id, model_revision, tokenizer_identity_sha256,
                      signal_id, stored_at, metadata_json
               FROM teacher_signals WHERE entry_key = ?""",
            (entry_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "entry_key": row[0],
            "artifact_digest": row[1],
            "request_digest": row[2],
            "payload_file_sha256": row[3],
            "signal_kind": row[4],
            "teacher_id": row[5],
            "model_revision": row[6],
            "tokenizer_identity_sha256": row[7],
            "signal_id": row[8],
            "stored_at": row[9],
            "metadata": json.loads(row[10]),
        }

    def lineage(self, experiment_id: str) -> tuple[str, ...]:
        lineage: list[str] = []
        current = experiment_id
        seen: set[str] = set()
        while current:
            if current in seen:
                raise ValueError("cycle detected in persisted lineage")
            seen.add(current)
            row = self._conn.execute(
                "SELECT parent_id FROM experiments WHERE experiment_id = ?", (current,)
            ).fetchone()
            if row is None:
                break
            parent = row[0]
            if parent is None:
                break
            lineage.append(parent)
            current = parent
        return tuple(lineage)

    def list_results(self) -> Iterable[ExperimentResult]:
        rows = self._conn.execute(
            "SELECT experiment_id, metrics_json, gpu_hours, artifact_ref, evidence_json FROM results ORDER BY rowid"
        )
        for experiment_id, metrics, gpu_hours, artifact_ref, evidence in rows:
            yield ExperimentResult(
                experiment_id=experiment_id,
                metrics=json.loads(metrics),
                gpu_hours=gpu_hours,
                artifact_ref=artifact_ref,
                evidence=json.loads(evidence),
            )

    def audit_stranded_results(self) -> list[dict[str, object]]:
        """Flag results stranded on a non-terminal experiment row.

        A result row whose experiment still reports ``planned`` or ``running``
        is a durable-evidence disagreement: the row carries a measured score
        while its status claims the experiment has not run (or has not
        finished). This is the audit for the class of defect the automatic-
        baseline settlement fixed for one writer — it keeps the class from
        recurring silently through any other writer.

        An orphan result (no experiment row at all) cannot exist here:
        ``results.experiment_id`` carries a foreign key into ``experiments``,
        so the schema refuses it at insert time — the audit only has to watch
        statuses.
        """
        terminal = {"passed", "failed", "rejected"}
        status_by_id = {
            row[0]: row[1]
            for row in self._conn.execute("SELECT experiment_id, status FROM experiments")
        }
        findings: list[dict[str, object]] = []
        for experiment_id, _metrics, gpu_hours, artifact_ref, _evidence in self._conn.execute(
            "SELECT experiment_id, metrics_json, gpu_hours, artifact_ref, evidence_json "
            "FROM results ORDER BY rowid"
        ):
            status = status_by_id.get(experiment_id)
            if status not in terminal:
                findings.append(
                    {
                        "experiment_id": experiment_id,
                        "status": status,
                        "gpu_hours": gpu_hours,
                        "artifact_ref": artifact_ref,
                    }
                )
        return findings

    def record_event(self, event: RunEventPayload) -> None:
        """Append one structured run event to the durable history.

        Unlike the other record_* methods, this is a plain append-only log,
        not an immutable single-record-per-key table -- many events share
        the same experiment_id, ordered by insertion, and there is nothing
        to compare a replay against.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO run_events (experiment_id, event_type, occurred_at, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    event_experiment_id(event),
                    event_type_name(event),
                    datetime.now(timezone.utc).isoformat(),
                    self._json(event_payload(event)),
                ),
            )

    def list_events(self, *, experiment_id: str | None = None) -> Iterable[RecordedEvent]:
        if experiment_id is None:
            rows = self._conn.execute(
                "SELECT event_id, experiment_id, event_type, occurred_at, payload_json "
                "FROM run_events ORDER BY event_id"
            )
        else:
            rows = self._conn.execute(
                "SELECT event_id, experiment_id, event_type, occurred_at, payload_json "
                "FROM run_events WHERE experiment_id = ? ORDER BY event_id",
                (experiment_id,),
            )
        for event_id, exp_id, event_type, occurred_at, payload_json in rows:
            yield RecordedEvent(
                event_id=event_id,
                experiment_id=exp_id,
                event_type=event_type,
                occurred_at=occurred_at,
                payload=json.loads(payload_json),
            )


@dataclass(frozen=True)
class RecordedEvent:
    """One durably-persisted run event, as read back from the registry.
    `payload` is the plain dict a RunEventPayload dataclass serialized to
    (see run_events.event_payload) -- reconstructing the exact original
    dataclass is the caller's job if it needs one, keyed on `event_type`.
    """

    event_id: int
    experiment_id: str | None
    event_type: str
    occurred_at: str
    payload: dict[str, Any]
