from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .executors import EvaluationOutcome, TrainingArtifact
from .failures import FailureRecord, FailureSourceRole, RepairPlan
from .models import Experiment, ExperimentResult
from .provenance import EvidenceManifest


SCHEMA = """
PRAGMA journal_mode=WAL;
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
"""


class RunRegistry:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunRegistry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def record_experiment(self, experiment: Experiment) -> None:
        self._conn.execute(
            """INSERT INTO experiments
               (experiment_id, parent_id, estimated_gpu_hours, hypothesis_json, config_json, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experiment.experiment_id,
                experiment.parent_id,
                experiment.estimated_gpu_hours,
                json.dumps(asdict(experiment.hypothesis), sort_keys=True),
                json.dumps(experiment.config_patch, sort_keys=True),
                experiment.status.value,
            ),
        )
        self._conn.commit()

    def record_training_artifact(self, artifact: TrainingArtifact) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO training_runs
               (run_id, experiment_id, artifact_ref, gpu_hours, telemetry_json, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                artifact.run_id,
                artifact.experiment_id,
                artifact.artifact_ref,
                artifact.gpu_hours,
                json.dumps(dict(artifact.telemetry), sort_keys=True),
                json.dumps(dict(artifact.evidence), sort_keys=True),
            ),
        )
        self._conn.commit()

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

    def record_evaluation_outcome(self, outcome: EvaluationOutcome) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO evaluation_runs
               (run_id, experiment_id, source_artifact_ref, metrics_json, gpu_hours, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                outcome.run_id,
                outcome.experiment_id,
                outcome.source_artifact_ref,
                json.dumps(dict(outcome.metrics), sort_keys=True),
                outcome.gpu_hours,
                json.dumps(dict(outcome.evidence), sort_keys=True),
            ),
        )
        self._conn.commit()

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

    def record_failure(self, failure: FailureRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO failure_records
               (failure_id, experiment_id, evaluation_run_id, evaluator, suite, row_index,
                protocol_sha256, artifact_sha256, source_role, prompt, expected, prediction,
                score, failure_kind, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
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
                json.dumps(dict(failure.metadata), sort_keys=True),
            ),
        )
        self._conn.commit()

    def record_failures(self, failures: Iterable[FailureRecord]) -> None:
        for failure in failures:
            self.record_failure(failure)

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
                failure_id,
                exp_id,
                evaluation_run_id,
                evaluator,
                suite,
                row_index,
                protocol_sha256,
                artifact_sha256,
                source_role,
                prompt,
                expected,
                prediction,
                score,
                failure_kind,
                metadata_json,
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
        self._conn.execute(
            """INSERT OR REPLACE INTO repair_plans
               (plan_id, cluster_id, observation, suspected_cause, intervention,
                source_failure_ids_json, direct_training_allowed, requires_independent_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan.plan_id,
                plan.cluster_id,
                plan.observation,
                plan.suspected_cause,
                plan.intervention,
                json.dumps(list(plan.source_failure_ids), sort_keys=True),
                int(plan.direct_training_allowed),
                int(plan.requires_independent_source),
            ),
        )
        self._conn.commit()

    def list_repair_plans(self) -> Iterable[RepairPlan]:
        rows = self._conn.execute(
            """SELECT plan_id, cluster_id, observation, suspected_cause, intervention,
                      source_failure_ids_json, direct_training_allowed, requires_independent_source
               FROM repair_plans ORDER BY rowid"""
        )
        for row in rows:
            (
                plan_id,
                cluster_id,
                observation,
                suspected_cause,
                intervention,
                source_failure_ids_json,
                direct_training_allowed,
                requires_independent_source,
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
        self._conn.execute(
            "UPDATE experiments SET status = ? WHERE experiment_id = ?",
            (status, experiment_id),
        )
        self._conn.commit()

    def record_result(self, result: ExperimentResult) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO results
               (experiment_id, metrics_json, gpu_hours, artifact_ref, evidence_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                result.experiment_id,
                json.dumps(dict(result.metrics), sort_keys=True),
                result.gpu_hours,
                result.artifact_ref,
                json.dumps(dict(result.evidence), sort_keys=True),
            ),
        )
        self._conn.commit()

    def record_manifest(self, manifest: EvidenceManifest) -> str:
        digest = manifest.digest()
        self._conn.execute(
            "INSERT OR REPLACE INTO manifests (experiment_id, digest, manifest_json) VALUES (?, ?, ?)",
            (manifest.experiment_id, digest, manifest.canonical_json()),
        )
        self._conn.commit()
        return digest

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
