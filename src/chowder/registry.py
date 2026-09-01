from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .executors import TrainingArtifact
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
CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT PRIMARY KEY,
    metrics_json TEXT NOT NULL,
    gpu_hours REAL NOT NULL,
    artifact_ref TEXT,
    evidence_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
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
