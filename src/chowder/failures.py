from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .executors import EvaluationOutcome


class FailureSourceRole(str, Enum):
    """Controls what Chowder may do with a failed example.

    Gate/holdout examples are diagnostic evidence only. They must never be
    copied into training data, otherwise Chowder contaminates its own benchmark.
    """

    GATE_HOLDOUT = "gate_holdout"
    DEVELOPMENT = "development"
    REPAIR_SOURCE = "repair_source"

    @property
    def training_eligible(self) -> bool:
        return self is FailureSourceRole.REPAIR_SOURCE


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    experiment_id: str
    evaluation_run_id: str
    evaluator: str
    suite: str
    row_index: int
    protocol_sha256: str
    artifact_sha256: str
    source_role: FailureSourceRole
    prompt: str
    expected: str
    prediction: str
    score: float
    failure_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def training_eligible(self) -> bool:
        return self.source_role.training_eligible


@dataclass(frozen=True)
class FailureCluster:
    cluster_id: str
    evaluator: str
    suite: str
    protocol_sha256: str
    source_role: FailureSourceRole
    failure_kind: str
    failure_ids: tuple[str, ...]


@dataclass(frozen=True)
class RepairPlan:
    plan_id: str
    cluster_id: str
    observation: str
    suspected_cause: str
    intervention: str
    source_failure_ids: tuple[str, ...]
    direct_training_allowed: bool
    requires_independent_source: bool


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _classify_failure(prediction: str, expected: str) -> str:
    if not prediction.strip():
        return "empty_prediction"
    if prediction.strip().casefold() in {"i don't know", "i do not know", "unknown"}:
        return "refusal_or_unknown"
    if len(prediction) > max(64, len(expected) * 4):
        return "overlong_mismatch"
    return "answer_mismatch"


def _failure_id(
    *,
    experiment_id: str,
    evaluation_run_id: str,
    suite: str,
    row_index: int,
    protocol_sha256: str,
    prompt: str,
    expected: str,
    prediction: str,
) -> str:
    return _canonical_digest(
        {
            "experiment_id": experiment_id,
            "evaluation_run_id": evaluation_run_id,
            "suite": suite,
            "row_index": row_index,
            "protocol_sha256": protocol_sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "expected_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
        }
    )


def harvest_transformers_text_failures(outcome: EvaluationOutcome) -> tuple[FailureRecord, ...]:
    """Harvest failed rows from the auditable transformers-text evaluator.

    The evaluator writes per-row prediction JSONL files. This function records
    only failed rows and preserves the protocol/artifact hashes needed to trace
    every repair hypothesis back to the exact evaluation evidence.
    """

    evidence = outcome.evidence
    if evidence.get("evaluator") != "transformers-text":
        raise ValueError("outcome is not from the transformers-text evaluator")
    protocol_sha = evidence.get("protocol_sha256")
    artifact_sha = evidence.get("artifact_sha256")
    suites = evidence.get("suite_evidence")
    if not isinstance(protocol_sha, str) or len(protocol_sha) != 64:
        raise ValueError("evaluation outcome is missing protocol_sha256")
    if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
        raise ValueError("evaluation outcome is missing artifact_sha256")
    if not isinstance(suites, Mapping):
        raise ValueError("evaluation outcome is missing suite_evidence")

    records: list[FailureRecord] = []
    for suite_name, suite_evidence in suites.items():
        if not isinstance(suite_evidence, Mapping):
            raise ValueError(f"suite evidence for {suite_name!r} is invalid")
        predictions_file = suite_evidence.get("predictions_file")
        if not isinstance(predictions_file, str):
            raise ValueError(f"suite {suite_name!r} is missing predictions_file")
        role_raw = str(suite_evidence.get("source_role", FailureSourceRole.GATE_HOLDOUT.value))
        try:
            source_role = FailureSourceRole(role_raw)
        except ValueError as exc:
            raise ValueError(f"unknown failure source role: {role_raw}") from exc

        path = Path(predictions_file)
        if not path.is_file():
            raise FileNotFoundError(f"prediction evidence not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"prediction row {row_index} is not an object")
                score = float(row.get("score", 0.0))
                if score >= 1.0:
                    continue
                prompt = str(row.get("prompt", ""))
                expected = str(row.get("expected", ""))
                prediction = str(row.get("prediction", ""))
                failure_kind = _classify_failure(prediction, expected)
                failure_id = _failure_id(
                    experiment_id=outcome.experiment_id,
                    evaluation_run_id=outcome.run_id,
                    suite=str(suite_name),
                    row_index=row_index,
                    protocol_sha256=protocol_sha,
                    prompt=prompt,
                    expected=expected,
                    prediction=prediction,
                )
                records.append(
                    FailureRecord(
                        failure_id=failure_id,
                        experiment_id=outcome.experiment_id,
                        evaluation_run_id=outcome.run_id,
                        evaluator="transformers-text",
                        suite=str(suite_name),
                        row_index=row_index,
                        protocol_sha256=protocol_sha,
                        artifact_sha256=artifact_sha,
                        source_role=source_role,
                        prompt=prompt,
                        expected=expected,
                        prediction=prediction,
                        score=score,
                        failure_kind=failure_kind,
                    )
                )
    return tuple(records)


def cluster_failures(failures: Iterable[FailureRecord]) -> tuple[FailureCluster, ...]:
    buckets: dict[tuple[str, str, str, FailureSourceRole, str], list[str]] = {}
    for failure in failures:
        key = (
            failure.evaluator,
            failure.suite,
            failure.protocol_sha256,
            failure.source_role,
            failure.failure_kind,
        )
        buckets.setdefault(key, []).append(failure.failure_id)

    clusters: list[FailureCluster] = []
    for key, failure_ids in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        evaluator, suite, protocol_sha, source_role, failure_kind = key
        ordered_ids = tuple(sorted(failure_ids))
        cluster_id = _canonical_digest(
            {
                "evaluator": evaluator,
                "suite": suite,
                "protocol_sha256": protocol_sha,
                "source_role": source_role.value,
                "failure_kind": failure_kind,
                "failure_ids": ordered_ids,
            }
        )
        clusters.append(
            FailureCluster(
                cluster_id=cluster_id,
                evaluator=evaluator,
                suite=suite,
                protocol_sha256=protocol_sha,
                source_role=source_role,
                failure_kind=failure_kind,
                failure_ids=ordered_ids,
            )
        )
    return tuple(clusters)


def plan_repairs(clusters: Iterable[FailureCluster]) -> tuple[RepairPlan, ...]:
    plans: list[RepairPlan] = []
    for cluster in clusters:
        direct = cluster.source_role.training_eligible
        if cluster.failure_kind == "empty_prediction":
            suspected = "generation or task-format failure"
            intervention = "add independently sourced examples that require a concise non-empty answer"
        elif cluster.failure_kind == "refusal_or_unknown":
            suspected = "over-refusal or insufficient task confidence"
            intervention = "add independently sourced solvable examples plus calibrated uncertainty negatives"
        elif cluster.failure_kind == "overlong_mismatch":
            suspected = "format or stopping-control weakness"
            intervention = "add independently sourced concise-answer examples and explicit output-format supervision"
        else:
            suspected = "knowledge, reasoning, or answer-selection weakness"
            intervention = "add independently sourced near-neighbor examples targeting this failure cluster"

        if not direct:
            intervention += "; never copy holdout prompt/answer pairs into training"

        plan_id = _canonical_digest(
            {
                "cluster_id": cluster.cluster_id,
                "intervention": intervention,
                "direct_training_allowed": direct,
            }
        )
        plans.append(
            RepairPlan(
                plan_id=plan_id,
                cluster_id=cluster.cluster_id,
                observation=f"{len(cluster.failure_ids)} {cluster.failure_kind} failures in {cluster.suite}",
                suspected_cause=suspected,
                intervention=intervention,
                source_failure_ids=cluster.failure_ids,
                direct_training_allowed=direct,
                requires_independent_source=not direct,
            )
        )
    return tuple(plans)


def write_direct_repair_dataset(
    failures: Iterable[FailureRecord],
    output_path: str | Path,
) -> str:
    """Write SFT rows only from explicitly designated repair-source examples.

    This function intentionally refuses mixed or holdout inputs. A later
    synthetic-data generator must use an independent knowledge/source corpus when
    repairing gate failures instead of laundering benchmark answers into training.
    """

    rows = tuple(failures)
    if not rows:
        raise ValueError("no failures supplied")
    blocked = [failure.failure_id for failure in rows if not failure.training_eligible]
    if blocked:
        raise ValueError("repair dataset contains non-training-eligible holdout/development failures")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for failure in rows:
            handle.write(
                json.dumps(
                    {
                        "text": f"User: {failure.prompt}\nAssistant: {failure.expected}",
                        "failure_id": failure.failure_id,
                        "source_role": failure.source_role.value,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()
