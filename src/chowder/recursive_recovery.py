from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .registry import RunRegistry
from .recursive_trace import RecursiveRepairTraceStore


class RecoveryDisposition(str, Enum):
    RESUMABLE = "resumable"
    ALREADY_TERMINAL = "already_terminal"
    TERMINAL_PENDING = "terminal_pending"
    SESSION_NOT_FOUND = "session_not_found"
    CHECKPOINT_INCONSISTENT = "checkpoint_inconsistent"
    MISSING_ENGINE_SNAPSHOT = "missing_engine_snapshot"
    INVALID_ENGINE_SNAPSHOT = "invalid_engine_snapshot"
    MISSING_REGISTRY_EVIDENCE = "missing_registry_evidence"
    AMBIGUOUS_REGISTRY_EVIDENCE = "ambiguous_registry_evidence"
    REGISTRY_EVIDENCE_MISMATCH = "registry_evidence_mismatch"
    ORPHANED_PROGRESS = "orphaned_progress"


@dataclass(frozen=True)
class RecursiveRecoveryReport:
    session_id: str
    disposition: RecoveryDisposition
    detail: str
    depth_completed: int = 0
    current_candidate_ids: tuple[str, ...] = ()
    orphaned_child_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    engine_snapshot: Mapping[str, Any] | None = None

    @property
    def resumable(self) -> bool:
        return self.disposition is RecoveryDisposition.RESUMABLE


_REQUIRED_ENGINE_SNAPSHOT_KEYS = {"goal", "baseline"}


def _experiment_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute(
            "SELECT experiment_id, parent_id, status, config_json FROM experiments"
        ).fetchall()
    finally:
        connection.close()
    return {
        experiment_id: {
            "experiment_id": experiment_id,
            "parent_id": parent_id,
            "status": status,
            "config": json.loads(config_json),
        }
        for experiment_id, parent_id, status, config_json in rows
    }


def _children_by_parent(
    experiments: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {}
    for experiment_id, row in experiments.items():
        parent_id = row.get("parent_id")
        if isinstance(parent_id, str):
            buckets.setdefault(parent_id, []).append(experiment_id)
    return {
        parent_id: tuple(sorted(children))
        for parent_id, children in buckets.items()
    }


def _group_by_experiment(rows, *, attribute: str = "experiment_id"):
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        experiment_id = getattr(row, attribute)
        grouped.setdefault(experiment_id, []).append(row)
    return grouped


def _expected_signature_counts(hops: tuple[Mapping[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hop in hops:
        signature = hop.get("failure_signature")
        if not isinstance(signature, str) or len(signature) != 64:
            raise ValueError("persisted recursive hop has an invalid failure signature")
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def _parse_portable_number(value: object) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("checkpoint previous_target_score is invalid")


def _validate_checkpoint(
    *,
    session: Mapping[str, Any],
    hops: tuple[Mapping[str, Any], ...],
) -> tuple[int, tuple[str, ...], float | None]:
    state = session.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("recursive session state is not a mapping")

    depth = state.get("depth_completed")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("checkpoint depth_completed is invalid")
    if depth != len(hops):
        raise ValueError("checkpoint depth does not match persisted hop count")

    hop_depths = tuple(hop.get("depth") for hop in hops)
    if hop_depths != tuple(range(1, len(hops) + 1)):
        raise ValueError("persisted recursive hop depths are not contiguous")

    current_raw = state.get("current_candidate_ids")
    if not isinstance(current_raw, list) or any(
        not isinstance(value, str) or not value for value in current_raw
    ):
        raise ValueError("checkpoint current_candidate_ids is invalid")
    current = tuple(current_raw)
    if len(current) != len(set(current)):
        raise ValueError("checkpoint current candidate IDs are not unique")

    expected_current = (
        tuple(hops[-1]["produced_candidate_ids"])
        if hops
        else tuple(session.get("initial_candidate_ids", ()))
    )
    if current != expected_current:
        raise ValueError("checkpoint current candidates disagree with hop history")

    signature_counts = state.get("signature_counts")
    if not isinstance(signature_counts, Mapping):
        raise ValueError("checkpoint signature_counts is invalid")
    normalized_counts: dict[str, int] = {}
    for key, value in signature_counts.items():
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("checkpoint contains invalid failure signature")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("checkpoint contains invalid signature occurrence count")
        normalized_counts[key] = value
    if normalized_counts != _expected_signature_counts(hops):
        raise ValueError("checkpoint signature counts disagree with hop history")

    previous_score = _parse_portable_number(state.get("previous_target_score"))
    expected_previous = float(hops[-1]["target_score"]) if hops else None
    if previous_score != expected_previous:
        raise ValueError("checkpoint previous target score disagrees with hop history")

    remaining = state.get("remaining_budget")
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
        raise ValueError("checkpoint remaining_budget is invalid")
    if not math.isfinite(float(remaining)) or float(remaining) < 0:
        raise ValueError("checkpoint remaining_budget must be finite and non-negative")

    promoted = state.get("promoted_experiment_id")
    if promoted is not None and (not isinstance(promoted, str) or not promoted):
        raise ValueError("checkpoint promoted_experiment_id is invalid")
    if hops:
        hop_promoted = hops[-1].get("promoted_experiment_id")
        if promoted != hop_promoted:
            raise ValueError("checkpoint promotion state disagrees with final hop")

    return depth, current, previous_score


def _raw_engine_snapshot(metadata: object) -> Mapping[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    snapshot = metadata.get("engine_snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    if not _REQUIRED_ENGINE_SNAPSHOT_KEYS.issubset(snapshot):
        return None
    return snapshot


def _finite_number(value: object, *, label: str, non_negative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if non_negative and number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _validate_engine_snapshot(
    snapshot: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    remaining_budget: float,
) -> None:
    goal = snapshot.get("goal")
    baseline = snapshot.get("baseline")
    if not isinstance(goal, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("engine snapshot goal/baseline must be mappings")

    metrics = goal.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("engine snapshot goal must contain metric targets")
    names: set[str] = set()
    for index, target in enumerate(metrics):
        if not isinstance(target, Mapping):
            raise ValueError(f"engine snapshot goal metric {index} is invalid")
        name = target.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"engine snapshot goal metric {index} has invalid name")
        if name in names:
            raise ValueError("engine snapshot goal contains duplicate metric names")
        names.add(name)
        direction = target.get("direction")
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"engine snapshot goal metric {name!r} has invalid direction")
        for bound_name in ("minimum", "maximum"):
            bound = target.get(bound_name)
            if bound is not None:
                _finite_number(bound, label=f"goal metric {name} {bound_name}")
        _finite_number(target.get("weight"), label=f"goal metric {name} weight")
        _finite_number(
            target.get("regression_tolerance"),
            label=f"goal metric {name} regression_tolerance",
            non_negative=True,
        )

    budget = _finite_number(
        goal.get("gpu_hour_budget"), label="goal gpu_hour_budget", non_negative=True
    )
    parallel = goal.get("max_parallel_candidates")
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel <= 0:
        raise ValueError("goal max_parallel_candidates must be positive")
    _finite_number(
        goal.get("minimum_promotion_gain"), label="goal minimum_promotion_gain"
    )
    if not isinstance(goal.get("require_protocol_match"), bool):
        raise ValueError("goal require_protocol_match must be boolean")
    if remaining_budget > budget + 1e-9:
        raise ValueError("checkpoint remaining budget exceeds snapshotted goal budget")

    baseline_id = baseline.get("experiment_id")
    if not isinstance(baseline_id, str) or not baseline_id:
        raise ValueError("engine snapshot baseline experiment_id is invalid")
    declared_baseline = metadata.get("baseline_experiment_id")
    if declared_baseline is not None and declared_baseline != baseline_id:
        raise ValueError("engine snapshot baseline ID disagrees with session metadata")
    baseline_metrics = baseline.get("metrics")
    if not isinstance(baseline_metrics, Mapping) or not baseline_metrics:
        raise ValueError("engine snapshot baseline metrics are invalid")
    for name, value in baseline_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("engine snapshot baseline metric name is invalid")
        _finite_number(value, label=f"baseline metric {name}")
    _finite_number(
        baseline.get("gpu_hours"), label="baseline gpu_hours", non_negative=True
    )
    artifact_ref = baseline.get("artifact_ref")
    if artifact_ref is not None and not isinstance(artifact_ref, str):
        raise ValueError("engine snapshot baseline artifact_ref is invalid")
    if not isinstance(baseline.get("evidence"), Mapping):
        raise ValueError("engine snapshot baseline evidence must be a mapping")


def list_interrupted_recursive_sessions(
    registry_path: str | Path,
) -> tuple[str, ...]:
    """List non-terminal recursive sessions without mutating recovery state."""

    path = str(registry_path)
    with RecursiveRepairTraceStore(path):
        pass
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """SELECT session_id FROM recursive_repair_sessions
               WHERE status = 'running' ORDER BY rowid"""
        ).fetchall()
    finally:
        connection.close()
    return tuple(row[0] for row in rows)


def analyze_recursive_repair_session(
    registry: RunRegistry,
    session_id: str,
) -> RecursiveRecoveryReport:
    """Reconcile one durable recursive checkpoint against persisted run evidence.

    This function is read-only. It never requeues training. A session is marked
    resumable only when checkpoint, engine snapshot, lineage, training artifact,
    evaluation, result, diagnostics, and compute evidence agree and no descendant
    experiment exists beyond the committed recursive checkpoint.
    """

    with RecursiveRepairTraceStore(registry.path) as store:
        session = store.get_session(session_id)
        if session is None:
            return RecursiveRecoveryReport(
                session_id,
                RecoveryDisposition.SESSION_NOT_FOUND,
                "recursive repair session does not exist",
            )
        hops = store.list_hops(session_id)

    status = session.get("status")
    if status in {"completed", "failed"}:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.ALREADY_TERMINAL,
            f"recursive repair session is already terminal with status={status}",
            depth_completed=len(hops),
        )
    if status != "running":
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.CHECKPOINT_INCONSISTENT,
            f"recursive repair session has unknown status {status!r}",
            depth_completed=len(hops),
        )

    try:
        depth, current_ids, _ = _validate_checkpoint(session=session, hops=hops)
    except (TypeError, ValueError, KeyError) as exc:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.CHECKPOINT_INCONSISTENT,
            str(exc),
            depth_completed=len(hops),
        )

    metadata = session.get("metadata")
    snapshot = _raw_engine_snapshot(metadata)
    if snapshot is None:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.MISSING_ENGINE_SNAPSHOT,
            "session predates or lacks the goal/baseline snapshot required for deterministic recovery",
            depth_completed=depth,
            current_candidate_ids=current_ids,
        )
    assert isinstance(metadata, Mapping)
    try:
        _validate_engine_snapshot(
            snapshot,
            metadata=metadata,
            remaining_budget=float(session["state"]["remaining_budget"]),
        )
    except (TypeError, ValueError) as exc:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.INVALID_ENGINE_SNAPSHOT,
            str(exc),
            depth_completed=depth,
            current_candidate_ids=current_ids,
            engine_snapshot=snapshot,
        )

    state = session["state"]
    policy = session.get("policy", {})
    max_depth = policy.get("max_depth") if isinstance(policy, Mapping) else None
    if state.get("promoted_experiment_id") is not None:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.TERMINAL_PENDING,
            "checkpoint already contains a promoted experiment and should be finalized, not resumed",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            engine_snapshot=snapshot,
        )
    if isinstance(max_depth, int) and depth >= max_depth:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.TERMINAL_PENDING,
            "checkpoint reached configured max depth and should be finalized, not resumed",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            engine_snapshot=snapshot,
        )
    if float(state.get("remaining_budget", 0.0)) <= 0:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.TERMINAL_PENDING,
            "checkpoint has no remaining GPU-hour budget and should be finalized",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            engine_snapshot=snapshot,
        )

    experiments = _experiment_rows(registry.path)
    missing_experiments = tuple(
        sorted(experiment_id for experiment_id in current_ids if experiment_id not in experiments)
    )
    if missing_experiments:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.MISSING_REGISTRY_EVIDENCE,
            "checkpoint references candidate experiments missing from the registry",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            missing_evidence=tuple(f"experiment:{value}" for value in missing_experiments),
            engine_snapshot=snapshot,
        )

    children = _children_by_parent(experiments)
    orphaned = tuple(
        sorted(
            {
                child
                for experiment_id in current_ids
                for child in children.get(experiment_id, ())
            }
        )
    )
    if orphaned:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.ORPHANED_PROGRESS,
            "registry contains child experiments beyond the committed recursive checkpoint; refusing to retrain",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            orphaned_child_ids=orphaned,
            engine_snapshot=snapshot,
        )

    training = _group_by_experiment(tuple(registry.list_training_artifacts()))
    evaluations = _group_by_experiment(tuple(registry.list_evaluation_outcomes()))
    results = {result.experiment_id: result for result in registry.list_results()}
    failures = {
        experiment_id: tuple(registry.list_failures(experiment_id=experiment_id))
        for experiment_id in current_ids
    }
    plans = tuple(registry.list_repair_plans())

    missing: list[str] = []
    ambiguous: list[str] = []
    mismatches: list[str] = []
    for experiment_id in current_ids:
        train_rows = training.get(experiment_id, ())
        eval_rows = evaluations.get(experiment_id, ())
        result = results.get(experiment_id)
        if len(train_rows) == 0:
            missing.append(f"training:{experiment_id}")
        elif len(train_rows) != 1:
            ambiguous.append(f"training:{experiment_id}:{len(train_rows)}")
        if len(eval_rows) == 0:
            missing.append(f"evaluation:{experiment_id}")
        elif len(eval_rows) != 1:
            ambiguous.append(f"evaluation:{experiment_id}:{len(eval_rows)}")
        if result is None:
            missing.append(f"result:{experiment_id}")

        if len(train_rows) == 1 and len(eval_rows) == 1 and result is not None:
            artifact = train_rows[0]
            evaluation = eval_rows[0]
            if evaluation.source_artifact_ref != artifact.artifact_ref:
                mismatches.append(f"evaluation_artifact:{experiment_id}")
            if result.artifact_ref != artifact.artifact_ref:
                mismatches.append(f"result_artifact:{experiment_id}")
            if result.evidence.get("training_run_id") != artifact.run_id:
                mismatches.append(f"training_run_id:{experiment_id}")
            if result.evidence.get("evaluation_run_id") != evaluation.run_id:
                mismatches.append(f"evaluation_run_id:{experiment_id}")
            expected_compute = artifact.gpu_hours + evaluation.gpu_hours
            if not math.isclose(result.gpu_hours, expected_compute, rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(f"result_compute:{experiment_id}")
            compute = result.evidence.get("compute")
            if not isinstance(compute, Mapping):
                mismatches.append(f"compute_evidence:{experiment_id}")
            else:
                try:
                    train_hours = float(compute.get("training_gpu_hours"))
                    eval_hours = float(compute.get("evaluation_gpu_hours"))
                    total_hours = float(compute.get("total_gpu_hours"))
                except (TypeError, ValueError):
                    mismatches.append(f"compute_evidence:{experiment_id}")
                else:
                    if not math.isclose(train_hours, artifact.gpu_hours, rel_tol=1e-9, abs_tol=1e-12):
                        mismatches.append(f"training_compute:{experiment_id}")
                    if not math.isclose(eval_hours, evaluation.gpu_hours, rel_tol=1e-9, abs_tol=1e-12):
                        mismatches.append(f"evaluation_compute:{experiment_id}")
                    if not math.isclose(total_hours, result.gpu_hours, rel_tol=1e-9, abs_tol=1e-12):
                        mismatches.append(f"total_compute:{experiment_id}")
            protocol = evaluation.evidence.get("protocol_sha256")
            result_protocol = result.evidence.get("evaluation_protocol_sha256")
            if isinstance(protocol, str) and len(protocol) == 64 and result_protocol != protocol:
                mismatches.append(f"evaluation_protocol:{experiment_id}")
            if experiments[experiment_id].get("status") != "rejected":
                mismatches.append(f"experiment_status:{experiment_id}")

        failure_ids = {failure.failure_id for failure in failures[experiment_id]}
        if failure_ids:
            matching_plans = [
                plan
                for plan in plans
                if set(plan.source_failure_ids).issubset(failure_ids)
                and plan.source_failure_ids
            ]
            diagnostic_error = None
            if result is not None:
                diagnostics = result.evidence.get("diagnostics")
                if isinstance(diagnostics, Mapping):
                    diagnostic_error = diagnostics.get("error")
            if not matching_plans and diagnostic_error is None:
                missing.append(f"repair_plan:{experiment_id}")

    if ambiguous:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.AMBIGUOUS_REGISTRY_EVIDENCE,
            "checkpoint candidates have multiple persisted training/evaluation records",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            missing_evidence=tuple(sorted(ambiguous)),
            engine_snapshot=snapshot,
        )
    if missing:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.MISSING_REGISTRY_EVIDENCE,
            "checkpoint candidates lack evidence required to reconstruct the generation",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            missing_evidence=tuple(sorted(missing)),
            engine_snapshot=snapshot,
        )
    if mismatches:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.REGISTRY_EVIDENCE_MISMATCH,
            "persisted candidate evidence disagrees across training/evaluation/result records",
            depth_completed=depth,
            current_candidate_ids=current_ids,
            missing_evidence=tuple(sorted(set(mismatches))),
            engine_snapshot=snapshot,
        )

    return RecursiveRecoveryReport(
        session_id,
        RecoveryDisposition.RESUMABLE,
        "checkpoint and persisted run evidence are consistent; no unrecorded descendants were found",
        depth_completed=depth,
        current_candidate_ids=current_ids,
        engine_snapshot=snapshot,
    )
