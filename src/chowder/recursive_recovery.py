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
    MISSING_REGISTRY_EVIDENCE = "missing_registry_evidence"
    AMBIGUOUS_REGISTRY_EVIDENCE = "ambiguous_registry_evidence"
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
    if isinstance(value, (int, float)):
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


def _engine_snapshot(metadata: object) -> Mapping[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    snapshot = metadata.get("engine_snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    if not _REQUIRED_ENGINE_SNAPSHOT_KEYS.issubset(snapshot):
        return None
    return snapshot


def list_interrupted_recursive_sessions(
    registry_path: str | Path,
) -> tuple[str, ...]:
    """List non-terminal recursive sessions without mutating recovery state."""

    path = str(registry_path)
    # Constructing the trace store also migrates older Chowder DBs by creating
    # the trace tables if they do not yet exist.
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
    resumable only when the checkpoint is internally self-consistent, the engine
    goal/baseline snapshot exists, every current candidate has one unambiguous
    training/evaluation/result record, diagnostic evidence can be recovered, and
    no child experiment exists beyond the committed checkpoint.
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

    snapshot = _engine_snapshot(session.get("metadata"))
    if snapshot is None:
        return RecursiveRecoveryReport(
            session_id,
            RecoveryDisposition.MISSING_ENGINE_SNAPSHOT,
            "session predates or lacks the goal/baseline snapshot required for deterministic recovery",
            depth_completed=depth,
            current_candidate_ids=current_ids,
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
    for experiment_id in current_ids:
        train_rows = training.get(experiment_id, ())
        eval_rows = evaluations.get(experiment_id, ())
        if len(train_rows) == 0:
            missing.append(f"training:{experiment_id}")
        elif len(train_rows) != 1:
            ambiguous.append(f"training:{experiment_id}:{len(train_rows)}")
        if len(eval_rows) == 0:
            missing.append(f"evaluation:{experiment_id}")
        elif len(eval_rows) != 1:
            ambiguous.append(f"evaluation:{experiment_id}:{len(eval_rows)}")
        if experiment_id not in results:
            missing.append(f"result:{experiment_id}")

        failure_ids = {failure.failure_id for failure in failures[experiment_id]}
        if failure_ids:
            matching_plans = [
                plan
                for plan in plans
                if set(plan.source_failure_ids).issubset(failure_ids)
                and plan.source_failure_ids
            ]
            result = results.get(experiment_id)
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

    return RecursiveRecoveryReport(
        session_id,
        RecoveryDisposition.RESUMABLE,
        "checkpoint and persisted run evidence are consistent; no unrecorded descendants were found",
        depth_completed=depth,
        current_candidate_ids=current_ids,
        engine_snapshot=snapshot,
    )
