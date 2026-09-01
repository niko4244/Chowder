from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from .models import Experiment, ExperimentStatus, Hypothesis
from .recursive_recovery import (
    RecoveryDisposition,
    RecursiveRecoveryReport,
    analyze_recursive_repair_session,
)
from .recursive_repair import (
    RecursiveRepairOutcome,
    RecursiveRepairPolicy,
    _engine_snapshot,
    _execution_snapshot,
    _policy_payload,
    _resume_bounded_autonomous_repair_from_checkpoint,
    _variant_metadata,
    _validated_variants,
)
from .recursive_trace import RecursiveRepairTraceStore
from .repair_candidates import RepairVariant
from .repair_requests import RepairSourceProvider
from .tournament import rank_candidates


class RecursiveResumeError(ValueError):
    """Raised when an interrupted recursive repair cannot be resumed safely."""


def _provider_identity(provider: RepairSourceProvider) -> tuple[str, str]:
    return (
        str(getattr(provider, "name", type(provider).__name__)),
        str(getattr(provider, "version", "unknown")),
    )


def _canonical_json_value(value: object) -> object:
    """Normalize tuples/other JSON-compatible containers to durable JSON types."""

    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise RecursiveResumeError("resume identity contains non-canonical JSON data") from exc


def _portable_score(value: object) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            raise RecursiveResumeError("checkpoint previous target score cannot be NaN")
        return number
    raise RecursiveResumeError("checkpoint previous target score is invalid")


def _require_resumable(report: RecursiveRecoveryReport) -> None:
    if report.disposition is not RecoveryDisposition.RESUMABLE:
        raise RecursiveResumeError(
            f"recursive repair session {report.session_id!r} is not resumable: "
            f"{report.disposition.value}: {report.detail}"
        )


def _verify_resume_identity(
    *,
    runner: ExperimentCycleRunner,
    session: Mapping[str, Any],
    provider: RepairSourceProvider,
    variants: tuple[RepairVariant, ...],
    policy: RecursiveRepairPolicy,
) -> None:
    metadata = session.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RecursiveResumeError("recursive session metadata is invalid")

    if _canonical_json_value(metadata.get("engine_snapshot")) != _canonical_json_value(
        _engine_snapshot(runner)
    ):
        raise RecursiveResumeError(
            "current engine goal/baseline does not match the interrupted session snapshot"
        )

    execution_snapshot = metadata.get("execution_snapshot")
    if not isinstance(execution_snapshot, Mapping):
        raise RecursiveResumeError(
            "interrupted session predates execution-context snapshots and cannot be auto-resumed"
        )
    if _canonical_json_value(execution_snapshot) != _canonical_json_value(
        _execution_snapshot(runner)
    ):
        raise RecursiveResumeError(
            "current executor/seed/base-config/hardware does not match the interrupted session"
        )

    if _canonical_json_value(session.get("policy")) != _canonical_json_value(
        _policy_payload(policy)
    ):
        raise RecursiveResumeError("resume policy does not match the interrupted session")
    if _canonical_json_value(metadata.get("variants")) != _canonical_json_value(
        _variant_metadata(variants)
    ):
        raise RecursiveResumeError("repair variants do not match the interrupted session")

    provider_name, provider_version = _provider_identity(provider)
    if metadata.get("provider_name") != provider_name:
        raise RecursiveResumeError("repair provider identity does not match the interrupted session")
    if metadata.get("provider_version") != provider_version:
        raise RecursiveResumeError("repair provider version does not match the interrupted session")


def _persisted_experiments(runner: ExperimentCycleRunner) -> tuple[Experiment, ...]:
    if runner.registry is None:
        raise RecursiveResumeError("recursive resume requires a RunRegistry")
    rows = runner.registry._conn.execute(
        """SELECT experiment_id, parent_id, estimated_gpu_hours,
                  hypothesis_json, config_json, status
           FROM experiments ORDER BY rowid"""
    )
    experiments: list[Experiment] = []
    for experiment_id, parent_id, hours, hypothesis_json, config_json, status in rows:
        hypothesis_raw = json.loads(hypothesis_json)
        if not isinstance(hypothesis_raw, Mapping):
            raise RecursiveResumeError(
                f"persisted hypothesis for {experiment_id!r} is invalid"
            )
        config_raw = json.loads(config_json)
        if not isinstance(config_raw, Mapping):
            raise RecursiveResumeError(
                f"persisted config for {experiment_id!r} is invalid"
            )
        try:
            hypothesis = Hypothesis(**dict(hypothesis_raw))
            experiment_status = ExperimentStatus(status)
        except (TypeError, ValueError) as exc:
            raise RecursiveResumeError(
                f"persisted experiment {experiment_id!r} cannot be reconstructed"
            ) from exc
        experiments.append(
            Experiment(
                experiment_id=experiment_id,
                parent_id=parent_id,
                hypothesis=hypothesis,
                config_patch=dict(config_raw),
                estimated_gpu_hours=float(hours),
                status=experiment_status,
            )
        )
    return tuple(experiments)


def _restore_engine_state(
    runner: ExperimentCycleRunner,
    *,
    remaining_budget: float,
) -> None:
    if runner.engine.graph.nodes:
        raise RecursiveResumeError(
            "resume requires a fresh EvolutionEngine graph; refusing to merge live and recovered state"
        )
    if runner.engine.reserved_gpu_hours != 0 or runner.engine.outstanding_candidates != 0:
        raise RecursiveResumeError("resume requires an engine with no active reservations")
    if runner.engine.spent_gpu_hours != 0:
        raise RecursiveResumeError("resume requires a fresh engine compute ledger")

    experiments = _persisted_experiments(runner)
    runner.engine.graph.add_many(experiments)
    spent = runner.engine.goal.gpu_hour_budget - float(remaining_budget)
    if not math.isfinite(spent) or spent < -1e-9:
        raise RecursiveResumeError("checkpoint remaining budget exceeds configured goal budget")
    runner.engine.spent_gpu_hours = max(0.0, spent)
    runner.engine.reserved_gpu_hours = 0.0
    runner.engine._reservations.clear()


def _one_by_experiment(rows: Iterable[Any], *, label: str) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.experiment_id].append(row)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if duplicates:
        raise RecursiveResumeError(
            f"recovered {label} evidence is ambiguous for experiments: {duplicates}"
        )
    return {key: values[0] for key, values in grouped.items()}


def _reconstruct_generation(
    runner: ExperimentCycleRunner,
    candidate_ids: tuple[str, ...],
) -> GenerationOutcome:
    if runner.registry is None:
        raise RecursiveResumeError("recursive resume requires a RunRegistry")
    if not candidate_ids:
        raise RecursiveResumeError("checkpoint contains no current candidate IDs")

    artifacts = _one_by_experiment(
        runner.registry.list_training_artifacts(), label="training"
    )
    evaluations = _one_by_experiment(
        runner.registry.list_evaluation_outcomes(), label="evaluation"
    )
    results = _one_by_experiment(runner.registry.list_results(), label="result")
    failures_by_experiment: dict[str, list[Any]] = defaultdict(list)
    for failure in runner.registry.list_failures():
        failures_by_experiment[failure.experiment_id].append(failure)
    plans = tuple(runner.registry.list_repair_plans())

    candidates: list[CandidateCycleOutcome] = []
    candidate_results = []
    for experiment_id in candidate_ids:
        artifact = artifacts.get(experiment_id)
        evaluation = evaluations.get(experiment_id)
        result = results.get(experiment_id)
        if artifact is None or evaluation is None or result is None:
            raise RecursiveResumeError(
                f"checkpoint candidate {experiment_id!r} lacks complete immutable run evidence"
            )
        failures = tuple(failures_by_experiment.get(experiment_id, ()))
        failure_ids = {failure.failure_id for failure in failures}
        relevant_plans = tuple(
            plan
            for plan in plans
            if plan.source_failure_ids
            and set(plan.source_failure_ids).issubset(failure_ids)
        )
        candidates.append(
            CandidateCycleOutcome(
                experiment_id=experiment_id,
                artifact=artifact,
                evaluation=evaluation,
                result=result,
                harvested_failures=failures,
                repair_plans=relevant_plans,
            )
        )
        candidate_results.append(result)

    ranking = rank_candidates(
        goal=runner.engine.goal,
        baseline=runner.engine.baseline,
        candidates=tuple(candidate_results),
    )
    if any(candidate.decision.accepted for candidate in ranking):
        raise RecursiveResumeError(
            "checkpoint reconstructs an accepted candidate; persisted gate state is inconsistent"
        )
    return GenerationOutcome(
        candidates=tuple(candidates),
        ranking=ranking,
        promoted=None,
    )


def resume_recursive_repair_session(
    *,
    runner: ExperimentCycleRunner,
    session_id: str,
    provider: RepairSourceProvider,
    variants: Iterable[RepairVariant],
    policy: RecursiveRepairPolicy | None = None,
) -> RecursiveRepairOutcome:
    """Safely resume one interrupted bounded recursive-repair session.

    The caller must supply a fresh runner whose immutable execution identity
    matches the checkpoint. The session is reconciled once before and once after
    an atomic recovery claim; only the claim holder may append new hops.
    """

    if runner.registry is None:
        raise RecursiveResumeError("recursive resume requires runner.registry")
    if not session_id.strip():
        raise RecursiveResumeError("recursive resume session_id is required")

    variant_rows = _validated_variants(variants)
    first_report = analyze_recursive_repair_session(runner.registry, session_id)
    _require_resumable(first_report)

    claim_token = uuid4().hex
    store = RecursiveRepairTraceStore(runner.registry.path)
    claimed = False
    loop_started = False
    try:
        session = store.get_session(session_id)
        if session is None:
            raise RecursiveResumeError("recursive repair session disappeared during recovery")
        chosen_policy = policy or RecursiveRepairPolicy(**dict(session["policy"]))
        _verify_resume_identity(
            runner=runner,
            session=session,
            provider=provider,
            variants=variant_rows,
            policy=chosen_policy,
        )

        store.claim_recovery(session_id=session_id, claim_token=claim_token)
        claimed = True

        second_report = analyze_recursive_repair_session(runner.registry, session_id)
        _require_resumable(second_report)
        session = store.get_session(session_id)
        if session is None:
            raise RecursiveResumeError("recursive repair session disappeared after claim")
        _verify_resume_identity(
            runner=runner,
            session=session,
            provider=provider,
            variants=variant_rows,
            policy=chosen_policy,
        )

        state = session.get("state")
        if not isinstance(state, Mapping):
            raise RecursiveResumeError("recursive repair checkpoint state is invalid")
        remaining = state.get("remaining_budget")
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
            raise RecursiveResumeError("checkpoint remaining budget is invalid")
        remaining_budget = float(remaining)
        if not math.isfinite(remaining_budget) or remaining_budget < 0:
            raise RecursiveResumeError("checkpoint remaining budget is invalid")

        signature_raw = state.get("signature_counts")
        if not isinstance(signature_raw, Mapping):
            raise RecursiveResumeError("checkpoint signature counts are invalid")
        signature_counts = {str(key): value for key, value in signature_raw.items()}
        previous_score = _portable_score(state.get("previous_target_score"))

        _restore_engine_state(runner, remaining_budget=remaining_budget)
        current_generation = _reconstruct_generation(
            runner, second_report.current_candidate_ids
        )

        loop_started = True
        return _resume_bounded_autonomous_repair_from_checkpoint(
            runner=runner,
            current_generation=current_generation,
            provider=provider,
            variants=variant_rows,
            policy=chosen_policy,
            session_id=session_id,
            starting_depth=second_report.depth_completed,
            signature_counts=signature_counts,
            previous_target_score=previous_score,
            recovery_claim_token=claim_token,
        )
    except Exception:
        if claimed and not loop_started:
            try:
                claim = store.get_recovery_claim(session_id)
                if claim is not None and claim.get("claim_token") == claim_token:
                    store.release_recovery_claim(
                        session_id=session_id, claim_token=claim_token
                    )
            except Exception:
                pass
        raise
    finally:
        store.close()
