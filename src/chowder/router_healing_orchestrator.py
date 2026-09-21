"""The caller that makes router healing a real Chowder experiment.

`router_healing.py` decides what trains. `router_healing_run.py` holds the
provenance, goal and gate plumbing. Neither *runs* anything, which is exactly
the gap an audit found: the only working healing run was a standalone script
that recorded no experiment, accounted no GPU-hours, and never reached the
gate. This module closes that by driving one bounded healing experiment
through Chowder's existing lifecycle, satisfying five acceptance conditions:

1. **Preflight measures reality, not labels.** `preflight_loaded_model`
   reports the dtype and byte footprint of the loaded tensors per category.
   This matters because "loaded in 4-bit" is misleading for this
   architecture: bitsandbytes only replaces `nn.Linear`, and
   `Qwen3_5MoeExperts` holds `gate_up_proj`/`down_proj` as raw
   `nn.Parameter`, so the bulk of the weight stays BF16. A run that assumes
   otherwise mis-budgets by tens of GiB.
2. **Every designated-trainable tensor must be able to learn.** Delegated to
   `assert_trainable_gradients_reachable`, which exists because a zero-init
   frozen shared expert silently made 64 "trainable" gates unlearnable.
3. **A delta plus resumable state is saved**, so a bounded run is a step in a
   sequence rather than a thing that must succeed in one sitting.
4. **Reload is independent.** The delta is re-read from disk and verified
   against the base manifest before evaluation, so evaluation cannot
   accidentally score the in-memory training model.
5. **The registry is the record.** Experiment row before any compute, real
   measured GPU-hours, terminal status on success, failure AND cancellation,
   and the gate's verdict -- never this module's opinion.

Training and evaluation arrive as injected callables. That is deliberate and
matches how this repo already tests orchestration: the trainer and evaluator
are the only fakes in the integration suites, while the engine, registry,
gate and ledger are the production implementations. It also keeps a
27B-specific training loop out of the library.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .cancellation import CancellationToken, OperationCancelled
from .executors import EvaluationOutcome
from .models import ExperimentStatus, GateDecision, Goal
from .router_healing import RouterHealingError, freeze_for_router_healing
from .router_healing_run import (
    RouterHealingSpec,
    build_healing_experiment,
    experiment_result_from_outcome,
    healing_experiment_id,
    judge_router_healing,
    load_router_delta,
    save_router_delta,
)

#: Categories reported by the preflight, in the order a reader wants them.
_PREFLIGHT_CATEGORIES = ("routed_expert", "router", "shared_expert", "attention", "embedding", "other")


class TrainFn(Protocol):
    """Runs bounded training and returns what was learned.

    Must honour `cancellation.raise_if_requested()` between steps so a stop
    request is a clean `OperationCancelled`, not a killed process. Returns a
    `TrainOutcome`; raising anything else is treated as a real failure.
    """

    def __call__(
        self, *, model: Any, spec: RouterHealingSpec, cancellation: CancellationToken
    ) -> "TrainOutcome": ...


class EvaluateFn(Protocol):
    """Scores a reloaded model and returns a registry-ready outcome.

    Receives the path the delta was reloaded onto, NOT the training model, so
    an implementation cannot accidentally score unsaved in-memory state.
    """

    def __call__(self, *, model_dir: str, experiment_id: str, spec: RouterHealingSpec) -> EvaluationOutcome: ...


@dataclass(frozen=True)
class TrainOutcome:
    """What a bounded training pass produced."""

    steps_completed: int
    trainable_tensors: Mapping[str, Any]
    resumable_state: Mapping[str, Any] = field(default_factory=dict)
    peak_gpu_mib: float | None = None

    def __post_init__(self) -> None:
        if self.steps_completed <= 0:
            raise RouterHealingError("a training outcome must report at least one completed step")
        if not self.trainable_tensors:
            raise RouterHealingError("a training outcome must carry the trained tensors")


@dataclass(frozen=True)
class HealingRunRecord:
    """The durable story of one healing experiment."""

    experiment_id: str
    status: str
    spec_digest: str
    preflight: Mapping[str, Any]
    steps_completed: int | None
    gpu_hours: float
    delta_path: str | None
    evaluation_run_id: str | None
    decision: GateDecision | None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "spec_digest": self.spec_digest,
            "preflight": dict(self.preflight),
            "steps_completed": self.steps_completed,
            "gpu_hours": self.gpu_hours,
            "delta_path": self.delta_path,
            "evaluation_run_id": self.evaluation_run_id,
            "decision": None if self.decision is None else {
                "accepted": self.decision.accepted,
                "score": self.decision.score,
                "reason": self.decision.reason,
                "regressions": dict(self.decision.regressions),
            },
            "failure": self.failure,
        }


def _categorise(name: str) -> str:
    if ".mlp.experts." in name:
        return "routed_expert"
    if name.endswith("mlp.gate.weight") or name.endswith("mlp.shared_expert_gate.weight"):
        return "router"
    if ".mlp.shared_expert." in name:
        return "shared_expert"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding"
    if any(token in name for token in (".self_attn.", ".attn.", "in_proj", "out_proj")):
        return "attention"
    return "other"


def preflight_loaded_model(model: Any) -> dict[str, Any]:
    """Report the dtypes and bytes actually resident, per category.

    Acceptance condition 1. The headline field is `quantized_fraction`: if a
    run believes it loaded in 4-bit, a low value here is the warning that most
    of the weight is still full precision, which is the difference between a
    model that fits and one that pages over PCIe for minutes per step.
    """
    by_category: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    quantized_bytes = 0

    for name, param in model.named_parameters():
        category = _categorise(name)
        entry = by_category.setdefault(
            category, {"tensors": 0, "parameters": 0, "bytes": 0, "dtypes": {}}
        )
        numel = int(param.numel())
        # bitsandbytes Params4bit stores packed uint8; element_size reflects that
        element_size = int(getattr(param, "element_size", lambda: 0)() or 0)
        nbytes = numel * element_size
        dtype_name = str(getattr(param, "dtype", "unknown")).replace("torch.", "")
        is_quantized = type(param).__name__ in {"Params4bit", "Int8Params"} or element_size == 1

        entry["tensors"] += 1
        entry["parameters"] += numel
        entry["bytes"] += nbytes
        entry["dtypes"][dtype_name] = entry["dtypes"].get(dtype_name, 0) + 1
        total_bytes += nbytes
        if is_quantized:
            quantized_bytes += nbytes

    ordered = {c: by_category[c] for c in _PREFLIGHT_CATEGORIES if c in by_category}
    for extra in sorted(set(by_category) - set(ordered)):
        ordered[extra] = by_category[extra]

    return {
        "categories": ordered,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "quantized_bytes": quantized_bytes,
        "quantized_fraction": round(quantized_bytes / total_bytes, 4) if total_bytes else 0.0,
        "note": (
            "bitsandbytes replaces nn.Linear only. Raw nn.Parameter expert tensors "
            "(Qwen3_5MoeExperts.gate_up_proj/down_proj) stay full precision even under "
            "load_in_4bit, so a low quantized_fraction with a '4-bit' label is expected "
            "and is the real memory budget."
        ),
    }


def run_router_healing_experiment(
    registry: Any,
    *,
    model: Any,
    spec: RouterHealingSpec,
    goal: Goal,
    baseline_outcome: Any,
    delta_dir: str | Path,
    train_fn: TrainFn,
    evaluate_fn: EvaluateFn,
    reload_model_dir: str | None = None,
    parent_experiment_id: str | None = None,
    estimated_gpu_hours: float = 1.0,
    trainable_suffixes: tuple[str, ...] | None = None,
    cancellation: CancellationToken | None = None,
) -> HealingRunRecord:
    """Drive one bounded healing experiment end to end.

    Ordering is the contract. The experiment row and the gradient-reachability
    check both land before any training compute, so a run can never spend GPU
    time that the registry has no record of, nor train a tensor that cannot
    learn. Cancellation and failure are recorded as terminal states rather
    than leaving a row stuck in RUNNING.
    """
    token = cancellation or CancellationToken()
    experiment = build_healing_experiment(
        spec, parent_experiment_id=parent_experiment_id, estimated_gpu_hours=estimated_gpu_hours
    )
    experiment_id = healing_experiment_id(spec)

    # Condition 5, first half: the row exists before anything is spent.
    if not registry.has_experiment(experiment_id):
        registry.record_experiment(experiment)

    preflight = preflight_loaded_model(model)
    started = time.time()

    def _elapsed_gpu_hours() -> float:
        return round((time.time() - started) / 3600.0, 6)

    def _terminal(status: ExperimentStatus, *, failure: str | None, steps: int | None,
                  delta_path: str | None = None, run_id: str | None = None,
                  decision: GateDecision | None = None) -> HealingRunRecord:
        registry.update_experiment_status(experiment_id, status.value)
        return HealingRunRecord(
            experiment_id=experiment_id,
            status=status.value,
            spec_digest=spec.digest(),
            preflight=preflight,
            steps_completed=steps,
            gpu_hours=_elapsed_gpu_hours(),
            delta_path=delta_path,
            evaluation_run_id=run_id,
            decision=decision,
            failure=failure,
        )

    try:
        token.raise_if_requested()
        # Conditions 2: freeze + refuse unreachable trainables, pre-compute.
        freeze_summary = freeze_for_router_healing(model, suffixes=trainable_suffixes)
        preflight["freeze"] = freeze_summary.to_dict()
        registry.update_experiment_status(experiment_id, ExperimentStatus.RUNNING.value)

        outcome = train_fn(model=model, spec=spec, cancellation=token)

        # Condition 3: delta + resumable state, written before evaluation so a
        # crash after training still leaves the learned weights recoverable.
        delta_path = Path(delta_dir) / f"{experiment_id}.delta.json"
        artifact = save_router_delta(
            outcome.trainable_tensors, delta_path,
            spec=spec, steps_completed=outcome.steps_completed,
        )
        if outcome.resumable_state:
            state_path = Path(delta_dir) / f"{experiment_id}.resume.json"
            import json as _json
            state_path.write_text(
                _json.dumps(dict(outcome.resumable_state), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8", newline="\n",
            )

        # Condition 4: reload from disk and verify provenance before scoring.
        token.raise_if_requested()
        load_router_delta(delta_path, expected_base_manifest_sha256=spec.base_manifest_sha256)
        evaluation = evaluate_fn(
            model_dir=reload_model_dir or spec.base_model_dir,
            experiment_id=experiment_id,
            spec=spec,
        )
        registry.record_evaluation_outcome(evaluation)

        # Condition 5, second half: the gate decides, not this module.
        candidate = experiment_result_from_outcome(evaluation)
        baseline = experiment_result_from_outcome(baseline_outcome)
        decision = judge_router_healing(goal=goal, baseline=baseline, candidate=candidate)

        status = ExperimentStatus.PASSED if decision.accepted else ExperimentStatus.REJECTED
        return _terminal(
            status, failure=None, steps=outcome.steps_completed,
            delta_path=artifact.delta_path, run_id=evaluation.run_id, decision=decision,
        )

    except OperationCancelled as exc:
        # A requested stop is not a defect, but it must not leave RUNNING.
        return _terminal(ExperimentStatus.FAILED, failure=f"cancelled: {exc}", steps=None)
    except Exception as exc:  # noqa: BLE001 - every failure must reach the registry
        return _terminal(ExperimentStatus.FAILED, failure=f"{type(exc).__name__}: {exc}", steps=None)
