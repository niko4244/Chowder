"""Wire router healing into Chowder's normal experiment spine.

`router_healing.py` holds the parameter-selection core (what trains, what
stays frozen). This module is the accountability layer around it: a healing
run becomes a real `Experiment` row with real GPU-hours, its evaluation is
the SAME protected-suite path the A/B/C tournament used, and promotion is
decided by `gate.evaluate_candidate` -- never by this module. The plan's
rule is explicit: "the existing hard regression gate judges every healing
run", so nothing here re-implements or softens that judgement.

Two deliberate shape decisions:

* **The healed artifact is a delta, not a checkpoint.** Only ~5.5M
  parameters train (64 layers x (router + shared-expert gate)) against a
  ~56 GiB base, so rewriting a full checkpoint per healing run would burn
  tens of GiB for ~11 MiB of real change. `save_router_delta` persists the
  trained tensors plus the provenance needed to refuse a mismatched base;
  `load_router_delta` fail-closes on base-manifest mismatch rather than
  silently applying a delta to the wrong weights.
* **Baseline and candidate share one metric space.** Both sides come from
  `record_parent_tournament_result`-shaped rows, whose evidence carries
  `evaluation_protocol_sha256` -- so a `Goal` with
  `require_protocol_match=True` rejects any cross-protocol comparison
  before scores are ever weighed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .gate import evaluate_candidate
from .models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    GateDecision,
    Goal,
    Hypothesis,
    MetricTarget,
    OptimizationDirection,
)
from .parent_eval import PARENT_DIMENSIONS
from .router_healing import RouterHealingError


class RouterHealingRunError(RouterHealingError):
    """A healing run cannot be recorded or judged honestly."""


@dataclass(frozen=True)
class RouterHealingSpec:
    """Everything that makes a healing run reproducible and attributable."""

    base_model_dir: str
    base_manifest_sha256: str
    corpus_path: str
    corpus_sha256: str
    max_steps: int
    learning_rate: float
    seq_len: int
    seed: int
    protocol_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("base_model_dir", self.base_model_dir),
            ("corpus_path", self.corpus_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"router healing spec {label} must be a non-empty string")
        for label, digest in (
            ("base_manifest_sha256", self.base_manifest_sha256),
            ("corpus_sha256", self.corpus_sha256),
            ("protocol_sha256", self.protocol_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"router healing spec {label} must be a 64-character sha256 hex digest"
                )
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ValueError("router healing spec max_steps must be a positive integer")
        if not isinstance(self.seq_len, int) or isinstance(self.seq_len, bool) or self.seq_len <= 0:
            raise ValueError("router healing spec seq_len must be a positive integer")
        if not (self.learning_rate > 0):
            raise ValueError("router healing spec learning_rate must be positive")

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RouterHealingArtifact:
    """A saved router delta: the only thing a healing run actually changes."""

    delta_path: str
    delta_sha256: str
    trainable_param_names: tuple[str, ...]
    trainable_param_count: int
    spec_digest: str
    steps_completed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_path": self.delta_path,
            "delta_sha256": self.delta_sha256,
            "trainable_param_names": list(self.trainable_param_names),
            "trainable_param_count": self.trainable_param_count,
            "spec_digest": self.spec_digest,
            "steps_completed": self.steps_completed,
        }


def healing_experiment_id(spec: RouterHealingSpec) -> str:
    """Stable, spec-derived id so a replayed identical run is not a new row."""
    return f"exp-router-healing-{spec.digest()[:16]}"


def build_healing_experiment(
    spec: RouterHealingSpec,
    *,
    parent_experiment_id: str | None,
    estimated_gpu_hours: float,
) -> Experiment:
    """The `Experiment` row a healing run must be anchored to.

    `estimated_gpu_hours` must be a real estimate, not a placeholder: the
    registry's GPU-hour accounting is how this program notices a run that
    quietly cost ten times what it claimed.
    """
    if not (estimated_gpu_hours > 0):
        raise RouterHealingRunError("estimated_gpu_hours must be positive for a healing experiment")
    return Experiment(
        experiment_id=healing_experiment_id(spec),
        parent_id=parent_experiment_id,
        hypothesis=Hypothesis(
            observation=(
                "The converted MoE checkpoint routes every token to every expert "
                "(num_experts_per_tok == num_experts), so conversion bought structure "
                "but no compute saving."
            ),
            suspected_cause=(
                "Router and shared-expert gate are zero-initialised by the "
                "exactness-preserving conversion, so routing carries no "
                "token-conditional signal yet."
            ),
            intervention=(
                f"Train only the router + shared-expert gate ({spec.max_steps} steps, "
                f"lr={spec.learning_rate}, seq_len={spec.seq_len}, seed={spec.seed}) "
                "with every expert weight frozen."
            ),
            expected_deltas={},
        ),
        config_patch={
            "router_healing": spec.to_dict(),
            "trains": ["mlp.gate.weight", "mlp.shared_expert_gate.weight"],
            "experts_frozen": True,
        },
        estimated_gpu_hours=float(estimated_gpu_hours),
        status=ExperimentStatus.PLANNED,
        tags=("router-healing", "qwen38-native-sparse", "phase6"),
    )


#: The metric healing actually improves. Recorded by the evaluation as the
#: model's measured `num_experts_per_tok`; MINIMIZE, so a genuine reduction
#: is the positive utility that earns promotion.
EFFICIENCY_METRIC = "num_experts_per_tok"


def build_router_healing_goal(
    *,
    gpu_hour_budget: float,
    regression_tolerance: float = 0.0,
    minimum_promotion_gain: float = 0.0,
    dimension_suite_names: Mapping[str, str],
    efficiency_weight: float = 1.0,
) -> Goal:
    """A Goal over the nine protected dimensions PLUS the efficiency metric.

    The nine capability/behavior dimensions are zero-regression guards, not
    sources of gain: healing is not expected to make the model smarter, and
    demanding that would be a made-up bar. But the gate accepts only when
    `score > minimum_promotion_gain` (strict), so a goal containing *only*
    capability metrics can never promote a healed model -- a flat,
    no-regression candidate scores exactly 0.0 and is correctly rejected as
    "did not improve enough". That is the gate being right: on those metrics
    there is no improvement.

    What healing actually buys is compute, so `EFFICIENCY_METRIC`
    (`num_experts_per_tok`, MINIMIZE) is part of the goal. A healed model
    that drops 16 experts-per-token to 3 while holding every protected
    dimension shows real positive utility and can be promoted; one that
    drops experts but breaks a dimension is still refused, because
    regressions hard-block acceptance independently of score.

    Note the score mixes a count-scale metric with 0-1 dimension scores and
    is therefore not normalised -- read it as "did something improve", not
    as a calibrated quantity. Capability protection does not depend on that
    scale: it comes from the per-metric `regression_tolerance`.

    `require_protocol_match` is always True: comparing a healed candidate
    against a baseline scored under a different protocol is exactly the
    incommensurable comparison the program forbids.

    `dimension_suite_names` maps each protected dimension to the suite-metric
    key the evaluation actually records (e.g. "reasoning" ->
    "suite-reasoning-v1"), because the gate reads metric names, not
    dimensions.
    """
    missing = [dim for dim in PARENT_DIMENSIONS if dim not in dimension_suite_names]
    if missing:
        raise RouterHealingRunError(
            f"dimension_suite_names is missing protected dimension(s): {missing}; "
            "refusing to build a goal that silently ignores part of the suite"
        )
    targets = [
        MetricTarget(
            name=dimension_suite_names[dim],
            weight=1.0,
            regression_tolerance=regression_tolerance,
        )
        for dim in PARENT_DIMENSIONS
    ]
    targets.append(
        MetricTarget(
            name=EFFICIENCY_METRIC,
            weight=efficiency_weight,
            regression_tolerance=0.0,
            direction=OptimizationDirection.MINIMIZE,
        )
    )
    return Goal(
        metrics=tuple(targets),
        gpu_hour_budget=float(gpu_hour_budget),
        minimum_promotion_gain=minimum_promotion_gain,
        require_protocol_match=True,
    )


def experiment_result_from_outcome(outcome: Any) -> ExperimentResult:
    """Bridge a persisted `evaluation_runs` row into the gate's input shape.

    Accepts anything with `experiment_id`/`metrics`/`gpu_hours`/`evidence`
    (a real `EvaluationOutcome`, or a mapping read back out of the
    registry), so both the frozen baseline and a fresh healed candidate
    reach the gate through one code path.
    """
    if isinstance(outcome, Mapping):
        experiment_id = outcome.get("experiment_id")
        metrics = outcome.get("metrics")
        gpu_hours = outcome.get("gpu_hours")
        artifact_ref = outcome.get("source_artifact_ref")
        evidence = outcome.get("evidence") or {}
    else:
        experiment_id = getattr(outcome, "experiment_id", None)
        metrics = getattr(outcome, "metrics", None)
        gpu_hours = getattr(outcome, "gpu_hours", None)
        artifact_ref = getattr(outcome, "source_artifact_ref", None)
        evidence = getattr(outcome, "evidence", None) or {}
    if not experiment_id or not metrics:
        raise RouterHealingRunError(
            "evaluation outcome lacks experiment_id/metrics; refusing to "
            "fabricate a gate input from an incomplete row"
        )
    return ExperimentResult(
        experiment_id=str(experiment_id),
        metrics={str(k): float(v) for k, v in dict(metrics).items()},
        gpu_hours=float(gpu_hours or 0.0),
        artifact_ref=str(artifact_ref) if artifact_ref else None,
        evidence=dict(evidence),
    )


def judge_router_healing(
    *,
    goal: Goal,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> GateDecision:
    """Hand the decision to the existing gate. Deliberately thin.

    This function exists so the healing path has one obvious call site for
    promotion, NOT so it can add conditions of its own. Every accept/reject
    reason comes from `gate.evaluate_candidate`.
    """
    return evaluate_candidate(goal=goal, baseline=baseline, candidate=candidate)


def save_router_delta(
    named_trainable_tensors: Mapping[str, Any],
    out_path: str | Path,
    *,
    spec: RouterHealingSpec,
    steps_completed: int,
) -> RouterHealingArtifact:
    """Persist only the trained router/shared-gate tensors, plus provenance.

    `named_trainable_tensors` maps parameter name -> a nested list of floats
    (caller converts from torch; this module stays torch-free so it can be
    unit tested on CPU without the dependency). The sidecar JSON is what
    `load_router_delta` verifies against before touching any base weights.
    """
    out = Path(out_path)
    if out.exists():
        raise RouterHealingRunError(f"router delta path already exists, refusing to overwrite: {out}")
    if not named_trainable_tensors:
        raise RouterHealingRunError("router delta is empty; nothing was trained")
    if steps_completed <= 0:
        raise RouterHealingRunError("steps_completed must be positive to save a delta")
    payload = {
        "kind": "router_healing_delta",
        "spec": spec.to_dict(),
        "spec_digest": spec.digest(),
        "steps_completed": int(steps_completed),
        "tensors": {str(name): value for name, value in named_trainable_tensors.items()},
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    count = 0
    for value in named_trainable_tensors.values():
        count += _count_leaves(value)
    return RouterHealingArtifact(
        delta_path=str(out),
        delta_sha256=digest,
        trainable_param_names=tuple(sorted(str(n) for n in named_trainable_tensors)),
        trainable_param_count=count,
        spec_digest=spec.digest(),
        steps_completed=int(steps_completed),
    )


def load_router_delta(
    delta_path: str | Path, *, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    """Read a delta back, refusing a mismatched base checkpoint.

    Applying a router delta to weights it was not trained against would
    produce a model whose provenance is a fiction, so this fails closed on
    base-manifest mismatch rather than warning.
    """
    path = Path(delta_path)
    if not path.is_file():
        raise RouterHealingRunError(f"router delta not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "router_healing_delta":
        raise RouterHealingRunError(f"{path} is not a router_healing_delta artifact")
    spec = payload.get("spec") or {}
    recorded_base = spec.get("base_manifest_sha256")
    if recorded_base != expected_base_manifest_sha256:
        raise RouterHealingRunError(
            "router delta base mismatch: delta was trained against base manifest "
            f"{recorded_base!r} but the target base is {expected_base_manifest_sha256!r}"
        )
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise RouterHealingRunError(f"{path} carries no tensors")
    return dict(payload)


def _count_leaves(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_count_leaves(item) for item in value)
    return 1
