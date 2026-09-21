"""A normalized intervention/outcome view over the run registry.

The Meta-controller (roadmap Priority 6) needs one durable, queryable table
of "what was tried, on what, at what cost, and what happened" before any
expected-improvement model or learned policy can be honestly evaluated
against held-out experiments. This module is that view and nothing more:
it joins evidence the registry *already* stores immutably -- it never runs
an experiment, never scores a not-yet-run candidate, and never predicts.

Three registry tables are joined per row:

  experiments   -> lineage, `config_patch` (the intervention), the
                   hypothesis text, and the persisted gate outcome
                   (`ExperimentStatus`)
  results       -> the real evaluation metrics and lifecycle GPU-hours
  training_runs -> the real training artifact's telemetry/evidence
                   (peak VRAM, wall time, base model, resolved Memory
                   Fabric mechanisms)

Only experiments that actually ran to a persisted `ExperimentResult` become
rows. An experiment that was preflight-rejected, cancelled, or crashed
during training never produced a scored outcome, so it is skipped entirely
rather than represented as a row full of `None` (see `censored_outcomes`
for the parallel view over exactly those).

Honesty rule
------------
Every field is read from real stored evidence, or is `None`. Nothing here
is imputed, defaulted, or estimated. The fields that are frequently `None`,
and exactly why:

  `training_run_id`, `peak_vram_gb`, `train_runtime_seconds`,
  `global_step`, `training_gpu_hours`
      All come from the joined `training_runs` row. `None` when no
      training artifact could be joined for this experiment (see
      `_join_artifact` for the join rule), and the three telemetry fields
      are additionally `None` when the joined artifact's telemetry simply
      does not carry that key -- a training backend is free to report
      whatever telemetry it measured, and only the real transformers-peft
      worker reports these three.

  `base_model`, `base_model_revision`, `recipe_sha256`,
  `min_device_vram_gb`, `memory_fabric_mechanisms`
      Read out of the training artifact's `evidence` mapping, whose shape
      is set by whichever training backend produced it. The real
      transformers-peft executor writes all of them; any other executor
      (including every hand-written test double) generally writes none, so
      these are `None` for such runs. They are also `None` when no
      training evidence could be located at all.

  `dataset_sha256`, `replay_dataset_sha256`
      The content digest of the dataset the run actually trained on (and
      of the replay selection, when one existed), read from the training
      evidence. Both real executors (transformers-peft and unsloth)
      verify the dataset on disk and record the digest they trained on,
      so these are the dataset *identity* half of the Priority-6
      "dataset context" gap: two runs shared training data exactly when
      the recorded digests match. `None` for an executor that does not
      record a digest, or when no training evidence could be located.
      No dataset path or file name exists in this view because the
      registry stores none -- a content digest is what "same data" means
      across runs.

  `dataset_format`, `primary_rows`, `replay_selected_rows`,
  `total_token_count`, `assistant_token_count`
      Dataset *scale and shape*, read from the training evidence's
      `data_provenance` block, which only the real transformers-peft
      worker writes. `None` for every other backend, for a run with no
      joined training evidence, and when a recorded block simply does
      not carry that key.

  `training_engine`
      Read from `evidence["engine"]` ("transformers" or "unsloth"), the
      literal Priority-6-roadmap ask that this evidence view be able to
      distinguish training engines. `None` for evidence recorded before
      the Transformers executor started writing this key (it always wrote
      "backend": "transformers-peft" but not a separate "engine" key) or
      for any other executor that does not record it -- not inferred from
      `backend`, so an older run's real absence of this evidence stays
      visible rather than being backfilled by assumption.

  `visible_accelerator_count`, `requested_active_accelerator_count`,
  `peak_vram_gb_by_accelerator`
      Hardware context beyond the single `active_accelerator_count`
      number: how many accelerators the run could even see (both real
      executors record it in `resource_usage`), how many the experiment
      request asked to use (transformers-peft evidence only), and the
      measured per-accelerator peak-VRAM map. The map is `None` unless
      the evidence's `resource_usage` block carries a well-formed one --
      the real executors always do; a block with any malformed entry is
      recorded as `None` in full rather than served partially. A row can
      legitimately show `active_accelerator_count=1` against
      `visible_accelerator_count=2`: the box has two GPUs and this run
      used one, which is exactly the multi-GPU telemetry context the
      roadmap flagged as missing from this view.

  `gate_accepted`
      `True`/`False` only when the experiment's persisted status is
      `PASSED`/`REJECTED` -- the status `EvolutionEngine.adjudicate()`
      writes from the real `GateDecision`. `None` for a result-bearing
      experiment left at `PLANNED`/`RUNNING`/`FAILED`, which means the
      gate outcome was never persisted for it; it is never guessed from
      the metrics.

  `gate_score_vs_parent`
      `None` when the experiment has no parent, or the parent has no
      persisted `ExperimentResult` to serve as a reference point. There is
      no substitute baseline and none is invented.

Two fields are *derived* rather than read, and both are derived only from
real stored numbers through the same hard gate every real candidate goes
through (`gate.evaluate_candidate`): `gate_score_vs_baseline` and
`gate_score_vs_parent`. The gate's weighted score is not itself persisted
anywhere -- it is meaningless without a reference point, so the caller
supplies the real `goal`/`baseline` and this module replays the real gate,
exactly as `candidate_selection.prioritize_candidates` already does for
its own bandit history.

Deliberately absent: any throughput *rate*. `train_runtime_seconds` and
`global_step` are the two raw measured numbers the registry actually
stores; a steps-per-second figure would be this module's arithmetic, not
stored evidence, so it is left to the caller. Likewise absent: any dataset
path or file name (the registry stores content digests, not locations),
any per-row or per-token derived rate, and any dataset-vs-replay mixture
ratio beyond the two digests and counts themselves -- all arithmetic a
caller can apply to the stored numbers, and none of it stored evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_selection import dotted_paths
from .executors import TrainingArtifact
from .gate import evaluate_candidate
from .models import ExperimentResult, ExperimentStatus, Goal
from .registry import RunRegistry

# The three Memory Fabric mechanisms the real transformers-peft executor
# records as `resolved_*` booleans in its training evidence. Kept as a
# literal tuple rather than imported from placement_policy so that reading
# historical evidence never drags in the experiment-running machinery
# (placement_policy imports the executor, which imports torch-adjacent code).
_MEMORY_FABRIC_MECHANISMS = (
    "activation_offload",
    "optimizer_tiering",
    "frozen_layer_streaming",
)


@dataclass(frozen=True)
class InterventionOutcome:
    """One historical experiment that actually ran, normalized.

    See this module's docstring for the honesty rule and for which fields
    are frequently `None` and why. Every `| None` below means "this
    evidence is genuinely not present for this run", never "unknown, so
    we picked a default".
    """

    # Identity / lineage -- always present (schema columns).
    experiment_id: str
    parent_id: str | None

    # The intervention.
    config_patch: Mapping[str, Any]
    arm: frozenset[str]
    intervention: str

    # What it ran on. The dataset fields are the Priority-6 "dataset
    # context" ask; the accelerator fields beyond `active_accelerator_count`
    # are the "hardware context" ask. All read from stored evidence.
    training_run_id: str | None
    training_engine: str | None
    base_model: str | None
    base_model_revision: str | None
    recipe_sha256: str | None
    dataset_sha256: str | None
    replay_dataset_sha256: str | None
    dataset_format: str | None
    primary_rows: int | None
    replay_selected_rows: int | None
    total_token_count: int | None
    assistant_token_count: int | None
    min_device_vram_gb: float | None
    active_accelerator_count: int | None
    visible_accelerator_count: int | None
    requested_active_accelerator_count: int | None
    peak_vram_gb_by_accelerator: Mapping[str, float] | None
    memory_fabric_mechanisms: frozenset[str] | None

    # What it cost.
    gpu_hours: float
    training_gpu_hours: float | None

    # What happened.
    metrics: Mapping[str, float]
    gate_accepted: bool | None
    gate_score_vs_baseline: float
    gate_score_vs_parent: float | None
    regressions_vs_baseline: Mapping[str, float]
    peak_vram_gb: float | None
    train_runtime_seconds: float | None
    global_step: int | None


def _number(value: object) -> float | None:
    """A stored JSON value as a float, or None if it is not a real number.

    Telemetry/evidence mappings are `float | int | str`-valued by contract
    (`TrainingArtifact.telemetry`), and a backend may legitimately store
    `None` for a phase it did not measure -- neither is coerced into a
    fabricated number here. `bool` is excluded explicitly: it is an `int`
    subclass in Python, and a flag is not a measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    """A stored JSON value as an int, or None. See `_number` for the rules."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _string(value: object) -> str | None:
    """A stored JSON value as a non-empty string, or None.

    See `_number` for the honesty rules. A whitespace-only string is
    treated as not recorded rather than preserved: it names nothing.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _digest(value: object) -> str | None:
    """A stored JSON value as a SHA-256 hex digest string, or None.

    Both real executors validate `dataset_sha256` to exactly this shape
    before storing it, so anything else in that slot means the evidence
    was not written by the real pipeline; serving it as a digest would
    fabricate a join key that no real run produced. A passing value is
    served verbatim -- never trimmed or case-normalized -- so what this
    view returns is byte-for-byte what the run stored.
    """
    if not isinstance(value, str) or len(value) != 64:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value


def _number_mapping(value: object) -> Mapping[str, float] | None:
    """A stored JSON object of finite non-negative numbers, or None.

    Used for `resource_usage["peak_vram_gb_by_accelerator"]`, whose real
    shape is validated by `ResourceUsage.__post_init__` before the
    executor ever stores it. Any malformed entry (bool, string, negative,
    non-finite, non-string key, empty key) poisons the whole block: the
    evidence is then not the shape this view promises to serve, and `None`
    records that honestly rather than silently dropping the bad entries
    and passing the remainder off as complete. An empty mapping is a real
    answer ("no per-accelerator peaks were recorded") and is kept.
    """
    if not isinstance(value, Mapping):
        return None
    cleaned: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            return None
        number = _number(item)
        if number is None or number < 0 or not math.isfinite(number):
            return None
        cleaned[key] = number
    return cleaned


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _join_artifact(
    result: ExperimentResult,
    *,
    by_run_id: Mapping[str, TrainingArtifact],
    by_experiment_id: Mapping[str, Sequence[TrainingArtifact]],
) -> TrainingArtifact | None:
    """The training artifact this result was actually evaluated from.

    `cycle.ExperimentCycleRunner` records the exact producing run under
    `evidence["training_run_id"]`, so that is the primary join key. When it
    is absent (a registry populated by something other than the cycle
    runner), a single unambiguous artifact for the experiment is used
    instead. Several artifacts and no recorded run id is genuinely
    ambiguous -- which one produced this result is not stored -- so it
    joins nothing rather than picking one.
    """
    run_id = result.evidence.get("training_run_id")
    if isinstance(run_id, str):
        artifact = by_run_id.get(run_id)
        if artifact is not None:
            return artifact
    candidates = by_experiment_id.get(result.experiment_id, ())
    return candidates[0] if len(candidates) == 1 else None


def _memory_fabric_mechanisms(training_evidence: Mapping[str, Any]) -> frozenset[str] | None:
    """Which Memory Fabric mechanisms were actually resolved on for this run.

    An empty frozenset is a real answer ("evidence says none were enabled")
    and is deliberately distinct from `None` ("this run recorded no Memory
    Fabric resolution at all"). All three `resolved_*` keys must be present
    -- the real executor always writes them together, and a partial block
    could not distinguish "off" from "not recorded" for the missing ones.
    """
    defaults = _mapping(training_evidence.get("hardware_aware_defaults"))
    keys = tuple(f"resolved_{name}" for name in _MEMORY_FABRIC_MECHANISMS)
    if not all(key in defaults for key in keys):
        return None
    return frozenset(
        name for name, key in zip(_MEMORY_FABRIC_MECHANISMS, keys) if bool(defaults[key])
    )


def _gate_accepted(status: ExperimentStatus) -> bool | None:
    """The persisted gate verdict, or None when one was never persisted.

    `EvolutionEngine.adjudicate()` writes `PASSED`/`REJECTED` straight from
    the real `GateDecision.accepted`, and `cycle.run_round` syncs that to
    the registry. Any other status on a result-bearing experiment means the
    gate outcome is simply not on record.
    """
    if status is ExperimentStatus.PASSED:
        return True
    if status is ExperimentStatus.REJECTED:
        return False
    return None


def build_intervention_outcomes(
    registry: RunRegistry,
    *,
    goal: Goal,
    baseline: ExperimentResult,
) -> tuple[InterventionOutcome, ...]:
    """Assemble the normalized intervention/outcome view from *registry*.

    Rows come back in the order the experiments were recorded
    (`RunRegistry.list_experiments()` is `ORDER BY rowid`), so this is
    deterministic for a given database.

    *goal* and *baseline* are the real reference point the gate score is
    computed against -- the same pair `candidate_selection` and
    `tournament` take. They are used only to replay `gate.evaluate_
    candidate` over already-stored metrics; nothing is promoted, ranked,
    or re-run.
    """
    results = {result.experiment_id: result for result in registry.list_results()}
    by_run_id: dict[str, TrainingArtifact] = {}
    by_experiment_id: dict[str, list[TrainingArtifact]] = defaultdict(list)
    for artifact in registry.list_training_artifacts():
        by_run_id[artifact.run_id] = artifact
        by_experiment_id[artifact.experiment_id].append(artifact)

    rows: list[InterventionOutcome] = []
    for experiment in registry.list_experiments():
        result = results.get(experiment.experiment_id)
        if result is None:
            # Never ran to a scored outcome -- no row, rather than a row of None.
            continue

        artifact = _join_artifact(
            result, by_run_id=by_run_id, by_experiment_id=by_experiment_id
        )
        if artifact is not None:
            training_evidence = _mapping(artifact.evidence)
            telemetry = _mapping(artifact.telemetry)
        else:
            # cycle.py nests the producing artifact's own evidence here, so a
            # result recorded without its artifact row still carries it.
            # Telemetry is never nested into the result, so it stays empty.
            training_evidence = _mapping(_mapping(result.evidence).get("training"))
            telemetry = {}

        provenance = _mapping(training_evidence.get("model_provenance"))
        base_model = provenance.get("requested_base_model")
        recipe_sha256 = training_evidence.get("recipe_sha256")
        resource_usage = _mapping(training_evidence.get("resource_usage"))
        hardware_defaults = _mapping(training_evidence.get("hardware_aware_defaults"))
        data_provenance = _mapping(training_evidence.get("data_provenance"))

        decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=result)
        parent_result = (
            results.get(experiment.parent_id) if experiment.parent_id is not None else None
        )
        gate_score_vs_parent = (
            evaluate_candidate(goal=goal, baseline=parent_result, candidate=result).score
            if parent_result is not None
            else None
        )

        rows.append(
            InterventionOutcome(
                experiment_id=experiment.experiment_id,
                parent_id=experiment.parent_id,
                config_patch=dict(experiment.config_patch),
                arm=dotted_paths(experiment.config_patch),
                intervention=experiment.hypothesis.intervention,
                training_run_id=artifact.run_id if artifact is not None else None,
                training_engine=(
                    training_evidence.get("engine")
                    if isinstance(training_evidence.get("engine"), str)
                    else None
                ),
                base_model=base_model if isinstance(base_model, str) else None,
                base_model_revision=_string(provenance.get("requested_revision")),
                recipe_sha256=recipe_sha256 if isinstance(recipe_sha256, str) else None,
                dataset_sha256=_digest(training_evidence.get("dataset_sha256")),
                replay_dataset_sha256=_digest(
                    training_evidence.get("replay_dataset_sha256")
                ),
                dataset_format=_string(data_provenance.get("dataset_format")),
                primary_rows=_integer(data_provenance.get("primary_rows")),
                replay_selected_rows=_integer(data_provenance.get("replay_selected_rows")),
                total_token_count=_integer(data_provenance.get("total_token_count")),
                assistant_token_count=_integer(
                    data_provenance.get("assistant_token_count")
                ),
                min_device_vram_gb=_number(hardware_defaults.get("min_device_vram_gb")),
                active_accelerator_count=_integer(
                    resource_usage.get("active_accelerator_count")
                ),
                visible_accelerator_count=_integer(
                    resource_usage.get("visible_accelerator_count")
                ),
                requested_active_accelerator_count=_integer(
                    training_evidence.get("requested_active_accelerator_count")
                ),
                peak_vram_gb_by_accelerator=_number_mapping(
                    resource_usage.get("peak_vram_gb_by_accelerator")
                ),
                memory_fabric_mechanisms=_memory_fabric_mechanisms(training_evidence),
                gpu_hours=result.gpu_hours,
                training_gpu_hours=artifact.gpu_hours if artifact is not None else None,
                metrics=dict(result.metrics),
                gate_accepted=_gate_accepted(experiment.status),
                gate_score_vs_baseline=decision.score,
                gate_score_vs_parent=gate_score_vs_parent,
                regressions_vs_baseline=dict(decision.regressions),
                peak_vram_gb=_number(telemetry.get("peak_vram_gb")),
                train_runtime_seconds=_number(telemetry.get("train_runtime_seconds")),
                global_step=_integer(telemetry.get("global_step")),
            )
        )
    return tuple(rows)


def filter_outcomes(
    outcomes: Sequence[InterventionOutcome],
    *,
    base_model: str | None = None,
    training_engine: str | None = None,
    dataset_sha256: str | None = None,
    touches_key_path: str | None = None,
    gate_accepted: bool | None = None,
    min_score_vs_baseline: float | None = None,
) -> tuple[InterventionOutcome, ...]:
    """Filter *outcomes*; every criterion given is ANDed, order preserved.

    `touches_key_path` matches rows whose intervention touched that exact
    dotted `config_patch` key-path (see `candidate_selection.dotted_paths`).

    `gate_accepted=True`/`False` matches only rows whose persisted gate
    verdict is exactly that -- a row whose verdict was never persisted
    (`gate_accepted is None`) is excluded by either value, and is reachable
    only by not passing this criterion at all. That is deliberate: "not on
    record" is not evidence of rejection.

    `training_engine` matches only rows whose evidence recorded that exact
    engine string -- a row with no recorded engine (`training_engine is
    None`) is excluded by either value, same "not on record" rule as
    `gate_accepted`.

    `dataset_sha256` matches only rows whose training evidence recorded
    that exact dataset digest -- the same-dataset selector the
    Priority-6 dataset context enables. A row whose evidence records no
    digest (`dataset_sha256 is None`) is excluded by either value, same
    "not on record" rule.
    """
    selected = tuple(outcomes)
    if base_model is not None:
        selected = tuple(row for row in selected if row.base_model == base_model)
    if training_engine is not None:
        selected = tuple(row for row in selected if row.training_engine == training_engine)
    if dataset_sha256 is not None:
        selected = tuple(row for row in selected if row.dataset_sha256 == dataset_sha256)
    if touches_key_path is not None:
        selected = tuple(row for row in selected if touches_key_path in row.arm)
    if gate_accepted is not None:
        selected = tuple(row for row in selected if row.gate_accepted is gate_accepted)
    if min_score_vs_baseline is not None:
        selected = tuple(
            row for row in selected if row.gate_score_vs_baseline >= min_score_vs_baseline
        )
    return selected


def group_by_arm(
    outcomes: Sequence[InterventionOutcome],
) -> dict[frozenset[str], tuple[InterventionOutcome, ...]]:
    """Group *outcomes* by intervention arm, preserving input order.

    The arm is the frozenset of dotted `config_patch` key-paths an
    experiment touched -- the same arm definition
    `candidate_selection.prioritize_candidates` already bandits over, via
    the same `dotted_paths` helper, so an arm identified here is the same
    arm identified there.

    Deliberately returns the grouped rows themselves and no summary
    statistic. Any per-arm expected improvement is a modelling question
    belonging to a later roadmap slice, and computing a mean here would
    quietly pre-empt it with an unvalidated one.
    """
    grouped: dict[frozenset[str], list[InterventionOutcome]] = defaultdict(list)
    for row in outcomes:
        grouped[row.arm].append(row)
    return {arm: tuple(rows) for arm, rows in grouped.items()}
