"""Real, SIMULTANEOUS measurement of two or three Memory Fabric mechanisms
enabled together in one real training run.

`placement_policy.py::build_placement_plan()` predicts a combination's
effect by summing each mechanism's own INDEPENDENT experiment (each
compares its own mechanism alone against a resident baseline; no experiment
anywhere in this codebase has ever run two of them simultaneously against
each other). This module closes that gap and answers, with real hardware
evidence: does the additive assumption hold?

Reuses the real production training path (`TransformersPeftExecutor.run()`,
the same entrypoint every real training run in this codebase goes through)
rather than inventing new dual/triple-mechanism measurement code: a
"combined" run is just a real training run with more than one mechanism's
`resolved_*` flag set to `"always"` simultaneously -- something
`transformers_worker.py` already supports independently per mechanism, just
never previously exercised with more than one on at once.

A real finding from this module's own development (tiny CI-safe smoke
model, batch_size=32, max_length=512, activation_offload +
frozen_layer_streaming together): the combined run showed a genuinely
~2 GB real `activation_offload_bytes_transferred` and real
`frozen_layer_streaming_bytes_transferred`, yet **zero net peak-VRAM
reduction** relative to the resident baseline, alongside a real,
order-independent ~1.65x wall-time penalty. The naive additive prediction
(each mechanism's own isolated forward+backward experiment) would have
predicted a real reduction -- a full `Trainer.train()` run's peak VRAM can
be dominated by fixed overhead (model load, optimizer/allocator
bookkeeping) that occurs at a different point in the run than either
mechanism's own savings would show, at small model scale. This is exactly
the kind of gap `build_placement_plan()` cannot see on its own, and exactly
why this module exists.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .activation_offload import run_activation_offload_experiment
from .backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from .executors import ExecutionContext
from .frozen_layer_streaming import run_frozen_layer_streaming_experiment
from .memory_preflight import _hardware_signature
from .models import Experiment, Hypothesis
from .optimizer_tiering import run_optimizer_tiering_experiment
from .placement_policy import _MECHANISM_NAMES, _mechanism_savings_gb

_DEFAULT_MAX_STEPS = 4
_DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class CombinedMechanismExperiment:
    """One real baseline run (every mechanism off) and one real combined
    run (every mechanism in `mechanisms` set to `"always"` simultaneously),
    both through the real production executor, compared against what
    naively summing each mechanism's own independent experiment predicted.

    `prediction_error_gb` is `predicted - actual`: positive means the
    additive assumption over-predicted real savings (the combined run used
    more real VRAM than summing each mechanism's own savings implied);
    negative means it under-predicted.
    """

    experiment_key: str
    mechanisms: tuple[str, ...]
    baseline_peak_vram_gb: float
    predicted_combined_peak_vram_gb: float
    actual_combined_peak_vram_gb: float
    prediction_error_gb: float
    baseline_wall_seconds: float
    combined_wall_seconds: float
    wall_time_penalty_ratio: float
    per_mechanism_predicted_savings_gb: Mapping[str, float]
    forward_seconds: float | None
    backward_seconds: float | None
    optimizer_seconds: float | None
    avg_gpu_utilization_percent: float | None
    optimizer_state_bytes: float | None
    frozen_layer_streaming_bytes_transferred: float | None
    activation_offload_bytes_transferred: float | None


def _combined_calibration_key(
    *, mechanisms: tuple[str, ...], spec: "TransformersPeftRunSpec", max_steps: int, context: ExecutionContext
) -> str:
    payload = {
        "mechanisms": sorted(mechanisms),
        "base_model": spec.base_model,
        "revision": spec.revision,
        "quantization": spec.quantization,
        "precision": spec.precision,
        "max_length": spec.max_length,
        "batch_size": spec.batch_size,
        "lora_r": spec.lora_r,
        "lora_alpha": spec.lora_alpha,
        "max_steps": max_steps,
        "hardware": _hardware_signature(context),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _config_with_mechanisms(
    resolved_config: Mapping[str, Any], *, mechanisms: frozenset[str], max_steps: int
) -> dict[str, Any]:
    """A deep copy of `resolved_config` with every Memory Fabric mechanism
    explicitly set (`"always"` for the chosen combination, `"off"` for the
    rest) and `detailed_timing_telemetry` turned on, so the baseline and
    combined runs are apples-to-apples: the real ~17% measured
    timing-telemetry overhead documented in `transformers_worker.py` must
    apply equally to both runs, or a wall-time comparison between them
    would be measuring telemetry overhead, not the mechanisms' own cost.
    """
    config = deepcopy(dict(resolved_config))
    backend = dict(config.get("backend", {}))
    training = dict(backend.get("training", {}))
    for name in _MECHANISM_NAMES:
        training[name] = "always" if name in mechanisms else "off"
    training["detailed_timing_telemetry"] = True
    training["max_steps"] = max_steps
    training["save_strategy"] = "no"
    backend["training"] = training
    config["backend"] = backend
    return config


def run_combined_mechanism_experiment(
    *,
    mechanisms: tuple[str, ...],
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    max_steps: int = _DEFAULT_MAX_STEPS,
    mechanism_experiment_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> CombinedMechanismExperiment:
    """Run one real baseline and one real combined training run and
    compare the combined run's real measured peak VRAM/timing against
    what naively summing each mechanism's own independent experiment
    would have predicted.

    A throwaway warmup run is NOT performed here -- unlike this module's
    own hands-on verification (which needed one to separate real
    combined-mechanism cost from one-time CUDA/cuDNN cold-start cost),
    baseline and combined here are each single real runs. Callers wanting
    an order-independent, warm comparison should run this function once
    to prime CUDA before treating a subsequent call's numbers as final --
    the same caution the individual single-mechanism experiment workers
    document for their own timed iterations.
    """
    mechanisms = tuple(mechanisms)
    unknown = set(mechanisms) - set(_MECHANISM_NAMES)
    if unknown:
        raise ValueError(f"unknown mechanism(s): {sorted(unknown)}")
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("mechanisms must not repeat")
    if len(mechanisms) < 2:
        raise ValueError("a combined-mechanism experiment needs at least 2 mechanisms")

    mechanism_set = frozenset(mechanisms)
    baseline_config = _config_with_mechanisms(resolved_config, mechanisms=frozenset(), max_steps=max_steps)
    combined_config = _config_with_mechanisms(resolved_config, mechanisms=mechanism_set, max_steps=max_steps)

    spec = TransformersPeftRunSpec.from_resolved_config(
        combined_config,
        work_dir=work_dir,
        output_dir=Path(work_dir) / ".chowder" / "_combined_mechanism_scratch",
        seed=context.seed,
        hardware=context.hardware,
    )
    key = _combined_calibration_key(mechanisms=mechanisms, spec=spec, max_steps=max_steps, context=context)

    trainer = TransformersPeftExecutor()
    baseline_experiment = Experiment(
        f"combined-baseline-{key[:16]}",
        None,
        Hypothesis("measuring resident baseline", "no Memory Fabric intervention", "run as-is"),
        {},
        1.0,
    )
    combined_experiment = Experiment(
        f"combined-{key[:16]}",
        None,
        Hypothesis(
            f"measuring {'+'.join(sorted(mechanisms))} simultaneously",
            "the additive-combination assumption is unverified for this pair/triple",
            "enable every listed mechanism at once in one real run",
        ),
        {},
        1.0,
    )
    baseline_artifact = trainer.run(baseline_experiment, replace(context, resolved_config=baseline_config))
    combined_artifact = trainer.run(combined_experiment, replace(context, resolved_config=combined_config))

    baseline_peak = float(baseline_artifact.telemetry["peak_vram_gb"])
    actual_combined_peak = float(combined_artifact.telemetry["peak_vram_gb"])
    baseline_wall = float(baseline_artifact.telemetry["train_runtime_seconds"])
    combined_wall = float(combined_artifact.telemetry["train_runtime_seconds"])

    activation_offload_exp = run_activation_offload_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir,
        timeout_seconds=mechanism_experiment_timeout_seconds,
    )
    optimizer_tiering_exp = run_optimizer_tiering_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir,
        timeout_seconds=mechanism_experiment_timeout_seconds,
    )
    frozen_layer_streaming_exp = run_frozen_layer_streaming_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir,
        timeout_seconds=mechanism_experiment_timeout_seconds,
    )
    per_mechanism_savings = _mechanism_savings_gb(
        activation_offload_exp=activation_offload_exp,
        optimizer_tiering_exp=optimizer_tiering_exp,
        frozen_layer_streaming_exp=frozen_layer_streaming_exp,
    )
    predicted_combined_peak = baseline_peak - sum(per_mechanism_savings[name] for name in mechanisms)

    return CombinedMechanismExperiment(
        experiment_key=key,
        mechanisms=tuple(sorted(mechanisms)),
        baseline_peak_vram_gb=baseline_peak,
        predicted_combined_peak_vram_gb=predicted_combined_peak,
        actual_combined_peak_vram_gb=actual_combined_peak,
        prediction_error_gb=predicted_combined_peak - actual_combined_peak,
        baseline_wall_seconds=baseline_wall,
        combined_wall_seconds=combined_wall,
        wall_time_penalty_ratio=combined_wall / baseline_wall if baseline_wall > 0 else float("inf"),
        per_mechanism_predicted_savings_gb={name: per_mechanism_savings[name] for name in mechanisms},
        forward_seconds=combined_artifact.telemetry.get("forward_seconds"),
        backward_seconds=combined_artifact.telemetry.get("backward_seconds"),
        optimizer_seconds=combined_artifact.telemetry.get("optimizer_seconds"),
        avg_gpu_utilization_percent=combined_artifact.telemetry.get("avg_gpu_utilization_percent"),
        optimizer_state_bytes=combined_artifact.telemetry.get("optimizer_state_bytes"),
        frozen_layer_streaming_bytes_transferred=combined_artifact.telemetry.get(
            "frozen_layer_streaming_bytes_transferred"
        ),
        activation_offload_bytes_transferred=combined_artifact.telemetry.get(
            "activation_offload_bytes_transferred"
        ),
    )
