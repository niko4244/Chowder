from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends.transformers_peft import TransformersPeftRunSpec, _min_device_vram_gb
from .executors import ExecutionContext
from .memory_preflight import _hardware_signature, estimate_memory_requirements

_CACHE_FILENAME = "optimizer_tiering.json"
_SAFETY_MARGIN_GB = 0.5
_DEFAULT_TIMEOUT_SECONDS = 300.0
# Same status as activation_offload's threshold: a documented starting
# point, not a claimed-optimal constant. Paging's real benefit is a
# capability (state CAN spill to host RAM under VRAM pressure without
# OOM-crashing), not a state-size reduction -- bnb.optim.PagedAdamW stores
# the identical exp_avg/exp_avg_sq as torch.optim.AdamW. So unlike
# activation offload, there is no "zero benefit, never recommend" case by
# size; the only cost worth weighing is the measured step-time overhead.
_MAX_ACCEPTABLE_PENALTY_RATIO = 1.2


@dataclass(frozen=True)
class OptimizerVariantMeasurement:
    """One real, measured optimizer.step() -- see
    optimizer_tiering_worker.run_experiment for how state_bytes is
    measured (direct tensor introspection, identical approach to
    memory_preflight_worker's own optimizer_state_bytes) and why timing
    needs warmup."""

    name: str
    step_seconds: float
    state_bytes: int


@dataclass(frozen=True)
class OptimizerTieringExperiment:
    """Real, measured comparison of torch.optim.AdamW (VRAM-resident
    optimizer state) against bitsandbytes' CUDA-unified-memory-paged
    optimizers (bnb.optim.PagedAdamW, bnb.optim.PagedAdamW8bit).

    `required`/`recommended` are computed against the paged_adamw variant
    specifically (the precision-preserving option) -- paged_adamw_8bit's
    numbers are still reported in `variants` for the caller's own
    reference, but its numeric-precision tradeoff is a materially
    different decision than pure memory tiering and is deliberately not
    folded into this recommendation. `required` means the combined
    real footprint (model_peak_vram_gb, from the same shared dry-run
    memory_preflight/telemetry already measure, plus the baseline AdamW's
    real state_bytes) would not fit in `per_rank_available_gb` --
    paging is the difference between fitting and OOMing in that case,
    promoted regardless of its measured step-time cost. Otherwise
    recommended only when that cost is negligible enough
    (wall_time_penalty_ratio under _MAX_ACCEPTABLE_PENALTY_RATIO) to use
    as free insurance against a workload that might grow.
    """

    device: str
    available: bool
    batch_size: int
    max_length: int
    variants: tuple[OptimizerVariantMeasurement, ...]
    model_peak_vram_gb: float
    per_rank_available_gb: float
    wall_time_penalty_ratio: float
    required: bool
    recommended: bool
    from_cache: bool = False
    reason: str | None = None

    def variant(self, name: str) -> OptimizerVariantMeasurement | None:
        return next((v for v in self.variants if v.name == name), None)


def _tiering_calibration_key(
    *, spec: TransformersPeftRunSpec, context: ExecutionContext, batch_size: int
) -> str:
    payload = {
        "base_model": spec.base_model,
        "revision": spec.revision,
        "quantization": spec.quantization,
        "precision": spec.precision,
        "max_length": spec.max_length,
        "batch_size": batch_size,
        "lora_r": spec.lora_r,
        "lora_alpha": spec.lora_alpha,
        "lora_dropout": spec.lora_dropout,
        "target_modules": list(spec.target_modules),
        "target_preset": spec.target_preset,
        "use_rslora": spec.use_rslora,
        "gradient_checkpointing": spec.gradient_checkpointing,
        "hardware": _hardware_signature(context),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / ".chowder" / _CACHE_FILENAME


def _read_cache_entry(work_dir: str | Path, key: str) -> Mapping[str, Any] | None:
    path = _cache_path(work_dir)
    if not path.is_file():
        return None
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cache, Mapping):
        return None
    entry = cache.get(key)
    return entry if isinstance(entry, Mapping) else None


def _write_cache_entry(work_dir: str | Path, key: str, entry: Mapping[str, Any]) -> None:
    path = _cache_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    cache[key] = dict(entry)
    path.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_optimizer_tiering_experiment(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    batch_size: int | None = None,
    use_cache: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> OptimizerTieringExperiment:
    """Real optimizer-state-tiering experiment for the given model+recipe
    on the given hardware. Requires bitsandbytes (chowder-ai[qlora]) and
    an available CUDA device; reports `available=False` otherwise.

    Reuses memory_preflight.estimate_memory_requirements() (itself
    already shared with telemetry.collect_runtime_telemetry(), see Phase
    7A) for the real model+activation peak-VRAM figure this experiment's
    `required` determination needs, rather than re-measuring it here.
    """
    spec = TransformersPeftRunSpec.from_resolved_config(
        resolved_config,
        work_dir=work_dir,
        output_dir=Path(work_dir) / ".chowder" / "_optimizer_tiering_scratch",
        seed=context.seed,
        hardware=context.hardware,
    )
    configured_batch_size = batch_size if batch_size is not None else spec.batch_size
    key = _tiering_calibration_key(spec=spec, context=context, batch_size=configured_batch_size)
    available_gb = _min_device_vram_gb(context.hardware)

    cached = _read_cache_entry(work_dir, key) if use_cache else None
    if cached is not None and "available" in cached:
        measured = dict(cached)
        from_cache = True
    else:
        measured = _run_tiering_worker(
            spec, batch_size=configured_batch_size, work_dir=work_dir, timeout_seconds=timeout_seconds
        )
        from_cache = False
        if use_cache:
            _write_cache_entry(work_dir, key, measured)

    if not measured.get("available", False):
        return OptimizerTieringExperiment(
            device=str(measured.get("device", "cpu")),
            available=False,
            batch_size=configured_batch_size,
            max_length=spec.max_length,
            variants=(),
            model_peak_vram_gb=0.0,
            per_rank_available_gb=available_gb,
            wall_time_penalty_ratio=0.0,
            required=False,
            recommended=False,
            from_cache=from_cache,
            reason=measured.get("reason"),
        )

    variants = tuple(
        OptimizerVariantMeasurement(
            name=name,
            step_seconds=float(entry["step_seconds"]),
            state_bytes=int(entry["state_bytes"]),
        )
        for name, entry in measured["variants"].items()
    )
    baseline = next(v for v in variants if v.name == "adamw")
    paged = next(v for v in variants if v.name == "paged_adamw")

    memory_estimate = estimate_memory_requirements(
        resolved_config=resolved_config, context=context, work_dir=work_dir, use_cache=use_cache
    )
    model_peak_vram_gb = memory_estimate.estimated_peak_gb

    combined_baseline_gb = model_peak_vram_gb + baseline.state_bytes / (1024**3)
    required = combined_baseline_gb > available_gb - _SAFETY_MARGIN_GB
    penalty_ratio = paged.step_seconds / baseline.step_seconds if baseline.step_seconds > 0 else float("inf")
    recommended = required or penalty_ratio <= _MAX_ACCEPTABLE_PENALTY_RATIO

    return OptimizerTieringExperiment(
        device=str(measured["device"]),
        available=True,
        batch_size=configured_batch_size,
        max_length=spec.max_length,
        variants=variants,
        model_peak_vram_gb=model_peak_vram_gb,
        per_rank_available_gb=available_gb,
        wall_time_penalty_ratio=penalty_ratio,
        required=required,
        recommended=recommended,
        from_cache=from_cache,
    )


def _run_tiering_worker(
    spec: TransformersPeftRunSpec, *, batch_size: int, work_dir: str | Path, timeout_seconds: float
) -> dict[str, Any]:
    scratch_dir = Path(work_dir) / ".chowder" / "_optimizer_tiering_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spec_path = scratch_dir / "spec.json"
    result_path = scratch_dir / "result.json"
    spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "chowder.backends.optimizer_tiering_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
        "--batch-size",
        str(batch_size),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    if process.returncode != 0:
        raise RuntimeError(
            f"optimizer tiering experiment worker failed with exit code {process.returncode}:\n"
            f"{process.stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise RuntimeError("optimizer tiering experiment worker exited successfully without a result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise RuntimeError("optimizer tiering experiment worker result is not a JSON object")
    return dict(result)
