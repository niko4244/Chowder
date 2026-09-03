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
from .memory_preflight import _hardware_signature

_CACHE_FILENAME = "activation_offload.json"
_SAFETY_MARGIN_GB = 0.5
_DEFAULT_TIMEOUT_SECONDS = 300.0
# A starting point, not a measured constant: "acceptable" throughput cost
# is inherently a judgment call. 20% wall-time overhead is a conservative
# bar for a technique that, per the real measurements this module is built
# on, can easily cost 5-30x on a workload with little real activation
# pressure to relieve -- this threshold exists so a merely-not-required
# recommendation still means something, not to claim it's the "correct"
# number for every workload.
_MAX_ACCEPTABLE_PENALTY_RATIO = 1.2


@dataclass(frozen=True)
class ActivationOffloadExperiment:
    """Real, measured comparison of one forward+backward pass with vs.
    without CPU activation offload (`torch.autograd.graph.saved_tensors_hooks`).
    Every *_gb/_seconds field here comes from an actual measured run, not a
    formula -- see activation_offload_worker.run_experiment for why
    wall-clock timing specifically requires warmup to mean anything.

    `required` and `recommended` follow the roadmap's own rule: "only
    promote when required or throughput acceptable". Required means the
    baseline configuration alone would not fit in `per_rank_available_gb`
    (offload is fitting-or-nothing, not optional, in that case).
    Recommended additionally allows the merely-not-required case when the
    measured wall_time_penalty_ratio stays under
    _MAX_ACCEPTABLE_PENALTY_RATIO -- never recommended when it produces no
    real VRAM savings, regardless of timing.
    """

    device: str
    available: bool
    batch_size: int
    max_length: int
    baseline_peak_vram_gb: float
    offload_peak_vram_gb: float
    vram_saved_gb: float
    baseline_wall_seconds: float
    offload_wall_seconds: float
    wall_time_penalty_ratio: float
    per_rank_available_gb: float
    required: bool
    recommended: bool
    from_cache: bool = False
    reason: str | None = None


def _offload_calibration_key(
    *, spec: TransformersPeftRunSpec, context: ExecutionContext, batch_size: int
) -> str:
    """Unlike memory_preflight's calibration key, this one includes
    batch_size: the offload experiment measures at one specific batch
    size (the caller's configured one, or an explicit override), not a
    batch-size-independent pair extrapolated afterward."""
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


def run_activation_offload_experiment(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    batch_size: int | None = None,
    use_cache: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> ActivationOffloadExperiment:
    """Real activation-offload experiment for the given model+recipe on
    the given hardware: a genuine forward+backward pass measured with and
    without CPU activation offload, in an isolated subprocess (the same
    isolation discipline every dry-run worker in this codebase uses).

    `batch_size` defaults to the resolved config's own configured batch
    size -- pass an explicit value to probe a different one without
    editing the config. Cached per (model+recipe+hardware+batch_size)
    combination under work_dir/.chowder/activation_offload.json.
    """
    spec = TransformersPeftRunSpec.from_resolved_config(
        resolved_config,
        work_dir=work_dir,
        output_dir=Path(work_dir) / ".chowder" / "_activation_offload_scratch",
        seed=context.seed,
        hardware=context.hardware,
    )
    configured_batch_size = batch_size if batch_size is not None else spec.batch_size
    key = _offload_calibration_key(spec=spec, context=context, batch_size=configured_batch_size)
    available_gb = _min_device_vram_gb(context.hardware)

    cached = _read_cache_entry(work_dir, key) if use_cache else None
    if cached is not None and "available" in cached:
        measured = dict(cached)
        from_cache = True
    else:
        measured = _run_offload_worker(
            spec, batch_size=configured_batch_size, work_dir=work_dir, timeout_seconds=timeout_seconds
        )
        from_cache = False
        if use_cache:
            _write_cache_entry(work_dir, key, measured)

    if not measured.get("available", False):
        return ActivationOffloadExperiment(
            device=str(measured.get("device", "cpu")),
            available=False,
            batch_size=configured_batch_size,
            max_length=spec.max_length,
            baseline_peak_vram_gb=0.0,
            offload_peak_vram_gb=0.0,
            vram_saved_gb=0.0,
            baseline_wall_seconds=0.0,
            offload_wall_seconds=0.0,
            wall_time_penalty_ratio=0.0,
            per_rank_available_gb=available_gb,
            required=False,
            recommended=False,
            from_cache=from_cache,
            reason=measured.get("reason"),
        )

    baseline_peak = float(measured["baseline_peak_vram_gb"])
    offload_peak = float(measured["offload_peak_vram_gb"])
    baseline_wall = float(measured["baseline_wall_seconds"])
    offload_wall = float(measured["offload_wall_seconds"])
    vram_saved = baseline_peak - offload_peak
    penalty_ratio = offload_wall / baseline_wall if baseline_wall > 0 else float("inf")
    required = baseline_peak > available_gb - _SAFETY_MARGIN_GB
    recommended = vram_saved > 0 and (required or penalty_ratio <= _MAX_ACCEPTABLE_PENALTY_RATIO)

    return ActivationOffloadExperiment(
        device=str(measured["device"]),
        available=True,
        batch_size=configured_batch_size,
        max_length=spec.max_length,
        baseline_peak_vram_gb=baseline_peak,
        offload_peak_vram_gb=offload_peak,
        vram_saved_gb=vram_saved,
        baseline_wall_seconds=baseline_wall,
        offload_wall_seconds=offload_wall,
        wall_time_penalty_ratio=penalty_ratio,
        per_rank_available_gb=available_gb,
        required=required,
        recommended=recommended,
        from_cache=from_cache,
    )


def _run_offload_worker(
    spec: TransformersPeftRunSpec, *, batch_size: int, work_dir: str | Path, timeout_seconds: float
) -> dict[str, Any]:
    scratch_dir = Path(work_dir) / ".chowder" / "_activation_offload_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spec_path = scratch_dir / "spec.json"
    result_path = scratch_dir / "result.json"
    spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "chowder.backends.activation_offload_worker",
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
            f"activation offload experiment worker failed with exit code {process.returncode}:\n"
            f"{process.stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise RuntimeError("activation offload experiment worker exited successfully without a result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise RuntimeError("activation offload experiment worker result is not a JSON object")
    return dict(result)
