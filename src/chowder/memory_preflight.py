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

_CACHE_FILENAME = "memory_calibration.json"
_SAFETY_MARGIN_GB = 0.5
_DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class MemoryEstimate:
    """A real, measured (not theoretical) VRAM estimate for one base_model +
    recipe + hardware combination, extrapolated from two tiny real
    forward+backward passes to the configured batch size.

    `per_rank_available_gb` is deliberately the smallest single accelerator's
    VRAM, not the sum across accelerators -- under DDP each rank holds a
    full model replica, so aggregate multi-GPU capacity is never the right
    number to compare a single replica's footprint against.
    """

    device: str
    frozen_params: int
    trainable_params: int
    max_length: int
    measured_peak_gb_at_batch_1: float
    measured_peak_gb_at_batch_2: float
    per_example_activation_gb: float
    configured_batch_size: int
    estimated_peak_gb: float
    per_rank_available_gb: float
    fits: bool
    recommendations: tuple[str, ...] = ()
    from_cache: bool = False


def _hardware_signature(context: ExecutionContext) -> str:
    pools = context.hardware.accelerator_vram_gb or (
        (context.hardware.vram_gb,) if context.hardware.vram_gb else ()
    )
    return "|".join(f"{pool:.2f}" for pool in sorted(pools)) or "cpu"


def _calibration_key(*, spec: TransformersPeftRunSpec, context: ExecutionContext) -> str:
    """Everything that plausibly changes real measured VRAM: the model
    identity, the recipe fields that affect what gets loaded/computed, and
    the hardware pool shape (see _hardware_signature). Training-length
    fields (epochs/max_steps/logging_steps/...) and anything checkpoint/
    dataset-related are irrelevant to a memory dry-run and excluded.
    """
    payload = {
        "base_model": spec.base_model,
        "revision": spec.revision,
        "quantization": spec.quantization,
        "precision": spec.precision,
        "max_length": spec.max_length,
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


def _recommendations(
    *, estimate_gb: float, available_gb: float, spec: TransformersPeftRunSpec
) -> tuple[str, ...]:
    if estimate_gb <= available_gb - _SAFETY_MARGIN_GB:
        return ()
    overage = estimate_gb - available_gb
    recommendations: list[str] = []
    if spec.batch_size > 1:
        recommendations.append(
            f"reduce backend.training.batch_size below {spec.batch_size} and raise "
            "gradient_accumulation_steps proportionally to keep the same effective batch size"
        )
    if not spec.gradient_checkpointing:
        recommendations.append("enable backend.training.gradient_checkpointing")
    if spec.quantization != "4bit":
        recommendations.append("switch backend.quantization to '4bit' (requires chowder-ai[qlora])")
    if spec.max_length > 128:
        recommendations.append(
            f"reduce backend.max_length below {spec.max_length} if the task tolerates shorter sequences"
        )
    recommendations.append(f"estimated to exceed available VRAM by {overage:.2f} GB")
    return tuple(recommendations)


def estimate_memory_requirements(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    use_cache: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> MemoryEstimate:
    """Real, measured VRAM estimate for the given config on the given
    hardware -- never a full training run, but a genuine model load plus
    two tiny forward+backward passes in an isolated subprocess, the same
    isolation discipline every other real-ML worker in this codebase uses.

    Cached per (model+recipe+hardware) combination under work_dir/.chowder,
    since the same combination measures the same way every time -- repeat
    preflight checks on an unchanged recipe don't need to pay for another
    real model load.
    """
    spec = TransformersPeftRunSpec.from_resolved_config(
        resolved_config,
        work_dir=work_dir,
        output_dir=Path(work_dir) / ".chowder" / "_memory_preflight_scratch",
        seed=context.seed,
        hardware=context.hardware,
    )
    key = _calibration_key(spec=spec, context=context)
    available_gb = _min_device_vram_gb(context.hardware)

    cached = _read_cache_entry(work_dir, key) if use_cache else None
    if cached is not None:
        measured = dict(cached)
        from_cache = True
    else:
        measured = _run_dry_run_worker(spec, work_dir=work_dir, timeout_seconds=timeout_seconds)
        from_cache = False
        if use_cache:
            _write_cache_entry(work_dir, key, measured)

    peak_bs1 = float(measured["peak_vram_gb_bs1"])
    peak_bs2 = float(measured["peak_vram_gb_bs2"])
    per_example = max(0.0, peak_bs2 - peak_bs1)
    estimated_peak = peak_bs1 + per_example * max(0, spec.batch_size - 1)
    fits = estimated_peak <= available_gb - _SAFETY_MARGIN_GB or measured["device"] != "cuda"

    return MemoryEstimate(
        device=str(measured["device"]),
        frozen_params=int(measured["frozen_params"]),
        trainable_params=int(measured["trainable_params"]),
        max_length=int(measured["max_length"]),
        measured_peak_gb_at_batch_1=peak_bs1,
        measured_peak_gb_at_batch_2=peak_bs2,
        per_example_activation_gb=per_example,
        configured_batch_size=spec.batch_size,
        estimated_peak_gb=estimated_peak,
        per_rank_available_gb=available_gb,
        fits=fits,
        recommendations=(
            () if fits else _recommendations(estimate_gb=estimated_peak, available_gb=available_gb, spec=spec)
        ),
        from_cache=from_cache,
    )


def _run_dry_run_worker(
    spec: TransformersPeftRunSpec, *, work_dir: str | Path, timeout_seconds: float
) -> dict[str, Any]:
    scratch_dir = Path(work_dir) / ".chowder" / "_memory_preflight_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spec_path = scratch_dir / "spec.json"
    result_path = scratch_dir / "result.json"
    spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "chowder.backends.memory_preflight_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_seconds
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"memory preflight worker failed with exit code {process.returncode}:\n"
            f"{process.stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise RuntimeError("memory preflight worker exited successfully without a result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise RuntimeError("memory preflight worker result is not a JSON object")
    return dict(result)
