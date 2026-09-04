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
_ALLOWED_MEMORY_PREFLIGHT_MODES = {"auto", "always", "cached", "off"}
# A documented starting point, not a claimed-optimal constant, matching
# every other threshold in this codebase's placement mechanisms
# (_MAX_ACCEPTABLE_PENALTY_RATIO in activation_offload.py/optimizer_
# tiering.py/frozen_layer_streaming.py). "auto" trusts a cached estimate
# outright when it shows comfortable headroom, and only pays for a fresh
# real dry-run when the cached number is close enough to the fit boundary
# that trusting a possibly-stale measurement carries real risk.
_AUTO_REFRESH_PRESSURE_RATIO = 0.9


def _effective_timeout_seconds(resolved_config: Mapping[str, Any]) -> float:
    """Never shrinks below `_DEFAULT_TIMEOUT_SECONDS`; only extends it when
    the recipe's own configured production-run timeout implies the user
    already expects operations at this scale to take longer. Duplicated
    (not imported) from the identical helper in placement_policy.py /
    combined_mechanism_experiment.py: that module imports FROM this one at
    module level, so importing back here would be a circular import --
    this logic is small enough that duplicating it is the right tradeoff.
    """
    backend = resolved_config.get("backend", {}) if isinstance(resolved_config, Mapping) else {}
    runtime = backend.get("runtime", {}) if isinstance(backend, Mapping) else {}
    configured = runtime.get("timeout_seconds") if isinstance(runtime, Mapping) else None
    if configured is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        configured = float(configured)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    return max(configured, _DEFAULT_TIMEOUT_SECONDS)


def _resolve_memory_preflight_policy(backend: Mapping[str, Any]) -> str:
    """Parse backend.memory_preflight into one of auto/always/cached/off.

    Defaults to "off" -- not "cached" or "auto" -- for the same reason
    every other production-wiring flag in this codebase defaults to its
    most conservative option: this preflight check did not exist at all
    before this policy was added (estimate_memory_requirements was only
    ever called from the TUI's manual "Estimate Memory" button, never
    from the real automated training/evaluation cycle). Defaulting to
    anything other than "off" would mean an existing project.json that
    never asked for this suddenly gets its candidates rejected by a
    check it never opted into -- exactly the "an existing project.json
    with no X key must train exactly as it always has" regression this
    session already hit once for real with activation_offload/
    optimizer_tiering defaulting to "auto" instead of "off".
    """
    raw = backend.get("memory_preflight", "off")
    if not isinstance(raw, str):
        raise ValueError(
            f"backend.memory_preflight must be one of {sorted(_ALLOWED_MEMORY_PREFLIGHT_MODES)}, "
            f"got {raw!r}"
        )
    value = raw.strip().lower()
    if value not in _ALLOWED_MEMORY_PREFLIGHT_MODES:
        raise ValueError(
            f"backend.memory_preflight must be one of {sorted(_ALLOWED_MEMORY_PREFLIGHT_MODES)}, "
            f"got {raw!r}"
        )
    return value


@dataclass(frozen=True)
class MemoryEstimate:
    """A real, measured (not theoretical) VRAM estimate for one base_model +
    recipe + hardware combination.

    `estimated_peak_gb` is a REAL, DIRECTLY MEASURED peak at the configured
    batch size whenever that differs from the two always-measured tiny
    points (batch 1 and 2) -- not a linear extrapolation. A real, measured
    finding motivated this: extrapolating from two tiny points to a much
    larger production batch size can be badly wrong (confirmed directly: a
    real Qwen2.5-1.5B run at batch_size=8 measured a real peak roughly 3x
    what the linear slope from batch=1/batch=2 would have predicted, and
    the real peak was reached entirely within the first training step, not
    a gradual multi-step climb). `measured_peak_gb_at_configured_batch_size`
    and `configured_batch_size_confirmed_oom` carry that direct measurement
    (or the fact that it genuinely CUDA-OOM'd) when it was taken; the
    linear-extrapolation fields (`measured_peak_gb_at_batch_1/2`,
    `per_example_activation_gb`) remain for callers wanting the
    per-example slope, and are the only thing `estimated_peak_gb` falls
    back to when the configured batch size happens to already be 1 or 2,
    or a real direct measurement is otherwise unavailable.

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
    measured_peak_gb_at_configured_batch_size: float | None = None
    configured_batch_size_confirmed_oom: bool = False


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
    timeout_seconds: float | None = None,
) -> MemoryEstimate:
    """Real, measured VRAM estimate for the given config on the given
    hardware -- never a full training run, but a genuine model load, two
    tiny forward+backward passes, and (when the configured batch size
    differs from those two tiny points) one real forward+backward pass at
    the actual configured batch size, all in an isolated subprocess, the
    same isolation discipline every other real-ML worker in this codebase
    uses.

    `timeout_seconds` defaults to whichever is larger: the module's own
    300s floor, or the recipe's own configured `backend.runtime.
    timeout_seconds` -- a real, large-batch direct measurement can
    legitimately take longer than 300s (confirmed directly: this is the
    same real production-timeout bug found and fixed in placement_policy.py
    /combined_mechanism_experiment.py for their own calibration calls).

    Cached per (model+recipe+hardware) combination under work_dir/.chowder,
    since the same combination measures the same way every time -- repeat
    preflight checks on an unchanged recipe don't need to pay for another
    real model load.
    """
    if timeout_seconds is None:
        timeout_seconds = _effective_timeout_seconds(resolved_config)
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
    extrapolated_peak = peak_bs1 + per_example * max(0, spec.batch_size - 1)

    # .get(...) with a safe default, not measured[...]: an older cache
    # entry written before this direct-measurement fix existed has neither
    # key, and a stale cache hit must not KeyError.
    direct_peak_raw = measured.get("peak_vram_gb_at_configured_batch_size")
    direct_peak = float(direct_peak_raw) if direct_peak_raw is not None else None
    confirmed_oom = bool(measured.get("configured_batch_size_oom", False))

    if confirmed_oom:
        # A real, direct measurement at the configured batch size genuinely
        # CUDA-OOM'd -- this recipe definitively does not fit, regardless
        # of what the linear extrapolation alone would have guessed.
        estimated_peak = max(extrapolated_peak, available_gb + _SAFETY_MARGIN_GB)
        fits = False
    elif direct_peak is not None:
        # Real, directly measured at the actual configured batch size --
        # see MemoryEstimate's own docstring for why this is preferred
        # over the linear extrapolation whenever it was taken.
        estimated_peak = direct_peak
        fits = estimated_peak <= available_gb - _SAFETY_MARGIN_GB or measured["device"] != "cuda"
    else:
        estimated_peak = extrapolated_peak
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
        measured_peak_gb_at_configured_batch_size=direct_peak,
        configured_batch_size_confirmed_oom=confirmed_oom,
    )


def resolve_memory_fit(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    timeout_seconds: float | None = None,
) -> MemoryEstimate | None:
    """The real preflight decision this policy governs: whether/how to
    call estimate_memory_requirements before a run, based on
    backend.memory_preflight ("auto" | "always" | "cached" | "off",
    default "off" -- see _resolve_memory_preflight_policy).

    Returns None for "off" (no measurement at all -- the caller should
    skip fit-checking entirely, matching today's pre-existing behavior
    for every config that doesn't opt in). Otherwise returns a real
    MemoryEstimate:
    - "always": always pays for a fresh real dry-run (use_cache=False).
    - "cached": prefers any existing cache, measuring fresh only on a
      cache miss (use_cache=True) -- a name for today's pre-existing
      estimate_memory_requirements default, now reachable through this
      policy layer too.
    - "auto": reads the cache first. If it hits and shows comfortable
      headroom (estimated_peak_gb is not within _AUTO_REFRESH_PRESSURE_
      RATIO of per_rank_available_gb), trusts it -- another cache-hit
      call, at effectively zero cost. If it misses, or if the cached
      estimate is close enough to the fit boundary that trusting a
      possibly-stale number carries real risk, pays for a fresh
      measurement instead. Config novelty on its own never needs special
      handling here -- estimate_memory_requirements' own calibration key
      already changes with the config, so a genuinely different recipe
      is already a cache miss by construction.
    """
    backend = resolved_config.get("backend", {})
    backend = backend if isinstance(backend, Mapping) else {}
    policy = _resolve_memory_preflight_policy(backend)
    if policy == "off":
        return None
    if policy == "always":
        return estimate_memory_requirements(
            resolved_config=resolved_config,
            context=context,
            work_dir=work_dir,
            use_cache=False,
            timeout_seconds=timeout_seconds,
        )
    if policy == "cached":
        return estimate_memory_requirements(
            resolved_config=resolved_config,
            context=context,
            work_dir=work_dir,
            use_cache=True,
            timeout_seconds=timeout_seconds,
        )

    # "auto"
    cached_estimate = estimate_memory_requirements(
        resolved_config=resolved_config,
        context=context,
        work_dir=work_dir,
        use_cache=True,
        timeout_seconds=timeout_seconds,
    )
    if not cached_estimate.from_cache:
        # Cache miss: estimate_memory_requirements already paid for a
        # real fresh measurement and cached it -- nothing more to do.
        return cached_estimate
    under_pressure = cached_estimate.estimated_peak_gb >= (
        cached_estimate.per_rank_available_gb * _AUTO_REFRESH_PRESSURE_RATIO
    )
    if not under_pressure:
        return cached_estimate
    return estimate_memory_requirements(
        resolved_config=resolved_config,
        context=context,
        work_dir=work_dir,
        use_cache=False,
        timeout_seconds=timeout_seconds,
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
