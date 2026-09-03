from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends.transformers_peft import TransformersPeftRunSpec
from .executors import ExecutionContext
from .memory_preflight import (
    _DEFAULT_TIMEOUT_SECONDS,
    _calibration_key,
    _read_cache_entry,
    _run_dry_run_worker,
    _write_cache_entry,
)


@dataclass(frozen=True)
class LayerTelemetry:
    """Real, forward-hook-measured telemetry for one leaf module (see
    memory_preflight_worker._leaf_modules): under a PEFT-wrapped model this
    lands one level below each LoRA wrapper (base_layer, lora_A, lora_B,
    lora_dropout are each their own entry), not at attention-projection
    granularity. Aggregating to a coarser level (e.g. by common name
    prefix) is the caller's job.
    """

    name: str
    module_type: str
    trainable_params: int
    frozen_params: int
    activation_bytes: int


@dataclass(frozen=True)
class RuntimeTelemetry:
    """Real, measured per-model runtime footprint: what Phase 7's later
    offload/tiering/streaming work (activation offload, optimizer-state
    tiering, frozen-layer residency) needs to know before it can make any
    placement decision. Nothing here is estimated from a formula --
    layer-level activation sizes come from real forward hooks, and
    optimizer_state_bytes comes from a real torch.optim.AdamW step, not an
    assumption about AdamW's internals.
    """

    device: str
    max_length: int
    frozen_params: int
    trainable_params: int
    optimizer_state_bytes: int
    layers: tuple[LayerTelemetry, ...]
    from_cache: bool = False

    @property
    def total_activation_bytes(self) -> int:
        return sum(layer.activation_bytes for layer in self.layers)

    def top_activation_layers(self, n: int = 10) -> tuple[LayerTelemetry, ...]:
        return tuple(sorted(self.layers, key=lambda layer: layer.activation_bytes, reverse=True)[:n])


def collect_runtime_telemetry(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
    use_cache: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RuntimeTelemetry:
    """Real per-layer/optimizer telemetry for the given model+recipe on the
    given hardware -- the same real subprocess dry-run
    memory_preflight.estimate_memory_requirements() uses, sharing its cache
    (work_dir/.chowder/memory_calibration.json, keyed by model+recipe+
    hardware): calling either function first warms the cache for the
    other, since one real measurement pass now produces both the VRAM-fit
    fields and the telemetry fields together.
    """
    spec = TransformersPeftRunSpec.from_resolved_config(
        resolved_config,
        work_dir=work_dir,
        output_dir=Path(work_dir) / ".chowder" / "_memory_preflight_scratch",
        seed=context.seed,
        hardware=context.hardware,
    )
    key = _calibration_key(spec=spec, context=context)

    cached = _read_cache_entry(work_dir, key) if use_cache else None
    # A cache entry written before this telemetry field existed (an older
    # Chowder version's memory_calibration.json) must not be trusted as if
    # it had telemetry -- force a fresh real measurement instead of
    # silently reporting zero layers.
    if cached is not None and "layer_telemetry" in cached:
        measured = dict(cached)
        from_cache = True
    else:
        measured = _run_dry_run_worker(spec, work_dir=work_dir, timeout_seconds=timeout_seconds)
        from_cache = False
        if use_cache:
            _write_cache_entry(work_dir, key, measured)

    raw_layers = measured.get("layer_telemetry", [])
    layers = tuple(
        LayerTelemetry(
            name=str(entry["name"]),
            module_type=str(entry["module_type"]),
            trainable_params=int(entry["trainable_params"]),
            frozen_params=int(entry["frozen_params"]),
            activation_bytes=int(entry["activation_bytes"]),
        )
        for entry in raw_layers
    )
    return RuntimeTelemetry(
        device=str(measured["device"]),
        max_length=int(measured["max_length"]),
        frozen_params=int(measured["frozen_params"]),
        trainable_params=int(measured["trainable_params"]),
        optimizer_state_bytes=int(measured.get("optimizer_state_bytes", 0)),
        layers=layers,
        from_cache=from_cache,
    )
