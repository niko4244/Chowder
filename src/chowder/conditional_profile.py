"""Opt-in profiler primitives for the isolated conditional-compute experiment.

The profiler operates on an already-loaded model and caller-supplied forward
call. It never downloads a model, mutates weights, changes a campaign config,
or infers attention/Mamba FLOPs from incomplete instrumentation. For GPU
measurements, run it in a dedicated process after an independent resource
preflight; torch profiler ranges and synchronized wall-time are not a substitute
for Nsight Compute HBM counters.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import math
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn


class ConditionalProfileError(RuntimeError):
    """Raised when the profiler cannot identify the requested model path."""


_LAYER_PATHS = (
    "model.language_model.layers",
    "language_model.layers",
    "model.layers",
    "transformer.h",
    "layers",
)


@dataclass
class _ModuleMeasure:
    name: str
    category: str
    calls: int = 0
    cpu_wall_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    projection_flops: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class _Instrumentation(AbstractContextManager):
    def __init__(self, model: nn.Module, *, call_kind: str, phase: str) -> None:
        self.model = model
        self.call_kind = call_kind
        self.phase = phase
        self.handles: list[Any] = []
        self.measures: dict[str, _ModuleMeasure] = {}
        self._stacks: dict[tuple[int, str], list[tuple[float, Any, str]]] = defaultdict(list)
        self._forward_calls = 0
        self.layers_path, self.layers = _find_decoder_layers(model)
        self._register()

    def _phase(self) -> str:
        if self.call_kind == "generation":
            return "prefill" if self._forward_calls <= 1 else "decode"
        if self.phase == "forward":
            return "forward"
        return self.phase

    def begin_call(self) -> None:
        """Reset the prefill/decode classifier for a measured request."""
        self._forward_calls = 0

    def reset_measurements(self) -> None:
        """Exclude warmup and the untimed latency pass from component totals."""
        for measure in self.measures.values():
            measure.calls = 0
            measure.cpu_wall_ms.clear()
            measure.projection_flops.clear()

    def _root_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module, args, kwargs
        self._forward_calls += 1

    def _make_hooks(
        self,
        module: nn.Module,
        measure: _ModuleMeasure,
        category: str,
    ) -> tuple[Callable[..., None], Callable[..., None]]:
        stack_key = (id(module), measure.name)

        def pre_hook(mod: nn.Module, args: tuple[Any, ...]) -> None:
            del mod
            phase_name = self._phase()
            started = time.perf_counter()
            range_context = torch.profiler.record_function(
                f"exp_c.{phase_name}.{category}.{measure.name}"
            )
            range_context.__enter__()
            self._stacks[stack_key].append((started, range_context, phase_name))
            if category == "linear" and isinstance(module, nn.Linear):
                measure.projection_flops[phase_name] += _linear_projection_flops(module, args)

        def post_hook(mod: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            del mod, args, output
            stack = self._stacks.get(stack_key)
            if not stack:
                return
            started, range_context, phase_name = stack.pop()
            range_context.__exit__(None, None, None)
            measure.calls += 1
            measure.cpu_wall_ms[phase_name].append((time.perf_counter() - started) * 1000.0)

        return pre_hook, post_hook

    def _register(self) -> None:
        if self.call_kind == "generation":
            try:
                self.handles.append(
                    self.model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True)
                )
            except TypeError as error:
                raise ConditionalProfileError(
                    "generation phase profiling needs forward hooks with kwargs support"
                ) from error

        names = {id(module): name for name, module in self.model.named_modules()}
        targets: dict[int, tuple[nn.Module, str, str]] = {}
        for layer_index, layer in enumerate(self.layers):
            layer_name = names.get(id(layer), f"{self.layers_path}.{layer_index}")
            targets[id(layer)] = (layer, f"layer:{layer_name}", "layer")
            for relative_name, child in layer.named_modules():
                if not relative_name:
                    continue
                path = f"{layer_name}.{relative_name}"
                category = _component_category(relative_name)
                if category is not None:
                    targets[id(child)] = (child, path, category)
                if isinstance(child, nn.Linear):
                    targets[id(child)] = (child, path, "linear")

        for module, name, category in targets.values():
            measure = self.measures.setdefault(
                name, _ModuleMeasure(name=name, category=category)
            )
            pre_hook, post_hook = self._make_hooks(module, measure, category)
            try:
                self.handles.append(module.register_forward_pre_hook(pre_hook))
                self.handles.append(
                    module.register_forward_hook(post_hook, always_call=True)
                )
            except TypeError as error:
                self.close()
                raise ConditionalProfileError(
                    "profiler requires PyTorch forward hooks with always_call support"
                ) from error

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for stack in self._stacks.values():
            while stack:
                _started, range_context, _phase = stack.pop()
                range_context.__exit__(None, None, None)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


def _path_value(root: Any, path: str) -> Any | None:
    value = root
    for segment in path.split("."):
        if not hasattr(value, segment):
            return None
        value = getattr(value, segment)
    return value


def _find_decoder_layers(model: nn.Module) -> tuple[str, list[nn.Module]]:
    candidates: list[tuple[str, list[nn.Module]]] = []
    for path in _LAYER_PATHS:
        value = _path_value(model, path)
        if value is None:
            continue
        if isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple)):
            layers = list(value)
            if layers and all(isinstance(layer, nn.Module) for layer in layers):
                candidates.append((path, layers))
    if not candidates:
        raise ConditionalProfileError(
            "cannot locate a non-empty decoder layer sequence at known paths: "
            + ", ".join(_LAYER_PATHS)
        )
    unique = {tuple(id(layer) for layer in layers) for _path, layers in candidates}
    if len(unique) > 1:
        raise ConditionalProfileError(
            f"ambiguous decoder layer paths: {[path for path, _ in candidates]}"
        )
    return candidates[0]


def _component_category(relative_name: str) -> str | None:
    names = relative_name.casefold().split(".")
    if any(
        any(key in part for key in ("linear_attn", "mamba", "ssm", "recurrent"))
        for part in names
    ):
        return "linear_attention_or_recurrent"
    if any(
        any(key in part for key in ("self_attn", "attention", "attn"))
        for part in names
    ):
        return "attention"
    if any(
        any(key in part for key in ("mlp", "ffn", "feed_forward"))
        for part in names
    ):
        return "ffn"
    if any("norm" in part for part in names):
        return "normalization"
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _linear_projection_flops(module: nn.Linear, args: tuple[Any, ...]) -> float:
    value = _first_tensor(args)
    if value is None or not value.is_floating_point() or value.ndim < 1:
        return 0.0
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return 0.0
    if value.shape[-1] != module.in_features:
        return 0.0
    vectors = value.numel() // value.shape[-1]
    # Two floating-point operations per multiply-accumulate. Bias, activation,
    # norm, attention-score and recurrent FLOPs are deliberately excluded.
    return float(2 * vectors * module.in_features * module.out_features)


def _storage_bytes(tensor: torch.Tensor) -> int:
    try:
        if tensor.device.type == "meta":
            return 0
        return int(tensor.untyped_storage().nbytes())
    except (RuntimeError, AttributeError):
        return int(tensor.numel() * tensor.element_size())


def _weight_residency(model: nn.Module) -> dict[str, Any]:
    devices: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "parameter_count": 0,
            "logical_parameter_bytes": 0,
            "storage_parameter_bytes": 0,
            "buffer_bytes": 0,
        }
    )
    seen_parameters: set[int] = set()
    seen_storage: set[tuple[str, int, int]] = set()
    for parameter in model.parameters():
        if id(parameter) in seen_parameters:
            continue
        seen_parameters.add(id(parameter))
        key = str(parameter.device)
        size = _storage_bytes(parameter)
        try:
            pointer = parameter.untyped_storage().data_ptr() if parameter.device.type != "meta" else id(parameter)
        except (RuntimeError, AttributeError):
            pointer = id(parameter)
        storage_id = (key, pointer, size)
        devices[key]["parameter_count"] += int(parameter.numel())
        devices[key]["logical_parameter_bytes"] += int(parameter.numel() * parameter.element_size())
        if storage_id not in seen_storage:
            devices[key]["storage_parameter_bytes"] += size
            seen_storage.add(storage_id)

    seen_buffers: set[tuple[str, int, int]] = set()
    for buffer in model.buffers():
        key = str(buffer.device)
        size = _storage_bytes(buffer)
        try:
            pointer = buffer.untyped_storage().data_ptr() if buffer.device.type != "meta" else id(buffer)
        except (RuntimeError, AttributeError):
            pointer = id(buffer)
        storage_id = (key, pointer, size)
        if storage_id not in seen_buffers:
            devices[key]["buffer_bytes"] += size
            seen_buffers.add(storage_id)

    layer_bytes: dict[str, Any] = {}
    try:
        _path, layers = _find_decoder_layers(model)
        for index, layer in enumerate(layers):
            seen: set[int] = set()
            by_device: dict[str, int] = defaultdict(int)
            for parameter in layer.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                by_device[str(parameter.device)] += int(
                    parameter.numel() * parameter.element_size()
                )
            layer_bytes[str(index)] = dict(by_device)
    except ConditionalProfileError:
        pass
    return {"by_device": dict(devices), "logical_layer_parameter_bytes": layer_bytes}


def _cuda_devices(model: nn.Module) -> list[torch.device]:
    devices: dict[int, torch.device] = {}
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            index = parameter.device.index
            if index is None:
                index = torch.cuda.current_device()
            devices[index] = torch.device("cuda", index)
    return [devices[index] for index in sorted(devices)]


def _synchronize(devices: Sequence[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _finite_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
        }
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
        "min_ms": ordered[0],
    }


def _profile_events(profiler: Any, *, device_type: str) -> dict[str, Any]:
    operators = []
    component_device_time_ms: dict[str, float] = defaultdict(float)
    component_operator_flops: dict[str, float] = defaultdict(float)
    for event in profiler.key_averages():
        device_us = getattr(event, "device_time_total", None)
        if device_us is None:
            device_us = getattr(event, "cuda_time_total", 0.0)
        event_name = str(event.key)
        flops = float(getattr(event, "flops", 0.0) or 0.0)
        device_total_ms = float(device_us or 0.0) / 1000.0
        operators.append(
            {
                "name": event_name,
                "calls": int(getattr(event, "count", 0)),
                "cpu_total_ms": float(getattr(event, "cpu_time_total", 0.0)) / 1000.0,
                "device_total_ms": device_total_ms,
                "self_device_total_ms": float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                    or 0.0
                )
                / 1000.0,
                "flops": flops,
            }
        )
        if event_name.startswith("exp_c."):
            component_device_time_ms[event_name] += device_total_ms
            component_operator_flops[event_name] += flops
    operators.sort(key=lambda row: row["device_total_ms"], reverse=True)
    kernel_events = []
    for event in profiler.events():
        event_device = str(getattr(event, "device_type", "")).casefold()
        if "cuda" not in event_device and "hip" not in event_device:
            continue
        device_ms = getattr(event, "device_time_total", None)
        if device_ms is None:
            device_ms = getattr(event, "cuda_time_total", 0.0)
        kernel_events.append(
            {"name": str(getattr(event, "name", "")), "device_time_ms": float(device_ms or 0.0) / 1000.0}
        )
    kernel_events.sort(key=lambda row: row["device_time_ms"], reverse=True)
    return {
        "operators_top_device_time": operators[:100],
        "component_range_device_time_ms_inclusive": dict(component_device_time_ms),
        "component_range_operator_flops_inclusive": dict(component_operator_flops),
        "raw_device_events_available": bool(kernel_events),
        "raw_device_events": kernel_events[:500],
        "device_type_requested": device_type,
    }


def _phase_component_rows(instrument: _Instrumentation) -> tuple[dict[str, Any], dict[str, Any]]:
    component_rows: dict[str, Any] = {}
    component_times: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for name, measure in instrument.measures.items():
        phases = set(measure.cpu_wall_ms) | set(measure.projection_flops)
        for phase in sorted(phases):
            timings = _finite_stats(measure.cpu_wall_ms.get(phase, []))
            row = {
                "name": name,
                "category": measure.category,
                "calls": len(measure.cpu_wall_ms.get(phase, [])),
                "cpu_wall_ms_total_inclusive": sum(measure.cpu_wall_ms.get(phase, [])),
                "cpu_wall_ms_mean_inclusive": timings["mean_ms"],
                "cpu_wall_ms_p95_inclusive": timings["p95_ms"],
                "linear_projection_flops": measure.projection_flops.get(phase, 0.0),
            }
            component_rows[f"{phase}:{name}"] = row
            if measure.category != "linear":
                component_times[phase][measure.category] += row["cpu_wall_ms_total_inclusive"]
    return component_rows, {phase: dict(times) for phase, times in component_times.items()}


def _linear_flops_by_layer(
    instrument: _Instrumentation,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # The recorded module name is rooted at the actual named decoder path;
    # use that path rather than assume HF's model.layers convention.
    module_names = list(instrument.model.named_modules())
    layer_prefixes = [
        next(
            (
                module_name
                for module_name, candidate in module_names
                if candidate is layer
            ),
            f"{instrument.layers_path}.{index}",
        )
        for index, layer in enumerate(instrument.layers)
    ]
    for name, measure in instrument.measures.items():
        if measure.category != "linear":
            continue
        layer_index = None
        for index, prefix in enumerate(layer_prefixes):
            if name.startswith(prefix + "."):
                layer_index = index
                break
        if layer_index is None:
            continue
        for phase, flops in measure.projection_flops.items():
            results[phase][str(layer_index)] += flops
    return {phase: dict(values) for phase, values in results.items()}


def _linear_flops_by_layer_and_component(
    instrument: _Instrumentation,
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    module_names = list(instrument.model.named_modules())
    layer_prefixes = [
        next(
            (
                module_name
                for module_name, candidate in module_names
                if candidate is layer
            ),
            f"{instrument.layers_path}.{index}",
        )
        for index, layer in enumerate(instrument.layers)
    ]
    for name, measure in instrument.measures.items():
        if measure.category != "linear":
            continue
        for layer_index, prefix in enumerate(layer_prefixes):
            if not name.startswith(prefix + "."):
                continue
            relative_name = name[len(prefix) + 1 :]
            component = _component_category(relative_name) or "other_linear"
            for phase, flops in measure.projection_flops.items():
                results[phase][str(layer_index)][component] += flops
            break
    return {
        phase: {layer: dict(components) for layer, components in layers.items()}
        for phase, layers in results.items()
    }


def profile_model_call(
    model: nn.Module,
    call: Callable[[], Any],
    *,
    phase: str = "prefill",
    call_kind: str = "forward",
    warmup: int = 1,
    iterations: int = 5,
    use_torch_profiler: bool = True,
    profile_cuda: bool = True,
) -> dict[str, Any]:
    """Profile a repeatable model call and return JSON-compatible evidence.

    ``call_kind='generation'`` expects ``call`` to execute one fresh,
    deterministic generation request through ``model.generate`` (or an
    equivalent generate loop). During the separate module-profiler pass, the
    first model forward is labelled prefill; subsequent forwards are decode.
    ``phase='decode'`` profiles a caller-constructed single-token cached
    forward independently. Each timed latency iteration must build its own
    fresh request/cache in the supplied callback.

    Latency is measured without hooks or torch-profiler overhead. A separate
    instrumented call measures module-inclusive CPU launch/wall time and
    profiler operator/kernel events. FLOPs count only observed ``nn.Linear``
    projection MACs. Full-attention score/value products, Mamba/linear-
    attention recurrent kernels, non-linear operations and actual HBM bytes
    are not inferred.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(call):
        raise TypeError("call must be callable")
    if phase not in {"prefill", "decode", "forward"}:
        raise ValueError("phase must be 'prefill', 'decode', or 'forward'")
    if call_kind not in {"forward", "generation"}:
        raise ValueError("call_kind must be 'forward' or 'generation'")
    if call_kind == "generation" and phase not in {"prefill", "forward"}:
        raise ValueError("generation call_kind labels phases internally")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    layers_path, layers = _find_decoder_layers(model)
    modules = list(model.modules())
    training_modes = [module.training for module in modules]
    model.eval()
    devices = _cuda_devices(model) if profile_cuda and torch.cuda.is_available() else []
    wall_times: list[float] = []
    profile_data: dict[str, Any] = {
        "operators_top_device_time": [],
        "raw_device_events_available": False,
        "raw_device_events": [],
        "device_type_requested": "cuda" if devices else "cpu",
    }
    cuda_before: dict[str, dict[str, int]] = {}
    instrumentation: _Instrumentation | None = None
    try:
        with torch.no_grad():
            for _ in range(warmup):
                call()
        _synchronize(devices)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
            cuda_before[str(device)] = {
                "allocated_bytes_before": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes_before": int(torch.cuda.memory_reserved(device)),
            }
        for _ in range(iterations):
            _synchronize(devices)
            started = time.perf_counter()
            with torch.no_grad():
                call()
            _synchronize(devices)
            wall_times.append((time.perf_counter() - started) * 1000.0)

        instrumentation = _Instrumentation(model, call_kind=call_kind, phase=phase)
        instrumentation.reset_measurements()
        instrumentation.begin_call()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if devices:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if use_torch_profiler:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                acc_events=True,
            ) as profiler:
                with torch.no_grad():
                    call()
            profile_data = _profile_events(
                profiler,
                device_type="cuda" if devices else "cpu",
            )
        else:
            with torch.no_grad():
                call()
        _synchronize(devices)
    finally:
        if instrumentation is not None:
            instrumentation.close()
        for module, was_training in zip(modules, training_modes):
            module.train(was_training)

    if instrumentation is None:
        raise ConditionalProfileError("module instrumentation was not initialized")
    component_rows, component_times = _phase_component_rows(instrumentation)
    layer_flops = _linear_flops_by_layer(instrumentation)
    layer_component_flops = _linear_flops_by_layer_and_component(instrumentation)
    component_flops: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for phase_name, by_layer in layer_component_flops.items():
        for components in by_layer.values():
            for component, flops in components.items():
                component_flops[phase_name][component] += flops
    for range_name, value in profile_data.get(
        "component_range_device_time_ms_inclusive", {}
    ).items():
        _prefix, phase_name, _category, module_name = range_name.split(".", 3)
        row = component_rows.get(f"{phase_name}:{module_name}")
        if row is not None:
            row["gpu_device_ms_inclusive"] = value
            row["gpu_device_time_source"] = "torch_profiler_record_function_range"
    for range_name, value in profile_data.get(
        "component_range_operator_flops_inclusive", {}
    ).items():
        _prefix, phase_name, _category, module_name = range_name.split(".", 3)
        row = component_rows.get(f"{phase_name}:{module_name}")
        if row is not None:
            row["torch_profiler_operator_flops_inclusive"] = value

    memory = _weight_residency(model)
    memory["cuda_allocator"] = {}
    for device in devices:
        key = str(device)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except (RuntimeError, TypeError):
            free_bytes, total_bytes = 0, 0
        before = cuda_before.get(key, {})
        allocated_after = int(torch.cuda.memory_allocated(device))
        reserved_after = int(torch.cuda.memory_reserved(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        memory["cuda_allocator"][key] = {
            **before,
            "allocated_bytes_after": allocated_after,
            "reserved_bytes_after": reserved_after,
            "peak_allocated_bytes": peak_allocated,
            "peak_delta_bytes_over_before": max(
                0, peak_allocated - before.get("allocated_bytes_before", 0)
            ),
            "free_bytes_after": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    layer_inventory = []
    named_modules = list(model.named_modules())
    for index, layer in enumerate(layers):
        name = next((name for name, module in named_modules if module is layer), f"{layers_path}.{index}")
        subtree = [
            {"name": child_name, "category": _component_category(relative)}
            for relative, child in layer.named_modules()
            if relative and (child_name := f"{name}.{relative}")
            and _component_category(relative) is not None
        ]
        layer_inventory.append(
            {
                "index": index,
                "name": name,
                "parameter_residency": _weight_residency(layer)["by_device"],
                "components": subtree,
                "conditional_ffn_candidate": any(
                    item["category"] == "ffn" for item in subtree
                ),
                "always_on_invariant_candidates": [
                    item["name"]
                    for item in subtree
                    if item["category"]
                    in {"attention", "linear_attention_or_recurrent", "normalization"}
                ],
            }
        )

    phase_flops = {
        phase_name: sum(by_layer.values())
        for phase_name, by_layer in layer_flops.items()
    }
    phase_components: dict[str, dict[str, float]] = {}
    for phase_name, values in component_times.items():
        phase_components[phase_name] = {}
        for category, total_ms in values.items():
            phase_components[phase_name][category] = total_ms
    return {
        "schema_version": 1,
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
            "decoder_layer_path": layers_path,
            "decoder_layer_count": len(layers),
            "layer_inventory": layer_inventory,
        },
        "request": {
            "phase": phase,
            "call_kind": call_kind,
            "warmup_calls": warmup,
            "measured_iterations": iterations,
            "torch_profiler_enabled": use_torch_profiler,
            "component_profile_calls": 1,
        },
        "latency": _finite_stats(wall_times),
        "latency_samples_ms": wall_times,
        "module_inclusive_timings": component_rows,
        "component_cpu_wall_ms_inclusive": component_times,
        "component_cpu_wall_ms_by_category": phase_components,
        "linear_projection_flops_by_layer": layer_flops,
        "linear_projection_flops_by_layer_and_component": layer_component_flops,
        "linear_projection_flops_by_component_and_phase": {
            phase_name: dict(components)
            for phase_name, components in component_flops.items()
        },
        "linear_projection_flops_by_phase": phase_flops,
        "linear_projection_flops_total": sum(phase_flops.values()),
        "weight_residency": memory,
        "torch_profiler": profile_data,
        "memory_bandwidth": {
            "measured": False,
            "reason": "PyTorch operator profiling does not provide reliable device HBM traffic counters; use Nsight Compute/CUPTI on an isolated run",
        },
        "flops_coverage": {
            "linear_projections": "counted from observed nn.Linear input and weight shapes",
            "supported_matmul_and_conv_ops": "torch.profiler.with_flops estimates supported operators and custom module-range totals when available",
            "attention_score_value_products": "only appears when PyTorch recognizes the underlying matmul/bmm; fused kernels may be absent",
            "linear_attention_or_mamba_recurrent_ops": "not counted generically",
            "nonlinear_norm_and_embedding_ops": "not counted",
            "attention_vs_ffn": "nn.Linear projection counts are split by discovered component name; fused/unnamed computations may be uncategorized or missing",
        },
        "limitations": [
            "module CPU times are inclusive wall/launch times and nested component values overlap",
            "linear projection FLOPs are theoretical shape counts; profiler operator FLOPs cover supported operators only, not hardware instruction counters",
            "GPU device times and kernel events depend on the local PyTorch profiler/CUDA build",
            "weight residency counts tensor storage by current device; it is not a whole-process VRAM census",
            "the standalone CUDA allocator peak delta is not a profiler counter for other processes",
        ],
    }


def write_profile_artifact(profile: Mapping[str, Any], path: str) -> str:
    """Write a new JSON profile artifact without overwriting previous evidence."""
    import json
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = output.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite profiler artifact: {output}") from None
    with handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(output)


def summarize_generation_profiles(
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate a set of request artifacts without hiding latency spread."""
    if not profiles:
        raise ValueError("at least one profile is required")
    latencies = [float(profile["latency"]["mean_ms"]) for profile in profiles]
    return {
        "requests": len(profiles),
        "latency_ms": _finite_stats(latencies),
        "profiles": list(profiles),
        "disclaimer": "request aggregation only; prompts, cache lengths, and batch shapes must be reported separately",
    }
