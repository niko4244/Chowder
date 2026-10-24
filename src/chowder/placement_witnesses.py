"""The four gate verdicts only a real training run can produce.

``chowder_batch/moe_int8_gates.py`` measures what a *placement* costs: the live
census, peak VRAM, host RSS, s/step. The prereg for that placement has four more
gates that no loader can answer, because they are properties of a fine-tune:
does every LoRA leaf actually receive a gradient, does 20 steps lower the loss,
does the expert load stay balanced, and did the card spill. Until this module
existed the table could never be completed -- the runner writes "not measured"
for those four, and nothing else could write anything, so a campaign declaring
the placement was refused forever.

This is that producer, and it is deliberately stingy:

- every verdict is computed from evidence the run itself collected, and the
  evidence travels in the witness file, so a reader can recompute the verdict;
- a run that did not collect enough (too few steps, no logged loss, routing
  never observed) records **not measured with the reason**, never a guess;
- ``wddm_spill`` is the one gate no tool can see: PyTorch cannot observe a
  driver paging the card, so the module records the numbers and leaves the
  verdict to an operator, who must give their basis. A recorded "pass" without a
  basis is refused where the witness is written into the table.

The thresholds below are the pre-registration's, not this module's taste: they
are quoted from ``docs/quals/MOE_INT8_PLACEMENT_PREREG_2026-10-20.md``.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from .moe_placement_gates import (
    EXPERT_PLACEMENT,
    WITNESS_GATES,
    WITNESS_SCHEMA,
    artifact_identity,
    code_commit,
)

#: The prereg's smoke gate: this many optimizer steps, and a loss drop.
SMOKE_MIN_STEPS = 20
#: Relative drop that counts as "the loss fell" rather than as noise.
SMOKE_MIN_RELATIVE_DROP = 0.01
#: The prereg's router-balance gate: expert-load entropy within this share of step 0.
ROUTER_ENTROPY_TOLERANCE = 0.20
#: Layers the gradient gate names: first, middle, last.
GRADIENT_LAYERS = "first, middle and last"

_LAYER = re.compile(r"\.layers\.(\d+)\.")
#: PEFT's LoRA B matrix, whatever adapter name it defaulted to.
_LORA_B = re.compile(r"\.lora_B(?:\.|$)")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# --- gradients on the LoRA leaves ---------------------------------------------


def lora_b_leaves(model: Any) -> dict[int | None, list[str]]:
    """Every LoRA B parameter, grouped by the decoder layer it sits in."""
    grouped: dict[int | None, list[str]] = {}
    for name, _ in model.named_parameters():
        if not _LORA_B.search(name):
            continue
        match = _LAYER.search(name)
        layer = int(match.group(1)) if match else None
        grouped.setdefault(layer, []).append(name)
    return grouped


def choose_layers(available: Iterable[int]) -> list[int]:
    """First, middle and last of the layers a LoRA leaf was actually found in."""
    layers = sorted(set(available))
    if not layers:
        return []
    return sorted({layers[0], layers[len(layers) // 2], layers[-1]})


def collect_lora_grads(model: Any, *, layers: Sequence[int] | None = None) -> dict[str, Any]:
    """Finite/non-zero gradient evidence for the LoRA B leaves of named layers.

    Called once, on the first step whose backward has populated ``.grad``: an
    inspection of what is really there, not a re-run of the backward.
    """
    grouped = lora_b_leaves(model)
    wanted = list(layers) if layers else choose_layers(index for index in grouped if index is not None)
    params = dict(model.named_parameters())
    per_layer: dict[str, dict[str, Any]] = {}
    missing = nonfinite = zero = total = 0
    smallest: float | None = None
    for layer in wanted:
        names = grouped.get(layer, [])
        entry = {"leaves": len(names), "with_grad": 0, "finite": 0, "nonzero": 0, "min_abs": None}
        for name in names:
            total += 1
            grad = params[name].grad
            if grad is None:
                missing += 1
                continue
            entry["with_grad"] += 1
            values = grad.detach().float()
            if not bool(values.isfinite().all()):
                nonfinite += 1
                continue
            entry["finite"] += 1
            peak = float(values.abs().max()) if values.numel() else 0.0
            entry["min_abs"] = peak if entry["min_abs"] is None else min(entry["min_abs"], peak)
            smallest = peak if smallest is None else min(smallest, peak)
            if peak > 0.0:
                entry["nonzero"] += 1
            else:
                zero += 1
        per_layer[str(layer)] = entry
    return {
        "layers": wanted,
        "leaves": total,
        "missing_grad": missing,
        "nonfinite": nonfinite,
        "zero": zero,
        "min_abs": smallest,
        "per_layer": per_layer,
    }


# --- router load --------------------------------------------------------------


def entropy(counts: Sequence[int] | Sequence[float]) -> float:
    """Shannon entropy (nats) of an expert-load histogram, 0.0 when empty."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


class RouterLoadRecorder:
    """Expert-load histograms per MoE layer, from the router's own output.

    Hooks ``mlp.gate`` -- whose forward returns ``(hidden, scores, indices)`` --
    and counts the indices. That is the routing decision itself, not a proxy for
    it, and it costs one ``bincount`` per layer per forward. The module shape is
    the one ``moe_instrumentation.audit_moe_architecture`` verifies; a model
    that does not match raises there rather than being guessed at here.
    """

    def __init__(self, model: Any) -> None:
        from .moe_instrumentation import audit_moe_architecture, _decoder_layers, _moe_block

        self._audit = audit_moe_architecture(model)
        self._layers = _decoder_layers(model)
        self._moe_block = _moe_block
        self._counts: dict[int, list[int]] = {
            structure.layer: [0] * structure.num_experts for structure in self._audit.moe_layers
        }
        self._handles: list[Any] = []

    def __enter__(self) -> "RouterLoadRecorder":
        import torch

        for structure in self._audit.moe_layers:
            block = self._moe_block(self._layers[structure.layer])
            if block is None:
                raise RuntimeError(
                    f"layer {structure.layer} no longer matches the audited MoE shape; "
                    "refusing to record expert load against a model that changed"
                )
            handle = block.gate.register_forward_hook(self._make_hook(structure.layer, torch))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer: int, torch_module: Any):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 3:
                return
            indices = output[2]
            if indices is None:
                return
            with torch_module.no_grad():
                flat = indices.detach().reshape(-1).to("cpu", torch_module.long)
                counts = self._counts[layer]
                if not counts:
                    return
                bin_counts = torch_module.bincount(flat, minlength=len(counts))
                for expert in range(len(counts)):
                    counts[expert] += int(bin_counts[expert])

        return hook

    def reset(self) -> None:
        for counts in self._counts.values():
            for expert in range(len(counts)):
                counts[expert] = 0

    def window(self) -> dict[str, list[int]]:
        """The counts since the last ``reset``, per layer."""
        return {str(layer): list(counts) for layer, counts in sorted(self._counts.items())}

    @property
    def layers(self) -> int:
        return len(self._counts)


# --- verdicts -----------------------------------------------------------------


def _not_measured(value: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"value": value, "verdict": "not measured", "evidence": dict(evidence or {})}


def _verdict(value: str, passed: bool, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {"value": value, "verdict": "pass" if passed else "FAIL", "evidence": dict(evidence)}


def gradient_gate(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every LoRA B leaf in the named layers saw a finite, non-zero gradient."""
    if not evidence:
        return _not_measured("not measured (no backward was inspected)")
    layers = evidence.get("layers") or []
    leaves = int(evidence.get("leaves") or 0)
    if not layers or not leaves:
        return _not_measured(
            "not measured (no LoRA B leaf found in a decoder layer)", evidence
        )
    missing = int(evidence.get("missing_grad") or 0)
    nonfinite = int(evidence.get("nonfinite") or 0)
    zero = int(evidence.get("zero") or 0)
    empty_layers = [
        layer
        for layer, entry in (evidence.get("per_layer") or {}).items()
        if not int(entry.get("leaves") or 0)
    ]
    passed = not (missing or nonfinite or zero or empty_layers)
    detail = (
        f"layers {', '.join(str(layer) for layer in layers)}: "
        f"{leaves - missing - nonfinite - zero}/{leaves} LoRA B leaves finite and non-zero"
    )
    if evidence.get("min_abs") is not None:
        detail += f" (smallest peak |g| {float(evidence['min_abs']):.3g})"
    if missing:
        detail += f", {missing} with no grad"
    if nonfinite:
        detail += f", {nonfinite} non-finite"
    if zero:
        detail += f", {zero} exactly zero"
    if empty_layers:
        detail += f", no LoRA leaf in layers {', '.join(sorted(empty_layers))}"
    return _verdict(detail, passed, evidence)


def smoke_gate(step_log: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    """The prereg's smoke gate: at least 20 steps, and the loss fell."""
    entries = step_log.get("entries") if isinstance(step_log, Mapping) else step_log
    points = [
        (int(entry["step"]), float(entry["loss"]))
        for entry in (entries or ())
        if isinstance(entry, Mapping) and entry.get("loss") is not None and _finite(entry.get("loss")) is not None
    ]
    if not points:
        return _not_measured("not measured (the run logged no finite training loss)")
    steps = max(step for step, _ in points)
    if steps < SMOKE_MIN_STEPS:
        return _not_measured(
            f"not measured ({steps} steps; the prereg's smoke gate needs {SMOKE_MIN_STEPS})",
            {"steps": steps, "points": points},
        )
    if len(points) < 2:
        return _not_measured(
            f"not measured (one logged loss point over {steps} steps)", {"steps": steps, "points": points}
        )
    window = max(1, len(points) // 3)
    first = sum(loss for _, loss in points[:window]) / window
    last = sum(loss for _, loss in points[-window:]) / window
    drop = (first - last) / abs(first) if first else 0.0
    detail = (
        f"{steps} steps, mean loss {first:.3f} -> {last:.3f} ({drop:+.1%}; "
        f"needs {SMOKE_MIN_RELATIVE_DROP:.0%})"
    )
    return _verdict(detail, drop >= SMOKE_MIN_RELATIVE_DROP, {"steps": steps, "points": points})


def router_balance_gate(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expert-load entropy stays within 20% of its step-0 value, every MoE layer."""
    if not evidence:
        return _not_measured("not measured (the router was never observed)")
    step0 = evidence.get("step0") or {}
    last = evidence.get("last") or {}
    if not step0 or not last:
        return _not_measured(
            "not measured (routing was seen for fewer than two steps)", dict(evidence)
        )
    shared = sorted(set(step0) & set(last), key=int)
    if not shared:
        return _not_measured("not measured (no layer was observed twice)", dict(evidence))
    per_layer: dict[str, dict[str, Any]] = {}
    worst = 0.0
    worst_layer = None
    degenerate = []
    for layer in shared:
        h0, h1 = entropy(step0[layer]), entropy(last[layer])
        change = abs(h1 - h0) / h0 if h0 > 0 else math.inf
        per_layer[layer] = {"step0_entropy": h0, "last_entropy": h1, "change": change}
        if h0 <= 0:
            degenerate.append(layer)
        if change > worst:
            worst, worst_layer = change, layer
    detail = (
        f"{len(shared)} MoE layers, worst change {worst:+.1%} in layer {worst_layer} "
        f"(tolerance {ROUTER_ENTROPY_TOLERANCE:.0%})"
    )
    if degenerate:
        detail += f"; zero-entropy at step 0 in layers {', '.join(degenerate)}"
    passed = not degenerate and worst <= ROUTER_ENTROPY_TOLERANCE
    return _verdict(detail, passed, {**dict(evidence), "per_layer": per_layer})


def wddm_gate(gpu: Mapping[str, Any] | None) -> dict[str, Any]:
    """The one gate no tool can see: record the numbers, not a verdict."""
    gpu = dict(gpu or {})
    peak = gpu.get("peak_vram_gb")
    reserved = gpu.get("reserved_gb")
    card = gpu.get("card_gb")
    if peak is None:
        return _not_measured(
            "not measured (no device memory was sampled)", gpu
        )
    detail = f"{peak} GB peak allocated, {reserved} GB reserved"
    if card is not None:
        detail += f" of a {card} GB card"
    detail += "; driver paging is not observable from PyTorch"
    return _not_measured(detail, gpu)


def gpu_snapshot(torch_module: Any, device: int = 0) -> dict[str, Any]:
    """Peak allocated/reserved device memory and the card's own size."""
    if not torch_module.cuda.is_available():
        return {}
    try:
        return {
            "peak_vram_gb": round(torch_module.cuda.max_memory_allocated(device) / 2**30, 3),
            "reserved_gb": round(torch_module.cuda.max_memory_reserved(device) / 2**30, 3),
            "card_gb": round(
                torch_module.cuda.get_device_properties(device).total_memory / 2**30, 2
            ),
        }
    except Exception:  # a device that cannot answer is not a measurement
        return {}


def build_witness_callback(callback_base: Any, model: Any, *, router_model: Any = None) -> Any:
    """A ``TrainerCallback`` collecting this run's evidence for the four gates.

    Three things are gathered, all of them free at the point they are read:

    - the LoRA B gradients from the **first** step whose backward populated
      ``.grad`` (``on_pre_optimizer_step`` if this transformers version has that
      hook, else ``on_step_end``), inspected once and never re-run;
    - expert-load histograms per MoE layer, reset at each step start and
      snapshotted at each step end, so the recorder holds step 0's window and
      the most recent one;
    - nothing else. Losses come from ``Trainer.state.log_history`` and device
      memory from ``gpu_snapshot``, both read after training.

    A model whose routing cannot be audited, or a step that never exposes a
    gradient, is recorded as an error the derivation turns into "not measured"
    with the reason: this callback never fails a run over instrumentation.
    """

    class _PlacementWitnessCallback(callback_base):
        def __init__(self) -> None:
            super().__init__()
            self._model = model
            # The adapter wrapper exposes no ``.model.layers`` of its own, so the
            # routing audit runs on the base model it wraps.
            self._router_model = router_model if router_model is not None else model
            self.recorder: RouterLoadRecorder | None = None
            self.router_error: str | None = None
            self.gradient: dict[str, Any] | None = None
            self.gradient_error: str | None = None
            self.step0: dict[str, list[int]] | None = None
            self.last: dict[str, list[int]] | None = None
            self._attempts = 0

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            try:
                self.recorder = RouterLoadRecorder(self._router_model).__enter__()
            except Exception as error:  # a non-MoE model, or an unverified shape
                self.router_error = f"{type(error).__name__}: {error}"

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.recorder is not None:
                self.recorder.__exit__(None, None, None)
                self.recorder = None

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.recorder is not None:
                self.recorder.reset()

        def _capture_gradient(self) -> None:
            if self.gradient is not None or self._attempts >= 3:
                return
            self._attempts += 1
            try:
                evidence = collect_lora_grads(self._model)
            except Exception as error:
                self.gradient_error = f"{type(error).__name__}: {error}"
                return
            if int(evidence.get("leaves") or 0) and int(evidence.get("missing_grad") or 0) == int(
                evidence.get("leaves") or 0
            ):
                return  # no backward has landed here yet; a later step may show one
            self.gradient = evidence

        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            self._capture_gradient()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._capture_gradient()
            if self.recorder is None:
                return
            window = self.recorder.window()
            if self.step0 is None:
                self.step0 = window
            self.last = window

        def evidence(self) -> dict[str, Any]:
            router: dict[str, Any] | None = None
            if self.recorder is not None or self.step0 is not None:
                router = {"step0": self.step0, "last": self.last}
            return {
                "gradient": self.gradient,
                "router": router,
                "router_error": self.router_error,
                "gradient_error": self.gradient_error,
            }

    return _PlacementWitnessCallback()


def derive(
    *,
    gradient: Mapping[str, Any] | None = None,
    router: Mapping[str, Any] | None = None,
    step_log: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    gpu: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The four witness gates, each from the evidence this run collected."""
    return {
        "gradient": gradient_gate(gradient),
        "smoke": smoke_gate(step_log),
        "router_balance": router_balance_gate(router),
        "wddm_spill": wddm_gate(gpu),
    }


def witness_record(
    gates: Mapping[str, Mapping[str, Any]],
    *,
    base_model: Any,
    device: str,
    steps: int,
    run: str | None = None,
) -> dict[str, Any]:
    """The file a worker writes and the runner reads: gates plus their binding.

    The artifact digest is the point: a witness measured on another checkpoint
    must not be recordable into a table that was measured on this one.
    """
    unknown = sorted(set(gates) - set(WITNESS_GATES))
    if unknown:
        raise ValueError(f"unknown witness gate(s) {unknown}; expected a subset of {WITNESS_GATES}")
    identity = artifact_identity(base_model)
    if not identity.get("content_sha256") or not identity.get("inventory_sha256"):
        raise ValueError("witness requires a verified SHA inventory for its exact checkpoint")
    return {
        "schema": WITNESS_SCHEMA,
        "placement": EXPERT_PLACEMENT,
        "base_model": str(base_model),
        "artifact": identity,
        "device": device,
        "steps": int(steps),
        "run": run,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": code_commit(),
        "gates": {gate: dict(verdict) for gate, verdict in gates.items()},
    }
