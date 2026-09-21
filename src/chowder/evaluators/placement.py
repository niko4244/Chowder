"""Evaluation-side placement: full-resident vs offload-resident generation.

The production text evaluators historically loaded the base model fully
onto the accelerator (``model.to("cuda:0")``) or quantized it onto it
(4-bit). For a dense ~9B bf16 parent (~18.6 GB) on a 16 GB card, both are
an honest OOM: the evaluator could not evaluate the exact model class this
program improves. Training is unaffected (the Memory Fabric streams frozen
LoRA base weights), so a trained candidate could exist with no production
way to score it.

``placement: "offload"`` mirrors the probe-qualified Generation-0 freeze
policy (3.94 GiB steady-state peak, measured 2026-09-17): bf16 weights
CPU-summoned, the decoder layers pinned to the CPU, everything else on the
accelerator, transient per-token copies streamed over PCIe. It is
deliberately evaluation-only: training placement stays the Memory Fabric's
own, already-judged domain. The placement a run used is reported in its
result payload, and the evaluators' protocol fingerprint carries it
(digest-additively) so a resident protocol and an offload protocol are
never silently compared as equals.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

PLACEMENT_MODES = frozenset({"resident", "offload"})

#: Where transient weight copies land. Configurable so a machine can keep
#: the streamed shards off a full system drive; never silently defaults to
#: a path the run does not report.
_OFFLOAD_DIR_ENV = "CHOWDER_EVAL_OFFLOAD_DIR"


def validate_placement(value: str, *, context: str) -> str:
    mode = str(value).strip().lower()
    if mode not in PLACEMENT_MODES:
        raise ValueError(
            f"{context} must be one of {sorted(PLACEMENT_MODES)}, got {value!r}"
        )
    return mode


def _offload_dir() -> str:
    configured = os.environ.get(_OFFLOAD_DIR_ENV, "").strip()
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return tempfile.mkdtemp(prefix="chowder-eval-offload-")


def dispatch_offloaded(model: Any, device_name: str) -> Any:
    """Place a causal LM for generation with the dense weights off the card.

    Requires ``accelerate``. Raises with the reason when the model's
    structure does not match the assumed boundary (one root submodule
    holding a ``layers`` ModuleList) instead of guessing a boundary and
    misplacing a silently different model.
    """
    from accelerate import dispatch_model

    main = device_name if str(device_name).startswith("cuda") else "cpu"
    if not main.startswith("cuda"):
        # A CPU-only host has nothing to offload from: the model is already
        # fully CPU-resident, which is exactly what "offload" asks for.
        return model

    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(root, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or not len(layers):
        raise RuntimeError(
            "evaluation placement 'offload' expects a decoder ModuleList at "
            "model.layers to pin to the CPU; this model does not match the "
            "assumed structure, and inventing a boundary would misplace "
            "tensors silently"
        )
    device_map: dict[Any, Any] = {"": main}
    device_map.update({f"model.layers.{i}": "cpu" for i in range(len(layers))})
    # ``offload_buffers=True`` puts the offloaded layers' buffers on the host
    # too. accelerate warns for exactly this model class that the buffers "do
    # not fit any GPU's remaining memory", and with the residual on the card a
    # 16 GB host OOMs on the first forward even with several GB nominally free
    # -- observed here as a 64 MiB allocation failure during generation. Letting
    # the buffers ride with their weights is what makes a base-only measurement
    # (the trusted-ancestor arm) run on this hardware at all.
    dispatch_model(
        model,
        device_map=device_map,
        offload_dir=_offload_dir(),
        main_device=0,
        offload_buffers=True,
    )
    return model


def needs_redispatch_after_adapter(
    *, quantization: str, placement: str, adapter: bool
) -> bool:
    """Whether the placement has to be re-applied once an adapter is attached.

    Attaching a PEFT adapter re-places the model it wraps: the wrapper
    re-dispatches the base against its existing device map, which materialises
    every offloaded parameter on the host and drops the ``offload_buffers``
    setting accelerate needs for this model class. The result is not an error
    but a silently unbounded measurement -- observed on the gen1 parent arm
    (2026-09-19): the base-only arm reported ``cuda=3 cpu=0 other=424`` and
    finished the math500 slice in 23 minutes, while the adapter arm reported
    ``cuda=0 cpu=683 other=0`` and was killed by the declared 7200 s worker
    timeout still inside that same slice.

    This is the one place that decides it, so the worker cannot quietly decide
    otherwise.
    """
    return bool(adapter) and quantization == "none" and placement == "offload"


def placement_note(model: Any) -> str:
    """Where the model's parameters actually live, counted from the model."""
    counts = {"cuda": 0, "cpu": 0, "other": 0}
    for param in model.parameters():
        device_type = param.device.type
        counts[device_type if device_type in counts else "other"] += 1
    return (
        f"params on cuda={counts['cuda']} cpu={counts['cpu']} other={counts['other']}"
    )
