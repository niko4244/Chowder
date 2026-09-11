"""Refuse to report a number for an adapter that cannot change the model.

`PeftModel.from_pretrained` succeeds even when NONE of the saved adapter weights
match the module names of the model it is being attached to. PEFT emits a
`UserWarning` about missing keys and returns a model whose LoRA `B` matrices are
all still zero-initialised -- which is mathematically the base model. Evaluation
then produces a perfectly plausible score for a candidate that was never applied.

Measured on the real 9B: an Unsloth-trained adapter loaded onto the model Chowder's
evaluator builds produced a **maximum logit delta of 0.000000** against 14.5 for a
Transformers-trained one, because Unsloth loads the full
`Qwen3_5ForConditionalGeneration` (decoder layers under
`model.language_model.layers`) while the evaluator loads the text-only CausalLM
(`model.layers`). Every key mismatched. The evaluation reported
`adapter_loaded: true` and a score equal to the baseline -- indistinguishable from
a genuinely useless adapter, for every Unsloth candidate on that architecture.

Two independent checks, both cheap and neither needing a forward pass:

1. **Key overlap.** At least one saved tensor must correspond to a real adapter
   parameter on the live model. Zero overlap means nothing was loaded.
2. **A non-zero `lora_B`.** PEFT zero-initialises `B` so a fresh adapter is an
   identity; training makes it non-zero. If every live `B` is exactly zero the
   adapter cannot alter any output, whatever the key bookkeeping says.

Saved keys are read straight from the safetensors header with stdlib, so this
module stays importable without torch and is testable without building a model.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


class AdapterNotLiveError(RuntimeError):
    """A loaded adapter cannot change the model, so any score would be a lie."""


def saved_adapter_keys(adapter_dir: str | Path) -> set[str]:
    """Tensor names inside an adapter directory's safetensors/bin weights."""
    directory = Path(adapter_dir)
    safetensors = directory / "adapter_model.safetensors"
    if safetensors.is_file():
        with safetensors.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        return {k for k in header if k != "__metadata__"}
    legacy = directory / "adapter_model.bin"
    if legacy.is_file():
        # A torch pickle; only reachable when torch is already loaded anyway.
        import torch

        return set(torch.load(legacy, map_location="cpu", weights_only=True))
    raise AdapterNotLiveError(
        f"no adapter weights in {directory} (expected adapter_model.safetensors "
        "or adapter_model.bin); there is nothing to evaluate"
    )


def _normalise(name: str) -> str:
    """Drop the adapter-name segment so saved and live names are comparable.

    Live parameters carry the active adapter's name (`...lora_B.default.weight`);
    saved tensors do not (`...lora_B.weight`).
    """
    return name.replace(".default.", ".")


def adapter_liveness_report(model: Any, adapter_dir: str | Path) -> dict[str, Any]:
    """Measure whether a loaded adapter can actually change `model`."""
    saved = saved_adapter_keys(adapter_dir)
    live_lora: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            live_lora[name] = param

    normalised_live = {_normalise(n) for n in live_lora}
    matched = {k for k in (_normalise(s) for s in saved) if k in normalised_live}

    b_params = {n: p for n, p in live_lora.items() if "lora_B" in n}
    nonzero_b = 0
    for param in b_params.values():
        try:
            if float(param.detach().float().abs().max()) > 0.0:
                nonzero_b += 1
        except Exception:  # pragma: no cover - exotic/quantised storage
            nonzero_b += 1  # unreadable: do not fail closed on a measurement gap

    return {
        "adapter_dir": str(adapter_dir),
        "saved_tensors": len(saved),
        "live_lora_parameters": len(live_lora),
        "matched_keys": len(matched),
        "lora_B_parameters": len(b_params),
        "lora_B_nonzero": nonzero_b,
        "example_saved_key": sorted(saved)[0] if saved else None,
        "example_live_parameter": sorted(live_lora)[0] if live_lora else None,
    }


def assert_adapter_is_live(model: Any, adapter_dir: str | Path) -> dict[str, Any]:
    """Raise unless the adapter attached to `model` can change its outputs.

    Returns the liveness report for provenance, so a run records that the check
    ran and what it measured rather than only that it passed.
    """
    report = adapter_liveness_report(model, adapter_dir)

    if report["matched_keys"] == 0:
        raise AdapterNotLiveError(
            f"adapter at {adapter_dir} shares NO parameter names with the loaded "
            f"model: {report['saved_tensors']} saved tensors, "
            f"{report['live_lora_parameters']} adapter parameters on the model, 0 "
            "matched. PeftModel.from_pretrained does not fail on this -- it warns "
            "about missing keys and leaves every LoRA B at zero, so the model is "
            "unchanged and any score would describe the BASE model.\n"
            f"  example saved key:      {report['example_saved_key']}\n"
            f"  example live parameter: {report['example_live_parameter']}\n"
            "A prefix difference here usually means the adapter was trained against "
            "a different model class than the one just loaded (e.g. a "
            "*ForConditionalGeneration wrapper, whose decoder layers sit under "
            "`language_model.`, versus a text-only CausalLM)."
        )

    if report["lora_B_parameters"] and report["lora_B_nonzero"] == 0:
        raise AdapterNotLiveError(
            f"adapter at {adapter_dir} loaded but every one of its "
            f"{report['lora_B_parameters']} LoRA B matrices is exactly zero, which "
            "is an identity transform: the adapter cannot change any output, so a "
            "score would describe the BASE model. Either the weights did not load "
            "or the adapter was never trained."
        )
    return report
