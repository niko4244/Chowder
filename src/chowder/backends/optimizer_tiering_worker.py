from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .memory_preflight_worker import load_dry_run_model
from .transformers_peft import TransformersPeftRunSpec

_WARMUP_ITERATIONS = 2


def _state_bytes(torch: Any, optimizer: Any) -> int:
    """Sum the actual state tensors an optimizer allocated -- the same
    real, device-agnostic tensor-introspection approach established in
    memory_preflight_worker.dry_run for AdamW's own optimizer_state_bytes,
    applied identically here so baseline and paged variants are measured
    the same way."""
    return sum(
        tensor.numel() * tensor.element_size()
        for state in optimizer.state.values()
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )


def run_experiment(spec: TransformersPeftRunSpec, *, batch_size: int = 2) -> dict[str, Any]:
    """Real, measured comparison of one optimizer.step() using the
    baseline torch.optim.AdamW (VRAM-resident state) against
    bitsandbytes' paged optimizers (bnb.optim.PagedAdamW,
    bnb.optim.PagedAdamW8bit), which use CUDA unified-memory paging to
    let optimizer state spill to host RAM under VRAM pressure rather than
    requiring it all resident on-device -- a real, already-implemented,
    battle-tested mechanism (not a hand-rolled CPU-offload step, which
    would carry real correctness risk for no proven benefit over an
    existing library).

    Requires bitsandbytes (chowder-ai[qlora]) and an available CUDA
    device -- paged optimizers are meaningless without CUDA unified
    memory. Returns available: False with a reason otherwise, rather than
    a meaningless comparison.

    Like activation_offload_worker, wall-clock timing requires warmup:
    both optimizer variants are run through several untimed steps before
    any timed measurement, so first-call CUDA/cuDNN kernel-selection cost
    doesn't get attributed to one variant over the other.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        # No model load needed to report this -- bitsandbytes' absence
        # doesn't depend on what device is present, and a full real model
        # load is real cost (10+ seconds) not worth paying just to name
        # the device in an error message nobody needs.
        return {
            "available": False,
            "device": "unknown",
            "reason": "bitsandbytes is not installed; install chowder-ai[qlora]",
        }

    import torch

    if not torch.cuda.is_available():
        # Same reasoning: CUDA availability is knowable without loading
        # the model at all.
        return {
            "available": False,
            "device": "cpu",
            "reason": "optimizer-state paging requires an available CUDA device",
        }

    model, tokenizer, device, _frozen_params, _trainable_params = load_dry_run_model(spec)

    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_param_list:
        return {
            "available": False,
            "device": device.type,
            "reason": "model has no trainable parameters to measure an optimizer step for",
        }

    input_ids = torch.randint(
        low=0, high=max(tokenizer.vocab_size, 2), size=(batch_size, spec.max_length), device=device
    )
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    outputs.loss.backward()
    # Every variant steps against the exact same real gradients -- computed
    # once, reused for each optimizer.step() below (each variant gets its
    # own optimizer instance, so this is a fair, identical starting point
    # for all three, not a cumulative sequence of updates).
    grad_snapshot = [p.grad.clone() if p.grad is not None else None for p in trainable_param_list]

    def _restore_grads() -> None:
        for param, grad in zip(trainable_param_list, grad_snapshot):
            param.grad = grad.clone() if grad is not None else None

    def _timed_step(optimizer_cls: Any, **kwargs: Any) -> tuple[float, int]:
        optimizer = optimizer_cls(trainable_param_list, lr=1e-4, **kwargs)
        _restore_grads()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state_bytes = _state_bytes(torch, optimizer)
        optimizer.zero_grad(set_to_none=True)
        return elapsed, state_bytes

    variants: dict[str, tuple[Any, dict[str, Any]]] = {
        "adamw": (torch.optim.AdamW, {}),
        "paged_adamw": (bnb.optim.PagedAdamW, {}),
        "paged_adamw_8bit": (bnb.optim.PagedAdamW8bit, {}),
    }

    for _ in range(_WARMUP_ITERATIONS):
        for optimizer_cls, kwargs in variants.values():
            _timed_step(optimizer_cls, **kwargs)

    measurements: dict[str, dict[str, Any]] = {}
    for name, (optimizer_cls, kwargs) in variants.items():
        elapsed, state_bytes = _timed_step(optimizer_cls, **kwargs)
        measurements[name] = {"step_seconds": elapsed, "state_bytes": state_bytes}

    model.zero_grad(set_to_none=True)
    return {
        "available": True,
        "device": device.type,
        "batch_size": batch_size,
        "max_length": spec.max_length,
        "variants": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = TransformersPeftRunSpec(**spec_data)
    result = run_experiment(spec, batch_size=args.batch_size)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
