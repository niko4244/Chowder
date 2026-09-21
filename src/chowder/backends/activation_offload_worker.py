from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .activation_offload_hooks import offload_pack, offload_unpack
from .memory_preflight_worker import load_dry_run_model
from .transformers_peft import TransformersPeftRunSpec

_WARMUP_ITERATIONS = 2
_TIMED_ITERATIONS = 3


def run_experiment(spec: TransformersPeftRunSpec, *, batch_size: int = 2) -> dict[str, Any]:
    """Real, measured comparison of one forward+backward pass with vs.
    without CPU activation offload (`torch.autograd.graph.saved_tensors_hooks`
    moving saved tensors to CPU during forward and back to the device
    during backward), at the configured `max_length` and the given
    `batch_size`.

    Wall-clock timing is only meaningful with warmup: a raw first call
    pays for CUDA/cuDNN kernel selection and allocator-cache warmup, which
    swamps the real offload-transfer cost and can even make an
    unwarmed offload measurement look *faster* than the unwarmed baseline
    -- verified directly against real hardware (a tiny model, first-call
    only: baseline 1.43s, offload 0.53s; the same comparison after
    warmup: baseline 0.036s, offload 0.195s -- a >5x real penalty). Both
    code paths are warmed up before any timed measurement.

    Requires an available CUDA device -- there is nothing to "offload
    from" without a GPU, so this returns None on CPU rather than reporting
    a meaningless comparison.
    """
    model, tokenizer, device, _frozen_params, _trainable_params = load_dry_run_model(spec)
    import torch

    if device.type != "cuda":
        return {
            "available": False,
            "device": device.type,
            "reason": "activation offload requires an available CUDA device",
        }

    def _offload_pack(tensor: Any) -> Any:
        packed, _bytes_moved = offload_pack(tensor)
        return packed

    def _offload_unpack(packed: Any) -> Any:
        tensor, _bytes_moved = offload_unpack(packed)
        return tensor

    def _forward_backward(*, offload: bool) -> float:
        input_ids = torch.randint(
            low=0, high=max(tokenizer.vocab_size, 2), size=(batch_size, spec.max_length), device=device
        )
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if offload:
            with torch.autograd.graph.saved_tensors_hooks(_offload_pack, _offload_unpack):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                outputs.loss.backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            outputs.loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        return elapsed

    for _ in range(_WARMUP_ITERATIONS):
        _forward_backward(offload=False)
        _forward_backward(offload=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_times = [_forward_backward(offload=False) for _ in range(_TIMED_ITERATIONS)]
    baseline_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    offload_times = [_forward_backward(offload=True) for _ in range(_TIMED_ITERATIONS)]
    offload_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        "available": True,
        "device": device.type,
        "batch_size": batch_size,
        "max_length": spec.max_length,
        "baseline_peak_vram_gb": baseline_peak_gb,
        "offload_peak_vram_gb": offload_peak_gb,
        "baseline_wall_seconds": statistics.median(baseline_times),
        "offload_wall_seconds": statistics.median(offload_times),
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
