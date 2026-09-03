from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .memory_preflight_worker import load_dry_run_model
from .transformers_peft import TransformersPeftRunSpec

_WARMUP_ITERATIONS = 2
_TIMED_ITERATIONS = 3


def run_experiment(spec: TransformersPeftRunSpec, *, batch_size: int = 2) -> dict[str, Any]:
    """Real, measured comparison of one forward+backward pass with vs.
    without frozen-layer CPU streaming (`chowder.memory_fabric.
    stream_frozen_layers`), at the configured `max_length` and the given
    `batch_size`. Same warmup discipline as activation_offload_worker.py:
    an unwarmed first call pays for CUDA/cuDNN kernel selection and
    allocator-cache warmup, which swamps the real streaming-transfer
    cost -- both code paths are warmed up before any timed measurement.

    Requires an available CUDA device -- there is nothing to stream a
    frozen layer's weight "off of" without a GPU, so this returns
    available=False on CPU rather than reporting a meaningless
    comparison (the same convention activation_offload/optimizer_tiering
    already use).
    """
    model, tokenizer, device, _frozen_params, _trainable_params = load_dry_run_model(spec)
    import torch

    from ..memory_fabric import stream_frozen_layers

    if device.type != "cuda":
        return {
            "available": False,
            "device": device.type,
            "reason": "frozen-layer streaming requires an available CUDA device",
        }

    def _random_inputs():
        input_ids = torch.randint(
            low=0, high=max(tokenizer.vocab_size, 2), size=(batch_size, spec.max_length), device=device
        )
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def _baseline_step() -> float:
        input_ids, attention_mask = _random_inputs()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        return elapsed

    for _ in range(_WARMUP_ITERATIONS):
        _baseline_step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_times = [_baseline_step() for _ in range(_TIMED_ITERATIONS)]
    baseline_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    streamed = stream_frozen_layers(model, device)

    def _streamed_step() -> float:
        input_ids, attention_mask = _random_inputs()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        streamed.start_step()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        return elapsed

    for _ in range(_WARMUP_ITERATIONS):
        _streamed_step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    streamed.runtime.bytes_transferred = 0
    streamed_times = [_streamed_step() for _ in range(_TIMED_ITERATIONS)]
    streamed_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    bytes_transferred_per_step = streamed.runtime.bytes_transferred // _TIMED_ITERATIONS

    streamed.restore()

    return {
        "available": True,
        "device": device.type,
        "batch_size": batch_size,
        "max_length": spec.max_length,
        "baseline_peak_vram_gb": baseline_peak_gb,
        "streamed_peak_vram_gb": streamed_peak_gb,
        "baseline_wall_seconds": statistics.median(baseline_times),
        "streamed_wall_seconds": statistics.median(streamed_times),
        "bytes_transferred_per_step": bytes_transferred_per_step,
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
