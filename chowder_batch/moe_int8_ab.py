"""A/B the int8 placement's two expert forms and the weight-retention switch.

One real artifact load, then the same training step (LoRA + AdamW + activation
checkpointing) under each configuration in turn, so the comparison cannot be
explained by a load or by a different seed:

    warmup (loop)  -> first touch of the 20 GB pageable store, not compared
    loop           -> grouped_experts(False), retain_recomputed_weights(False)
    grouped        -> grouped_experts(True),  retain_recomputed_weights(False)
    grouped+retain -> grouped_experts(True),  retain_recomputed_weights(True)
    loop (repeat)  -> the first arm again, to expose drift between phases

Why the step includes LoRA and the optimizer: those are what a campaign pays for,
and the placement's own cost is only meaningful inside them. The gate runner's
``_placement_steps`` deliberately measures the placement alone; this script is the
other half -- the throughput and peak memory of the step the gates are about.

Every number is measured on the card the run reports; nothing here is a pass/fail
gate. A phase that did not finish is reported as incomplete, not as a time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def summarize_phases(phases: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-phase mean step time (warm steps only) and the comparison between them.

    The first step of a phase is kept in ``step_seconds`` but left out of the mean:
    it pays allocator growth and, for the first phase, the store's first read. A
    phase with fewer than two steps has no warm mean and says so.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for phase in phases:
        seconds = [float(value) for value in phase.get("step_seconds") or []]
        warm = seconds[1:]
        by_name[phase["name"]] = {
            "s_per_step": round(sum(warm) / len(warm), 4) if warm else None,
            "steps_measured": len(warm),
            "peak_alloc_gb": phase.get("peak_alloc_gb"),
            "peak_reserved_gb": phase.get("peak_reserved_gb"),
            "loss_first": (phase.get("losses") or [None])[0],
            "loss_last": (phase.get("losses") or [None])[-1],
        }

    def speedup(numerator: str, denominator: str) -> float | None:
        a, b = by_name.get(numerator, {}).get("s_per_step"), by_name.get(denominator, {}).get("s_per_step")
        if not a or not b:
            return None
        return round(a / b, 3)

    def delta(field: str, a: str, b: str) -> float | None:
        left, right = by_name.get(a, {}).get(field), by_name.get(b, {}).get(field)
        if left is None or right is None:
            return None
        return round(float(left) - float(right), 3)

    return {
        "phases": by_name,
        "grouped_speedup": speedup("loop", "grouped"),
        "retain_speedup": speedup("grouped", "grouped+retain"),
        "drift": speedup("loop", "loop (repeat)"),
        "grouped_peak_reserved_delta_gb": delta("peak_reserved_gb", "grouped", "loop"),
        "retain_peak_reserved_delta_gb": delta("peak_reserved_gb", "grouped+retain", "grouped"),
        "retain_peak_alloc_delta_gb": delta("peak_alloc_gb", "grouped+retain", "grouped"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="", help="real int8-loadable checkpoint directory")
    parser.add_argument("--calib", default="", help="calibration JSONL used for the training text")
    parser.add_argument("--out", default="")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps-per-phase", type=int, default=3)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--self-check", action="store_true", help="print the pure comparison on fake phases")
    args = parser.parse_args(argv)

    if args.self_check:
        fake = [
            {"name": "loop", "step_seconds": [99.0, 10.0, 12.0], "peak_alloc_gb": 1.0, "peak_reserved_gb": 2.0},
            {"name": "grouped", "step_seconds": [40.0, 6.0, 7.0], "peak_alloc_gb": 1.1, "peak_reserved_gb": 2.1},
            {"name": "grouped+retain", "step_seconds": [30.0, 5.0, 5.0], "peak_alloc_gb": 2.0, "peak_reserved_gb": 3.0},
            {"name": "loop (repeat)", "step_seconds": [20.0, 11.0, 11.0], "peak_alloc_gb": 1.0, "peak_reserved_gb": 2.0},
        ]
        print(json.dumps(summarize_phases(fake), indent=2))
        return 0

    missing = [name for name, value in (("--base", args.base), ("--calib", args.calib), ("--out", args.out)) if not value]
    if missing:
        parser.error(f"missing {', '.join(missing)} (or pass --self-check)")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer

    from chowder.backends import moe_int8_experts as mie
    from chowder.backends.moe_transient_grad import grouped_experts
    from chowder.backends.transformers_worker import _resolve_target_modules

    out = Path(args.out)
    report: dict[str, Any] = {
        "base": args.base,
        "seq": args.seq,
        "steps_per_phase": args.steps_per_phase,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    torch.manual_seed(20261018)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model, census = mie.load_model_int8_experts(args.base, device=0)
    report["load_seconds"] = round(time.perf_counter() - started, 1)
    report["load_census"] = census
    report["census"] = mie.prepare_int8_placement(model)
    save()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    targets = _resolve_target_modules(model, explicit=(), preset="moe_frozen_experts")
    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules=targets, task_type=TaskType.CAUSAL_LM),
    )
    report["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = [json.loads(line) for line in Path(args.calib).read_text(encoding="utf-8").splitlines() if line.strip()]
    stream = tokenizer("\n\n".join(row["text"] for row in rows), return_tensors="pt")["input_ids"][0]
    batches = [
        stream[i * args.seq:(i + 1) * args.seq].unsqueeze(0).cuda()
        for i in range(min(args.examples, len(stream) // args.seq))
    ]
    report["examples"] = len(batches)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    phases = [
        ("warmup", False, False),
        ("loop", False, False),
        ("grouped", True, False),
        ("grouped+retain", True, True),
        ("loop (repeat)", False, False),
    ]
    results: list[dict[str, Any]] = []
    for name, grouped, retain in phases:
        grouped_experts(grouped)
        mie.retain_recomputed_weights(retain)
        seconds: list[float] = []
        losses: list[float] = []
        peak_alloc = peak_reserved = 0.0
        for step in range(args.steps_per_phase):
            ids = batches[step % len(batches)]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds.append(round(time.perf_counter() - start, 4))
            losses.append(round(float(loss), 6))
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**30)
            del loss
            print(
                f"{name} step {step}: {seconds[-1]:.2f}s "
                f"alloc {peak_alloc:.2f} GB reserved {peak_reserved:.2f} GB",
                flush=True,
            )
        results.append(
            {
                "name": name,
                "grouped": grouped,
                "retain": retain,
                "step_seconds": seconds,
                "losses": losses,
                "peak_alloc_gb": round(peak_alloc, 3),
                "peak_reserved_gb": round(peak_reserved, 3),
            }
        )
        report["phases"] = results
        report["comparison"] = summarize_phases(results)
        save()

    grouped_experts(True)
    mie.retain_recomputed_weights(False)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["note"] = (
        "step_seconds is one optimizer step on one sequence of --seq tokens, LoRA + AdamW, "
        "activation checkpointing on; the first step of a phase is excluded from s_per_step"
    )
    save()
    print(json.dumps(report["comparison"], indent=2))
    if not all(math.isfinite(value) for phase in results for value in phase["step_seconds"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
