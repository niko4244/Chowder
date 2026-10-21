"""Produce the four witness gates from a real int8 placement fine-tune.

``chowder_batch/moe_int8_gates.py`` can measure ``load_census``, ``host_rss``,
``memory_vram`` and ``throughput`` on the card, but the ``gradient``, ``smoke``
and ``router_balance`` gates need an optimizer, a dataset and two router
windows -- i.e. a training run. This script is the smallest run that is still
one: one artifact load, LoRA on the frozen-expert preset, AdamW, activation
checkpointing, and the evidence ``chowder.placement_witnesses`` derives those
gates from, written as the ``placement-witnesses.json`` the gate runner's
``--witness`` mode records.

It takes no measurement the gate runner could have taken, and it never writes
the table itself: the runner refuses a witness whose artifact digest differs
from the table's, so this file can only ever narrow the table to what this card
actually observed.

    python chowder_batch/moe_int8_witness_run.py --base <artifact> \
        --calib calibration.jsonl --steps 24 --out placement-witnesses.json

``--self-check`` exercises the evidence-to-verdict path on synthetic evidence
(no card, no weights) and prints the four verdicts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _fake_witness() -> dict[str, Any]:
    """The evidence a healthy run produces, at the shapes ``derive`` expects."""
    return {
        "gradient": {
            "layers": [0, 19, 39],
            "leaves": 20,
            "missing_grad": 0,
            "nonfinite": 0,
            "zero": 0,
            "min_abs": 1e-5,
            "per_layer": {str(layer): {"leaves": 7} for layer in (0, 19, 39)},
        },
        "router": {
            "step0": {"0": [8] * 160},
            "last": {"0": [9, 7] + [8] * 158},
        },
        "step_log": [{"step": step + 1, "loss": 2.0 - 0.02 * step} for step in range(20)],
        "gpu": {"peak_allocated_gb": 10.0, "peak_reserved_gb": 13.0, "total_gb": 15.93},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="", help="real int8-loadable checkpoint directory")
    parser.add_argument("--calib", default="", help="calibration JSONL used for the training text")
    parser.add_argument("--out", default="", help="where to write placement-witnesses.json")
    parser.add_argument("--losses", default="", help="optional path for the raw step log")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=24, help="optimizer steps; the smoke gate needs 20")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    from chowder.placement_witnesses import derive

    if args.self_check:
        verdicts = derive(**_fake_witness())
        for name, verdict in verdicts.items():
            print(f"{name}: {verdict.get('verdict')} - {verdict.get('value')}")
        if verdicts["gradient"].get("verdict") != "pass":
            raise SystemExit("self-check: a healthy gradient evidence set did not pass")
        if verdicts["smoke"].get("verdict") != "pass":
            raise SystemExit("self-check: a falling 20-step loss did not pass the smoke gate")
        print("self-check ok")
        return 0

    missing = [name for name, value in (("--base", args.base), ("--calib", args.calib), ("--out", args.out)) if not value]
    if missing:
        parser.error(f"missing {', '.join(missing)} (or pass --self-check)")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer

    from chowder.backends import moe_int8_experts as mie
    from chowder.backends.transformers_worker import _resolve_target_modules
    from chowder.placement_witnesses import (
        RouterLoadRecorder,
        artifact_identity,
        collect_lora_grads,
        gpu_snapshot,
        witness_record,
    )

    out = Path(args.out)
    report: dict[str, Any] = {"base": args.base, "seq": args.seq, "steps": args.steps, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    torch.manual_seed(20261018)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model, census = mie.load_model_int8_experts(args.base, device=0)
    report["load_seconds"] = round(time.perf_counter() - started, 1)
    report["census"] = census

    # The router recorder audits ``model.model.layers``, which is the text stack
    # only *before* PEFT wraps the model: after ``get_peft_model`` the trained
    # object's ``.model`` is the LoRA wrapper and the audit refuses it (as it
    # should -- it will not guess an alternate path). Keep the base model, and
    # hand that to the recorder. The hooks land on the same ``mlp.gate`` modules
    # either way: PEFT wraps leaves, it does not rebuild the block.
    placement_model = model

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
    batches = [stream[i * args.seq:(i + 1) * args.seq].unsqueeze(0).cuda() for i in range(min(args.examples, len(stream) // args.seq))]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    entries: list[dict[str, Any]] = []
    gradient_evidence: dict[str, Any] | None = None
    step0_window: dict[str, list[int]] = {}
    last_window: dict[str, list[int]] = {}

    with RouterLoadRecorder(placement_model) as recorder:
        for step in range(args.steps):
            ids = batches[step % len(batches)]
            torch.cuda.synchronize()
            begin = time.perf_counter()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            if gradient_evidence is None:
                gradient_evidence = collect_lora_grads(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            entries.append({"step": step + 1, "loss": round(float(loss), 6), "seconds": round(time.perf_counter() - begin, 4)})
            del loss
            if step == 0:
                step0_window = recorder.window()
                recorder.reset()
            print(f"step {step}: {entries[-1]['seconds']:.2f}s loss {entries[-1]['loss']}", flush=True)
        last_window = recorder.window()

    evidence = {
        "gradient": gradient_evidence,
        "router": {"step0": step0_window, "last": last_window},
        "step_log": entries,
        "gpu": gpu_snapshot(torch, 0),
    }
    gates = derive(**evidence)
    record = witness_record(
        gates, base_model=args.base, device="cuda:0", steps=args.steps, run="moe_int8_witness_run"
    )
    record["evidence"] = evidence
    record["artifact_identity"] = artifact_identity(args.base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    report["witness"] = str(out)
    report["gates"] = {name: verdict.get("verdict") for name, verdict in gates.items()}
    if args.losses:
        Path(args.losses).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
