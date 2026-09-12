"""Does the router-healing arm fit in 16 GiB, and does the router actually learn?

The B1 LoRA arm as configured does not fit: `prepare_model_for_kbit_training`
casts every non-quantized parameter to fp32 (peft's own comment: "cast all non
INT8 parameters to fp32"), and the routed expert bank is raw `nn.Parameter`, so
7.95 GB of FROZEN, un-adaptable weight becomes 15.9 GB. Measured jump 12.64 ->
23.84 GiB allocated on a 15.93 GiB card; one 4x400 rollout then ran >14 min
without finishing, against 133 s for the whole dense step.

This probes the alternative, which is the only path that trains the thing the
MoE exists for:

  * no `prepare_model_for_kbit_training`, so the expert bank stays BF16;
  * `freeze_for_router_healing` -- trains ONLY `mlp.gate.weight` and
    `mlp.shared_expert_gate.weight`, every other tensor frozen;
  * gradient checkpointing enabled directly;
  * AdamW over the ~2.2M trainable params;
  * a real forward/backward at several sequence lengths.

It also checks two things that are claims, not assumptions:
  1. `assert_trainable_gradients_reachable` should now PASS. It refuses a freeze
     plan whose shared expert is all-zero and frozen, because then
     d/d(shared_expert_gate) is exactly 0. The hot core makes the shared expert
     non-zero, so the gate becomes learnable -- the claimed corollary of putting
     the core there. This is where that claim gets tested.
  2. The router must receive a non-zero gradient. A zero-init router with tied
     logits is reproducible but cold-starts the bank; if the gradient is zero the
     arm cannot work regardless of memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

MOE = r"F:\llm-models\Qwen3.8-9B-HotCore-E16-k2-h2176"
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MOE)
    ap.add_argument("--lengths", default="384,512,768,1024,1536,2048")
    ap.add_argument("--out", default=str(Path(__file__).with_name("healing_fit.json")))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.router_healing import (
        QUANTIZATION_SKIP_MODULES,
        RouterHealingError,
        assert_trainable_gradients_reachable,
        freeze_for_router_healing,
        select_trainable_parameter_names,
    )

    GIB = 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    stages: dict[str, dict] = {}

    def mark(label: str) -> None:
        stages[label] = {
            "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
        }
        s = stages[label]
        log(f"  {label:<26} alloc {s['allocated_gib']:>6.2f}  peak_res {s['peak_reserved_gib']:>6.2f} GiB")

    log(f"card {total:.2f} GiB; {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES)),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    mark("after load")

    # NOTE: deliberately NOT prepare_model_for_kbit_training -- that is the
    # fp32 upcast that breaks the LoRA arm.
    summary = freeze_for_router_healing(model)
    log(f"  trainable tensors: {len(summary.trainable_param_names)} "
        f"({summary.trainable_param_count/1e6:.3f}M params), "
        f"frozen {summary.frozen_param_count/1e9:.3f}B")
    log(f"  layers with router gate: {summary.layers_with_trainable_gate}, "
        f"with shared_expert_gate: {summary.layers_with_trainable_shared_expert_gate}")

    # claim under test #1: the hot core makes shared_expert_gate reachable
    names = select_trainable_parameter_names(list(model.named_parameters()))
    try:
        report = assert_trainable_gradients_reachable(model, names)
        reachable, reach_err = True, None
        log(f"  gradient reachability: PASS ({report['checked']} checked, 0 dead)")
        log("    => the hot core in shared_expert removed the zero-gradient defect")
    except RouterHealingError as exc:
        reachable, reach_err = False, str(exc)
        log(f"  gradient reachability: REFUSED -- {exc}")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    mark("after freeze + ckpt")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

    prompts = [json.loads(l)["prompt"] for l in
               CALIB.read_text(encoding="utf-8").splitlines()[:160] if l.strip()]
    text = "\n\n".join(prompts)

    results = []
    router_grad_norms = []
    shared_gate_grad_norms = []
    for length in [int(x) for x in args.lengths.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        actual = int(enc["input_ids"].shape[-1])
        model.train()
        t0 = time.time()
        try:
            out = model(**enc, labels=enc["input_ids"])
            out.loss.backward()
            # claim under test #2: does the router get gradient?
            rn = sum(float(p.grad.float().norm()) ** 2
                     for n, p in model.named_parameters()
                     if p.grad is not None and n.endswith("mlp.gate.weight")) ** 0.5
            sn = sum(float(p.grad.float().norm()) ** 2
                     for n, p in model.named_parameters()
                     if p.grad is not None and n.endswith("mlp.shared_expert_gate.weight")) ** 0.5
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_reserved() / GIB
            router_grad_norms.append(rn)
            shared_gate_grad_norms.append(sn)
            results.append({"seq_len": actual, "ok": True, "seconds": round(dt, 2),
                            "peak_reserved_gib": round(peak, 3),
                            "headroom_gib": round(total - peak, 3),
                            "router_grad_norm": rn, "shared_gate_grad_norm": sn})
            log(f"  len {actual:<5} step {dt:>6.2f}s  peak {peak:>6.2f} GiB "
                f"(headroom {total-peak:>5.2f})  |grad| router {rn:.3e} "
                f"shared_gate {sn:.3e}")
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"seq_len": actual, "ok": False,
                            "error": "CUDA OOM", "detail": str(exc)[:200]})
            log(f"  len {actual:<5} OOM")
            torch.cuda.empty_cache()

    # "Did it OOM?" is the WRONG instrument on this platform. Windows WDDM lets
    # CUDA oversubscribe VRAM into system RAM, so torch.cuda.OutOfMemoryError
    # never fires -- the step silently pages over PCIe instead. An earlier
    # version of this script therefore reported "FITS" at seq 2048 while that
    # step took 137 s against 7 s at 768. Judge on headroom and step-time blowup.
    THRASH_FACTOR = 3.0
    ran = [r for r in results if r.get("ok")]
    fits = [r for r in ran if r["headroom_gib"] > 0]
    baseline = min((r["seconds"] for r in fits), default=None)
    for r in ran:
        if r["headroom_gib"] <= 0:
            r["verdict"] = ("thrashing" if baseline and r["seconds"] > THRASH_FACTOR * baseline
                            else "oversubscribed")
        else:
            r["verdict"] = "fits"
    log("")
    if fits:
        ceiling = max(fits, key=lambda r: r["seq_len"])
        log(f"VERDICT: fits to seq {ceiling['seq_len']} at {ceiling['seconds']:.2f}s/step, "
            f"peak {ceiling['peak_reserved_gib']:.2f} of {total:.2f} GiB "
            f"(headroom {ceiling['headroom_gib']:.2f} GiB)")
        over = [r for r in ran if r["headroom_gib"] <= 0]
        if over:
            log(f"         beyond that it oversubscribes rather than erroring: "
                + ", ".join(f"seq {r['seq_len']} {r['seconds']:.0f}s ({r['verdict']})"
                            for r in over))
    else:
        log("VERDICT: does NOT fit at any tested length")
    if ran and all(r["router_grad_norm"] > 0 for r in ran):
        log("         router gradient is non-zero at every length -- it can learn")
    else:
        log("         WARNING: zero router gradient somewhere -- it cannot learn")
    ok = fits

    Path(args.out).write_text(json.dumps({
        "model": args.model, "card_total_gib": round(total, 3),
        "recipe": {"prepare_model_for_kbit_training": False,
                   "trainable": "mlp.gate.weight + mlp.shared_expert_gate.weight",
                   "gradient_checkpointing": True, "optimizer": "AdamW",
                   "quantization_skip_modules": list(QUANTIZATION_SKIP_MODULES)},
        "trainable_params": summary.trainable_param_count,
        "frozen_params": summary.frozen_param_count,
        "layers_with_router_gate": summary.layers_with_trainable_gate,
        "layers_with_shared_expert_gate": summary.layers_with_trainable_shared_expert_gate,
        "gradient_reachability_pass": reachable,
        "gradient_reachability_error": reach_err,
        "stages": stages, "steps": results,
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
