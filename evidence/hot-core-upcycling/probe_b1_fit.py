"""Does B1's upcycle arm fit in 16 GiB with optimizer state? Measure, don't estimate.

The converted model's inference peak was 15.07 GiB on a 15.93 GiB card, so the
question is not academic. This reproduces the B1 trainer's exact memory recipe
(training/train_grpo_minimal.py + run_grpo_v5_borderline.py):

  nf4 / bf16 compute / double quant, prepare_model_for_kbit_training with
  gradient checkpointing, LoRA r=16 alpha=32 on the "broad" target set, AdamW,
  then a GRPO step shape: num_generations=4 rollouts of up to 400 new tokens,
  followed by forward+backward over those completions.

It reports peak allocated/reserved at each stage and, just as important, WHICH
modules LoRA actually attached to. That second question has a structural answer
for a converted MoE: PEFT wraps `nn.Linear`, and `Qwen3_5MoeExperts.gate_up_proj`
/ `down_proj` plus the router's `mlp.gate.weight` are raw `nn.Parameter`. So the
"broad" set's gate_proj/up_proj/down_proj can only land on the SHARED expert, and
neither the routed expert bank nor the router can receive a LoRA adapter at all.

Run with --which dense to get the reference arm's numbers on the same card.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
MOE = r"F:\llm-models\Qwen3.8-9B-HotCore-E16-k2-h2176"
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")

BROAD_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj",
                 "in_proj_qkv", "in_proj_z", "out_proj"]
RANK, NUM_GEN, MAX_COMP = 16, 4, 400


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["moe", "dense"], default="moe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_dir = MOE if args.which == "moe" else DENSE

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    GIB = 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    stages: dict[str, dict[str, float]] = {}

    def mark(label: str) -> None:
        stages[label] = {
            "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
        }
        s = stages[label]
        log(f"  {label:<28} alloc {s['allocated_gib']:>6.2f}  "
            f"reserved {s['reserved_gib']:>6.2f}  peak_res {s['peak_reserved_gib']:>6.2f} GiB")

    log(f"card total {total:.2f} GiB; probing {args.which} at {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=qcfg, dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True, trust_remote_code=False)
    mark("after load")

    # what kind of parameter holds the bulk, and is it quantized?
    by_kind: dict[str, dict[str, float]] = {}
    for name, p in model.named_parameters():
        if ".mlp.experts." in name:
            kind = "routed_expert_bank"
        elif name.endswith("mlp.gate.weight"):
            kind = "router"
        elif ".mlp.shared_expert." in name or name.endswith("mlp.shared_expert_gate.weight"):
            kind = "shared_expert"
        elif ".mlp." in name:
            kind = "dense_mlp"
        else:
            kind = "other"
        e = by_kind.setdefault(kind, {"params": 0, "bytes": 0, "quantized_bytes": 0})
        n = p.numel()
        esz = p.element_size()
        e["params"] += n
        e["bytes"] += n * esz
        if type(p).__name__ in {"Params4bit", "Int8Params"} or esz == 1:
            e["quantized_bytes"] += n * esz
    log("  resident weight by kind:")
    for kind, e in sorted(by_kind.items(), key=lambda kv: -kv[1]["bytes"]):
        log(f"    {kind:<20} {e['params']/1e9:>6.3f}B params  {e['bytes']/1e9:>6.2f} GB  "
            f"quantized {100*e['quantized_bytes']/max(e['bytes'],1):>5.1f}%")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    mark("after kbit prep")

    lcfg = LoraConfig(r=RANK, lora_alpha=RANK * 2, target_modules=BROAD_MODULES,
                      lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lcfg)
    mark("after LoRA attach")

    # WHERE did LoRA land? For a converted MoE this is the structural question.
    hit = Counter()
    for name, _ in model.named_modules():
        if not name.endswith("lora_A.default"):
            continue
        base = name[: -len(".lora_A.default")]
        leaf = base.split(".")[-1]
        if ".mlp.shared_expert." in base:
            hit[f"shared_expert.{leaf}"] += 1
        elif ".mlp.experts." in base:
            hit[f"ROUTED_EXPERT.{leaf}"] += 1
        elif ".visual." in base:
            hit[f"vision.{leaf}"] += 1
        elif ".linear_attn." in base:
            hit[f"linear_attn.{leaf}"] += 1
        elif ".self_attn." in base:
            hit[f"self_attn.{leaf}"] += 1
        else:
            hit[f"other.{leaf}"] += 1
    log("  LoRA adapters attached:")
    for k, v in sorted(hit.items()):
        log(f"    {k:<32} x{v}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trainable params {trainable/1e6:.2f}M")
    log(f"  routed expert bank adapters: {sum(v for k, v in hit.items() if 'ROUTED_EXPERT' in k)}"
        "  <- PEFT wraps nn.Linear only; the bank is raw nn.Parameter")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)

    prompt = json.loads(CALIB.read_text(encoding="utf-8").splitlines()[0])["prompt"]
    enc = tok(prompt, return_tensors="pt").to(model.device)
    log(f"  prompt {enc['input_ids'].shape[-1]} tokens; generating {NUM_GEN}x{MAX_COMP}")

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=True,
                             temperature=0.5, num_return_sequences=NUM_GEN,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    mark("after generation")
    log(f"  generation {time.time()-t0:.1f}s -> {tuple(gen.shape)}")

    model.train()
    t0 = time.time()
    out = model(input_ids=gen, labels=gen)
    mark("after forward")
    out.loss.backward()
    mark("after backward")
    optimizer.step()
    mark("after optimizer.step")
    optimizer.zero_grad(set_to_none=True)
    log(f"  fwd+bwd+step {time.time()-t0:.1f}s; loss {out.loss.item():.4f}")

    peak = stages["after optimizer.step"]["peak_reserved_gib"]
    headroom = total - peak
    log("")
    log(f"VERDICT {args.which}: peak reserved {peak:.2f} GiB of {total:.2f} GiB "
        f"=> headroom {headroom:.2f} GiB ({'FITS' if headroom > 0 else 'OOM'})")

    payload = {"which": args.which, "model_dir": model_dir,
               "card_total_gib": round(total, 3), "stages": stages,
               "weight_by_kind": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                  for k, v in by_kind.items()},
               "lora_attachments": dict(hit), "trainable_params": trainable,
               "recipe": {"rank": RANK, "targets": BROAD_MODULES,
                          "num_generations": NUM_GEN, "max_completion": MAX_COMP,
                          "gradient_checkpointing": True, "optimizer": "AdamW"},
               "peak_reserved_gib": peak, "headroom_gib": round(headroom, 3)}
    out_path = args.out or str(Path(__file__).with_name(f"b1_fit_{args.which}.json"))
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log(f"OOM: {exc}")
            log("VERDICT: DOES NOT FIT")
            raise SystemExit(2)
        raise
