"""Controlled A/B: does the LoRA adapter explain the candidate eval's 2x slowdown?

The observation to explain: on the re-run, the baseline arm evaluated 50 GSM8K
problems at 83 s/problem holding ~5.8 GiB, while the candidate arm -- same model,
same prompts, same settings, plus a 200-module r=16 adapter -- ran at 166 s/problem
holding ~15.1 GiB with 559 MiB of card left.

Two explanations, and they imply different verdicts against the pre-registration:

  H1 ADAPTER COMPUTE. 200 unfused LoRA modules add two small matmuls per module per
     token. Batch-1 autoregressive decoding is launch-latency bound, so this can
     cost ~2x on its own. Nothing is wrong; the arms are just not equal work.

  H2 VRAM PRESSURE. Windows WDDM oversubscribes VRAM into system RAM rather than
     raising, so a model that does not fit silently pages and crawls. This is the
     pre-registered FAIL condition ("judged by headroom and step-time blowup, never
     by an OOM exception").

The discriminator is headroom: run both arms with the whole card free. If the
adapter arm is still ~2x slower with GiB to spare, H1. If it is close to parity, the
2x during the run was pressure and H2 stands.

THE CONFOUND I ALMOST MISSED: seconds per PROBLEM is not comparable between arms if
the arms generate different numbers of tokens. A trained adapter can change where
generation stops. So the primary measure here is seconds per GENERATED TOKEN, and
token counts are reported per arm so the reader can see whether that mattered.

DECISION RULE, FIXED BEFORE THE RUN:
  * s/token ratio (adapter / no-adapter) >= 1.5 with >= 4 GiB free in both arms
        -> H1: adapter compute explains it, no oversubscription
  * ratio <= 1.2
        -> H2: adapter compute does NOT explain it; the run's 2x was VRAM pressure
  * 1.2 < ratio < 1.5
        -> partial / inconclusive, reported as such, no verdict claimed
Separately: if the adapter arm's peak reserved VRAM is within 1 GiB of the
no-adapter arm, then the 15.1 GiB seen during the run is not inherent to the
adapter and needs its own explanation.

Refuses to start unless the GPU is essentially free, because the whole point is to
measure with headroom -- running it against a busy card would reproduce the
ambiguity instead of resolving it.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

MODEL = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
ADAPTER = r"F:\llm-models\_a4b\realtrain-gsm8k-2\.chowder\runs\realtrain-unsloth-a9e4dbb91fa5\adapter"
EVAL = r"F:\llm-models\_a4b\gsm8k_test_50.jsonl"
OUT = Path(r"F:\llm-models\_a4b\ab-adapter-overhead.json")

N_PROBLEMS = 5
MAX_NEW_TOKENS = 768
SEED = 123
MIN_FREE_GIB_TO_START = 13.0
MIN_HEADROOM_GIB = 4.0
RATIO_H1 = 1.5
RATIO_H2 = 1.2


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_gib() -> float:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used = (float(x) for x in out.split(","))
    return (total - used) / 1024.0


def run_arm(label: str, *, with_adapter: bool, rows: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    from chowder.evaluators.generation import resolve_eos_token_ids

    log(f"--- arm {label}: loading (adapter={with_adapter})")
    set_seed(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Identical to transformers_text_worker's 4-bit path.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    eos = resolve_eos_token_ids(tokenizer, model)

    after_load_reserved = torch.cuda.max_memory_reserved() / 1024**3
    headroom_after_load = free_gib()
    log(f"    loaded: reserved {after_load_reserved:.2f} GiB, card free {headroom_after_load:.2f} GiB")

    per_problem: list[dict] = []
    min_headroom = headroom_after_load
    with torch.inference_mode():
        for i, row in enumerate(rows):
            encoded = tokenizer(str(row["prompt"]), return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            prompt_tokens = encoded["input_ids"].shape[1]
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            new_tokens = int(generated.shape[1] - prompt_tokens)
            min_headroom = min(min_headroom, free_gib())
            per_problem.append({
                "index": i,
                "seconds": seconds,
                "new_tokens": new_tokens,
                "seconds_per_token": seconds / max(new_tokens, 1),
                "hit_cap": new_tokens >= MAX_NEW_TOKENS,
            })
            log(f"    [{i+1}/{len(rows)}] {seconds:6.1f}s  {new_tokens:4d} tok  "
                f"{seconds/max(new_tokens,1)*1000:5.1f} ms/tok")

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    tokens = sum(p["new_tokens"] for p in per_problem)
    seconds = sum(p["seconds"] for p in per_problem)
    arm = {
        "label": label,
        "with_adapter": with_adapter,
        "problems": len(per_problem),
        "total_seconds": seconds,
        "total_new_tokens": tokens,
        "seconds_per_problem": seconds / len(per_problem),
        "seconds_per_token": seconds / max(tokens, 1),
        "seconds_per_token_stdev": (
            statistics.stdev([p["seconds_per_token"] for p in per_problem])
            if len(per_problem) > 1 else 0.0
        ),
        "hit_cap_count": sum(1 for p in per_problem if p["hit_cap"]),
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "min_card_free_gib": min_headroom,
        "per_problem": per_problem,
    }
    log(f"    arm {label}: {arm['seconds_per_token']*1000:.1f} ms/tok, "
        f"peak alloc {peak_alloc:.2f} / reserved {peak_reserved:.2f} GiB, "
        f"min free {min_headroom:.2f} GiB")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return arm


def main() -> int:
    free = free_gib()
    log(f"GPU free {free:.2f} GiB (need >= {MIN_FREE_GIB_TO_START})")
    if free < MIN_FREE_GIB_TO_START:
        log("REFUSING: the card is busy. Measuring with a busy card reproduces the "
            "ambiguity instead of resolving it.")
        return 2
    for path in (Path(MODEL) / "config.json", Path(ADAPTER) / "adapter_config.json", Path(EVAL)):
        if not path.exists():
            log(f"MISSING {path}")
            return 2

    rows = [json.loads(l) for l in Path(EVAL).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:N_PROBLEMS]
    log(f"{len(rows)} problems, max_new_tokens {MAX_NEW_TOKENS}, greedy, seed {SEED}")

    # No-adapter first: it is the cheaper arm, so a surprise here costs less.
    arm_a = run_arm("no-adapter", with_adapter=False, rows=rows)
    arm_b = run_arm("adapter", with_adapter=True, rows=rows)

    ratio_token = arm_b["seconds_per_token"] / arm_a["seconds_per_token"]
    ratio_problem = arm_b["seconds_per_problem"] / arm_a["seconds_per_problem"]
    headroom_ok = min(arm_a["min_card_free_gib"], arm_b["min_card_free_gib"]) >= MIN_HEADROOM_GIB
    vram_delta = arm_b["peak_reserved_gib"] - arm_a["peak_reserved_gib"]

    if ratio_token >= RATIO_H1 and headroom_ok:
        verdict = "H1: adapter compute explains the slowdown; no oversubscription"
    elif ratio_token <= RATIO_H2:
        verdict = "H2: adapter compute does NOT explain it; the run's 2x was VRAM pressure"
    else:
        verdict = f"INCONCLUSIVE: ratio {ratio_token:.2f} falls between {RATIO_H2} and {RATIO_H1}"

    report = {
        "model": MODEL, "adapter": ADAPTER, "problems": len(rows),
        "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
        "decision_rule": {
            "h1_if_ratio_at_least": RATIO_H1, "h2_if_ratio_at_most": RATIO_H2,
            "min_headroom_gib": MIN_HEADROOM_GIB,
        },
        "arms": [arm_a, arm_b],
        "ratio_seconds_per_token": ratio_token,
        "ratio_seconds_per_problem": ratio_problem,
        "headroom_sufficient_in_both_arms": headroom_ok,
        "adapter_vram_cost_gib": vram_delta,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    log("=== A/B RESULT ===")
    for arm in (arm_a, arm_b):
        log(f"  {arm['label']:<12} {arm['seconds_per_problem']:7.1f} s/problem  "
            f"{arm['seconds_per_token']*1000:6.1f} ms/tok  "
            f"{arm['total_new_tokens']:5d} tok  cap {arm['hit_cap_count']}/{arm['problems']}  "
            f"reserved {arm['peak_reserved_gib']:.2f} GiB  min free {arm['min_card_free_gib']:.2f}")
    log(f"  ratio per token   : {ratio_token:.2f}x")
    log(f"  ratio per problem : {ratio_problem:.2f}x")
    log(f"  adapter VRAM cost : {vram_delta:+.2f} GiB")
    log(f"  VERDICT: {verdict}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
