"""End-to-end validation of the hot-core converter against a prior prediction.

The mask sweep (A4B-UPCYCLING-DESIGN.md) measured, by masking channels in
activation space on the dense parent, that keeping f=0.28 of channels with a
50-75% hot core costs 2.09x-1.81x baseline perplexity. The design converted here
(E=16, top_k=2, hot core 2176, cold 632/expert) has a 63.3% core at f=0.2800, so
the sweep predicts roughly **1.9x**.

That prediction was made before the converter existed, from a completely
different code path (a forward_pre_hook on the dense model, no weight surgery at
all). If the converted checkpoint lands on it, the converter is validated
end-to-end by agreement between two independent implementations of the same
arithmetic. If it does not, one of them is wrong and the gap says which.

Stages are skippable so a failure late does not discard GPU work:
  1  rank channels on the SELECT half of the calibration split
  2  convert (stdlib byte surgery, no torch)
  3  score dense and converted on the disjoint EVAL half, same nf4 load
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

BASE = Path(r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16")
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
WORK = Path(r"F:\llm-models\_a4b")
RANK_PATH = WORK / "rank-9b-select-v1.json"
OUT_DIR = Path(r"F:\llm-models\Qwen3.8-9B-HotCore-E16-k2-h2176")
REPORT = WORK / "hot-core-validation.json"

NUM_EXPERTS, TOP_K, HOT_CORE = 16, 2, 2176
PROMPTS, MAX_LEN = 64, 384
PREDICTED_RATIO = (1.81, 2.09)   # the sweep's 75%-core and 50%-core bounds


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_texts() -> list[str]:
    rows: list[str] = []
    with CALIB.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("prompt")
            if isinstance(t, str) and t.strip():
                rows.append(t)
            if len(rows) >= PROMPTS:
                break
    return rows


def main() -> int:
    from chowder.channel_importance import (
        ChannelRanking, measure_channel_importance, split_disjoint,
    )
    from chowder.hot_core_upcycle import (
        convert_checkpoint_hot_core, plan_hot_core_conversion,
    )

    WORK.mkdir(parents=True, exist_ok=True)
    texts = load_texts()
    select, evaluate = split_disjoint(texts)
    log(f"calibration {len(texts)} -> {len(select)} select / {len(evaluate)} eval (disjoint)")

    # ---- stage 1: ranking -------------------------------------------------
    if RANK_PATH.is_file():
        ranking = ChannelRanking.read(RANK_PATH)
        log(f"reusing ranking {RANK_PATH.name} (digest {ranking.digest()[:12]})")
    else:
        log("measuring channel importance on the SELECT half")
        t0 = time.time()
        ranking = measure_channel_importance(
            BASE, select, max_length=MAX_LEN, load_in_4bit=True, split_label="even-index",
        )
        ranking.write(RANK_PATH)
        log(f"ranked in {time.time()-t0:.0f}s; top-10% mass "
            f"{ranking.concentration['top10pct_mass_mean']:.3f}; "
            f"digest {ranking.digest()[:12]}")

    # ---- stage 2: conversion ----------------------------------------------
    plan = plan_hot_core_conversion(
        BASE, ranking, num_experts=NUM_EXPERTS, top_k=TOP_K, hot_core_size=HOT_CORE)
    log(f"plan: E={plan.num_experts} top_k={plan.top_k} core={plan.hot_core_size} "
        f"c={plan.moe_intermediate_size} active={plan.active_channels} "
        f"(f={plan.active_channels/plan.intermediate_size:.4f})")
    log(f"      stored FFN {plan.stored_ffn_params/1e9:.3f}B "
        f"({plan.stored_ffn_params/plan.dense_ffn_params:.3f}x dense), "
        f"active FFN {plan.active_ffn_params/1e9:.3f}B")

    if (OUT_DIR / "conversion.provenance.json").is_file():
        provenance = json.loads((OUT_DIR / "conversion.provenance.json").read_text(encoding="utf-8"))
        log("reusing existing conversion")
    else:
        log(f"converting -> {OUT_DIR}")
        t0 = time.time()
        provenance = convert_checkpoint_hot_core(
            BASE, OUT_DIR, ranking,
            num_experts=NUM_EXPERTS, top_k=TOP_K, hot_core_size=HOT_CORE)
        log(f"converted in {time.time()-t0:.0f}s; "
            f"{provenance['storage']['actual_output_bytes']/1e9:.2f} GB out")

    # ---- stage 3: score both on the EVAL half -----------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.router_healing import QUANTIZATION_SKIP_MODULES

    def score(model_dir: Path, label: str) -> float:
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        log(f"loading {label}")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES)),
            device_map="cuda:0", dtype=torch.bfloat16)
        model.eval()
        log(f"  loaded in {time.time()-t0:.0f}s; "
            f"{torch.cuda.memory_allocated()/2**30:.2f} GiB allocated")
        total, n_tok = 0.0, 0
        with torch.no_grad():
            for text in evaluate:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 8:
                    continue
                out = model(**enc, labels=enc["input_ids"])
                n = enc["input_ids"].shape[1] - 1
                loss = out.loss.float().item()
                if not math.isfinite(loss):
                    log(f"  {label}: non-finite loss; reporting inf")
                    return float("inf")
                total += loss * n
                n_tok += n
        ppl = math.exp(total / n_tok)
        log(f"  {label} ppl = {ppl:.4f} over {n_tok} tokens")
        del model
        torch.cuda.empty_cache()
        return ppl

    dense_ppl = score(BASE, "dense parent")
    moe_ppl = score(OUT_DIR, "hot-core MoE")
    ratio = moe_ppl / dense_ppl
    lo, hi = PREDICTED_RATIO
    verdict = "AGREES" if lo * 0.9 <= ratio <= hi * 1.1 else "DISAGREES"

    log("")
    log(f"dense {dense_ppl:.4f}   converted {moe_ppl:.4f}   ratio {ratio:.3f}x")
    log(f"mask-sweep prediction for f=0.28 at 50-75% core: {lo}x-{hi}x  =>  {verdict}")

    REPORT.write_text(json.dumps({
        "base": str(BASE), "converted": str(OUT_DIR),
        "design": {"num_experts": NUM_EXPERTS, "top_k": TOP_K,
                   "hot_core_size": HOT_CORE,
                   "moe_intermediate_size": plan.moe_intermediate_size,
                   "active_channels": plan.active_channels,
                   "active_fraction": plan.active_channels / plan.intermediate_size,
                   "core_share_of_active": HOT_CORE / plan.active_channels},
        "params": plan.to_dict(),
        "eval": {"prompts": len(evaluate), "max_length": MAX_LEN,
                 "split": "odd-index (disjoint from ranking)",
                 "load": "nf4, shared_expert_gate skipped"},
        "dense_ppl": dense_ppl, "converted_ppl": moe_ppl, "ratio": ratio,
        "prediction_from_mask_sweep": {"low": lo, "high": hi, "verdict": verdict},
        "ranking_digest": ranking.digest(),
        "output_manifest_sha256": provenance["output_manifest_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
