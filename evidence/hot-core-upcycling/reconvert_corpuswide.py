"""Re-convert the hot core from the corpus-wide ranking. Ranking is the ONLY change.

Design held identical to the first checkpoint (E=16, top_k=2, core 2176, cold
632/expert, 3,440 active of 12,288) so nothing confounds the comparison. The
pilot's gate asked for a larger core and the design sweep agrees, but changing two
things at once would make the result unreadable; a larger-core variant is a
separate experiment.

Why this run exists: the first checkpoint was built on a ranking measured over 32
CONTIGUOUS prompts, and that ranking turned out to be parochial — static pruning
under it cost 1.691x near the ranking data and 3.646x far from it, while the same
number of prompts spread across the corpus cost 1.859x and 3.011x. The trained MoE
then LOST to the corpus-wide static prune by 13.2%. So the open question is whether
the hot-core init is any good once the ranking is not sabotaging it.

Three stages, and stage 1 matters as much as stage 3:

  1 PREDICT, before any bytes move. Mask the dense parent with exactly the channel
    set the new checkpoint will compute at init (top 2,176 UNION the round-robin
    cold slices of experts 0 and 1) and score it. This is arithmetically what the
    converted model computes, so it is a real prediction, made first.
  2 CONVERT.
  3 VERIFY the real checkpoint against that prediction, and against every
    reference now on the table.

Agreement between stages 1 and 3 is what validated the first converter (predicted
1.81-2.09x, measured 1.853x). Disagreement here would mean the converter mishandles
this ranking, and the drift is the diagnostic.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
RANK = Path(r"F:\llm-models\_a4b\rank-9b-corpuswide-v1.json")
OUT_DIR = Path(r"F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176")
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
REPORT = Path(r"F:\llm-models\_a4b\reconvert-corpuswide.json")

NUM_EXPERTS, TOP_K, HOT_CORE, EVAL_MAXLEN = 16, 2, 2176, 384

# everything already measured, for the comparison table
REF = {
    "eval_a": {"dense": 5.2707, "static_contig": 8.9111, "static_cw": 9.7968,
               "converted_contig_init": 9.8444},
    "eval_b": {"dense": 4.3044, "static_contig": 15.6918, "static_cw": 12.9605,
               "converted_contig_init": 19.4255, "contig_trained_final": 14.9314},
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.channel_importance import ChannelRanking
    from chowder.hot_core_upcycle import (
        HotCoreScheme, convert_checkpoint_hot_core, plan_hot_core_conversion,
    )
    from chowder.router_healing import QUANTIZATION_SKIP_MODULES

    ranking = ChannelRanking.read(RANK)
    log(f"ranking {ranking.digest()[:12]} split={ranking.calibration.get('split')}")
    rows = [json.loads(l)["prompt"] for l in
            CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    evals = {"eval_a": [rows[i] for i in range(1, 64, 2)],
             "eval_b": [rows[i] for i in range(632, 696)]}

    plan = plan_hot_core_conversion(
        DENSE, ranking, num_experts=NUM_EXPERTS, top_k=TOP_K, hot_core_size=HOT_CORE)
    log(f"plan: E={plan.num_experts} k={plan.top_k} core={plan.hot_core_size} "
        f"c={plan.moe_intermediate_size} active={plan.active_channels} "
        f"stored {plan.stored_ffn_params/1e9:.3f}B "
        f"({plan.stored_ffn_params/plan.dense_ffn_params:.3f}x dense)")

    def score(model, tok, prompts) -> float:
        total, n = 0.0, 0
        with torch.no_grad():
            for text in prompts:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=EVAL_MAXLEN)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 8:
                    continue
                loss = model(**enc, labels=enc["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    return float("inf")
                total += loss * (enc["input_ids"].shape[1] - 1)
                n += enc["input_ids"].shape[1] - 1
        return math.exp(total / n)

    # ---------------- stage 1: predict via mask on the dense parent ----------------
    log("STAGE 1 - predicting the init by masking the dense parent")
    tok = AutoTokenizer.from_pretrained(DENSE, local_files_only=True, trust_remote_code=False)
    dense = AutoModelForCausalLM.from_pretrained(
        DENSE, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    dense.eval()
    dense.config.use_cache = False
    inner = getattr(dense, "model", dense)
    inner = getattr(inner, "language_model", inner)
    mlps = [(i, l.mlp) for i, l in enumerate(inner.layers)
            if getattr(l, "mlp", None) is not None and hasattr(l.mlp, "down_proj")
            and hasattr(l.mlp, "gate_proj")]
    inter = mlps[0][1].down_proj.in_features
    spec: dict = {"mask": None}

    def hook(idx):
        def f(_m, inputs):
            if spec["mask"] is None:
                return None
            return (inputs[0] * spec["mask"][idx],) + tuple(inputs[1:])
        return f
    handles = [m.down_proj.register_forward_pre_hook(hook(i)) for i, m in mlps]

    predicted = {}
    try:
        masks = {}
        for i, _ in mlps:
            scheme = HotCoreScheme.from_ranking(
                ranking, i, num_experts=NUM_EXPERTS, top_k=TOP_K, hot_core_size=HOT_CORE)
            kept = list(scheme.hot) + [c for e in range(TOP_K) for c in scheme.cold_by_expert[e]]
            assert len(set(kept)) == plan.active_channels
            m = torch.zeros(inter, device=dense.device, dtype=torch.bfloat16)
            m[torch.tensor(kept, device=dense.device)] = 1.0
            masks[i] = m
        spec["mask"] = masks
        for key, prompts in evals.items():
            predicted[key] = score(dense, tok, prompts)
            log(f"  predicted init {key}: {predicted[key]:.4f} "
                f"({predicted[key]/REF[key]['dense']:.3f}x dense)")
        spec["mask"] = None
    finally:
        for h in handles:
            h.remove()
    del dense
    torch.cuda.empty_cache()

    # ---------------- stage 2: convert ----------------
    if (OUT_DIR / "conversion.provenance.json").is_file():
        provenance = json.loads((OUT_DIR / "conversion.provenance.json").read_text(encoding="utf-8"))
        log("STAGE 2 - reusing existing conversion")
    else:
        log(f"STAGE 2 - converting -> {OUT_DIR}")
        t0 = time.time()
        provenance = convert_checkpoint_hot_core(
            DENSE, OUT_DIR, ranking,
            num_experts=NUM_EXPERTS, top_k=TOP_K, hot_core_size=HOT_CORE)
        log(f"  converted in {time.time()-t0:.0f}s, "
            f"{provenance['storage']['actual_output_bytes']/1e9:.2f} GB")

    # ---------------- stage 3: verify the real checkpoint ----------------
    log("STAGE 3 - scoring the real converted checkpoint")
    tok2 = AutoTokenizer.from_pretrained(OUT_DIR, local_files_only=True, trust_remote_code=False)
    moe = AutoModelForCausalLM.from_pretrained(
        OUT_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES)),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    moe.eval()
    moe.config.use_cache = False
    measured = {}
    for key, prompts in evals.items():
        measured[key] = score(moe, tok2, prompts)
        drift = measured[key] - predicted[key]
        log(f"  measured {key}: {measured[key]:.4f} "
            f"({measured[key]/REF[key]['dense']:.3f}x dense)  "
            f"vs predicted {predicted[key]:.4f} (drift {drift:+.4f})")

    # ---------------- the comparison table ----------------
    log("")
    log("init perplexity, contiguous vs corpus-wide ranking (design identical):")
    log(f"{'':<34}{'eval A':>12}{'eval B':>12}")
    for label, ka, kb in (
        ("dense parent", REF["eval_a"]["dense"], REF["eval_b"]["dense"]),
        ("static prune, contiguous rank", REF["eval_a"]["static_contig"], REF["eval_b"]["static_contig"]),
        ("static prune, corpus-wide rank", REF["eval_a"]["static_cw"], REF["eval_b"]["static_cw"]),
        ("converted init, contiguous", REF["eval_a"]["converted_contig_init"], REF["eval_b"]["converted_contig_init"]),
        ("converted init, CORPUS-WIDE", measured["eval_a"], measured["eval_b"]),
    ):
        log(f"  {label:<32}{ka:>12.4f}{kb:>12.4f}")
    log("")
    for key in ("eval_a", "eval_b"):
        old_init = REF[key]["converted_contig_init"]
        new_init = measured[key]
        cw_static = REF[key]["static_cw"]
        log(f"{key}: init {old_init:.4f} -> {new_init:.4f} "
            f"({100*(new_init/old_init-1):+.1f}%); vs its corpus-wide static "
            f"reference {cw_static:.4f} -> {100*(new_init/cw_static-1):+.1f}%")
    log("")
    gap_b = measured["eval_b"] / REF["eval_b"]["static_cw"] - 1.0
    log(f"What healing must now close on eval B: {100*gap_b:+.1f}% from init "
        f"{measured['eval_b']:.4f} down to {REF['eval_b']['static_cw']:.4f} just to "
        "MATCH static pruning, before any win.")
    log(f"For reference the previous healing run moved eval B "
        f"{REF['eval_b']['converted_contig_init']:.4f} -> "
        f"{REF['eval_b']['contig_trained_final']:.4f} "
        f"({100*(REF['eval_b']['contig_trained_final']/REF['eval_b']['converted_contig_init']-1):+.1f}%).")

    REPORT.write_text(json.dumps({
        "dense_model": DENSE, "converted": str(OUT_DIR),
        "ranking_digest": ranking.digest(),
        "ranking_split": ranking.calibration.get("split"),
        "design": {"num_experts": NUM_EXPERTS, "top_k": TOP_K,
                   "hot_core_size": HOT_CORE,
                   "moe_intermediate_size": plan.moe_intermediate_size,
                   "active_channels": plan.active_channels,
                   "note": "identical to the contiguous-ranked checkpoint; only the "
                           "ranking changed"},
        "params": plan.to_dict(),
        "predicted_init": predicted, "measured_init": measured,
        "prediction_drift": {k: measured[k] - predicted[k] for k in measured},
        "references": REF,
        "output_manifest_sha256": provenance["output_manifest_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
