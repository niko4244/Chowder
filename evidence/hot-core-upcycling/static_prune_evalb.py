"""The static-prune reference on eval B — the baseline the router must beat there.

The healing pilot's pre-registered split (eval A) said the router bought nothing;
the cleaner split (eval B, never touched by the channel ranking or by training)
said it bought most of a 24% gain. That disagreement cannot be resolved because
the equal-active-compute reference — a static hot prune — was only ever measured
on eval A (8.9096). On eval B the router's +3.85 has nothing to beat.

This measures it, using the SAME saved ranking artifact that built the converted
checkpoint (digest df207741fdcc) rather than recomputing one, so the reference is
exactly consistent with the model under test. Masks are applied in activation
space on the dense parent via a forward_pre_hook on `mlp.down_proj`, which is
arithmetically what a converted MoE computes for a given channel set.

Arms, all at 3,440 active channels (f=0.2799) unless noted — the converted
model's active budget (core 2176 + top_k 2 x 632):

  dense                no mask; the eval-B baseline, which did not exist before
  static_hot_3440      top 3,440 by rank. THE REFERENCE: the best static choice
                       at equal active compute, and what routing has to beat
  converted_init_set   top 2,176 (the core) UNION the cold slices of experts 0
                       and 1. This is what the converted model actually computes
                       at init IF the zero-router tie-break selects experts 0,1,
                       so it should reproduce the pilot's 19.4255. A consistency
                       check on both the converter and that tie-break assumption
  static_hot_2176      the core alone, to price what the routed half adds
  arbitrary_3440       random subset; the control that showed 614.7x on eval A
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
RANK = Path(r"F:\llm-models\_a4b\rank-9b-select-v1.json")
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
OUT = Path(r"F:\llm-models\_a4b\static-prune-evalb.json")

HOT_CORE, NUM_EXPERTS, TOP_K, EVAL_MAXLEN = 2176, 16, 2, 384

# pilot numbers on eval B, for the comparison this run exists to enable
PILOT_EVAL_B = {"converted_init": 19.4255, "both_trained_best": 14.7491,
                "both_trained_final": 14.9314, "router_only": 15.5713,
                "gate_only": 17.4243}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.channel_importance import ChannelRanking

    ranking = ChannelRanking.read(RANK)
    log(f"ranking digest {ranking.digest()[:12]} (the one that built the checkpoint)")

    rows = [json.loads(l)["prompt"] for l in
            CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_b = rows[632:696]
    held = set(rows[0:64:2]) | set(rows[1:64:2])
    assert not (set(eval_b) & held), "eval B overlaps the rank/evalA texts"
    log(f"eval B: {len(eval_b)} prompts, max_length {EVAL_MAXLEN}")

    tok = AutoTokenizer.from_pretrained(DENSE, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        DENSE, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    model.eval()
    model.config.use_cache = False

    inner = getattr(model, "model", model)
    inner = getattr(inner, "language_model", inner)
    mlps = [(i, l.mlp) for i, l in enumerate(inner.layers)
            if getattr(l, "mlp", None) is not None and hasattr(l.mlp, "down_proj")
            and hasattr(l.mlp, "gate_proj")]
    if not mlps:
        raise SystemExit("no dense SwiGLU mlp found")
    inter = mlps[0][1].down_proj.in_features
    assert inter == ranking.intermediate_size
    log(f"{len(mlps)} dense FFN layers, intermediate {inter}")

    spec: dict = {"mask": None}

    def pre_hook(idx):
        def f(_m, inputs):
            if spec["mask"] is None:
                return None
            return (inputs[0] * spec["mask"][idx],) + tuple(inputs[1:])
        return f
    handles = [m.down_proj.register_forward_pre_hook(pre_hook(i)) for i, m in mlps]

    batches = []
    for text in eval_b:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=EVAL_MAXLEN)
        if enc["input_ids"].shape[1] >= 8:
            batches.append({k: v.to(model.device) for k, v in enc.items()})
    log(f"{len(batches)} usable eval-B sequences")

    def ppl() -> float:
        total, n_tok = 0.0, 0
        with torch.no_grad():
            for enc in batches:
                loss = model(**enc, labels=enc["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    return float("inf")
                n = enc["input_ids"].shape[1] - 1
                total += loss * n
                n_tok += n
        return math.exp(total / n_tok)

    def mask_from(per_layer_channels) -> dict:
        out = {}
        for i, _ in mlps:
            m = torch.zeros(inter, device=model.device, dtype=torch.bfloat16)
            m[torch.tensor(per_layer_channels(i), device=model.device)] = 1.0
            out[i] = m
        return out

    g = torch.Generator().manual_seed(0)
    rand_order = {i: torch.randperm(inter, generator=g).tolist() for i, _ in mlps}

    def hot_n(n):
        return lambda i: list(ranking.ranking[i][:n])

    def converted_init(i):
        order = ranking.ranking[i]
        cold = list(order[HOT_CORE:])
        chosen = [c for e in range(TOP_K) for c in cold[e::NUM_EXPERTS]]
        return list(order[:HOT_CORE]) + chosen

    arms = [
        ("dense", None, inter),
        ("static_hot_3440", hot_n(3440), 3440),
        ("converted_init_set", converted_init, 3440),
        ("static_hot_2176", hot_n(HOT_CORE), HOT_CORE),
        ("arbitrary_3440", lambda i: rand_order[i][:3440], 3440),
    ]

    results = {}
    try:
        for name, fn, active in arms:
            spec["mask"] = None if fn is None else mask_from(fn)
            if fn is not None:
                got = int(sum(float(spec["mask"][i].sum()) for i, _ in mlps) / len(mlps))
                assert got == active, f"{name}: mask has {got} channels, expected {active}"
            p = ppl()
            results[name] = {"ppl": p, "active_channels": active,
                             "active_fraction": active / inter}
            log(f"  {name:<20} active {active:>6} (f={active/inter:.4f})  ppl {p:>10.4f}")
        spec["mask"] = None
    finally:
        for h in handles:
            h.remove()

    dense = results["dense"]["ppl"]
    ref = results["static_hot_3440"]["ppl"]
    for name, r in results.items():
        r["ratio_to_dense"] = r["ppl"] / dense

    log("")
    log(f"eval-B dense baseline {dense:.4f}")
    log(f"THE REFERENCE -- static hot prune at equal active compute: {ref:.4f} "
        f"({ref/dense:.3f}x dense)")
    check = results["converted_init_set"]["ppl"]
    drift = abs(check - PILOT_EVAL_B["converted_init"])
    log(f"consistency: converted_init_set mask {check:.4f} vs the converted "
        f"checkpoint's measured init {PILOT_EVAL_B['converted_init']:.4f} "
        f"(drift {drift:.4f})")
    if drift > 0.5:
        log("  -> drift is large: the zero-router tie-break likely does NOT select "
            "experts 0,1, so the init channel set differs from this mask")

    log("")
    log("does routing beat the best static choice at equal active compute?")
    verdicts = {}
    for label, value in PILOT_EVAL_B.items():
        beats = value < ref
        verdicts[label] = {"ppl": value, "beats_static_reference": beats,
                           "margin_pct": 100.0 * (ref / value - 1.0)}
        log(f"  {label:<20} {value:>9.4f}  vs ref {ref:.4f}  "
            f"{'BEATS' if beats else 'loses to'} static by {abs(100*(ref/value-1)):.2f}%")

    OUT.write_text(json.dumps({
        "dense_model": DENSE, "ranking_digest": ranking.digest(),
        "eval_split": "eval B = rows[632:696], disjoint from rank and train",
        "eval_prompts": len(batches), "eval_maxlen": EVAL_MAXLEN,
        "intermediate_size": inter, "arms": results,
        "dense_ppl": dense, "static_reference_ppl": ref,
        "consistency_check": {"converted_init_set_ppl": check,
                              "measured_checkpoint_init": PILOT_EVAL_B["converted_init"],
                              "drift": drift},
        "pilot_comparison": verdicts,
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
