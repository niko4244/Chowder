"""Does ranking channels corpus-wide instead of contiguously cut the OOD penalty?

The healing pilot's 4.85% equal-active-compute win on eval B rests on the static
reference there being weak: static hot pruning at f=0.28 costs 1.677x on eval A
but 3.646x on eval B. The ranking was measured on rows[0:64:2] -- a contiguous
region adjacent to eval A -- so the obvious suspicion is that the reference is
weak because the RANKING is parochial, not because static pruning is inherently
bad out of distribution.

If that is right, a ranking measured across the whole corpus should lower eval B's
static cost, which would deflate or erase the win routing is currently credited
with. That is the uncomfortable direction and the reason to run this.

Controls:
  * The ranking split stays at **32 prompts**, the same size as the contiguous one,
    so the only variable is WHERE they come from, not how many.
  * Neither eval A nor eval B is ever used for ranking, under either scheme.
  * The two rankings are compared directly (top-3,440 set overlap per layer), so
    "the rankings barely differ" is distinguishable from "the rankings differ and
    it matters".

Also recorded because it complicates the story: on the dense model eval B is
EASIER than eval A (ppl 4.3044 vs 5.3137), yet far harder once pruned. Whatever
explains the 2.17x spread has to be consistent with that.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
OLD_RANK = Path(r"F:\llm-models\_a4b\rank-9b-select-v1.json")
NEW_RANK = Path(r"F:\llm-models\_a4b\rank-9b-corpuswide-v1.json")
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
OUT = Path(r"F:\llm-models\_a4b\corpus-wide-ranking.json")

ACTIVE, EVAL_MAXLEN, RANK_N = 3440, 384, 32

# contiguous-ranking references, already measured
PRIOR = {"eval_a": {"dense": 5.3137, "static": 8.9096},
         "eval_b": {"dense": 4.3044, "static": 15.6918}}
PILOT_EVAL_B_TRAINED = 14.9314   # hot-core MoE, step 150, built on the OLD ranking


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.channel_importance import ChannelRanking, measure_channel_importance

    rows = [json.loads(l)["prompt"] for l in
            CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_a_idx = set(range(1, 64, 2))
    eval_b_idx = set(range(632, 696))
    eval_a = [rows[i] for i in sorted(eval_a_idx)]
    eval_b = [rows[i] for i in sorted(eval_b_idx)]
    held = set(eval_a) | set(eval_b)

    # corpus-wide ranking split: 32 prompts evenly spread over everything that is
    # not an eval prompt, deduped on text (the file has 690 distinct in 696 rows)
    pool, seen = [], set()
    for i, text in enumerate(rows):
        if i in eval_a_idx or i in eval_b_idx or text in held or text in seen:
            continue
        seen.add(text)
        pool.append(text)
    stride = len(pool) / RANK_N
    wide = [pool[min(len(pool) - 1, int(round(k * stride)))] for k in range(RANK_N)]
    wide = list(dict.fromkeys(wide))
    log(f"pool {len(pool)} eligible prompts -> corpus-wide rank split {len(wide)} "
        f"(contiguous was {RANK_N} from rows[0:64:2])")
    assert not (set(wide) & held), "corpus-wide rank split touches an eval set"

    # ---- stage 1: the new ranking ----
    if NEW_RANK.is_file():
        new = ChannelRanking.read(NEW_RANK)
        log(f"reusing corpus-wide ranking {new.digest()[:12]}")
    else:
        log("measuring corpus-wide channel importance")
        new = measure_channel_importance(
            DENSE, wide, max_length=EVAL_MAXLEN, load_in_4bit=True,
            split_label="corpus-wide-stride")
        new.write(NEW_RANK)
    old = ChannelRanking.read(OLD_RANK)
    log(f"old (contiguous) {old.digest()[:12]}  new (corpus-wide) {new.digest()[:12]}")
    log(f"top-10% mass: contiguous {old.concentration.get('top10pct_mass_mean', float('nan')):.3f}"
        f"  corpus-wide {new.concentration.get('top10pct_mass_mean', float('nan')):.3f}")

    # ---- how different are the two rankings, where it matters? ----
    overlaps = []
    for layer in sorted(set(old.ranking) & set(new.ranking)):
        a, b = set(old.ranking[layer][:ACTIVE]), set(new.ranking[layer][:ACTIVE])
        overlaps.append(len(a & b) / ACTIVE)
    log(f"top-{ACTIVE} set overlap between the two rankings: "
        f"mean {sum(overlaps)/len(overlaps):.3f} "
        f"(min {min(overlaps):.3f} / max {max(overlaps):.3f})")

    # ---- stage 2: score static pruning under each ranking, on both evals ----
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
    inter = mlps[0][1].down_proj.in_features

    spec: dict = {"mask": None}

    def pre_hook(idx):
        def f(_m, inputs):
            if spec["mask"] is None:
                return None
            return (inputs[0] * spec["mask"][idx],) + tuple(inputs[1:])
        return f
    handles = [m.down_proj.register_forward_pre_hook(pre_hook(i)) for i, m in mlps]

    def encode(prompts):
        out = []
        for t in prompts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=EVAL_MAXLEN)
            if e["input_ids"].shape[1] >= 8:
                out.append({k: v.to(model.device) for k, v in e.items()})
        return out
    batches = {"eval_a": encode(eval_a), "eval_b": encode(eval_b)}
    log(f"eval A {len(batches['eval_a'])} seqs, eval B {len(batches['eval_b'])} seqs")

    def ppl(key) -> float:
        total, n = 0.0, 0
        with torch.no_grad():
            for enc in batches[key]:
                loss = model(**enc, labels=enc["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    return float("inf")
                k = enc["input_ids"].shape[1] - 1
                total += loss * k
                n += k
        return math.exp(total / n)

    def mask_top(ranking, n):
        out = {}
        for i, _ in mlps:
            m = torch.zeros(inter, device=model.device, dtype=torch.bfloat16)
            m[torch.tensor(list(ranking.ranking[i][:n]), device=model.device)] = 1.0
            out[i] = m
        return out

    results: dict = {}
    try:
        spec["mask"] = None
        for key in ("eval_a", "eval_b"):
            results.setdefault(key, {})["dense"] = ppl(key)
        log(f"dense: eval A {results['eval_a']['dense']:.4f}  "
            f"eval B {results['eval_b']['dense']:.4f}")
        for label, ranking in (("contiguous", old), ("corpus_wide", new)):
            spec["mask"] = mask_top(ranking, ACTIVE)
            for key in ("eval_a", "eval_b"):
                p = ppl(key)
                results[key][label] = p
                log(f"  static hot {ACTIVE} / {label:<12} {key}  ppl {p:>9.4f} "
                    f"({p/results[key]['dense']:.3f}x dense)")
        spec["mask"] = None
    finally:
        for h in handles:
            h.remove()

    log("")
    for key in ("eval_a", "eval_b"):
        r = results[key]
        log(f"{key}: contiguous {r['contiguous']/r['dense']:.3f}x  ->  "
            f"corpus-wide {r['corpus_wide']/r['dense']:.3f}x  "
            f"({100*(r['corpus_wide']/r['contiguous']-1):+.1f}% ppl change)")
    spread_old = (results["eval_b"]["contiguous"] / results["eval_b"]["dense"]) / \
                 (results["eval_a"]["contiguous"] / results["eval_a"]["dense"])
    spread_new = (results["eval_b"]["corpus_wide"] / results["eval_b"]["dense"]) / \
                 (results["eval_a"]["corpus_wide"] / results["eval_a"]["dense"])
    log(f"A->B cost spread: contiguous {spread_old:.2f}x  ->  corpus-wide {spread_new:.2f}x")

    log("")
    new_ref = results["eval_b"]["corpus_wide"]
    beats = PILOT_EVAL_B_TRAINED < new_ref
    log(f"THE CONSEQUENCE: the pilot's trained MoE scored {PILOT_EVAL_B_TRAINED:.4f} on "
        f"eval B (built on the CONTIGUOUS ranking).")
    log(f"  vs contiguous static reference 15.6918 -> beat it by 4.85%")
    log(f"  vs corpus-wide static reference {new_ref:.4f} -> "
        f"{'still beats it by' if beats else 'LOSES to it by'} "
        f"{abs(100*(new_ref/PILOT_EVAL_B_TRAINED-1)):.2f}%")
    if not beats:
        log("  => fixing the ranking beats adding routing. The 4.85% win does not")
        log("     survive a better static baseline.")

    OUT.write_text(json.dumps({
        "dense_model": DENSE, "active_channels": ACTIVE, "eval_maxlen": EVAL_MAXLEN,
        "rank_split_size": {"contiguous": RANK_N, "corpus_wide": len(wide)},
        "rankings": {"contiguous_digest": old.digest(), "corpus_wide_digest": new.digest(),
                     "top_set_overlap_mean": sum(overlaps)/len(overlaps),
                     "top_set_overlap_min": min(overlaps),
                     "top_set_overlap_max": max(overlaps),
                     "concentration_contiguous": old.concentration,
                     "concentration_corpus_wide": new.concentration},
        "results": results,
        "cost_spread_a_to_b": {"contiguous": spread_old, "corpus_wide": spread_new},
        "consequence": {"pilot_trained_eval_b": PILOT_EVAL_B_TRAINED,
                        "contiguous_reference": PRIOR["eval_b"]["static"],
                        "corpus_wide_reference": new_ref,
                        "still_beats_static": beats},
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
