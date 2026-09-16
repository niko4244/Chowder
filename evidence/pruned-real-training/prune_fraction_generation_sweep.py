"""At which prune fraction does GENERATION survive on this 9B?

Established: at f=0.28 the pruned checkpoint degenerates on 100/100 responses across
two arms and never emits an EOS token, while the dense parent terminates on 5 of 8
and scores 0.375 (docs/PRUNED_9B_RERUN_RESULT.md, the dense-vs-pruned control). Every
other fraction has only ever been measured on PERPLEXITY, which turned out not to
predict whether a checkpoint can terminate -- f=0.56 costs just 1.14x dense
perplexity and is completely untested for generation.

WHY A MASK RATHER THAN A BUILT CHECKPOINT
Building one checkpoint per fraction costs ~15 GB each on a disk with 51 GB free, and
tests one guess. Masking sweeps every fraction in ONE model load, which answers the
actual question -- where does it break -- instead of one point on the curve. The mask
is the same mechanism the perplexity sweeps used: a forward_pre_hook on each
`mlp.down_proj` zeroing the intermediate channels that a prune would delete. Pruning
those channels and zeroing them are equivalent in the forward pass; what differs is
only that nf4 quantises a narrowed matrix slightly differently, which moved perplexity
by +1.3% to +2.5% in four prior mask-vs-checkpoint comparisons. For a qualitative
"does it terminate at all" question that drift is immaterial.

Masking keeps the CHANNELS THE REAL CONVERTER WOULD KEEP: the persisted corpus-wide
ranking, digest 304fffa858ea, which is the ranking both built checkpoints used. No
scaling is applied, matching `chowder.static_prune`, which scales nowhere.

PRE-REGISTERED READING, fixed before the run. "Generation survives at f" iff
  * at least 4 of 8 responses TERMINATE (do not hit the 768-token cap), and
  * at most 2 of 8 are flagged degenerate.
Calibrated against the known dense control, which terminated 5/8 and had 1/8
degenerate -- so f=1.0 must pass this bar, and if it does not the harness is at fault
rather than the prune. f=0.28 must fail it; it degenerated 8/8 with 0/8 terminating.
Those two anchors make the sweep falsifiable at both ends before any new number
exists.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

# The dense parent the ranking and both checkpoints are bound to. My first draft
# pointed at Qwen3.5-9B, which is a DIFFERENT model that is also 32 layers x
# 12288 intermediate with the same architecture -- it would have loaded, masked
# cleanly, and produced plausible numbers about the wrong weights. Asserted
# against the ranking's own source_dir below rather than trusted.
DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
RANKING = r"F:\llm-models\_a4b\rank-9b-corpuswide-v1.json"
EXPECTED_RANKING_DIGEST = "304fffa858ea"
EVAL = r"F:\llm-models\_a4b\gsm8k_test_50.jsonl"
OUT = Path(r"F:\llm-models\_a4b\prune-fraction-generation-sweep.json")

FRACTIONS = (1.0, 0.75, 0.5625, 0.40, 0.28)
N_PROMPTS = 8
MAX_NEW_TOKENS = 768
SEED = 123
#: Pre-registered survival bar, and the two anchors that make it falsifiable.
MIN_TERMINATED = 4
MAX_DEGENERATE = 2
TRIGRAM_FLOOR = 0.30
COMPRESSION_FLOOR = 0.20


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def distinct_trigram_ratio(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(grams)) / len(grams) if grams else 0.0


def compression_ratio(text: str) -> float:
    blob = text.encode("utf-8")
    return len(zlib.compress(blob)) / len(blob) if blob else 0.0


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    from chowder.channel_importance import ChannelRanking
    from chowder.evaluators.generation import resolve_eos_token_ids
    from chowder.evaluators.scoring import score as strict_score

    ranking = ChannelRanking.read(RANKING)
    if not ranking.digest().startswith(EXPECTED_RANKING_DIGEST):
        raise SystemExit(
            f"ranking digest {ranking.digest()[:12]} != {EXPECTED_RANKING_DIGEST}; "
            "this would not be comparable to the built checkpoints"
        )
    if Path(ranking.source_dir) != Path(DENSE):
        raise SystemExit(
            f"ranking is bound to {ranking.source_dir!r} but this script would load "
            f"{DENSE!r}; masking one model with another's channel order is silently wrong"
        )
    inter = ranking.intermediate_size
    log(f"ranking {ranking.digest()[:12]}, bound to {Path(ranking.source_dir).name}, "
        f"intermediate={inter}, layers={len(ranking.ranking)}")

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(DENSE, trust_remote_code=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    log("loading dense in nf4")
    model = AutoModelForCausalLM.from_pretrained(
        DENSE, trust_remote_code=False, dtype=torch.bfloat16, local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    model.eval()
    device = next(model.parameters()).device
    eos = resolve_eos_token_ids(tok, model)

    # Decoder MLPs only, by exact module path -- ".mlp." substring matching once swept
    # 108 vision-tower blocks into a decoder path (docs/HOT_CORE_UPCYCLING.md).
    mlps: list[tuple[int, object]] = []
    for idx, layer in enumerate(model.model.language_model.layers
                                if hasattr(model.model, "language_model")
                                else model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            mlps.append((idx, mlp))
    log(f"{len(mlps)} decoder MLPs found")
    if len(mlps) != len(ranking.ranking):
        raise SystemExit(f"{len(mlps)} MLPs but ranking covers {len(ranking.ranking)} layers")

    spec: dict = {"mask": None}

    def make_pre_hook(layer_idx: int):
        def pre_hook(_module, inputs):
            m = spec["mask"]
            if m is None:
                return None
            return (inputs[0] * m[layer_idx],) + tuple(inputs[1:])
        return pre_hook

    for idx, mlp in mlps:
        mlp.down_proj.register_forward_pre_hook(make_pre_hook(idx))

    def set_fraction(frac: float) -> int:
        if frac >= 1.0:
            spec["mask"] = None
            return inter
        keep_n = int(round(frac * inter))
        masks = {}
        for idx, _ in mlps:
            keep = torch.zeros(inter, dtype=torch.bfloat16, device=device)
            order = ranking.ranking[idx][:keep_n]
            keep[torch.tensor(order, dtype=torch.long, device=device)] = 1.0
            masks[idx] = keep
        spec["mask"] = masks
        return keep_n

    rows = [json.loads(l) for l in Path(EVAL).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:N_PROMPTS]

    report: dict = {
        "dense_model": DENSE, "ranking_digest": ranking.digest(),
        "prompts": len(rows), "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
        "survival_rule": {
            "min_terminated_of_8": MIN_TERMINATED, "max_degenerate_of_8": MAX_DEGENERATE,
            "trigram_floor": TRIGRAM_FLOOR, "compression_floor": COMPRESSION_FLOOR,
            "anchors": "f=1.0 must pass (dense control: 5/8 terminated, 1/8 degenerate); "
                       "f=0.28 must fail (8/8 degenerate, 0/8 terminated)",
        },
        "arms": [],
    }

    with torch.inference_mode():
        for frac in FRACTIONS:
            keep_n = set_fraction(frac)
            log(f"--- f={frac:.4f}  keeping {keep_n}/{inter} channels per layer")
            per: list[dict] = []
            started = time.perf_counter()
            for i, row in enumerate(rows):
                enc = tok(str(row["prompt"]), return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                prompt_tokens = enc["input_ids"].shape[1]
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos)
                new_tokens = int(gen.shape[1] - prompt_tokens)
                text = tok.decode(gen[0, prompt_tokens:], skip_special_tokens=True)
                tg, cp = distinct_trigram_ratio(text), compression_ratio(text)
                per.append({
                    "index": i, "new_tokens": new_tokens,
                    "terminated": new_tokens < MAX_NEW_TOKENS,
                    "score": strict_score(text, str(row["expected"]), "final_number_match"),
                    "distinct_trigram_ratio": tg, "compression_ratio": cp,
                    "degenerate": tg < TRIGRAM_FLOOR or cp < COMPRESSION_FLOOR,
                    "tail": text[-120:],
                })
                log(f"    [{i+1}/{len(rows)}] {new_tokens:>4} tok  "
                    f"{'term' if per[-1]['terminated'] else 'CAP '}  "
                    f"score {per[-1]['score']:.0f}  trig {tg:.3f}  "
                    f"{'DEGEN' if per[-1]['degenerate'] else 'ok'}")
            n = len(per)
            terminated = sum(1 for p in per if p["terminated"])
            degenerate = sum(1 for p in per if p["degenerate"])
            arm = {
                "fraction": frac, "keep_channels": keep_n,
                "gsm8k_final_number_match": sum(p["score"] for p in per) / n,
                "terminated": terminated, "degenerate": degenerate,
                "mean_distinct_trigram_ratio": sum(p["distinct_trigram_ratio"] for p in per) / n,
                "mean_compression_ratio": sum(p["compression_ratio"] for p in per) / n,
                "mean_new_tokens": sum(p["new_tokens"] for p in per) / n,
                "survives": terminated >= MIN_TERMINATED and degenerate <= MAX_DEGENERATE,
                "seconds": time.perf_counter() - started,
                "per_prompt": per,
            }
            report["arms"].append(arm)
            log(f"    f={frac:.4f}: GSM8K {arm['gsm8k_final_number_match']:.3f}  "
                f"terminated {terminated}/{n}  degenerate {degenerate}/{n}  "
                f"trig {arm['mean_distinct_trigram_ratio']:.3f}  "
                f"=> {'SURVIVES' if arm['survives'] else 'FAILS'}")
            OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    surviving = [a["fraction"] for a in report["arms"] if a["survives"]]
    report["lowest_surviving_fraction"] = min(surviving) if surviving else None
    anchors_ok = (
        report["arms"][0]["survives"]
        and not report["arms"][-1]["survives"]
    )
    report["anchors_behaved"] = anchors_ok
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    log("=== SWEEP RESULT ===")
    log(f"  {'f':>7} {'keep':>6} {'GSM8K':>7} {'term':>6} {'degen':>7} {'trig':>7}  verdict")
    for a in report["arms"]:
        log(f"  {a['fraction']:>7.4f} {a['keep_channels']:>6} "
            f"{a['gsm8k_final_number_match']:>7.3f} {a['terminated']:>3}/{N_PROMPTS} "
            f"{a['degenerate']:>4}/{N_PROMPTS} {a['mean_distinct_trigram_ratio']:>7.3f}  "
            f"{'SURVIVES' if a['survives'] else 'FAILS'}")
    log(f"  anchors behaved as pre-registered: {anchors_ok}")
    log(f"  lowest surviving fraction: {report['lowest_surviving_fraction']}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
