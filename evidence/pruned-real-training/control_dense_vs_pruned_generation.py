"""Is the repetition a property of PRUNING, or of my prompt/harness?

The pruned model degenerated into repetition loops on GSM8K ("The number of eggs for
the farmers' market is 2." repeated until the 768-token budget ran out), scoring 0.00
on the baseline pass. That is either (a) pruning destroying multi-step generation, or
(b) my prompt format / sampling settings being bad for this model family.

(b) is the null hypothesis and it has to be excluded, because the whole
static-prune recommendation rests on the pruned model being usable. Same prompts,
same settings, same scorer -- only the weights differ.
"""
import json, sys, time
from pathlib import Path
import zlib
sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")
DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
PRUNED = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
EVAL = Path(r"F:\llm-models\_a4b\gsm8k_test_50.jsonl")
N = 8
MAX_NEW = 768

def degeneration(text: str) -> dict:
    """Line-AGNOSTIC degeneration, because the line-based version undercounted.

    A duplicate-LINE ratio was tried first and missed the dominant failure mode: the
    baseline's "least degenerate" response scored 0.00 on it while actually being one
    unbroken line repeating "200 / 20 = 200 / 20 = ..." to the token cap. On the real
    50-problem baseline the line metric flagged 41/50; these two flag 48/50, and even
    the two they spare cycle forever with only surface variety from line numbering.

    Two independent measures on purpose -- trusting a single proxy is what went wrong.
    """
    words = text.split()
    if len(words) >= 4:
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        distinct = len(set(grams)) / len(grams)
    else:
        distinct = 1.0
    raw = text.encode("utf-8", "replace")
    compression = len(zlib.compress(raw, 9)) / max(len(raw), 1)
    # thresholds from the measured baseline, whose medians were 0.079 and 0.080
    return {"distinct_trigram_ratio": distinct, "compression_ratio": compression,
            "degenerate": bool(distinct < 0.25 or compression < 0.15)}

def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.evaluators.transformers_text_worker import _score, _final_number

    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()][:N]
    out = {}
    for label, path in (("dense", DENSE), ("pruned", PRUNED)):
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, device_map="cuda:0",
            local_files_only=True, trust_remote_code=False)
        model.eval()
        scores, reps, lens = [], [], []
        t0 = time.time()
        for r in rows:
            enc = tok(r["prompt"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            scores.append(_score(text, r["expected"], "final_number_match"))
            reps.append(degeneration(text))
            lens.append(int(gen.shape[-1] - enc["input_ids"].shape[-1]))
        out[label] = {
            "n": len(rows),
            "gsm8k_final_number_match": sum(scores) / len(scores),
            "mean_distinct_trigram_ratio": sum(d["distinct_trigram_ratio"] for d in reps) / len(reps),
            "mean_compression_ratio": sum(d["compression_ratio"] for d in reps) / len(reps),
            "degenerate_count": sum(1 for d in reps if d["degenerate"]),
            "mean_generated_tokens": sum(lens) / len(lens),
            "hit_token_cap_fraction": sum(1 for x in lens if x >= MAX_NEW) / len(lens),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"\n=== {label} ===")
        for k, v in out[label].items():
            print(f"  {k}: {v}")
        del model
        torch.cuda.empty_cache()

    print("\nCONTROL VERDICT")
    d, p = out["dense"], out["pruned"]
    print(f"  dense  GSM8K {d['gsm8k_final_number_match']:.3f}  degen {d['degenerate_count']}/{d['n']}  cap-hit {d['hit_token_cap_fraction']:.2f}")
    print(f"  pruned GSM8K {p['gsm8k_final_number_match']:.3f}  degen {p['degenerate_count']}/{p['n']}  cap-hit {p['hit_token_cap_fraction']:.2f}")
    if d["gsm8k_final_number_match"] > p["gsm8k_final_number_match"] and d["degenerate_count"] < p["degenerate_count"]:
        print("  => the harness is fine; PRUNING caused the degeneration")
    elif d["degenerate_count"] >= len(rows) - 1:
        print("  => the DENSE model loops too: the prompt/harness is implicated, not pruning")
    else:
        print("  => mixed; neither explanation is clean")
    Path(r"F:\llm-models\_a4b\control-dense-vs-pruned-generation.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
