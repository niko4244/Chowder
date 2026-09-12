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
sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")
DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
PRUNED = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
EVAL = Path(r"F:\llm-models\_a4b\gsm8k_test_50.jsonl")
N = 8
MAX_NEW = 768

def repetition_ratio(text: str) -> float:
    """Share of generated lines that are exact duplicates of an earlier line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0
    seen, dupes = set(), 0
    for l in lines:
        if l in seen:
            dupes += 1
        seen.add(l)
    return dupes / len(lines)

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
            reps.append(repetition_ratio(text))
            lens.append(int(gen.shape[-1] - enc["input_ids"].shape[-1]))
        out[label] = {
            "n": len(rows),
            "gsm8k_final_number_match": sum(scores) / len(scores),
            "mean_duplicate_line_ratio": sum(reps) / len(reps),
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
    print(f"  dense  GSM8K {d['gsm8k_final_number_match']:.3f}  dup-lines {d['mean_duplicate_line_ratio']:.3f}  cap-hit {d['hit_token_cap_fraction']:.2f}")
    print(f"  pruned GSM8K {p['gsm8k_final_number_match']:.3f}  dup-lines {p['mean_duplicate_line_ratio']:.3f}  cap-hit {p['hit_token_cap_fraction']:.2f}")
    if d["gsm8k_final_number_match"] > p["gsm8k_final_number_match"] and d["mean_duplicate_line_ratio"] < p["mean_duplicate_line_ratio"]:
        print("  => the harness is fine; PRUNING caused the degeneration")
    elif d["mean_duplicate_line_ratio"] > 0.3:
        print("  => the DENSE model loops too: the prompt/harness is implicated, not pruning")
    else:
        print("  => mixed; neither explanation is clean")
    Path(r"F:\llm-models\_a4b\control-dense-vs-pruned-generation.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
