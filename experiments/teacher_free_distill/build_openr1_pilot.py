"""Build a complete-trace SFT pilot from released DeepSeek-R1 traces (OpenR1-Math-220k).

Replaces the chunked-OT3 pipeline that produced Condition A's GSM8K regression:
every row here is ONE whole teacher trace -- closed reasoning, a boxed final
answer the dataset's math_verify marked correct -- and its rendered length is
measured with the trainer's own `_build_chat_example`, so nothing is silently
truncated at `max_length`. Problems overlapping the eval sets are dropped, and
splits are by problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO = "open-r1/OpenR1-Math-220k"
REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
BOXED = re.compile(r"\\boxed\{")
NGRAM = 13


def audit_target(generation: str) -> str | None:
    """Why a teacher generation is not a complete trace, or None when it is."""
    if not generation.lstrip().startswith("<think>"):
        return "no_opening_think"
    if generation.count("<think>") != 1 or generation.count("</think>") != 1:
        return "think_markers_not_single_pair"
    reasoning, answer = generation.split("</think>", 1)
    if not reasoning.replace("<think>", "").strip():
        return "empty_reasoning"
    if not BOXED.search(answer):
        return "no_boxed_answer_after_think"
    return None


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def contaminated(problem: str, eval_norm: set[str], eval_grams: set[tuple[str, ...]]) -> bool:
    return " ".join(_words(problem)) in eval_norm or bool(ngrams(problem) & eval_grams)


def _write_lf(path: Path, rows: list[dict]) -> str:
    data = b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=Path, required=True, help="OpenR1-Math-220k default-config parquet shard")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from chowder.backends.training_data import _build_chat_example

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    evals = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")]
    evals += [r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    eval_norm = {" ".join(_words(q)) for q in evals}
    eval_grams = set().union(*(ngrams(q) for q in evals))

    shard_bytes = args.shard.read_bytes()
    rows = pq.read_table(args.shard).to_pylist()
    reasons: dict[str, int] = {}
    kept = []
    for r in rows:
        prompt = r["messages"][0]["content"]
        if contaminated(r["problem"], eval_norm, eval_grams) or contaminated(prompt, eval_norm, eval_grams):
            reasons["eval_overlap"] = reasons.get("eval_overlap", 0) + 1
            continue
        best = None
        for gen, complete, ok in zip(r["generations"], r["is_reasoning_complete"], r["correctness_math_verify"]):
            why = "not_complete" if not complete else "math_verify_false" if not ok else audit_target(gen)
            if why:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": gen}]
            ex = _build_chat_example(tok, msgs, max_length=10**7, row_index=0)
            n = len(ex["input_ids"])
            if n > args.max_length:
                reasons["over_max_length"] = reasons.get("over_max_length", 0) + 1
                continue
            if best is None or n < best[0]:
                best = (n, msgs)
        if best is None:
            reasons["problem_without_usable_trace"] = reasons.get("problem_without_usable_trace", 0) + 1
            continue
        kept.append({"uuid": r["uuid"], "tokens": best[0], "problem_type": r["problem_type"], "messages": best[1]})

    random.Random(args.seed).shuffle(kept)
    need = args.train + args.dev
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} usable problems; need {need}")
    dev, train = kept[: args.dev], kept[args.dev : need]
    assert not {r["uuid"] for r in dev} & {r["uuid"] for r in train}

    args.out.mkdir(parents=True, exist_ok=True)
    digests = {
        name: _write_lf(args.out / f"{name}.jsonl", [{"messages": r["messages"]} for r in split])
        for name, split in (("train", train), ("dev", dev))
    }
    toks = sorted(r["tokens"] for r in train)
    manifest = {
        "format": "chowder-openr1-complete-trace-pilot/v1",
        "source": {"repo": REPO, "revision": REVISION, "shard": args.shard.name,
                   "shard_sha256": hashlib.sha256(shard_bytes).hexdigest(), "license": "apache-2.0",
                   "teacher": "DeepSeek-R1 (released traces; no teacher run locally)"},
        "selection": {"per_problem": "shortest generation that is_reasoning_complete, math_verify-correct, "
                                     "single closed non-empty <think>, boxed answer after it, rendered <= max_length",
                      "max_length": args.max_length, "renderer": "chowder.backends.training_data._build_chat_example",
                      "tokenizer": "Qwen/Qwen3-1.7B@70d244cc", "seed": args.seed},
        "decontamination": {"against": ["openai/gsm8k main test (1319)", "HuggingFaceH4/MATH-500 test (500)"],
                            "rule": f"normalized exact match or any shared {NGRAM}-word n-gram"},
        "counts": {"shard_problems": len(rows), "usable_problems": len(kept), "train": len(train), "dev": len(dev),
                   "rejections": dict(sorted(reasons.items()))},
        "train_tokens": {"min": toks[0], "p50": toks[len(toks) // 2], "p90": toks[int(len(toks) * 0.9)], "max": toks[-1],
                         "truncated_rows": 0},
        "output_sha256": digests,
    }
    (args.out / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest["counts"] | {"train_tokens": manifest["train_tokens"], "sha256": digests}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
