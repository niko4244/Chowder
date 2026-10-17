"""Build the REAP calibration set for Qwen3.6-35B-A3B from local workload data.

REAP decides which experts survive, so its calibration text shapes the pruned
weights. Rules, all enforced here rather than by convention:

* workload-matched mix: agentic coding trajectories, code reasoning, math, and
  general prose (quotas below);
* no Fable-derived text (reference/judge only, per the Ornith/Fable plan);
* 13-gram dedupe against every protected evaluation text (GSM8K test, MATH-500,
  MGSM, HumanEval) -- a calibration row sharing any 13-gram is dropped;
* deterministic (fixed seed, sorted inputs) with a sha256 manifest.

Usage:
    python chowder_batch/reap_calibration.py --tokenizer <model_dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

SOURCES = {
    # name: (path, quota)
    "swe_smith_trajectories": (r"C:\Users\nikma\chowder_teacher_free\smith_target_trajectories_b2.jsonl", 192),
    "ot3_code_reasoning": (r"C:\Users\nikma\chowder_teacher_free\ot3_sft.jsonl", 96),
    "openr1_math": (r"C:\Users\nikma\chowder_teacher_free\supply\openr1_math_sample500.jsonl", 48),
    "openstax_prose": (r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\openstax_broad_clean_train.jsonl", 48),
}
FORBIDDEN_SOURCE = re.compile(r"fable", re.IGNORECASE)
NGRAM = 13
SEED = 20261017


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def protected_ngrams() -> tuple[set[tuple[str, ...]], dict[str, int]]:
    """Every 13-gram of every protected evaluation text, from the offline cache."""
    import os

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    specs = [
        ("openai/gsm8k", "main", "test", ("question", "answer")),
        ("HuggingFaceH4/MATH-500", None, "test", ("problem", "solution", "answer")),
        ("juletxara/mgsm", "en", "test", ("question", "answer")),
        ("openai/openai_humaneval", None, "test", ("prompt", "canonical_solution", "test")),
    ]
    grams: set[tuple[str, ...]] = set()
    counts: dict[str, int] = {}
    for name, config, split, fields in specs:
        ds = load_dataset(name, config, split=split)
        counts[name] = len(ds)
        for row in ds:
            for field in fields:
                value = row.get(field)
                if isinstance(value, str):
                    grams |= _ngrams(value)
    return grams, counts


def _render(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _texts(name: str, path: str, tokenizer):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if FORBIDDEN_SOURCE.search(str(row.get("source", "")) + str(row.get("model", ""))):
                continue
            if name == "swe_smith_trajectories":
                messages = row["messages"]
                if isinstance(messages, str):
                    messages = json.loads(messages)
                messages = [
                    {"role": m.get("role", "user"), "content": m["content"] if isinstance(m.get("content"), str)
                     else json.dumps(m.get("content"))}
                    for m in messages
                ]
                yield _render(tokenizer, messages)
            elif name == "ot3_code_reasoning":
                roles = {"human": "user", "gpt": "assistant"}
                yield _render(tokenizer, [
                    {"role": roles.get(c["from"], c["from"]), "content": c["value"]} for c in row["conversations"]
                ])
            elif name == "openr1_math":
                yield _render(tokenizer, [
                    {"role": "user", "content": row["problem"]},
                    {"role": "assistant", "content": row["solution"]},
                ])
            else:
                yield row["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    protected, eval_counts = protected_ngrams()
    rng = random.Random(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, stats = [], {}
    for name, (path, quota) in SOURCES.items():
        pool = list(_texts(name, path, tokenizer))
        rng.shuffle(pool)
        kept = dropped = 0
        for text in pool:
            if kept >= quota:
                break
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][: args.max_tokens]
            if len(ids) < 256:
                continue
            clipped = tokenizer.decode(ids)
            if _ngrams(clipped) & protected:
                dropped += 1
                continue
            rows.append({"source": name, "n_tokens": len(ids), "text": clipped})
            kept += 1
        stats[name] = {"path": path, "quota": quota, "kept": kept, "dropped_for_contamination": dropped,
                       "pool": len(pool)}

    calib = out / "calibration.jsonl"
    with calib.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "seed": SEED,
        "ngram": NGRAM,
        "max_tokens": args.max_tokens,
        "protected_eval_rows": eval_counts,
        "protected_ngram_count": len(protected),
        "sources": stats,
        "rows": len(rows),
        "tokens": sum(r["n_tokens"] for r in rows),
        "calibration_sha256": hashlib.sha256(calib.read_bytes()).hexdigest(),
        "excluded": "any row whose source/model mentions Fable",
    }
    (out / "calibration-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "tokens", "calibration_sha256")}, indent=2))
    for name, s in stats.items():
        print(name, s["kept"], "kept,", s["dropped_for_contamination"], "dropped")


if __name__ == "__main__":
    main()
