"""Build the external-benchmark eval set: GSM8K test split under the frozen protocol.

Selection rules (recorded, deterministic):

  - Source: the `openai/gsm8k` test split (1319 problems), an external
    benchmark entirely disjoint from the training corpus by provenance (MIT
    license; different questions, different distribution than OpenThoughts3).
    The plan-time separation check still verifies prompt disjointness.
  - Gold answers: GSM8K's own `#### <number>` marker, normalized (commas and
    currency symbols stripped) — never re-derived from the question text.
  - Determinism: a seeded sample (seed 2026) of 120 problems; every row
    carries its test-split index.
  - Immutability: rewriting a pinned set requires --allow-rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

FORMAT = "chowder-teacher-free-external-eval/v1"
GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"
LICENSE_NOTE = "MIT (dataset card; external benchmark, not training-corpus material)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gold_from_answer(answer: str) -> str | None:
    match = GOLD_RE.search(answer or "")
    if not match:
        return None
    value = match.group(1).replace(",", "").replace("$", "")
    if not value or not any(c.isdigit() for c in value):
        return None
    try:
        float(value)
    except ValueError:
        return None
    return value


def build(out_dir: Path, *, size: int = 120, seed: int = 2026,
          allow_rewrite: bool = False) -> dict:
    from datasets import load_dataset  # optional dependency; only this builder needs it
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET)
    revision = info.sha
    data = load_dataset(DATASET, CONFIG, split=SPLIT)
    if len(data) < size:
        raise RuntimeError(f"{DATASET}:{SPLIT} has only {len(data)} rows")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(data)), size))
    rows = []
    for index in indices:
        row = data[index]
        gold = gold_from_answer(row.get("answer", ""))
        if gold is None:
            raise RuntimeError(f"gsm8k test row {index} has no parseable gold")
        rows.append({
            "problem_id": f"gsm8k-{SPLIT}-{index}",
            "split_index": index,
            "prompt": str(row["question"]).strip(),
            "expected": gold,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gsm8k_eval_prompts.jsonl"
    if out_path.is_file() and not allow_rewrite:
        prior = out_path.read_bytes()
        new_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows).encode("utf-8")
        if prior and prior != new_bytes:
            raise SystemExit(
                f"refusing to rewrite the pinned eval set {out_path}; "
                "pass --allow-rewrite deliberately")
    with out_path.open("wb") as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False) + "\n")
                         .encode("utf-8"))
    summary = {
        "format": FORMAT,
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "dataset_revision": revision,
        "license": LICENSE_NOTE,
        "sample_seed": seed,
        "sampled": size,
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "gold_source": "the dataset's own #### answer marker, normalized",
        "external": True,
    }
    (out_dir / "gsm8k_eval_set.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("C:/Users/nikma/chowder_teacher_free/gsm8k_eval"))
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-rewrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.out, size=args.size, seed=args.seed,
                           allow_rewrite=args.allow_rewrite), indent=2))


if __name__ == "__main__":
    main()
