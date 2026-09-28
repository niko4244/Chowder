"""Build the harder external eval set: a stratified MATH-500 subset (A4 plan).

Selection rules (recorded, deterministic):

  - Source: `HuggingFaceH4/MATH-500` test (500 problems, levels 1-5). Training
    data is decontaminated against all 500, so any subset is disjoint.
  - Gold: the dataset's own `answer` field, verbatim LaTeX; scored with
    `math_verify_match` (symbolic equivalence), never final-number matching.
  - Stratified: `per_level` problems from each level 1-5, seeded (2026), so
    the subset keeps MATH-500's difficulty spread instead of drifting easy.
  - Immutability: rewriting a pinned set requires --allow-rewrite.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gsm8k_eval_set import sha256_file

FORMAT = "chowder-teacher-free-external-eval/v1"
DATASET = "HuggingFaceH4/MATH-500"
SPLIT = "test"


def build(out_dir: Path, *, per_level: int = 30, seed: int = 2026, allow_rewrite: bool = False) -> dict:
    from datasets import load_dataset

    data = load_dataset(DATASET, split=SPLIT)
    rng = random.Random(seed)
    rows = []
    for level in range(1, 6):
        pool = sorted(i for i, r in enumerate(data) if int(r["level"]) == level)
        if len(pool) < per_level:
            raise RuntimeError(f"level {level} has only {len(pool)} problems")
        for index in sorted(rng.sample(pool, per_level)):
            r = data[index]
            rows.append({"problem_id": r["unique_id"], "split_index": index, "level": level,
                         "subject": r["subject"], "prompt": r["problem"].strip(), "expected": r["answer"].strip()})
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "math_eval_prompts.jsonl"
    new_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    if out_path.is_file() and not allow_rewrite and out_path.read_bytes() not in (b"", new_bytes):
        raise SystemExit(f"refusing to rewrite the pinned eval set {out_path}; pass --allow-rewrite deliberately")
    out_path.write_bytes(new_bytes)
    summary = {
        "format": FORMAT, "dataset": DATASET, "split": SPLIT,
        "dataset_fingerprint": getattr(data, "_fingerprint", None),
        "license": "MIT (MATH / PRM800K subset; external benchmark, decontaminated from training)",
        "sample_seed": seed, "per_level": per_level, "sampled": len(rows),
        "output": str(out_path), "output_sha256": sha256_file(out_path),
        "gold_source": "the dataset's own answer field, verbatim LaTeX", "scoring": "math_verify_match",
        "external": True,
    }
    (out_dir / "math_eval_set.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("C:/Users/nikma/chowder_teacher_free/math_eval"))
    parser.add_argument("--per-level", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-rewrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.out, per_level=args.per_level, seed=args.seed,
                           allow_rewrite=args.allow_rewrite), indent=2))


if __name__ == "__main__":
    main()
