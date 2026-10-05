"""Experiment E task suite: GSM8K + factual-retrieval + repair tasks.

Every task carries validated ground truth and a difficulty label:

* ``gsm8k``   -- numeric word problems from the cached GSM8K validation split;
                 ground truth is the dataset's verified answer.
* ``factual`` -- questions whose verified answers appear verbatim in the
                 experiment corpus (built by exp_e_corpus); grading is
                 answer-match plus citation-of-source-doc.
* ``repair``  -- runtime repair tasks executed through the chowder runtime
                 harness; success is a green test observation, not a claim.

Tasks are split dev/eval deterministically; routers may only calibrate on dev.
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Runtime repair tasks reused from the batch-009 harness (single-file family).
REPAIR_TASKS = (
    {"name": "repair_version_parser", "target": "version.py",
     "initial": {"version.py": "def parse_version(s):\n    return tuple(int(p) for p in s.split('.'))\n"},
     "expected_fix": "while len(parts) < 3", "test_count": 2, "test_success": "2 passed", "difficulty": "hard"},
    {"name": "repair_slugify", "target": "slugify.py",
     "initial": {"slugify.py": "def slugify(value):\n    return value\n"},
     "expected_fix": ".strip().lower()", "test_count": 2, "test_success": "2 passed", "difficulty": "hard"},
    {"name": "repair_sum_text", "target": "sum_text.py",
     "initial": {"sum_text.py": "def total(values):\n    return sum(values) - 1\n"},
     "expected_fix": "__SUM_TEXT_CHECK__", "test_count": 3, "test_success": "3 passed", "difficulty": "easy"},
    {"name": "repair_config_defaults", "target": "config_defaults.py",
     "initial": {"config_defaults.py": "def with_defaults(config):\n    return {}\n"},
     "expected_fix": "setdefault", "test_count": 2, "test_success": "2 passed", "difficulty": "easy"},
)

# Factual tasks: ground truth lives in docs built by exp_e_corpus.build_corpus.
# doc_id is the verified source; grading requires citing it.
FACTUAL_TASKS = (
    {"name": "fact_spark_vocab", "question": "How many entries does the Spark tokenizer vocabulary have?", "answer": "131072", "doc_id": "spark_model_card", "difficulty": "easy"},
    {"name": "fact_teacher_vocab", "question": "What is the vocabulary size of the Qwen3.8-9B teacher model?", "answer": "248320", "doc_id": "teacher_model_card", "difficulty": "easy"},
    {"name": "fact_teacher_hidden", "question": "What is the hidden size of the Qwen3.8-9B teacher model?", "answer": "4096", "doc_id": "teacher_model_card", "difficulty": "easy"},
    {"name": "fact_teacher_layers", "question": "How many decoder layers does the Qwen3.8-9B teacher have?", "answer": "32", "doc_id": "teacher_model_card", "difficulty": "easy"},
    {"name": "fact_spark_params", "question": "How many parameters does the Spark model have (in billions)?", "answer": "4", "doc_id": "spark_model_card", "difficulty": "easy"},
    {"name": "fact_version_parse", "question": "In version.py, how many parts must a version string have after parsing?", "answer": "3", "doc_id": "runtime_version_doc", "difficulty": "easy"},
    {"name": "fact_slug_rule", "question": "What transformation does slugify apply before joining words?", "answer": "strip and lowercase", "doc_id": "runtime_slug_doc", "difficulty": "hard"},
    {"name": "fact_gsm_rule", "question": "According to the arithmetic notes, what is 13 times 24?", "answer": "312", "doc_id": "arith_notes", "difficulty": "easy"},
    {"name": "fact_gsm_rule2", "question": "According to the arithmetic notes, what is 47 plus 86?", "answer": "133", "doc_id": "arith_notes", "difficulty": "easy"},
    {"name": "fact_runtime_green", "question": "What exact observation string marks a green test run in the runtime harness?", "answer": "2 passed", "doc_id": "runtime_harness_doc", "difficulty": "hard"},
    {"name": "fact_runtime_tools", "question": "Name the three tools the runtime repair harness exposes.", "answer": "read_file, write_file, run_tests", "doc_id": "runtime_harness_doc", "difficulty": "hard"},
    {"name": "fact_rrsi", "question": "Which paper does the campaign follow for harness evolution regularizers?", "answer": "RRSI", "doc_id": "campaign_doc", "difficulty": "hard"},
)

GSM8K_SPLIT_SIZE = 20  # 10 dev / 10 eval


def _extract_answer(gsm_answer: str) -> str:
    match = re.findall(r"-?\d[\d,]*\.?\d*", gsm_answer.split("####")[-1].replace(",", ""))
    return match[-1] if match else ""


def load_gsm8k_tasks() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(20260924)
    indices = rng.sample(range(len(rows)), min(GSM8K_SPLIT_SIZE, len(rows)))
    tasks = []
    for i in indices:
        row = rows[i]
        tasks.append({
            "name": f"gsm8k_{i}",
            "question": row["question"].strip(),
            "answer": _extract_answer(row["answer"]),
            "difficulty": "hard" if len(row["question"]) > 600 else "easy",
        })
    return tasks


def build_tasks(out_path: Path) -> dict:
    """Build the full task suite with a deterministic dev/eval split."""
    gsm = load_gsm8k_tasks()
    split = {t["name"]: ("dev" if idx % 2 == 0 else "eval") for idx, t in enumerate(gsm)}
    tasks = []
    for t in gsm:
        tasks.append({**t, "kind": "gsm8k", "split": split[t["name"]]})
    for i, t in enumerate(FACTUAL_TASKS):
        tasks.append({**t, "kind": "factual", "split": "dev" if i % 3 == 0 else "eval"})
    for t in REPAIR_TASKS:
        tasks.append({**t, "kind": "repair", "question": f"Repair {t['target']} so its tests pass.", "split": "eval"})
    payload = {"tasks": tasks}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for t in tasks:
        counts[(t["kind"], t["split"])] = counts.get((t["kind"], t["split"]), 0) + 1
    print("task suite:", dict(counts))
    return payload


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exp_e_tasks.json")
    build_tasks(out)
