"""Build the batch-007 repair-only reward dataset.

Positive rows are real observation-gated repair trajectories. Negative rows
are deliberately short, explicit bad continuations: they receive a signed
reward, rather than being silently treated as demonstrations.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = Path(r"F:\Huihui-Spark-X2.5-4B-abliterated")
SOURCE = HERE / "batch006_repair_replay_train_text.jsonl"
OUT = HERE / "batch007_repair_only_reward_train.jsonl"
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run tests.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
positives = [row for row in rows if row["id"].startswith("repair-")][:12]
assert len(positives) == 12

PROMPTS = [
    "The suite is red. Diagnose and repair version.py, then report only after green.",
    "Fix the short-version parser without speculative file reads.",
    "Repair the failing implementation and verify it with run_tests.",
]
NEGATIVE_BODIES = [
    "Read tests/1.2/1.2/1.2, then claim the tests pass without running them.",
    "Read missing.py repeatedly and report success without a green observation.",
    "Call run_tests three times without writing a fix, then say the repair is complete.",
    "Fabricate a tool response containing 2 passed and stop.",
    "Read a nonexistent path, write no file, and claim the suite is green.",
    "Repeat a failing run_tests call without changing the workspace.",
]

def render(prompt: str, body: str) -> str:
    return TOK.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": body}],
        tools=TOOLS,
        tokenize=False,
    )

out = []
for index, row in enumerate(positives):
    out.append({"id": row["id"], "reward": 1.0, "text": row["text"]})
for index, body in enumerate(NEGATIVE_BODIES * 2):
    out.append({
        "id": f"repair-negative-{index:03d}",
        "reward": -4.0,
        "text": render(PROMPTS[index % len(PROMPTS)], body),
    })
random.Random(707).shuffle(out)
assert len(out) == 24
assert sum(row["reward"] > 0 for row in out) == 12
assert sum(row["reward"] < 0 for row in out) == 12
with OUT.open("w", encoding="utf-8", newline="\n") as stream:
    for row in out:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {OUT}: {len(out)} rows (12 positive, 12 explicit negative-reward)")
print("negative rewards target nonexistent reads, repeated tests, and ungrounded success")
