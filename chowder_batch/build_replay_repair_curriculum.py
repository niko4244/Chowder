"""Build batch-005: gen-2 replay plus grounded repair-loop supervision.

The repair examples deliberately contain a red observation, a file read, a
write, and a final run_tests observation. The assistant report is present only
after the green summary, so the dataset teaches the stopping boundary rather
than merely teaching another envelope shape.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path(r"F:\Huihui-Spark-X2.5-4B-abliterated")
REPLAY = Path(r"C:\Users\nikma\Chowder\.chowder-spark-calib\gsm8k\gen2_train.jsonl")
OUT_RECORDS = HERE / "chowder_agent_batch_005_replay_repair.jsonl"
OUT_TEXT = HERE / "replay_repair_train_text.jsonl"

if not MODEL.is_dir() or not REPLAY.is_file():
    sys.exit("SKIP: Spark model or gen-2 replay data is unavailable")

from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Return the full text of a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the workspace test suite and return its summary.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]
RULES = ("Workspace tools: read_file, write_file, run_tests. Emit one tool call "
         "per turn. Wait for the runtime tool observation before the next action. "
         "Never invent a tool_response or claim success before a passing run_tests "
         "summary is observed. ")


def tool_call(name: str, args: dict) -> str:
    rendered = TOK.apply_chat_template([{"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": args}}]}], tokenize=False)
    match = re.search(r"<tool_call>.*?</tool_call>", rendered, re.S)
    assert match, rendered[:200]
    return match.group(0)


def make_repair(i: int, failure: str, path: str, buggy: str, fixed: str, green: str) -> dict:
    turns = [
        ("run_tests", {}, failure),
        ("read_file", {"path": path}, buggy),
        ("write_file", {"path": path, "content": fixed}, f"OK {path} written ({fixed.count(chr(10)) + 1} lines)"),
        ("run_tests", {}, green),
    ]
    messages = [{"role": "user", "content": RULES + "Diagnose the red suite, fix the responsible file, and report only after the observed green summary."}]
    for name, args, observation in turns:
        messages.append({"role": "assistant", "content": tool_call(name, args)})
        messages.append({"role": "tool", "content": observation})
    messages.append({"role": "assistant", "content": f"Fixed {path}; the observed test summary is {green}."})
    assert messages[-2]["content"].endswith(green)
    assert "tool_response" not in messages[-1]["content"]
    return {"id": f"repair-{i:02d}", "track": "CHOWDER_AGENT", "domain": "agent_tool_use_envelope", "task_family": "spark_envelope_observation_gated_loop", "source": "teacher_synthetic", "tools": TOOLS, "messages": messages, "verification": {"method": "executable_test", "expected": "one action per turn; final report follows green run_tests", "status": "passed"}, "failure_mode": "premature success, fabricated observation, or batched calls", "difficulty": "hard", "estimated_tokens": 900}


CASES = [
    ("test_version.py::test_short - ValueError: not enough values to unpack", "version.py", "def parse_version(s):\n    a, b = s.split('.')\n    return (int(a), int(b))\n", "def parse_version(s):\n    parts = s.split('.')\n    return tuple(int(p) for p in parts) + (0,) * (3 - len(parts))\n", "2 passed"),
    ("test_total.py::test_empty - IndexError: list index out of range", "src/total.py", "def total(xs):\n    return xs[0] + sum(xs[1:])\n", "def total(xs):\n    return sum(xs)\n", "4 passed"),
    ("test_parse.py::test_decimal - ValueError: invalid literal for int", "src/parse.py", "def parse(s):\n    return int(s)\n", "def parse(s):\n    return int(s.replace('$', '').replace(',', ''))\n", "3 passed"),
    ("test_auth.py::test_missing - KeyError: 'role'", "src/auth.py", "def role(user):\n    return user['role']\n", "def role(user):\n    return user.get('role', 'guest')\n", "6 passed"),
    ("test_queue.py::test_peek - IndexError: pop from empty list", "src/queue.py", "def peek(xs):\n    return xs.pop(0)\n", "def peek(xs):\n    return xs[0] if xs else None\n", "5 passed"),
    ("test_config.py::test_flag - TypeError: unhashable type: list", "src/config.py", "def enabled(cfg, key):\n    return cfg[key] in ('on', True)\n", "def enabled(cfg, key):\n    return cfg.get(key) in ('on', True)\n", "7 passed"),
    ("test_date.py::test_year - IndexError: tuple index out of range", "src/date.py", "def year(parts):\n    return int(parts[:3][2])\n", "def year(parts):\n    return int(parts[2])\n", "2 passed"),
    ("test_text.py::test_trim - AttributeError: 'NoneType' object has no attribute 'strip'", "src/text.py", "def clean(value):\n    return value.strip().lower()\n", "def clean(value):\n    return (value or '').strip().lower()\n", "8 passed"),
    ("test_math.py::test_divisor - ZeroDivisionError: division by zero", "src/math.py", "def mean(xs):\n    return sum(xs) / len(xs)\n", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0\n", "3 passed"),
    ("test_json.py::test_field - KeyError: 'id'", "src/json_api.py", "def public(row):\n    return {'value': row['value']}\n", "def public(row):\n    return {'id': row.get('id'), 'value': row.get('value')}\n", "5 passed"),
    ("test_cli.py::test_flag - TypeError: int() argument must be a string", "src/cli.py", "def port(value):\n    return int(value or 8080)\n", "def port(value):\n    return int(value) if value else 8080\n", "4 passed"),
    ("test_retry.py::test_limit - ValueError: timeout must be positive", "src/retry.py", "def delay(attempt, timeout):\n    return timeout / attempt\n", "def delay(attempt, timeout):\n    return timeout / max(1, attempt)\n", "6 passed"),
]
repairs = [make_repair(i, *case) for i, case in enumerate(CASES, 1)]

with REPLAY.open(encoding="utf-8") as f:
    replay = [json.loads(line) for line in f if line.strip()]
rng = random.Random(705)
rng.shuffle(replay)
replay = replay[:48]
assert len(replay) == 48 and len(repairs) == 12

# Interleave replay and repair so neither mode monopolizes the optimizer.
text_rows = []
for i in range(12):
    text_rows.append({"id": replay[i * 4]["id"] if "id" in replay[i * 4] else f"gen2-replay-{i * 4:02d}", "text": replay[i * 4]["text"]})
    text_rows.append({"id": replay[i * 4 + 1].get("id", f"gen2-replay-{i * 4 + 1:02d}"), "text": replay[i * 4 + 1]["text"]})
    text_rows.append({"id": replay[i * 4 + 2].get("id", f"gen2-replay-{i * 4 + 2:02d}"), "text": replay[i * 4 + 2]["text"]})
    text_rows.append({"id": replay[i * 4 + 3].get("id", f"gen2-replay-{i * 4 + 3:02d}"), "text": replay[i * 4 + 3]["text"]})
    repair = repairs[i]
    text_rows.append({"id": repair["id"], "text": TOK.apply_chat_template(repair["messages"], tools=repair["tools"], tokenize=False)})

with OUT_RECORDS.open("w", encoding="utf-8", newline="\n") as f:
    for rec in repairs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with OUT_TEXT.open("w", encoding="utf-8", newline="\n") as f:
    for row in text_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"wrote {OUT_RECORDS} ({len(repairs)} repair records)")
print(f"wrote {OUT_TEXT} ({len(text_rows)} rows: 48 replay + 12 repair)")
print("repair families: green-gated diagnose/read/write/verify/report")
