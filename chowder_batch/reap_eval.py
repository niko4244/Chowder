"""Score one llama-server arm on the frozen REAP gate suites.

Suites (all from the offline HF cache):
  gsm8k      openai/gsm8k main/test, 1,319 items -- Gate 0 (termination,
             degeneracy) and Gate 1 (accuracy)
  humaneval  openai/openai_humaneval test, 164 items -- Gate 2 (pass@1, executed)
  math500    HuggingFaceH4/MATH-500 test, 500 items -- reported only

Scoring reuses Chowder production: ``chowder.evaluators.scoring.score`` with
``final_number_match`` for GSM8K (answer after the last ``</think>``, last
comma-aware number) and normalized exact match on the boxed answer for
MATH-500. An item whose generation did not stop on its own (finish_reason !=
"stop") scores 0 on every suite. Decoding is greedy, thinking is disabled, and
max_tokens is fixed per suite; the same settings apply to every arm.

HumanEval completions are executed with their unit tests in a fresh
``python -I`` subprocess in an empty temp dir, with a 10 s timeout.

Usage:
  python chowder_batch/reap_eval.py --port 8790 --arm dense --suites gsm8k,humaneval \
      --out C:/llm-work/reap/eval [--limit N] [--parallel 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MAX_TOKENS = {"gsm8k": 1024, "humaneval": 768, "math500": 2048}
SYSTEM = "You are a helpful assistant."


def _load(suite: str):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    if suite == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"id": f"gsm8k-{i}", "prompt": r["question"] + "\n\nSolve step by step, then give the final answer as a number on the last line.",
                 "expected": r["answer"].split("####")[-1].strip()} for i, r in enumerate(ds)]
    if suite == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [{"id": r["task_id"], "prompt": "Complete this Python function. Reply with the full function in one ```python code block.\n\n```python\n" + r["prompt"] + "```",
                 "entry_point": r["entry_point"], "test": r["test"], "header": r["prompt"]} for r in ds]
    if suite == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": r["unique_id"], "prompt": r["problem"] + "\n\nPut the final answer in \\boxed{}.", "expected": r["answer"]} for r in ds]
    raise SystemExit(f"unknown suite {suite}")


def _chat(port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.0, "top_k": 1, "seed": 7, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    return {"text": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
            "completion_tokens": data.get("usage", {}).get("completion_tokens"), "seconds": time.time() - t0,
            "timings": data.get("timings")}


def _trigram_ratio(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(grams)) / len(grams) if grams else 1.0


def _boxed(text: str) -> str:
    i = text.rfind("\\boxed{")
    if i < 0:
        return ""
    depth, j = 0, i + len("\\boxed{")
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            if depth == 0:
                return text[j:k]
            depth -= 1
    return ""


def _humaneval_pass(item: dict, text: str) -> bool:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    code = blocks[-1] if blocks else text
    if f"def {item['entry_point']}" not in code:
        code = item["header"] + code
    program = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, timeout=10)
            return done.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _score(suite: str, item: dict, out: dict) -> dict:
    from chowder.evaluators.scoring import final_answer, normalize, score

    terminated = out["finish_reason"] == "stop"
    text = out["text"]
    if suite == "gsm8k":
        correct = terminated and score(text, item["expected"], "final_number_match") == 1.0
    elif suite == "math500":
        correct = terminated and normalize(_boxed(final_answer(text))) == normalize(item["expected"])
    else:
        correct = terminated and _humaneval_pass(item, final_answer(text))
    return {"id": item["id"], "correct": bool(correct), "terminated": terminated,
            "trigram_ratio": round(_trigram_ratio(text), 4), "completion_tokens": out["completion_tokens"],
            "seconds": round(out["seconds"], 2),
            "predicted_per_second": (out.get("timings") or {}).get("predicted_per_second"), "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--suites", default="gsm8k,humaneval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for suite in args.suites.split(","):
        items = _load(suite)
        if args.limit:
            items = items[: args.limit]
        path = out_dir / f"{suite}.jsonl"
        done = {}
        if path.exists():  # resumable: keep already-scored items
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                done[row["id"]] = row
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with path.open("a", encoding="utf-8") as handle, cf.ThreadPoolExecutor(args.parallel) as pool:
            futures = {pool.submit(_chat, args.port, it["prompt"], MAX_TOKENS[suite]): it for it in todo}
            for n, fut in enumerate(cf.as_completed(futures), 1):
                it = futures[fut]
                row = _score(suite, it, fut.result())
                done[row["id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if n % 50 == 0:
                    print(f"{args.arm}/{suite}: {len(done)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        rows = [done[it["id"]] for it in items]
        summary[suite] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "terminated_rate": sum(r["terminated"] for r in rows) / len(rows),
            "degenerate_rate": sum(r["trigram_ratio"] < 0.5 for r in rows) / len(rows),
            "mean_completion_tokens": sum((r["completion_tokens"] or 0) for r in rows) / len(rows),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(args.arm, suite, json.dumps(summary[suite]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
