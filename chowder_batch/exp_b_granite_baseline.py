"""Experiment B: Granite 4.0 H Tiny student baseline (Phase 1).

Benchmarks the *unmodified* student on the same eval tasks the teacher and
small model faced in Experiment E, using Granite's native chat template served
by llama.cpp. This is the pretrained-then-benchmarked reference every
fine-tuned student must beat.

Also records the tokenizer-alignment facts between teacher (Qwen3.8, 248320
vocab) and student (Granite, 100352 vocab): sequences do NOT align, which is
why token-level KL over shared indices is invalid for this pair (Phase 3
condition C).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chowder_batch"))

GRANITE_PORT = 18083
TEACHER_PORT = 18081
SPARK_PORT = 18082


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def chat(port: int, messages: list[dict], *, max_tokens: int) -> tuple[str, dict]:
    """Chat completion returning full visible output (content + reasoning)."""
    t0 = time.perf_counter()
    data = _post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {"messages": messages, "max_tokens": max_tokens, "temperature": 0},
    )
    wall = time.perf_counter() - t0
    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    visible = reasoning + "\n" + content if reasoning and content else (content or reasoning)
    return visible, {
        "wall_seconds": round(wall, 3),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
        "tokens_per_second": round(usage.get("completion_tokens", 0) / (timings.get("predicted_ms", 1) / 1000), 2),
    }


def grade_answer(answer: str, response: str) -> bool:
    return str(answer).lower().replace(",", "").strip() in response.lower().replace(",", "")


def tokenizer_report() -> dict:
    """Token-level alignment facts for the teacher/student pair."""
    from transformers import AutoTokenizer

    teacher = AutoTokenizer.from_pretrained("F:/llm-models/Qwen3.8-9B-abliterated-25-bf16", local_files_only=True)
    student = AutoTokenizer.from_pretrained(
        "ibm-granite/granite-4.0-h-tiny", local_files_only=False
    )
    sample = "def parse_version(s):\n    return tuple(int(p) for p in s.split('.'))\n"
    t_ids = teacher(sample, add_special_tokens=False)["input_ids"]
    s_ids = student(sample, add_special_tokens=False)["input_ids"]
    return {
        "teacher_vocab": len(teacher),
        "student_vocab": len(student),
        "identical_tokenization": t_ids == s_ids,
        "sample_token_counts": {"teacher": len(t_ids), "student": len(s_ids)},
        "token_level_kl_valid": False,
        "reason": "vocabularies differ; shared indices do not denote shared tokens",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=640)
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))["tasks"]
    eval_tasks = [t for t in tasks if t["split"] == "eval" and t["kind"] != "repair"]

    rows = []
    for task in eval_tasks:
        content, meta = chat(GRANITE_PORT, [{"role": "user", "content": task["question"]}], max_tokens=args.max_tokens)
        rows.append({
            "task": task["name"], "kind": task["kind"], "difficulty": task["difficulty"],
            "correct": grade_answer(task["answer"], content),
            "latency_s": meta["wall_seconds"], "tokens": meta["completion_tokens"],
            "tok_s": meta["tokens_per_second"], "finish": meta["finish_reason"],
            "response": content[-200:],
        })
        print(f"{task['name']}: correct={rows[-1]['correct']} ({meta['tokens_per_second']} tok/s)", flush=True)

    summary = {}
    for kind in ("gsm8k", "factual"):
        subset = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "accuracy": sum(r["correct"] for r in subset) / max(len(subset), 1),
            "latency_s": round(sum(r["latency_s"] for r in subset) / max(len(subset), 1), 2),
            "tokens": round(sum(r["tokens"] for r in subset) / max(len(subset), 1), 1),
            "n": len(subset),
        }
    payload = {"student": "ibm-granite/granite-4.0-h-tiny (Q3_K_M, llama.cpp)", "summary": summary, "tasks": rows}
    try:
        payload["tokenizer_alignment"] = tokenizer_report()
    except Exception as error:  # tokenizer fetch can fail offline; not fatal
        payload["tokenizer_alignment"] = {"error": str(error)[:200]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
