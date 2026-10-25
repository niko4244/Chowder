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
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

MAX_TOKENS = {"gsm8k": 1024, "humaneval": 768, "math500": 2048}
SYSTEM = "You are a helpful assistant."
IDENTITY_SCHEMA = "chowder.reap-eval-input-identity.v1"


def _digest(value: Any) -> str:
    """Hash the frozen rendered test prompts, answers and scoring inputs."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bind_evaluation_inputs(out_dir: Path, suite: str, items: list[dict]) -> dict:
    """Refuse resuming outputs from an unclaimed model, prompt or decode policy.

    The serving wrapper previously hashes its GGUF once while creating
    serve.json. Reuse that model witness rather than hashing huge weight
    files again. This is not independently verified process/weight attestation.
    """
    serving = out_dir / "serve.json"
    if not serving.is_file():
        raise ValueError("REAP evaluation needs a bound serve.json from reap_serve; "
                         "standalone unverified evaluation cannot certify a model")
    try:
        model = json.loads(serving.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("REAP serving model identity is unreadable") from exc
    if (not isinstance(model, dict) or not isinstance(model.get("gguf"), str)
            or not model["gguf"].strip() or not isinstance(model.get("flags"), list)
            or not all(isinstance(v, str) for v in model["flags"])
            or not isinstance(model.get("gguf_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", model["gguf_sha256"])):
        raise ValueError("REAP serving model identity lacks valid GGUF content digest/flags")
    ids = [item.get("id") if isinstance(item, dict) else None for item in items]
    if any(not isinstance(ident, str) or not ident for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError("REAP benchmark source has missing/duplicate item ids")
    context = {
        "schema": IDENTITY_SCHEMA,
        "model": {"gguf": model["gguf"], "gguf_sha256": model["gguf_sha256"],
                  "flags": model["flags"]},
        "suite": suite,
        "source_items_sha256": _digest(items),
        "source_items_n": len(items),
        "decode": {"system": SYSTEM, "temperature": 0.0, "top_k": 1,
                   "seed": 7, "max_tokens": MAX_TOKENS[suite],
                   "enable_thinking": False},
        "scoring": "frozen reap_eval production scoring: final_number_match/boxed/HumanEval",
    }
    witness = out_dir / f"{suite}.inputs.json"
    results = out_dir / f"{suite}.jsonl"
    if witness.is_file():
        try:
            recorded = json.loads(witness.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError("REAP resume identity witness is unreadable") from exc
        if recorded != context:
            raise ValueError("REAP resume identity or frozen source/decode inputs changed; "
                             "use a new experiment output directory")
    else:
        if results.exists():
            raise ValueError("REAP unclaimed existing results have no original resume identity; "
                             "use a new experiment output directory")
        # Exclusive creation preserves an earlier interrupted experiment witness.
        with witness.open("x", encoding="utf-8") as stream:
            json.dump(context, stream, sort_keys=True, indent=2)
            stream.write("\n")
    return context


def _read_resume_rows(path: Path, allowed_ids: set[str], summary_path: Path,
                      suite: str, *, expected_summary: dict | None = None) -> dict[str, dict]:
    """Reject duplicate, unrelated and post-summary edited outputs."""
    if path.is_symlink():
        raise ValueError("REAP result JSONL cannot be a symlink")
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    if expected_summary is None:
        summary = {}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError("REAP saved evaluation summary cannot be parsed") from exc
            if not isinstance(summary, dict):
                raise ValueError("REAP saved evaluation summary is malformed")
        earlier = summary.get(suite)
    else:
        earlier = expected_summary
    if earlier is not None and (
        not isinstance(earlier, dict)
        or earlier.get("file_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("REAP existing result SHA256 disagrees with earlier summary")
    done: dict[str, dict] = {}
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"REAP existing result row {index} is malformed") from exc
        ident = row.get("id") if isinstance(row, dict) else None
        if not isinstance(ident, str) or ident not in allowed_ids:
            raise ValueError(f"REAP existing result row {index} has unexpected item id")
        if ident in done:
            raise ValueError(f"REAP existing result contains duplicate id {ident!r}")
        done[ident] = row
    if isinstance(earlier, dict) and earlier.get("n") != len(done):
        raise ValueError("REAP existing summary row count differs from recorded results")
    return done


def _summarize_rows(rows: list[dict], path: Path) -> dict:
    """Recompute the original summary fields from witnessed scored rows."""
    if not rows:
        raise ValueError("REAP cannot summarize zero scored rows")
    for row in rows:
        ratio = row.get("trigram_ratio")
        tokens = row.get("completion_tokens")
        if (not isinstance(row.get("correct"), bool)
                or not isinstance(row.get("terminated"), bool)
                or isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio) or not 0 <= ratio <= 1
                or (tokens is not None and (
                    isinstance(tokens, bool) or not isinstance(tokens, (int, float))
                    or not math.isfinite(tokens) or tokens < 0))):
            raise ValueError("REAP existing result has invalid scored row fields")
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(row["correct"] for row in rows) / n,
        "terminated_rate": sum(row["terminated"] for row in rows) / n,
        "degenerate_rate": sum(row["trigram_ratio"] < 0.5 for row in rows) / n,
        "mean_completion_tokens": sum((row["completion_tokens"] or 0) for row in rows) / n,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _summary_bytes(path: Path) -> bytes | None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("REAP evaluation summary must be a regular file")
    return path.read_bytes() if path.is_file() else None


def _validated_saved_summary(out_dir: Path, *, skip: str | None = None) -> tuple[dict, bytes | None]:
    """Admit only summaries backed by the current input identity and JSONL bytes.

    `skip` is the suite currently being appended, whose *old* summary digest
    necessarily differs until the new summary is atomically published.
    """
    summary_path = out_dir / "summary.json"
    raw = _summary_bytes(summary_path)
    if raw is None:
        return {}, None
    try:
        saved = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("REAP saved evaluation summary cannot be parsed") from exc
    if not isinstance(saved, dict):
        raise ValueError("REAP saved evaluation summary is malformed")
    for suite, entry in saved.items():
        if suite == skip:
            continue
        if suite not in MAX_TOKENS or not isinstance(entry, dict):
            raise ValueError("REAP saved evaluation summary contains unsupported suite or fields")
        path = out_dir / f"{suite}.jsonl"
        witness = out_dir / f"{suite}.inputs.json"
        if (not path.is_file() or path.is_symlink()
                or not witness.is_file() or witness.is_symlink()):
            raise ValueError(f"REAP saved {suite} summary lacks a result or identity witness")
        full_items = _load(suite)
        _bind_evaluation_inputs(out_dir, suite, full_items)
        allowed = {it["id"] for it in full_items}
        done = _read_resume_rows(path, allowed, summary_path, suite)
        expected = _summarize_rows([done[it["id"]] for it in full_items
                                    if it["id"] in done], path)
        if expected != entry:
            raise ValueError(f"REAP saved {suite} summary metrics or SHA256 mismatch")
    return saved, raw


@contextmanager
def _summary_write_guard(out_dir: Path):
    """Serialize suite re-entry; an orphan lock fails closed after an unclean exit."""
    lock = out_dir / ".summary.json.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
    # Windows prevents unlinking an open file, including our own lock.
    try:
        yield
    finally:
        lock.unlink()


def _publish_summary(out_dir: Path, summary: dict, *, prior_bytes: bytes | None,
                     updated_suite: str) -> bytes:
    """Reject stale changes before atomic replacement of one completed suite."""
    path = out_dir / "summary.json"
    existing, raw = _validated_saved_summary(out_dir, skip=updated_suite)
    if raw != prior_bytes:
        raise ValueError("REAP summary changed while scoring; refuse stale overwrite")
    if any(summary.get(suite) != entry for suite, entry in existing.items()
           if suite != updated_suite):
        raise ValueError("REAP other suite summary changed during merge")
    selected = out_dir / f"{updated_suite}.jsonl"
    if (selected.is_symlink() or not selected.is_file()
            or summary[updated_suite]["file_sha256"]
            != hashlib.sha256(selected.read_bytes()).hexdigest()):
        raise ValueError("REAP latest suite output SHA256 changed before publication")
    full_items = _load(updated_suite)
    _bind_evaluation_inputs(out_dir, updated_suite, full_items)
    done = _read_resume_rows(
        selected, {it["id"] for it in full_items}, path, updated_suite,
        expected_summary=summary[updated_suite],
    )
    expected = _summarize_rows([done[it["id"]] for it in full_items
                                if it["id"] in done], selected)
    if expected != summary[updated_suite]:
        raise ValueError("REAP new suite metrics/rows changed before publication")
    encoded = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    pending = out_dir / ".summary.json.pending"
    with pending.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
    try:
        if _summary_bytes(path) != prior_bytes:
            raise ValueError("REAP summary changed before atomic publication")
        os.replace(pending, path)
    finally:
        if pending.exists():
            pending.unlink()
    return encoded


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


def _run_suites(args: argparse.Namespace, out_dir: Path) -> None:
    suites = args.suites.split(",")
    if (not suites or len(suites) != len(set(suites))
            or any(suite not in MAX_TOKENS for suite in suites)):
        raise ValueError("REAP evaluation requires unique known suite names")
    if args.limit < 0 or args.parallel < 1:
        raise ValueError("REAP evaluation limit/parallel values must be nonnegative/positive")
    with _summary_write_guard(out_dir):
        # Validate ALL previously summarized suites before any new model call.
        # A run for HumanEval must never silently discard corrupt GSM8K proof.
        summary, summary_bytes = _validated_saved_summary(out_dir)
        for suite in suites:
            summary_bytes = _run_suite(args, out_dir, suite, summary, summary_bytes)


def _run_suite(args: argparse.Namespace, out_dir: Path, suite: str,
               summary: dict, summary_bytes: bytes | None) -> bytes:
    items = _load(suite)
    # Bind the complete source population, not just an optional limited
    # first batch, so legitimate partial runs can resume full evaluation.
    input_witness = _bind_evaluation_inputs(out_dir, suite, items)
    if args.limit:
        items = items[: args.limit]
    path = out_dir / f"{suite}.jsonl"
    done = _read_resume_rows(path, {str(it["id"]) for it in items},
                             out_dir / "summary.json", suite)
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
    if _bind_evaluation_inputs(out_dir, suite, _load(suite)) != input_witness:
        raise ValueError("REAP evaluation source identity changed while scoring")
    summary[suite] = _summarize_rows(rows, path)
    summary_bytes = _publish_summary(out_dir, summary, prior_bytes=summary_bytes,
                                     updated_suite=suite)
    print(args.arm, suite, json.dumps(summary[suite]), flush=True)
    return summary_bytes


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
    _run_suites(args, out_dir)


if __name__ == "__main__":
    main()
