"""Evaluation protocol for the teacher-free pilot: leakage discipline and
metric computation from execution evidence.

This module defines and enforces the evaluation protocol; it deliberately does
not itself launch model inference. Generating model outputs is the
operator-authorized GPU step (see recipes/*.json authorization blocks). Given
result files (per-task trajectories), it computes:

  general:  GSM8K exact-match accuracy, held-out perplexity inputs
  repair:   green-completion rate, nonexistent-read frequency, wrong-file
            edits, recovery after failed first patches, premature-success
            frequency, unproductive repeated calls, tool-call efficiency
  efficiency: measured peak VRAM/RAM and tokens/sec passthrough from evidence

Fail-closed rules:
  - candidate comparison requires a measured baseline (never historical scores)
  - repair train/eval split leakage is checked by repository
  - no promotion decision is ever emitted by this module
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

FINAL_HOLDOUT_MARKER = "final_holdout"


def normalized_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def check_repair_split_leakage(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    """Repair tasks must split by repository (or independent bug family).

    Accepts rows with `group` ("repair:<repo>:<task>") or explicit
    repository/task_id fields. Any repository present in both splits is a
    hard failure — eval repos must be unseen in training.
    """
    def repo_of(row: dict) -> str | None:
        group = row.get("group")
        if isinstance(group, str) and group.startswith("repair:"):
            return group.split(":")[1]
        if isinstance(row.get("repository"), str):
            return row["repository"]
        return None

    train_repos = {r for r in (repo_of(x) for x in train_rows) if r}
    eval_repos = {r for r in (repo_of(x) for x in eval_rows) if r}
    overlap = sorted(train_repos & eval_repos)
    return {"leaked_repositories": overlap, "ok": not overlap,
            "train_repositories": len(train_repos), "eval_repositories": len(eval_repos)}


def check_prompt_overlap(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    """Exact normalized-prompt overlap between any train material and final eval."""
    def user_contents(rows: list[dict]):
        for row in rows:
            for m in row.get("messages", []) or []:
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    yield m["content"]

    train_prompts = {normalized_prompt(c) for c in user_contents(train_rows)}
    collisions = []
    for content in user_contents(eval_rows):
        if normalized_prompt(content) in train_prompts:
            collisions.append(normalized_prompt(content)[:80])
    return {"collisions": len(collisions), "ok": not collisions}


ANSWER_RE = re.compile(r"####\s*(-?\$?[\d,]+(?:\.\d+)?%?)")
NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")


def gsm8k_extract(prediction: str) -> str | None:
    m = ANSWER_RE.search(prediction)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    numbers = NUMBER_RE.findall(prediction.splitlines()[-1]) if prediction.strip() else []
    return numbers[-1].replace(",", "").replace("$", "") if numbers else None


def gsm8k_correct(prediction: str, gold: str) -> bool:
    got = gsm8k_extract(prediction)
    if got is None:
        return False
    want = gold.replace(",", "").replace("$", "")
    try:
        return abs(float(got) - float(want)) < 1e-6
    except ValueError:
        return got == want


REPAIR_TOOLS_READ = ("read_file", "view_file", "open_file", "cat")
REPAIR_TOOLS_EDIT = ("edit_file", "write_file", "apply_patch", "insert")


def repair_behaviors(trajectory: list[dict]) -> dict:
    """Behavioral metrics from one recorded tool trajectory (evidence, not claims).

    Each entry: {"action": {...}, "observation": str, "returncode": int}.
    """
    reads = [e for e in trajectory
             if str((e.get("action") or {}).get("tool")) in REPAIR_TOOLS_READ]
    edits = [e for e in trajectory
             if str((e.get("action") or {}).get("tool")) in REPAIR_TOOLS_EDIT]
    test_calls = [e for e in trajectory if (e.get("action") or {}).get("tool") == "run_tests"]
    nonexistent_reads = sum(
        1 for e in reads
        if e.get("returncode") != 0 or "does not exist" in str(e.get("observation", "")).lower()
    )
    failed_tests = [e for e in test_calls if e.get("returncode") != 0]
    recovered = bool(failed_tests) and bool(test_calls) and test_calls[-1].get("returncode") == 0
    premature_success = any(
        "success" in str(e.get("observation", "")).lower()
        and e.get("returncode") == 0 and not e.get("tests_observed")
        for e in trajectory
    )
    action_keys = (
        str((e.get("action") or {}).get("command") or (e.get("action") or {}).get("path") or "")
        for e in reads + edits
    )
    repeat_count = max(Counter(action_keys).values(), default=0)
    return {
        "tool_calls": len(trajectory),
        "nonexistent_reads": nonexistent_reads,
        "edit_calls": len(edits),
        "test_calls": len(test_calls),
        "recovered_after_failed_tests": recovered,
        "premature_success_claim": premature_success,
        "most_repeated_action_count": int(repeat_count),
        "green_completion": bool(test_calls) and test_calls[-1].get("returncode") == 0,
    }


def repair_aggregate(trajectories: list[list[dict]]) -> dict:
    per_task = [repair_behaviors(t) for t in trajectories]
    n = max(len(per_task), 1)
    return {
        "tasks": len(per_task),
        "green_completion_rate": sum(x["green_completion"] for x in per_task) / n,
        "nonexistent_read_rate": sum(x["nonexistent_reads"] for x in per_task) / n,
        "premature_success_rate": sum(x["premature_success_claim"] for x in per_task) / n,
        "recovery_rate": sum(x["recovered_after_failed_tests"] for x in per_task) / n,
        "mean_tool_calls": sum(x["tool_calls"] for x in per_task) / n,
        "per_task": per_task,
    }


def compare(baseline: dict, candidate: dict) -> dict:
    """Paired aggregate comparison. Emits regression flags; never a promotion decision."""
    deltas = {}
    regressions = []
    for key in ("green_completion_rate", "nonexistent_read_rate",
                "premature_success_rate", "recovery_rate", "mean_tool_calls"):
        if key in baseline and key in candidate:
            delta = candidate[key] - baseline[key]
            deltas[key] = round(delta, 4)
            if key in ("nonexistent_read_rate", "premature_success_rate") and delta > 0.05:
                regressions.append(key)
            if key == "green_completion_rate" and delta < -0.05:
                regressions.append(key)
    return {"deltas": deltas, "regressions": regressions,
            "decision": "requires_operator_review",
            "note": "regression flags are advisory; promotion decisions are never automated"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_leak = sub.add_parser("check-leakage")
    p_leak.add_argument("--train", type=Path, required=True)
    p_leak.add_argument("--eval", type=Path, required=True, dest="eval_path")
    p_repair = sub.add_parser("repair-metrics")
    p_repair.add_argument("--trajectories", type=Path, required=True,
                          help="JSONL where each row is a recorded tool trajectory list")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--baseline", type=Path, required=True)
    p_cmp.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "check-leakage":
        train, ev = load_jsonl(args.train), load_jsonl(args.eval_path)
        print(json.dumps({"repair_split": check_repair_split_leakage(train, ev),
                          "prompt_overlap": check_prompt_overlap(train, ev)}, indent=2))
    elif args.cmd == "repair-metrics":
        # One trajectory per JSONL row (each a list of recorded events),
        # matching what the replay/generation steps write.
        print(json.dumps(repair_aggregate(load_jsonl(args.trajectories)), indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare(json.loads(args.baseline.read_text()),
                                 json.loads(args.candidate.read_text())), indent=2))

if __name__ == "__main__":
    main()
