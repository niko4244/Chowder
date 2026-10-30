"""Independent source-locked original AutomationBench600 Q4 live scorer.

Never starts inference, reads incomplete evaluator output as a finished score,
alters original verifier metadata or mixes model sessions. Each completed task
has its own immutable official JSON, original task name, model identity and
recomputed strict WorldState pass. Active tasks are UNKNOWN until scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chowder_batch import reap_q4_automationbench_full600_resumable_20261027 as run

SCHEMA = "chowder.original-automationbench600-live-independent-audit.v1"


def digest(path: Path) -> str:
    return run._digest(path)


def inspect(root: Path) -> dict:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("audit must use an existing ordinary original run root")
    actual_source = run.q4.verify_official_source()
    if (actual_source["source_git_commit"] != run.q4.BENCH_COMMIT
            or actual_source["benchmark_version"] != run.q4.BENCH_VERSION):
        raise ValueError("live original evaluator source drifted after run admission")
    owner_file = root / "owner.json"
    if owner_file.is_symlink():
        raise ValueError("linked source owner is untrusted")
    owner_bytes = owner_file.read_bytes()
    owner = run._json(owner_file)
    if (run._canonical(owner) != owner_bytes
            or owner.get("schema") != run.SCHEMA
            or owner.get("model_sha256") != run.q4.GGUF_SHA
            or owner.get("official_benchmark_commit") != run.q4.BENCH_COMMIT
            or owner.get("original_task_prompt_sha256")
                != run.source_ids.PUBLIC_PROMPTS_SHA256
            or owner.get("original_task_population") != 600
            or owner.get("native_flags") != run.q4.FLAGS
            or owner.get("context_guard_policy") != run.q4.POLICY_NAME
            or owner.get("release_approved") is not False
            or owner.get("chosen") is not None):
        raise ValueError("original source/model/protocol owner drift")
    items = owner.get("tasks")
    if (not isinstance(items, list) or len(items) != 600
            or any(not isinstance(i, dict) or
                   not isinstance(i.get("name"), str) or
                   not isinstance(i.get("id"), str) for i in items)
            or len({i["name"] for i in items}) != 600):
        raise ValueError("authentic original 600-task owner broken")
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    expected = {f"task-{i:03d}" for i in range(600)}
    if any(name not in expected and ".unfinished-" not in name for name in directories):
        raise ValueError("unowned output directory in original benchmark")
    owner_sha = hashlib.sha256(owner_bytes).hexdigest()
    records: list[dict] = []
    active: list[int] = []
    total = {"model_calls": 0, "tool_calls": 0, "input_tokens": 0,
             "output_tokens": 0, "model_time_s": 0.0, "tool_time_s": 0.0,
             "strict_passes": 0, "positive_assertions": 0, "positive_assertions_passed": 0,
             "excluded_assertions": 0, "excluded_assertions_passed": 0}
    for index, task in enumerate(items):
        item_dir = root / f"task-{index:03d}"
        if not item_dir.exists():
            continue
        if item_dir.is_symlink():
            raise ValueError("linked original item is untrusted")
        if not (item_dir / "receipt.json").exists():
            active.append(index)
            continue
        saved = run.verify_item(item_dir, index, task, owner_sha, str(run.q4.GGUF))
        if not saved:
            raise ValueError("empty supposedly completed original item")
        original = run._json(item_dir / "public-automationbench.private.json")
        if (len(original["tasks"]) != 1 or
                len(original["usage_by_task"]) != 1 or
                original["usage_by_task"][0].get("task_name") != task["name"]):
            raise ValueError("original task/usage identity corrupted")
        row = original["tasks"][0]
        assert record_type_is(row)
        if (row["name"] != task["name"]
                or row["passed"] != bool(saved["independent_score"]["passed_tasks"])
                or row["input_tokens"] != original["usage_by_task"][0]["input_tokens"]
                or row["output_tokens"] != original["usage_by_task"][0]["output_tokens"]):
            raise ValueError("scored source row differs from official usage or receipt")
        positive = [r for r in row["assertion_results"] if not r.get("excluded")]
        excluded = [r for r in row["assertion_results"] if r.get("excluded")]
        if (len(positive) != row["assertions_total"]
                or row["assertions_passed"] != sum(r["passed"] for r in positive)):
            raise ValueError("original official positive assertion counts inconsistent")
        for key, value in (
            ("model_calls", row["num_model_calls"]), ("tool_calls", row["num_tool_calls"]),
            ("input_tokens", row["input_tokens"]), ("output_tokens", row["output_tokens"]),
            ("model_time_s", row["model_time_s"]), ("tool_time_s", row["tool_time_s"]),
            ("strict_passes", int(row["passed"])),
            ("positive_assertions", len(positive)),
            ("positive_assertions_passed", sum(r["passed"] for r in positive)),
            ("excluded_assertions", len(excluded)),
            ("excluded_assertions_passed", sum(r["passed"] for r in excluded)),
        ):
            total[key] += value
        records.append({"index": index, "name": task["name"],
                        "strict_pass": row["passed"], "positive_passed": row["assertions_passed"],
                        "positive_total": row["assertions_total"],
                        "model_calls": row["num_model_calls"],
                        "tool_calls": row["num_tool_calls"],
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "original_scorer_json_sha256": digest(item_dir / "public-automationbench.private.json"),
                        "checkpoint_sha256": digest(item_dir / "receipt.json")})
    indexes = [r["index"] for r in records]
    if indexes != list(range(len(records))):
        raise ValueError("source population has a skipped ungraded task, cannot silently pool")
    if any(x != len(records) for x in active) or len(active) > 1:
        raise ValueError("only the immediate next original index may be unscored")
    return {"schema": SCHEMA, "status": "PARTIAL_AUTHENTIC_OFFICIAL_TOOL_EVALUATOR_EXECUTION"
            if len(records) < 600 else "COMPLETE_600_SOURCE_BOUND_INDEPENDENT_REPRODUCTION",
            "official_source_commit": run.q4.BENCH_COMMIT,
            "official_version": run.q4.BENCH_VERSION,
            "model": "REAP128_v2 Q4_K_M", "model_gguf_sha256": run.q4.GGUF_SHA,
            "original_prompt_sha256": run.source_ids.PUBLIC_PROMPTS_SHA256,
            "owner_sha256": owner_sha,
            "registered_population": 600, "completed_scored_tasks": len(records),
            "active_unscored_indices": active,
            "strict_pass_rate_completed": total["strict_passes"] / len(records) if records else None,
            "totals": total, "rows": records,
            "official_complete_600_claim_allowed": len(records) == 600,
            "noncomparable_to_frozen_four_slot": True,
            "frozen_gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def record_type_is(row: dict) -> bool:
    if not isinstance(row, dict):
        raise ValueError("official original task row is not an object")
    for key in ("model_time_s", "tool_time_s"):
        value = row.get(key)
        if type(value) not in (int, float) or not 0 <= value < 1_000_000:
            raise ValueError("official timing is nonfinite, negative or implausible")
    if (type(row.get("assertions_total")) is not int or
            type(row.get("assertions_passed")) is not int or
            not isinstance(row.get("assertion_results"), list) or
            type(row.get("input_tokens")) is not int or
            type(row.get("output_tokens")) is not int or
            row["input_tokens"] < 0 or row["output_tokens"] < 0):
        raise ValueError("original official scalar/positive assertion types malformed")
    for a in row["assertion_results"]:
        if not isinstance(a, dict) or type(a.get("passed")) is not bool or type(a.get("excluded")) is not bool:
            raise ValueError("original grader assertion types malformed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path,
                        help="Exclusive NEW source-bound evidence JSON; never rewrite source run")
    args = parser.parse_args()
    try:
        result = inspect(args.original_run)
        if args.snapshot:
            if args.snapshot.exists() or args.snapshot.is_symlink():
                raise ValueError("snapshot destination is not exclusive")
            args.snapshot.write_bytes(run._canonical(result))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                         indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_AUDIT",
                          "reason": str(exc)[:400], "no_model_rerun": True,
                          "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
