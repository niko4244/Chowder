"""Strict source-bound AutomationBench public-600 export census.

This verifies the dataset identity, prompt prefixes, assertion declarations,
partial-credit arithmetic, and exported summary. It deliberately does *not*
claim that exported worldstates were independently replayed by the official
assertion engine, or that original generated actions were actually executed.
The upstream v1.0.6 exporter drops source example_id and task-contract hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def verify_population(sources: list[dict], tasks: list[dict], usage: list[dict],
                      summary: dict) -> dict:
    """Reject forged/replaced 600 rows without upgrading raw data to gate scores.

    `sources` must be regenerated from the pinned official dataset, NOT taken
    from the benchmark result. The returned descriptor is not a Gen-0 pass.
    """
    if (type(sources) is not list or len(sources) != 600
            or any(type(x) is not dict for x in sources)):
        raise ValueError("official source expected 600 items")
    names = [x.get("name") for x in sources]
    ids = [x.get("example_id") for x in sources]
    # Upstream v1.0.6 legitimately REUSES numeric example_id values between
    # domains: its original public 600 has only 570 globally distinct IDs.
    # A source identity is (domain, example_id, canonical task_name), never
    # the export's id (an ordinal), nor numeric example_id alone.
    scoped_ids = [(names[i].split(".", 1)[0], value)
                  for i, value in enumerate(ids) if isinstance(names[i], str)]
    if (any(type(name) is not str or not name or
            name.split(".", 1)[0] != DOMAINS[i // 100]
            for i, name in enumerate(names))
            or len(set(names)) != 600 or
            any(type(x) is not int for x in ids)
            or len(set(scoped_ids)) != 600):
        raise ValueError("official source item identity/domain invalid")
    if (type(tasks) is not list or len(tasks) != 600
            or type(usage) is not list or len(usage) != 600
            or type(summary) is not dict):
        raise ValueError("official result population or usage missing")

    score_sum = 0.0
    passes = 0
    for i, (source, task, usage_row) in enumerate(zip(sources, tasks, usage), 1):
        if type(task) is not dict or task.get("name") != source["name"]:
            raise ValueError("official export source name/order mismatch")
        if type(task.get("id")) is not int or task["id"] != i:
            raise ValueError("official export sequential ordinal invalid (not source ID)")
        if (type(usage_row) is not dict or usage_row.get("task_id") != i
                or usage_row.get("task_name") != source["name"]):
            raise ValueError("official export usage task ID/name mismatch")
        if (type(task.get("steps")) is not int or task["steps"] < 0):
            raise ValueError("official export step count malformed")
        prompt = source.get("prompt")
        messages = task.get("messages")
        if (type(prompt) is not list or not prompt or type(messages) is not list
                or len(messages) < len(prompt)):
            raise ValueError("official source prompt missing from transcript")
        for expected, received in zip(prompt, messages):
            if (type(expected) is not dict or type(received) is not dict
                    or any(received.get(key) != expected.get(key)
                           for key in ("role", "content"))):
                raise ValueError("official source prompt differs from exported transcript")
        # Pinned upstream export.py::is_aborted: a pending final assistant
        # tool call with >2 steps of unused budget is an unscored infra abort.
        last = messages[-1]
        if (type(last) is not dict or (last.get("role") == "assistant"
                and last.get("tool_calls") and task["steps"] < 48)):
            raise ValueError("official aborted transcript cannot be counted as scored")

        assertions = source.get("assertions")
        results = task.get("assertion_results")
        if (type(assertions) is not list or not assertions or
                type(results) is not list or len(assertions) != len(results)):
            raise ValueError("official assertion population missing or different")
        scored = 0
        passed = 0
        for original, actual in zip(assertions, results):
            if (type(original) is not dict or type(actual) is not dict
                    or type(actual.get("passed")) is not bool
                    or type(actual.get("excluded")) is not bool
                    or actual.get("type") != original.get("type")
                    or _canonical(actual.get("params")) != _canonical({
                        k: v for k, v in original.items() if k != "type"})):
                raise ValueError("official assertion declaration/result mismatch")
            if not actual["excluded"]:
                scored += 1
                passed += int(actual["passed"])

        score = passed / scored if scored else 0.0
        row_score = task.get("score")
        if (type(task.get("assertions_total")) is not int
                or type(task.get("assertions_passed")) is not int
                or type(row_score) not in (int, float)
                or not math.isfinite(row_score)
                or abs(row_score - score) > 1e-10
                or task["assertions_total"] != scored
                or task["assertions_passed"] != passed
                or type(task.get("passed")) is not bool
                or task["passed"] != (score == 1.0)):
            raise ValueError("official binary/partial task metric inconsistency")
        if (type(task.get("num_model_calls")) is not int
                or type(task.get("num_tool_calls")) is not int
                or task["num_model_calls"] < 0 or task["num_tool_calls"] < 0):
            raise ValueError("official call metric invalid")
        for key in ("input_tokens", "output_tokens"):
            if (type(task.get(key)) is not int or task[key] < 0
                    or type(usage_row.get(key)) is not int
                    or usage_row[key] != task[key]):
                raise ValueError("official usage token counters inconsistent")
        if (type(usage_row.get("total_tokens")) is not int
                or usage_row["total_tokens"] != task["input_tokens"] + task["output_tokens"]):
            raise ValueError("official usage total tokens inconsistent")
        score_sum += score
        passes += int(task["passed"])

    # A complete-scored 600 cohort has *no* ignored/cancelled task rows.
    if summary.get("aborted_tasks") != []:
        raise ValueError("official aborted tasks cannot be laundered into a full score")
    if (type(summary.get("passed_count")) is not int or
            type(summary.get("failed_count")) is not int or
            summary["passed_count"] != passes or
            summary["failed_count"] != 600 - passes or
            type(summary.get("pass_rate")) not in (int, float) or
            type(summary.get("avg_score")) not in (int, float) or
            not math.isfinite(summary["pass_rate"]) or
            not math.isfinite(summary["avg_score"]) or
            abs(summary["pass_rate"] - passes / 600) > 1e-10 or
            abs(summary["avg_score"] - score_sum / 600) > 1e-10):
        raise ValueError("official exported summary metric inconsistency")
    return {
        "population_count": 600,
        "source_name_order_sha256": hashlib.sha256(_canonical(names).encode()).hexdigest(),
        "verified_raw_strict_passes": passes,
        "binding": "EXACT_PUBLIC_SOURCE_NAMES_PROMPTS_ASSERTIONS_NOT_WORLDSTATE_CERTIFIED",
        "worldstate_recomputed": False,
        "human_executed_code_certified": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }
