"""Tests for the corrected 64k/45-step runner's pure verification layer."""

from __future__ import annotations

import json

import pytest

from chowder_batch.reap_q4_automationbench_ctx64k_20261026 import (
    CONTEXT,
    MAX_STEPS,
    TASKS,
    audit_official,
    audit_truncation,
    verify_cuda,
    verify_geometry,
)

BASE_PROMPT = 1067
STEP_GROWTH = 1372.7


def clean_log(steps: int, *, context: int = CONTEXT) -> str:
    lines = []
    prompt = BASE_PROMPT
    for i in range(steps):
        lines.append(
            f"1.{i:02d}.000.000 I slot   operator(): id  0 | task {1000 + i} | "
            f"new prompt, n_ctx_slot = {context}, n_keep = 0, task.n_tokens = {int(prompt)}"
        )
        lines.append(
            f"1.{i:02d}.500.000 I slot print_timing: id  0 | task {1000 + i} | "
            f"eval time = 3000.00 ms / 300 tokens"
        )
        prompt += STEP_GROWTH
    lines.append(f"1.99.000.000 I srv  main: n_ctx = {context}")
    return "\n".join(lines) + "\n"


def official_result(**overrides) -> dict:
    rows = [
        {"name": name, "passed": (i % 2 == 0), "score": 1.0 if i % 2 == 0 else 0.0,
         "num_model_calls": 5, "num_tool_calls": 4}
        for i, name in enumerate(TASKS)
    ]
    doc = {
        "meta": {"benchmark_version": "1.0.6", "model": "reap128_v2_q4",
                 "toolset": "api", "max_steps": MAX_STEPS, "total_tasks": len(TASKS)},
        "summary": {"passed_count": sum(r["passed"] for r in rows), "aborted_tasks": []},
        "tasks": rows,
        "usage_by_task": {},
    }
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        doc[section][field] = value
    return doc


def blob(doc: dict) -> bytes:
    return json.dumps(doc).encode("utf-8")


# --------------------------------------------------------------------------
# geometry / cuda
# --------------------------------------------------------------------------


def test_geometry_accepts_declared_context():
    geometry = verify_geometry(clean_log(3), context=CONTEXT)
    assert geometry["observed_context_per_seq"] == CONTEXT
    assert geometry["observed_sequences"] == 1


def test_geometry_rejects_silent_fit_shrink():
    with pytest.raises(ValueError, match="slot geometry"):
        verify_geometry(clean_log(3, context=32768), context=CONTEXT)


def test_cuda_requires_full_offload():
    assert verify_cuda("load_tensors: offloaded 41/41 layers to GPU") == {
        "layers_cuda": "41/41", "measured_experts_offloaded": 0,
    }


def test_cuda_rejects_partial_offload():
    with pytest.raises(ValueError):
        verify_cuda("load_tensors: offloaded 30/41 layers to GPU")


# --------------------------------------------------------------------------
# official audit
# --------------------------------------------------------------------------


def test_audit_rederives_aggregates_from_rows():
    audit = audit_official(blob(official_result()), served_id="reap128_v2_q4",
                           count=len(TASKS), max_steps=MAX_STEPS, expected=TASKS)
    assert audit["completed_tasks"] == 6
    assert audit["passed_tasks"] == 3
    assert audit["completed_pass_rate"] == pytest.approx(0.5)
    assert audit["aborted_unscored_tasks"] == 0
    assert audit["chosen"] is None and audit["gen0_status"] == "BLOCKED"


def test_audit_rejects_the_old_step_budget():
    doc = official_result()
    doc["meta"]["max_steps"] = 50
    with pytest.raises(ValueError, match="max_steps"):
        audit_official(blob(doc), served_id="reap128_v2_q4", count=len(TASKS),
                       max_steps=MAX_STEPS, expected=TASKS)


def test_audit_rejects_summary_that_disagrees_with_rows():
    doc = official_result()
    doc["summary"]["passed_count"] = 5
    with pytest.raises(ValueError, match="passed_count"):
        audit_official(blob(doc), served_id="reap128_v2_q4", count=len(TASKS),
                       max_steps=MAX_STEPS, expected=TASKS)


def test_audit_rejects_passed_flag_contradicting_score():
    doc = official_result()
    doc["tasks"][0]["passed"] = not doc["tasks"][0]["passed"]
    with pytest.raises(ValueError, match="passed flag"):
        audit_official(blob(doc), served_id="reap128_v2_q4", count=len(TASKS),
                       max_steps=MAX_STEPS, expected=TASKS)


def test_audit_rejects_a_different_served_model():
    with pytest.raises(ValueError, match="scored model"):
        audit_official(blob(official_result()), served_id="some_other_model",
                       count=len(TASKS), max_steps=MAX_STEPS, expected=TASKS)


def test_audit_excludes_aborted_rows_from_the_rate():
    doc = official_result()
    aborted = TASKS[0]
    doc["summary"]["aborted_tasks"] = [aborted]
    doc["tasks"][0]["passed"] = False
    doc["tasks"][0]["score"] = 0.0
    doc["summary"]["passed_count"] = sum(
        row["passed"] for row in doc["tasks"] if row["name"] != aborted
    )
    audit = audit_official(blob(doc), served_id="reap128_v2_q4", count=len(TASKS),
                           max_steps=MAX_STEPS, expected=TASKS)
    assert audit["completed_tasks"] == 5
    assert audit["aborted_unscored_tasks"] == 1
    assert audit["requested_task_pass_rate"] == pytest.approx(2 / 6)


def test_audit_accepts_the_same_slice_in_a_different_order():
    # Only membership is pinned; row order inside the export is not semantic.
    audit = audit_official(blob(official_result()), served_id="reap128_v2_q4",
                           count=len(TASKS), max_steps=MAX_STEPS,
                           expected=tuple(reversed(TASKS)))
    assert audit["completed_tasks"] == 6


def test_audit_rejects_a_substituted_task():
    swapped = TASKS[:-1] + ("finance.reconcile_ledger",)
    with pytest.raises(ValueError, match="six-task slice"):
        audit_official(blob(official_result()), served_id="reap128_v2_q4",
                       count=len(TASKS), max_steps=MAX_STEPS, expected=swapped)


# --------------------------------------------------------------------------
# truncation audit: the whole point of this lane
# --------------------------------------------------------------------------


def test_clean_episode_proves_truncation_eliminated():
    result = audit_truncation(clean_log(MAX_STEPS), context=CONTEXT, max_steps=MAX_STEPS)
    assert result["truncation_eliminated"] is True
    assert result["truncated_generations"] == 0
    assert result["tool_call_json_failures"] == 0
    assert result["affordable_steps"] >= MAX_STEPS
    assert result["max_prompt_tokens"] < CONTEXT


def test_same_episode_at_32k_still_shows_truncation():
    text = clean_log(MAX_STEPS, context=32768)
    text += (
        "2.29.054.431 I slot      release: id  0 | task 11911 | stop processing: "
        "n_tokens = 32767, truncated = 1\n"
        '2.29.060.808 W srv    operator(): got exception: {"error":{"code":500,'
        '"message":"Failed to parse tool call arguments as JSON: parse error at column 89"}}\n'
    )
    result = audit_truncation(text, context=32768, max_steps=MAX_STEPS)
    assert result["truncation_eliminated"] is False
    assert result["truncated_generations"] == 1
    assert result["tool_call_json_failures"] == 1
    assert result["affordable_steps"] < MAX_STEPS
