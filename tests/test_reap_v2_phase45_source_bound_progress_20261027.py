"""Fail-closed bridge tests; synthetic fixtures are not benchmark results."""

from copy import deepcopy

import pytest

from chowder_batch import reap_v2_phase45_source_bound_progress_20261027 as bridge


def synthetic_read_only_receipts():
    q = {
        "status": "FULL_164_INDEPENDENT_RESEARCH_PAIRED_Q8_Q4_ONLY",
        "paired_original_case_count": 164,
        "original_q4_correct_in_paired_items": 151,
        "original_q8_correct_in_paired_items": 152,
        "both_correct": 147, "q8_only_correct": 5,
        "q4_only_correct": 4, "both_incorrect": 8,
        "remaining_unscored_q8_original_cases": 0,
        "original_q4_model_sha256": bridge.HUMAN_Q4_SHA,
        "original_q8_model_sha256": bridge.HUMAN_Q8_SHA,
        "original_q4_generated_rows_sha256": "e" * 64,
        "original_q8_generated_rows_sha256": "f" * 64,
        "original_q4_witnessed_journal_sha256": "a" * 64,
        "original_q8_witnessed_journal_sha256": "b" * 64,
        "original_q4_serving_session": "q4-source-session",
        "original_q8_serving_session": "q8-source-session",
        "is_original_frozen_Q8_vs_dense_gate2": False,
        "is_amendment4_paired_mcnemar_ci": False,
        "is_confirmatory_quantization_comparison": False,
        "gen0_status": "BLOCKED", "release_approved": False,
    }
    a = {
        "status": "PARTIAL_AUTHENTIC_OFFICIAL_TOOL_EVALUATOR_EXECUTION",
        "completed_scored_tasks": 87,
        "totals": {"strict_passes": 3, "model_calls": 2278, "tool_calls": 3294},
        "registered_population": 600,
        "active_unscored_indices": [87],
        "official_complete_600_claim_allowed": False,
        "official_source_commit": bridge.AUTOMATION_ORIGINAL_COMMIT,
        "original_prompt_sha256": bridge.AUTOMATION_PROMPT_SHA,
        "model_gguf_sha256": bridge.HUMAN_Q4_SHA,
        "historical_git_commit": bridge.auto.GIT_COMMIT,
        "historical_guard_policy": "guard.v3",
        "historical_git_runner_source_sha256": "c" * 64,
        "owner_sha256": "d" * 64,
        "noncomparable_to_frozen_four_slot": True,
        "frozen_gen0_status": "BLOCKED", "release_approved": False,
    }
    return q, a


def test_synthetic_source_compatible_bridge_is_research_only():
    h, a = synthetic_read_only_receipts()
    result = bridge.build(h, a)
    assert result["human_eval"]["q4_correct"] == 151
    assert result["human_eval"]["q8_correct"] == 152
    assert result["automationbench"]["scored"] == 87
    assert result["automationbench"]["strict_passes"] == 3
    assert result["eleven_frontier_original_suite_status"] == "INCOMPLETE"
    assert result["gate2"] == "UNKNOWN"
    assert result["gen0"] == {"status": "BLOCKED", "chosen": None, "release_approved": False}


@pytest.mark.parametrize("arm,field,value", [
    ("human", "status", "PARTIAL_PREFIX_SOURCE_BOUND_NOT_A_FULL_BENCHMARK"),
    ("human", "paired_original_case_count", 163),
    ("human", "both_correct", 148),
    ("human", "original_q8_model_sha256", "0" * 64),
    ("human", "is_original_frozen_Q8_vs_dense_gate2", True),
    ("human", "gen0_status", "PASS"),
    ("auto", "completed_scored_tasks", 600),
    ("auto", "totals", {"strict_passes": 601, "model_calls": 10, "tool_calls": 20}),
    ("auto", "historical_guard_policy", "guard.v4"),
    ("auto", "model_gguf_sha256", "1" * 64),
    ("auto", "official_source_commit", "wrong"),
    ("auto", "frozen_gen0_status", "PASS"),
    ("auto", "noncomparable_to_frozen_four_slot", False),
])
def test_mutated_source_receipt_fails_closed(arm, field, value):
    h, a = deepcopy(synthetic_read_only_receipts())
    (h if arm == "human" else a)[field] = value
    with pytest.raises(bridge.ReceiptRefusal):
        bridge.build(h, a)


def test_original_witnesses_recomputed_from_live_archived_files():
    result = bridge.audit_actual()
    assert result["human_eval"]["population"] == 164
    assert result["human_eval"]["q4_correct"] == 151
    assert result["human_eval"]["q8_correct"] == 152
    assert result["automationbench"]["scored"] >= 87
    assert result["automationbench"]["strict_passes"] >= 3
    assert result["gen0"]["status"] == "BLOCKED"
