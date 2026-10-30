"""Read-only current REAP128_v2 original HumanEval + AutomationBench progress.

Every count comes from original, independently rescored per-item receipts,
never copied from prose, fabricated fixtures, or unrelated model sessions.
This is a RESEARCH ledger, not the preregistered Gen-0 selector.
"""

from __future__ import annotations

import json

from chowder_batch import reap_humaneval_v38_q4q8_paired_original_20261027 as human
from chowder_batch import reap_original600_guardv3_git_memory_audit_worker11_20261027 as auto

HUMAN_Q4_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
HUMAN_Q8_SHA = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
AUTOMATION_ORIGINAL_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
AUTOMATION_PROMPT_SHA = "1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5"


class ReceiptRefusal(ValueError):
    """Read-only archived evidence does not satisfy this bridge's identity rules."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ReceiptRefusal(reason)


def build(human_result: dict, automation_result: dict) -> dict:
    """Validate both full and partial source-authenticated studies independently."""
    paired_n = human_result.get("paired_original_case_count")
    correct4 = human_result.get("original_q4_correct_in_paired_items")
    correct8 = human_result.get("original_q8_correct_in_paired_items")
    cells = [human_result.get(k) for k in
             ("both_correct", "q8_only_correct", "q4_only_correct", "both_incorrect")]
    require(
        human_result.get("status") == "FULL_164_INDEPENDENT_RESEARCH_PAIRED_Q8_Q4_ONLY"
        and type(paired_n) is int and paired_n == 164
        and all(type(x) is int and x >= 0 for x in cells)
        and sum(cells) == 164
        and type(correct4) is int and type(correct8) is int
        and correct4 == cells[0] + cells[2]
        and correct8 == cells[0] + cells[1]
        and human_result.get("remaining_unscored_q8_original_cases") == 0
        and human_result.get("original_q4_model_sha256") == HUMAN_Q4_SHA
        and human_result.get("original_q8_model_sha256") == HUMAN_Q8_SHA
        and human_result.get("is_original_frozen_Q8_vs_dense_gate2") is False
        and human_result.get("is_amendment4_paired_mcnemar_ci") is False
        and human_result.get("is_confirmatory_quantization_comparison") is False
        and human_result.get("gen0_status") == "BLOCKED"
        and human_result.get("release_approved") is False,
        "original complete HumanEval source, population or gate guard violated",
    )
    n = automation_result.get("completed_scored_tasks")
    strict = automation_result.get("totals", {}).get("strict_passes")
    require(
        automation_result.get("status") == "PARTIAL_AUTHENTIC_OFFICIAL_TOOL_EVALUATOR_EXECUTION"
        and type(n) is int and 0 <= n < 600
        and type(strict) is int and 0 <= strict <= n
        and automation_result.get("registered_population") == 600
        and automation_result.get("active_unscored_indices") in ([], [n])
        and automation_result.get("official_complete_600_claim_allowed") is False
        and automation_result.get("official_source_commit") == AUTOMATION_ORIGINAL_COMMIT
        and automation_result.get("original_prompt_sha256") == AUTOMATION_PROMPT_SHA
        and automation_result.get("model_gguf_sha256") == HUMAN_Q4_SHA
        and automation_result.get("historical_git_commit") == auto.GIT_COMMIT
        and automation_result.get("historical_guard_policy") == "guard.v3"
        and automation_result.get("noncomparable_to_frozen_four_slot") is True
        and automation_result.get("frozen_gen0_status") == "BLOCKED"
        and automation_result.get("release_approved") is False,
        "original AutomationBench source, scoring or frozen gate violated",
    )
    return {
        "schema": "chowder.reap-v2-phase45-original-receipt-progress.v1",
        "source": "independent archived original item witnesses; no new inference",
        "human_eval": {
            "status": "COMPLETE_ORIGINAL_MODEL_RESEARCH_ONLY",
            "population": 164,
            "q4_correct": correct4,
            "q8_correct": correct8,
            "both_correct": cells[0],
            "q8_only_correct": cells[1],
            "q4_only_correct": cells[2],
            "neither_correct": cells[3],
            "q4_model_sha256": human_result["original_q4_model_sha256"],
            "q8_model_sha256": human_result["original_q8_model_sha256"],
            "q4_generated_rows_sha256": human_result["original_q4_generated_rows_sha256"],
            "q8_generated_rows_sha256": human_result["original_q8_generated_rows_sha256"],
            "q4_journal_sha256": human_result["original_q4_witnessed_journal_sha256"],
            "q8_journal_sha256": human_result["original_q8_witnessed_journal_sha256"],
            "q4_serving_session": human_result["original_q4_serving_session"],
            "q8_serving_session": human_result["original_q8_serving_session"],
            "frozen_gate2_eligible": False,
        },
        "automationbench": {
            "status": "SOURCE_SCORED_PARTIAL_NOT_COMPLETE_600",
            "scored": n,
            "registered_population": 600,
            "strict_passes": strict,
            "next_unscored": automation_result["active_unscored_indices"],
            "model_sha256": automation_result["model_gguf_sha256"],
            "original_source_commit": automation_result["official_source_commit"],
            "original_prompt_sha256": automation_result["original_prompt_sha256"],
            "historic_guard_source_git": automation_result["historical_git_commit"],
            "owner_sha256": automation_result["owner_sha256"],
            "original_scorer_source_sha256": automation_result["historical_git_runner_source_sha256"],
            "model_calls": automation_result["totals"]["model_calls"],
            "tool_calls": automation_result["totals"]["tool_calls"],
            "frozen_four_slot_eligible": False,
        },
        "eleven_frontier_original_suite_status": "INCOMPLETE",
        "gate0": "UNKNOWN",
        "gate1": "UNKNOWN",
        "gate2": "UNKNOWN",
        "random_control": "UNKNOWN_FROZEN_COMPARABLE",
        "throughput_gate": "UNKNOWN_PAIRED_DENSE_MISSING",
        "reap160_speed": "UNKNOWN",
        "amendment4": "BLOCKED_PENDING_PROSPECTIVE_CI_SPECIFICATION",
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }


def audit_actual() -> dict:
    return build(human.audit_actual(), auto.inspect())


def main() -> None:
    print(json.dumps(audit_actual(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
