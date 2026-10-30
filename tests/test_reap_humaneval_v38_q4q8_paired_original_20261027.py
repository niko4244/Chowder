"""Genuine original Q4/Q8 research pairing and fail-closed adversarial cases."""

from copy import deepcopy

import pytest

from chowder_batch import reap_humaneval_v38_q4q8_paired_original_20261027 as pairing


def pretend(arm, count, flags):
    rows = [{"index": i, "id": f"HumanEval/{i}",
             "program_sha256": (("a" if arm == "q4" else "b") * 63 + format(i % 16, "x")),
             "research_correct": flag}
            for i, flag in enumerate(flags)]
    assert count == len(rows)
    return {"arm": arm, "rows": rows,
            "verified_contiguous_rows": count,
            "verified_actual_executions": count,
            "verified_source_unterminated_nonpasses": 0,
            "verified_independent_correct_prefix": sum(flags),
            "full_164_independent_correct": sum(flags) if count == 164 else None,
            "unknown_remaining_rows": 164 - count,
            "full_164_denominator": 164 if count == 164 else None,
            "status": ("COMPLETE_164_RESEARCH_ONLY" if count == 164 else "VERIFIED_PREFIX_ONLY"),
            "original_model_sha256": ("1" if arm == "q4" else "2") * 64,
            "original_rows_sha256": ("3" if arm == "q4" else "4") * 64,
            "ordered_program_manifest_sha256": ("5" if arm == "q4" else "6") * 64,
            "journal_snapshot_sha256": ("7" if arm == "q4" else "8") * 64,
            "original_session": arm + "-original",
            "gen0_status": "BLOCKED"}


def cohort():
    flags4 = [True] * 164
    flags4[2] = False
    flags4[4] = False
    a = pretend("q4", 164, flags4)
    b = pretend("q8", 5, [True, False, True, False, True])
    return a, b


def test_synthetic_derived_pairs_verify_direction_without_frozen_gate():
    a, b = cohort()
    z = pairing.verified_pair(a, b, q8_full_certified=False)
    assert z["paired_original_case_count"] == 5
    assert (z["both_correct"], z["q8_only_correct"],
            z["q4_only_correct"], z["both_incorrect"]) == (1, 2, 2, 0)
    assert z["observed_difference_q8_minus_q4_on_verified_original_prefix"] == 0
    assert z["remaining_unscored_q8_original_cases"] == 159
    assert z["status"] == "PARTIAL_PREFIX_SOURCE_BOUND_NOT_A_FULL_BENCHMARK"
    assert z["is_original_frozen_Q8_vs_dense_gate2"] is False
    assert z["is_amendment4_paired_mcnemar_ci"] is False
    assert z["gen0_status"] == "BLOCKED"


def test_real_original_Q4_Q8_generated_programs_paired_by_independent_witness():
    result = pairing.audit_actual()
    assert result["paired_original_case_count"] >= 5
    assert result["original_q4_model_sha256"] != result["original_q8_model_sha256"]
    assert result["original_q4_generated_rows_sha256"] != result["original_q8_generated_rows_sha256"]
    assert result["both_correct"] + result["q8_only_correct"] + result["q4_only_correct"] + result["both_incorrect"] == result["paired_original_case_count"]
    assert result["release_approved"] is False
    assert result["is_confirmatory_quantization_comparison"] is False


@pytest.mark.parametrize("broken", [
    "shuffled", "wrong_id", "wrong_index", "bad_source", "false_frozen",
    "fake_score", "same_model", "bad_boolean", "extra_unverified"])
def test_cross_arm_fails_closed_on_corrupted_identity_or_score(broken):
    a, b = cohort()
    if broken == "shuffled":
        b["rows"][0], b["rows"][1] = b["rows"][1], b["rows"][0]
    elif broken == "wrong_id":
        b["rows"][2]["id"] = "HumanEval/800"
    elif broken == "wrong_index":
        b["rows"][3]["index"] = 0
    elif broken == "bad_source":
        b["rows"][0]["program_sha256"] = ""
    elif broken == "false_frozen":
        b["gen0_status"] = "APPROVED"
    elif broken == "fake_score":
        b["verified_independent_correct_prefix"] = 5
    elif broken == "same_model":
        b["original_model_sha256"] = a["original_model_sha256"]
    elif broken == "bad_boolean":
        b["rows"][0]["research_correct"] = 1
    elif broken == "extra_unverified":
        b["rows"].append(deepcopy(b["rows"][-1]))
    with pytest.raises(ValueError):
        pairing.verified_pair(a, b, q8_full_certified=False)


def test_false_full_population_without_Q8_final_certificate_refused():
    a, b = cohort()
    b = pretend("q8", 164, [True] * 164)
    with pytest.raises(ValueError, match="lacks independent"):
        pairing.verified_pair(a, b, q8_full_certified=False)
    z = pairing.verified_pair(a, b, q8_full_certified=True)
    assert z["paired_original_case_count"] == 164
    assert z["status"] == "FULL_164_INDEPENDENT_RESEARCH_PAIRED_Q8_Q4_ONLY"
    assert z["is_original_frozen_Q8_vs_dense_gate2"] is False


def test_cannot_promote_partial_Q8_to_completed_before_original164():
    a, b = cohort()
    with pytest.raises(ValueError, match="partial Q8"):
        pairing.verified_pair(a, b, q8_full_certified=True)


def test_actual_unterminated_original_program_is_nonpass_not_missing_or_reexecuted():
    a, _ = cohort()
    # Canonical HumanEval/129 was genuinely not terminated by the Q4 model:
    # the independent candidate manifest correctly has NO executable SHA.
    a["rows"][129]["program_sha256"] = None
    a["rows"][129]["research_correct"] = False
    a["verified_independent_correct_prefix"] -= 1
    a["full_164_independent_correct"] -= 1
    a["verified_actual_executions"] -= 1
    a["verified_source_unterminated_nonpasses"] += 1
    b = pretend("q8", 130, [True] * 130)
    result = pairing.verified_pair(a, b, q8_full_certified=False)
    assert result["paired_original_case_count"] == 130
    assert result["q8_only_correct"] >= 1
    source_pair = result["per_item_original_source_bound_pairs"][129]
    assert source_pair["q4_original_program_sha256"] is None
    assert source_pair["q4_independent_correct"] is False


def test_forged_pass_with_no_generated_program_is_rejected():
    a, b = cohort()
    a["rows"][0]["program_sha256"] = None
    with pytest.raises(ValueError, match="census mismatch|exact canonical"):
        pairing.verified_pair(a, b, q8_full_certified=False)
