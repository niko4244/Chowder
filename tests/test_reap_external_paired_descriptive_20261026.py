"""No-GPU adversarial tests of source-bound external Q8/Q4/control comparisons."""

import copy
import hashlib
import json

import pytest

from chowder_batch.reap_external_paired_descriptive import compare, read_verified_cohort
from chowder_batch.reap_external_full_audit import Q4_SHA, Q8_SHA, RANDOM_SHA


def _side(model, session, row_correct, suite="gsm8k"):
    ids = ("gsm8k-0", "gsm8k-1", "gsm8k-2", "gsm8k-3") if suite == "gsm8k" else (
        "HumanEval/0", "HumanEval/1", "HumanEval/2", "HumanEval/3")
    rows = [{"id": ident, "correct": answer, "terminated": True,
             "eval_session_id": session} for ident, answer in zip(ids, row_correct)]
    row_bytes = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    row_sha = hashlib.sha256(row_bytes).hexdigest()
    audit = {"schema": "chowder.reap-v2-original-population-external-independent-audit.v1",
             "status": "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE",
             "original_windows_gate_eligible": False, "speed_ratio_certified": False,
             "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
             "original_machine_receipt_sha256": "e" * 64,
             "model": model, "model_sha256": {
                 "reap128_v2_q4": Q4_SHA, "reap128_v2_q8": Q8_SHA,
                 "random128_v2_q8": RANDOM_SHA}[model],
             "session_id": session,
             "original_native_cuda_placement": {"native_gpu_offloaded_layers": "41/41",
                                                "n_seq_max": 4, "n_ctx": 16384,
                                                "n_ctx_seq": 4096,
                                                "total_kv_buffer_mib": 320.0,
                                                "loopback_bound": True},
             "suites": {suite: {"n": len(rows), "correct": sum(row_correct),
                                 "terminated": len(rows), "row_sha256": row_sha}}}
    if model != "reap128_v2_q4":
        audit["original_measured_postload_resource_witness"] = {
            "status": "COMPLETE_LINUX_SOURCE_MEASURED_POSTLOAD_RESOURCE_WITNESS",
            "initial_actual_gpu_free_mib": [14912, 14912],
            "measured_postload_recovery_gib": 17.93,
            "measured_file_cache_decrease_gib": 17.889,
            "minimum_measured_cgroup_unused_gib": 10.471,
            "minimum_measured_physical_available_gib": 29.8,
            "monitored_sample_count": 4,
            "monitored_runtime_sample_count": 2,
            "minimum_observed_gpu_free_mib": [4700, 4800],
            "largest_sampled_drop_from_start_gpu_free_mib": [10212, 10112],
            "gpu_peak_instantaneous_claimed": False,
            "original_windows_commit_attested": False,
            "model_fit_windows_gate_eligible": False,
        }
    else:
        audit["original_measured_postload_resource_witness"] = None
    if suite == "humaneval":
        audit["human_eval_independent_execution"] = (
            "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:" + "a" * 64
        )
    return audit, row_bytes


def test_descriptive_full_paired_offdiagonal_counts_and_no_gate():
    q4, q4_bytes = _side("reap128_v2_q4", "a" * 32, [True, True, False, False])
    q8, q8_bytes = _side("reap128_v2_q8", "b" * 32, [True, False, True, False])
    result = compare("gsm8k", q8, q8_bytes, q4, q4_bytes, expected_n=4)
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["candidate_only_correct"] == 1
    assert result["reference_only_correct"] == 1
    assert result["delta_accuracy"] == 0
    assert result["paired_confidence_interval"] is None
    assert result["amendment4_eligibility"] is False


def test_independent_frozen_negative_control_can_be_compared_descriptively():
    control, raw = _side("random128_v2_q8", "c" * 32, [False, False, True, False])
    q4, q4_raw = _side("reap128_v2_q4", "d" * 32, [True, True, True, False])
    result = compare("gsm8k", control, raw, q4, q4_raw, expected_n=4)
    assert result["candidate_only_correct"] == 0
    assert result["reference_only_correct"] == 2
    assert result["delta_accuracy"] == -0.5


@pytest.mark.parametrize("mutation", [
    lambda a, rows: a.update(status="REFUSED_OR_ABORTED"),
    lambda a, rows: a.update(original_windows_gate_eligible=True),
    lambda a, rows: a.update(gen0_status="SELECTED"),
    lambda a, rows: a.update(release_approved=True),
    lambda a, rows: a.update(speed_ratio_certified=True),
    lambda a, rows: a.update(model_sha256="0" * 64),
    lambda a, rows: a.update(session_id="f" * 32),
    lambda a, rows: a["original_native_cuda_placement"].update(native_gpu_offloaded_layers="40/41"),
    lambda a, rows: a["suites"]["gsm8k"].update(row_sha256="0" * 64),
    lambda a, rows: a["suites"]["gsm8k"].update(correct=4),
    lambda a, rows: rows.pop(),
    lambda a, rows: rows.append(copy.deepcopy(rows[0])),
    lambda a, rows: rows[0].update(eval_session_id="f" * 32),
    lambda a, rows: rows[0].update(correct=False, terminated=False),
])
def test_unauthenticated_reused_aborted_or_edited_source_rows_rejected(mutation):
    audit, raw = _side("reap128_v2_q8", "b" * 32, [True, False, True, False])
    rows = [json.loads(line) for line in raw.splitlines()]
    mutation(audit, rows)
    # Even if an attacker recomputes the row hash, the immutable prior audit
    # identity/summary is independently validated first.
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    with pytest.raises(ValueError):
        read_verified_cohort("gsm8k", audit, raw, expected_n=4)


def test_missing_item_and_self_comparison_fail_closed():
    audit, raw = _side("reap128_v2_q8", "b" * 32, [True, False, True, False])
    q4, q4_raw = _side("reap128_v2_q4", "c" * 32, [True, False, True, False])
    with pytest.raises(ValueError):
        compare("gsm8k", audit, raw, audit, raw, expected_n=4)
    q4_rows = [json.loads(row) for row in q4_raw.splitlines()]
    q4_rows[0]["id"] = "other"
    poisoned = ("\n".join(json.dumps(row) for row in q4_rows) + "\n").encode()
    with pytest.raises(ValueError):
        compare("gsm8k", audit, raw, q4, poisoned, expected_n=4)


@pytest.mark.parametrize("unverified", [
    None, "", "PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION",
    "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:",
    "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:" + "0" * 63,
    "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:" + "z" * 64,
    "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:" + "0" * 65,
])
def test_humaneval_pair_refuses_missing_or_forged_independent_execution(unverified):
    q8, q8_bytes = _side("reap128_v2_q8", "b" * 32, [True, False, True, False],
                         suite="humaneval")
    q4, q4_bytes = _side("reap128_v2_q4", "c" * 32, [True, True, False, False],
                         suite="humaneval")
    if unverified is None:
        q8.pop("human_eval_independent_execution")
    else:
        q8["human_eval_independent_execution"] = unverified
    with pytest.raises(ValueError, match="HumanEval"):
        compare("humaneval", q8, q8_bytes, q4, q4_bytes, expected_n=4)


def test_humaneval_pair_admitted_only_after_both_independent_replays():
    q8, q8_bytes = _side("reap128_v2_q8", "b" * 32, [True, False, True, False],
                         suite="humaneval")
    q4, q4_bytes = _side("reap128_v2_q4", "c" * 32, [True, True, False, False],
                         suite="humaneval")
    result = compare("humaneval", q8, q8_bytes, q4, q4_bytes, expected_n=4)
    assert result["candidate_only_correct"] == 1
    assert result["reference_only_correct"] == 1
    assert result["amendment4_eligibility"] is False


@pytest.mark.parametrize("mutation", [
    lambda a: a.pop("original_measured_postload_resource_witness"),
    lambda a: a.update(original_measured_postload_resource_witness=None),
    lambda a: a["original_measured_postload_resource_witness"].update(
        status="MAY_ATTEMPT_PROVISIONAL_SMOKE_ONLY"),
    lambda a: a["original_measured_postload_resource_witness"].update(
        original_windows_commit_attested=True),
    lambda a: a["original_measured_postload_resource_witness"].update(
        model_fit_windows_gate_eligible=True),
    lambda a: a["original_measured_postload_resource_witness"].update(
        minimum_measured_cgroup_unused_gib=9.99),
    lambda a: a["original_measured_postload_resource_witness"].update(
        measured_postload_recovery_gib=3.99),
    lambda a: a["original_measured_postload_resource_witness"].update(
        measured_file_cache_decrease_gib=3.99),
    lambda a: a["original_measured_postload_resource_witness"].update(
        monitored_sample_count=0),
    lambda a: a["original_measured_postload_resource_witness"].update(
        monitored_runtime_sample_count=0),
    lambda a: a["original_measured_postload_resource_witness"].update(
        minimum_observed_gpu_free_mib=[4700]),
    lambda a: a["original_measured_postload_resource_witness"].update(
        minimum_observed_gpu_free_mib=[4700, -1]),
    lambda a: a["original_measured_postload_resource_witness"].update(
        largest_sampled_drop_from_start_gpu_free_mib=[10212, -1]),
    lambda a: a["original_measured_postload_resource_witness"].update(
        gpu_peak_instantaneous_claimed=True),
])
def test_paired_q8_or_control_refuses_missing_or_forged_resource_admission(mutation):
    q8, raw = _side("reap128_v2_q8", "b" * 32, [True, False, True, False])
    mutation(q8)
    with pytest.raises(ValueError, match="resource"):
        read_verified_cohort("gsm8k", q8, raw, expected_n=4)
