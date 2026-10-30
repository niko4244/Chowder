"""Adversarial independent FULL research scorer and immutable certificate join."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_humaneval_v38_full_164_completion_20261027 as a


def source():
    root = a.ROOTS["q4"]
    audited = a.independent.audit_live("q4", root / "q4-actual-independent-per-case.private.jsonl")
    assert audited["verified_contiguous_rows"] == 164
    raw = (root / "q4-complete.private.json").read_bytes()
    return audited, raw


def encode(obj):
    return (json.dumps(obj, sort_keys=True) + "\n").encode()


def test_actual_q4_full164_original_native_score_independently_admitted():
    if not a.ROOTS["q4"].is_dir():
        pytest.skip("private original Q4 research evidence unavailable")
    result = a.certify("q4")
    assert result["status"] == "INDEPENDENTLY_VERIFIED_COMPLETE_164_RESEARCH_ONLY"
    assert result["scored_original_rows"] == 164
    assert result["actual_independently_witnessed_executions"] == 162
    assert result["original_unterminated_nonpasses"] == 2
    assert result["independent_correct"] == 151
    assert result["independent_accuracy"] == 151 / 164
    assert len(result["incorrect_case_ids"]) == 13
    assert result["original_frozen_v2_gate2_eligible"] is False
    assert result["dense_paired_gate2_complete"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("field,bad", [
    ("independently_research_scored_correct", 152),
    ("executed_rows", 164),
    ("original_rows", 163),
    ("per_case_evidence_sha256", "0" * 64),
    ("original_model_sha256", "0" * 64),
    ("candidate_identity_manifest_sha256", "1" * 64),
    ("canonical_sha256", "2" * 64),
    ("original_v2_attestation_approved", True),
    ("windows_native_gate_eligible", True),
    ("gen0_status", "PASS"),
    ("chosen", "reap128_v2_q4"),
    ("release_approved", True),
    ("status", "RELEASE_APPROVED"),
])
def test_forged_complete_source_or_frozen_claim_fail_closed(field, bad):
    audited, raw = source()
    forged = json.loads(raw)
    forged[field] = bad
    with pytest.raises(ValueError, match="certificate mismatch"):
        a.validate_completion_json("q4", encode(forged), audited, row_count=164)


def test_fake_reported_model_score_nonfinite_or_rounded_refused():
    audited, raw = source()
    fake = json.loads(raw)
    for score in [0.95, 1, "0.92", float("nan"), float("inf")]:
        fake["independent_score"] = score
        with pytest.raises(ValueError):
            a.validate_completion_json("q4", encode(fake), audited, row_count=164)


def test_partial_population_never_claims_full_score():
    audited, raw = source()
    with pytest.raises(ValueError, match="not a fully"):
        a.validate_completion_json("q4", raw, audited, row_count=163)
    forged = dict(audited)
    forged["verified_contiguous_rows"] = 163
    with pytest.raises(ValueError, match="not a fully"):
        a.validate_completion_json("q4", raw, forged, row_count=164)


def test_duplicate_fields_and_extra_self_attestation_refused():
    audited, raw = source()
    forged = raw.rstrip(b"\n").replace(
        b'"release_approved": false', b'"release_approved": false, "release_approved": true')
    with pytest.raises(ValueError, match="duplicate"):
        a.validate_completion_json("q4", forged, audited, row_count=164)
    modified = json.loads(raw)
    modified["official_score"] = "1"
    with pytest.raises(ValueError, match="fields"):
        a.validate_completion_json("q4", encode(modified), audited, row_count=164)


def test_wrong_arm_or_foreign_root_refused(tmp_path, monkeypatch):
    audited, raw = source()
    with pytest.raises(ValueError, match="wrong arm"):
        a.validate_completion_json("unknown", raw, audited, row_count=164)
    monkeypatch.setitem(a.ROOTS, "q4", Path(tmp_path / "absent"))
    with pytest.raises(ValueError, match="physical model research root"):
        a.certify("q4")


def test_incomplete_q8_is_not_approved_as_completed_original_suite():
    if not a.ROOTS["q8"].is_dir():
        pytest.skip("Q8 execution root unavailable")
    if (a.ROOTS["q8"] / "q8-complete.private.json").exists():
        pytest.skip("Q8 reached 164 while test was running")
    with pytest.raises(ValueError, match="no authentic final"):
        a.certify("q8")
