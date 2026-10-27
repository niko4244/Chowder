"""CPU-only RED/GREEN checks: 164 original rows cannot bypass v2 admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38
from chowder_batch import reap_humaneval_v2_attestation as v2
from chowder_batch import reap_humaneval_independent_witness_20261027 as v37


def test_both_actual_original_complete_cohorts_are_distinct_and_sha_pinned():
    original_4 = v38.original_program_manifest("q4")
    original_8 = v38.original_program_manifest("q8")
    assert original_4["arm"] == "reap128_v2_q4"
    assert original_8["arm"] == "reap128_v2_q8"
    for result in (original_4, original_8):
        assert result["source_rows"] == 164
        assert result["canonical_raw_sha256"] == v38.CANONICAL_SHA256
        assert result["terminated_original_rows"] == 162
        assert result["unterminated_original_rows"] == 2
        assert result["v38_independently_executed_programs"] == 0
        assert result["v38_independent_verified_correct"] is None
        assert result["gen0_status"] == "BLOCKED"
        assert result["v2_approved_execution"] is False
        assert len(result["source_rows_in_original_order"]) == 164
        assert [row["id"] for row in result["source_rows_in_original_order"]] == [
            f"HumanEval/{i}" for i in range(164)]
        assert sum(row["candidate_program_sha256"] is not None for row in
                   result["source_rows_in_original_order"]) == 162
        assert sum(row["candidate_program_sha256"] is None for row in
                   result["source_rows_in_original_order"]) == 2
    assert original_4["original_recorded_research_correct_unverified"] == 151
    assert original_8["original_recorded_research_correct_unverified"] == 152
    assert original_4["original_rows_raw_sha256"] == v38.Q4_ROW_SHA256
    assert original_8["original_rows_raw_sha256"] == v38.Q8_ROW_SHA256
    assert original_4["ordered_program_manifest_sha256"] != original_8["ordered_program_manifest_sha256"]
    assert original_4["original_serving_session"] != original_8["original_serving_session"]
    assert original_4["original_model_sha256"] != original_8["original_model_sha256"]


def test_entire_original_program_identity_digest_reproducible_without_execution():
    for arm in ("q4", "q8"):
        record = v38.original_program_manifest(arm)
        source = json.dumps(record["source_rows_in_original_order"], sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        assert hashlib.sha256(source).hexdigest() == record["ordered_program_manifest_sha256"]
        assert all((row["candidate_program_sha256"] is None
                    or len(row["candidate_program_sha256"]) == 64)
                   for row in record["source_rows_in_original_order"])


def test_original_attestation_approval_pins_are_still_unset_and_not_inferred_from_v37():
    assert v2.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert v2.APPROVED_INDEPENDENT_WITNESS_SHA256 is None
    assert v37.EXECUTION_APPROVED is False
    gaps = v38.independent_execution_gaps()
    assert "APPROVED_SANDBOX_CAPABILITY_SHA256_REQUIRED" in gaps
    assert "APPROVED_INDEPENDENT_WITNESS_SHA256_REQUIRED" in gaps
    assert "ONLY_FIXED_BENIGN_CANARIES_VERIFIED_UNTRUSTED_GENERATED_CODE_NOT_ADMITTED" in gaps
    with pytest.raises(RuntimeError, match="REFUSED original generated HumanEval execution"):
        v38.require_independent_generated_code_admission()


@pytest.mark.parametrize("arm", ["q4", "q8"])
def test_actual_164_execution_attempt_fails_before_any_candidate_source_is_read(arm, monkeypatch):
    monkeypatch.setattr(v38, "original_program_manifest",
                        lambda *_: (_ for _ in ()).throw(AssertionError("candidate source read after refusal")))
    with pytest.raises(RuntimeError, match="APPROVED_SANDBOX_CAPABILITY_SHA256_REQUIRED"):
        v38.execute_original_164(arm)


def test_untrusted_caller_supplied_manifest_or_new_session_cannot_promote_generated_code():
    with pytest.raises(ValueError, match="unsupported"):
        v38.execute_original_164("q8-foreign")
    with pytest.raises(ValueError, match="forged"):
        v38.execute_original_164("q8", manifest=[{"self_signed": "PASS"}])
    with pytest.raises(RuntimeError):
        v38.execute_original_164("q8", manifest={"status": "PASS", "executed": 164})


@pytest.mark.parametrize("arm", ["q4", "q8"])
def test_bound_row_missing_or_modified_is_rejected_not_implicitly_zero(arm, tmp_path, monkeypatch):
    path = tmp_path / "original.jsonl"
    path.write_text('{"id":"HumanEval/0"}\n', encoding="utf-8")
    spec = v38.ARMS[arm]
    monkeypatch.setitem(v38.ARMS, arm, v38.OriginalArm(
        name=spec.name, original_model_sha256=spec.original_model_sha256,
        source=path, source_sha256=spec.source_sha256, session=spec.session,
        original_recorded_correct=spec.original_recorded_correct))
    with pytest.raises(ValueError, match="original HumanEval bytes differ"):
        v38.original_program_manifest(arm)


def test_missing_or_renamed_canonical_refuses_complete_score(tmp_path, monkeypatch):
    monkeypatch.setattr(v38, "CANONICAL", tmp_path / "missing-humaneval.json")
    with pytest.raises(ValueError, match="missing, linked"):
        v38.original_program_manifest("q4")


def test_original_v37_host_observer_sha_forgery_is_not_an_admission(monkeypatch):
    monkeypatch.setattr(v37, "VM_HOST_OBSERVER_SHA256", "0" * 64)
    assert "ORIGINAL_V37_EXTERNAL_HOST_EXECUTABLE_OR_SOURCE_PIN_MISMATCH" in v38.independent_execution_gaps()
    with pytest.raises(RuntimeError, match="HOST_EXECUTABLE_OR_SOURCE_PIN_MISMATCH"):
        v38.execute_original_164("q4")


def test_v37_fixed_benign_execution_approved_true_is_still_not_v38_admission(monkeypatch):
    monkeypatch.setattr(v37, "EXECUTION_APPROVED", True)
    assert "V37_FIXED_BENIGN_ONLY_GATE_CHANGED_NOT_A_REPLAY_APPROVAL" in v38.independent_execution_gaps()
    with pytest.raises(RuntimeError, match="REFUSED"):
        v38.execute_original_164("q8")


def test_forged_original_v2_pins_are_not_sufficient_to_bypass_new_executor(monkeypatch):
    monkeypatch.setattr(v2, "APPROVED_SANDBOX_CAPABILITY_SHA256", "f" * 64)
    monkeypatch.setattr(v2, "APPROVED_INDEPENDENT_WITNESS_SHA256", "e" * 64)
    assert not any("REQUIRED" in name for name in v38.independent_execution_gaps())
    with pytest.raises(RuntimeError, match="ONLY_FIXED_BENIGN_CANARIES_VERIFIED"):
        v38.execute_original_164("q8")


def test_malicious_duplicate_json_property_refused_before_loading_candidate(tmp_path, monkeypatch):
    data = b'{"id":"HumanEval/0","id":"HumanEval/0"}\n'
    fake = tmp_path / "fake.jsonl"
    fake.write_bytes(data)
    old = v38.ARMS["q4"]
    monkeypatch.setitem(v38.ARMS, "q4", v38.OriginalArm(
        name=old.name, original_model_sha256=old.original_model_sha256, source=fake,
        source_sha256=hashlib.sha256(data).hexdigest(), session=old.session,
        original_recorded_correct=old.original_recorded_correct))
    with pytest.raises(ValueError, match="duplicate JSON evidence key"):
        v38.original_program_manifest("q4")


def test_source_only_mode_reports_unknown_actual_score_not_model_zero(capsys):
    assert v38.main(["--arm", "both"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SOURCE_BOUND_ORIGINAL_164_EXECUTION_REFUSED_PENDING_INDEPENDENT_APPROVAL"
    assert payload["actual_scored_164"] is False
    assert all(row["v38_independent_verified_correct"] is None for row in payload["arms"])
    assert all("source_rows_in_original_order" not in row for row in payload["arms"])
    assert payload["gen0_status"] == "BLOCKED"


def test_execute_cli_refuses_without_calling_podman_or_model(monkeypatch, capsys):
    monkeypatch.setattr(v38, "original_program_manifest", lambda *_: (_ for _ in ()).throw(
        AssertionError("generated-code source must never be read after admission refusal")))
    assert v38.main(["--arm", "both", "--execute"]) == 2
    outcome = json.loads(capsys.readouterr().err)
    assert outcome["generated_programs_executed"] == 0
    assert outcome["independent_score"] is None
    assert outcome["status"] == "REFUSED_ACTUAL_ORIGINAL_HUMANEVAL_EXECUTION"


def test_only_owned_v38_source_approval_pins_and_no_legacy_pin_writes():
    source = Path(v38.__file__).read_text(encoding="utf-8")
    assert 'subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)' in source
    assert 'shell=True' not in source
    assert "APPROVED_SANDBOX_CAPABILITY_SHA256 =" not in source
    assert "APPROVED_INDEPENDENT_WITNESS_SHA256 =" not in source
    assert 'parser.add_argument("--force"' not in source


def test_research_v38_fixed_holdout_source_identity_requires_exact_no_model_code():
    assert v38._sha(v38.V38_FIXED_HOLDOUT.encode()) == v38.V38_FIXED_HOLDOUT_SHA256
    assert v38.V38_FIXED_HOLDOUT.startswith("print(")
    assert v38._research_source_gaps() == [
        "INDEPENDENT_V38_GENERATED_CODE_LOADER_SOURCE_REVIEW_REQUIRED",
        "INDEPENDENT_V38_OUTSIDE_HOST_OBSERVER_SOURCE_REVIEW_REQUIRED",
    ]
    with pytest.raises(ValueError, match="only exact fixed benign holdout"):
        v38._run_research_candidate("print('foreign model output')", fixed_holdout_only=True)


def test_independent_v38_research_source_review_still_fail_closed(monkeypatch):
    assert v38.APPROVED_V38_RESEARCH_LOADER_SHA256 is None
    assert v38.APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 is None
    assert v38.V37_IMAGE_ID == v37.IMAGE_ID
    assert v38.V37_PYTHON_EXECUTABLE_SHA256 == v37.PYTHON_SHA256
    assert v2.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert v2.APPROVED_INDEPENDENT_WITNESS_SHA256 is None
    with pytest.raises(RuntimeError, match="INDEPENDENT_V38_GENERATED_CODE_LOADER_SOURCE_REVIEW_REQUIRED"):
        v38.original_research_admission()


def test_benign_source_transfer_profile_has_exclusive_noegress_only_owned_podman():
    cmd = v38._new_research_container_argv("w10-he-v38-1234567890abcdef")
    assert cmd[:3] == ["podman", "run", "--detach"]
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--user=65534:65534" in cmd
    assert "--pids-limit=16" in cmd
    assert "--memory=268435456" in cmd
    assert "--tmpfs=/work:rw,nosuid,nodev,noexec,size=16m,mode=1777" in cmd
    assert "--tmpfs=/tmp:ro,nosuid,nodev,noexec,size=64k,mode=0555" in cmd
    assert not any(x.startswith("--volume") for x in cmd)
    assert "--pull=never" in cmd and "--privileged" not in cmd
    with pytest.raises(ValueError, match="exclusive"):
        v38._new_research_container_argv("hermes-brainz-stack_litellm_1")


def test_research_cli_does_not_open_or_mutate_private_output_when_not_approved(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(v38, "APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256", None)
    dest = tmp_path / "reap-v2-human164-v38-private-forbidden"
    assert v38.main(["--arm", "q8", "--research-execute", "--private-output", str(dest)]) == 2
    data = json.loads(capsys.readouterr().err)
    assert data["generated_programs_executed"] == 0
    assert data["independent_score"] is None
    assert not dest.exists()
