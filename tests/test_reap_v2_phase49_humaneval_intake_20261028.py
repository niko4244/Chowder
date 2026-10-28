"""CPU-only original Gate2 intake tests: all fixtures are SYNTHETIC and unapproved."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_v2_frozen_gate_evidence_bridge_20261027 as bridge
from chowder_batch import reap_v2_humaneval_certificate_verifier_20261027 as cert


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path):
    return {
        field: str(tmp_path / (field + ".fixture"))
        for field in bridge.CERTIFICATE_INPUT_FIELDS
    }


def _stage(tmp_path, monkeypatch):
    root = tmp_path / "fixtures"
    root.mkdir()
    serving = {}
    pins = {}
    cases = {}
    for arm in ("reap128_v2", "dense_v2"):
        arm_root = root / arm
        arm_root.mkdir()
        session = "a" * 32 if arm == "reap128_v2" else "b" * 32
        serving[arm] = {"session_id": session, "physical_host_sha256": "c" * 64,
                        "hardware_profile_sha256": "e" * 64}
        cases[arm] = {
            f"HumanEval/{i}": {"id": f"HumanEval/{i}", "correct": i != 7, "terminated": i != 7}
            for i in range(164)
        }
        path = root / (arm + ".frozen-humaneval-certificate-inputs.json")
        input_paths = _inputs(tmp_path)
        input_paths["serve_path"] = str(arm_root / "serve.json")
        raw = (json.dumps(input_paths, sort_keys=True) + "\n").encode()
        path.write_bytes(raw)
        pins[arm] = _sha(raw)
        (arm_root / "humaneval.jsonl").write_bytes(b"SYNTHETIC ORIGINAL ROW BYTE TEST ONLY\n")
    monkeypatch.setattr(bridge, "APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256", pins)
    monkeypatch.setattr(cert, "approval_gaps", lambda: [])
    monkeypatch.setattr(
        bridge.reap_select, "_attested_rows",
        lambda _, arm, suite, count: cases[arm],
    )
    certified_identity = {arm: dict(row) for arm, row in serving.items()}
    certified_row_hash = {arm: "f" * 64 for arm in serving}
    # This fixture explicitly imitates EXTERNALLY reviewed original evaluator
    # bytes. These are not approved production pins and never write real output.
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256", {
        str(root / arm / "humaneval.jsonl"):
        _sha((root / arm / "humaneval.jsonl").read_bytes()) for arm in serving
    })

    def mock_verified_original_trace(arm, **kwargs):
        assert set(kwargs) == bridge.CERTIFICATE_INPUT_FIELDS
        assert all(isinstance(p, Path) and p.is_absolute() for p in kwargs.values())
        return {
            "status": "INDEPENDENT_164_CERTIFICATE_RECOMPUTED_NOT_RELEASE",
            "model_sha256": bridge.MODEL_SHA256[arm],
            "session_id": certified_identity[arm]["session_id"],
            "physical_host_sha256": certified_identity[arm]["physical_host_sha256"],
            "population": 164,
            "canonical_source_sha256": cert.CANONICAL_RAW_SHA256,
            "frozen_source_items_sha256": bridge.reap_gates.FROZEN_SOURCE_SHA256["humaneval"],
            "selector_rows_sha256": certified_row_hash[arm],
            "chosen": None, "release_approved": False,
            "outcomes": [(f"HumanEval/{i}", i != 7) for i in range(164)],
            "termination_outcomes": [(f"HumanEval/{i}", i != 7) for i in range(164)],
        }

    monkeypatch.setattr(cert, "verify_arm", mock_verified_original_trace)
    return root, serving, cases


def test_original_production_trace_approval_maps_remain_empty():
    assert bridge.APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256 == {}
    assert cert.APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256 == {}


def test_unapproved_external_traces_never_enter_frozen_selector(tmp_path, monkeypatch):
    root, serving, _ = _stage(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "approval_gaps", lambda: ["INDEPENDENT_TRACE_REVIEW_ABSENT"])
    with pytest.raises(bridge.FrozenEvidenceError, match="outside-host"):
        bridge._verify_humaneval_certificates(root, serving)


def test_missing_independent_manifest_pins_fail_before_trace_read(tmp_path, monkeypatch):
    root, serving, _ = _stage(tmp_path, monkeypatch)
    monkeypatch.setattr(bridge, "APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256", {})
    with pytest.raises(bridge.FrozenEvidenceError, match="certificate input approvals absent"):
        bridge._verify_humaneval_certificates(root, serving)


def test_source_locked_trace_and_original_item_join_synthetic_only(tmp_path, monkeypatch):
    root, serving, _ = _stage(tmp_path, monkeypatch)
    assert bridge._verify_humaneval_certificates(root, serving) is None


@pytest.mark.parametrize("tamper", ["manifest", "correct", "terminated", "missing", "session", "host", "rows", "serve_path", "cross_host", "cross_profile", "same_session"])
def test_original_trace_intake_never_promotes_disagreed_certificate(
    tmp_path, monkeypatch, tamper
):
    root, serving, cases = _stage(tmp_path, monkeypatch)
    if tamper == "manifest":
        path = root / "reap128_v2.frozen-humaneval-certificate-inputs.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "correct":
        cases["reap128_v2"]["HumanEval/6"]["correct"] = False
    elif tamper == "terminated":
        cases["dense_v2"]["HumanEval/5"]["terminated"] = False
    elif tamper == "missing":
        del cases["dense_v2"]["HumanEval/2"]
    elif tamper == "session":
        serving["dense_v2"]["session_id"] = "wrong"
    elif tamper == "host":
        serving["reap128_v2"]["physical_host_sha256"] = "d" * 64
    elif tamper == "cross_host":
        serving["dense_v2"]["physical_host_sha256"] = "d" * 64
    elif tamper == "cross_profile":
        serving["dense_v2"]["hardware_profile_sha256"] = "d" * 64
    elif tamper == "same_session":
        serving["dense_v2"]["session_id"] = serving["reap128_v2"]["session_id"]
    elif tamper == "rows":
        path = root / "dense_v2" / "humaneval.jsonl"
        path.write_bytes(path.read_bytes() + b"FORGED NEW ROW BYTES\n")
    elif tamper == "serve_path":
        arm = "reap128_v2"
        path = root / (arm + ".frozen-humaneval-certificate-inputs.json")
        values = json.loads(path.read_bytes())
        values["serve_path"] = str(root / "wrong-original-serving.json")
        raw = (json.dumps(values, sort_keys=True) + "\n").encode()
        path.write_bytes(raw)
        monkeypatch.setitem(bridge.APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256, arm, _sha(raw))
    with pytest.raises(bridge.FrozenEvidenceError):
        bridge._verify_humaneval_certificates(root, serving)
