"""CPU-only, no-execution adversarial tests for private Q8/control replay admission."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_humaneval_private_preflight as entry
from chowder_batch.reap_external_full_audit import MODELS, Q8_SHA, RANDOM_SHA


def _fixture(monkeypatch, arm="reap128_v2_q8"):
    canonical = [
        {"id": f"HumanEval/{i}", "header": f"def test_{i}():\n",
         "test": "def check(f): pass\n", "entry_point": f"test_{i}"}
        for i in range(164)
    ]
    canonical_bytes = json.dumps(canonical, sort_keys=True).encode()
    monkeypatch.setattr(entry, "CANONICAL_HUMANEVAL_RAW_SHA",
                        hashlib.sha256(canonical_bytes).hexdigest())
    session = "a" * 32
    rows = [
        {"id": item["id"], "text": "a private generated program",
         "correct": i % 2 == 0, "terminated": True, "eval_session_id": session}
        for i, item in enumerate(canonical)
    ]
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    machine = {
        "schema": MODELS[arm][0],
        "status": MODELS[arm][1],
        "session_id": session,
        "data": {MODELS[arm][2]: MODELS[arm][3]},
        "suite_results": {
            "gsm8k": {"n": 1319, "session_id": session, "file_sha256": "c" * 64},
            "humaneval": {"n": 164, "session_id": session,
                          "file_sha256": hashlib.sha256(raw).hexdigest()},
        },
    }
    machine_bytes = json.dumps(machine, sort_keys=True).encode()
    witness = {"status": "COMPLETE_LINUX_SOURCE_MEASURED_POSTLOAD_RESOURCE_WITNESS",
               "minimum_measured_cgroup_unused_gib": 10.5}
    monkeypatch.setattr(entry, "_check_machine", lambda receipt, selected: None)
    monkeypatch.setattr(entry, "_check_postload_resource_evidence",
                        lambda receipt: witness)
    audit = {
        "schema": "chowder.reap-v2-original-population-external-independent-audit.v1",
        "status": "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE",
        "model": arm, "model_sha256": MODELS[arm][3],
        "session_id": session,
        "original_machine_receipt_sha256": hashlib.sha256(machine_bytes).hexdigest(),
        "original_measured_postload_resource_witness": witness,
        "original_native_cuda_placement": {
            "native_gpu_offloaded_layers": "41/41", "n_seq_max": 4,
            "n_ctx": 16384, "n_ctx_seq": 4096, "total_kv_buffer_mib": 320.0,
            "loopback_bound": True,
        },
        "human_eval_independent_execution": "PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION",
        "suites": {
            "gsm8k": {"n": 1319, "row_sha256": "c" * 64},
            "humaneval": {"n": 164, "row_sha256": hashlib.sha256(raw).hexdigest(),
                          "correct": 82, "terminated": 164},
        },
        "original_windows_gate_eligible": False,
        "speed_ratio_certified": False, "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }
    params = {
        "arm": arm, "machine": machine,
        "machine_sha": hashlib.sha256(machine_bytes).hexdigest(),
        "trusted_audit": audit, "audit_sha": "d" * 64,
        "canonical_raw": canonical_bytes, "rows_raw": raw,
        "evaluator_sha": entry.EVALUATOR_SHA, "scorer_sha": entry.SCORER_SHA,
    }
    return params


@pytest.mark.parametrize("arm,sha", [
    ("reap128_v2_q8", Q8_SHA),
    ("random128_v2_q8", RANDOM_SHA),
])
def test_only_original_q8_and_negative_control_are_admitted_without_execution(
        monkeypatch, arm, sha):
    params = _fixture(monkeypatch, arm)
    result = entry.preflight(**params)
    assert result["status"] == "PREFLIGHT_PRIVATE_164_SOURCE_BOUND_SANDBOX_REQUIRED"
    assert result["arm"] == arm and result["model_sha256"] == sha
    assert result["source_rows"] == 164
    assert result["reported_correct"] == 82
    assert result["candidate_code_executed"] is False
    assert result["sandbox_verified"] is False
    assert result["release_approved"] is False
    assert "a private generated program" not in json.dumps(result)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(arm="reap128_v2_q4"),
    lambda p: p["machine"].update(session_id="b" * 32),
    lambda p: p["trusted_audit"].update(session_id="b" * 32),
    lambda p: p["trusted_audit"].update(model_sha256="0" * 64),
    lambda p: p["trusted_audit"].update(original_machine_receipt_sha256="0" * 64),
    lambda p: p["trusted_audit"].update(human_eval_independent_execution="DONE"),
    lambda p: p["trusted_audit"].update(original_windows_gate_eligible=True),
    lambda p: p["trusted_audit"].update(release_approved=True),
    lambda p: p["trusted_audit"]["original_native_cuda_placement"].update(n_ctx=8192),
    lambda p: p["trusted_audit"].update(original_measured_postload_resource_witness=None),
    lambda p: p["trusted_audit"]["suites"]["humaneval"].update(correct=83),
    lambda p: p["machine"]["suite_results"]["humaneval"].update(n=163),
    lambda p: p.update(scorer_sha="0" * 64),
    lambda p: p.update(canonical_raw=p["canonical_raw"] + b" "),
    lambda p: p.update(rows_raw=p["rows_raw"] + b" "),
    lambda p: p.update(rows_raw=p["rows_raw"][:-1]),
    lambda p: p.update(rows_raw=p["rows_raw"].replace(b"a private generated program",
                                                      b"altered private output", 1)),
    lambda p: p.update(rows_raw=p["rows_raw"].replace(b'"terminated": true',
                                                      b'"terminated": false', 1)),
])
def test_untrusted_or_incomplete_164_code_receipts_refused_before_any_execution(
        monkeypatch, mutate):
    params = _fixture(monkeypatch)
    mutate(params)
    with pytest.raises(ValueError):
        entry.preflight(**params)


def test_preflight_calls_original_source_bound_native_machine_verification(monkeypatch):
    params = _fixture(monkeypatch)
    def refuse(*args):
        raise ValueError("original native model/cgroup proof invalid")
    monkeypatch.setattr(entry, "_check_machine", refuse)
    with pytest.raises(ValueError, match="original native"):
        entry.preflight(**params)


def test_untrusted_file_paths_and_symlinks_refused_without_touching_the_source(tmp_path):
    protected = tmp_path / "source.jsonl"
    protected.write_text('{"secret":"never execute"}\n', encoding="utf8")
    link = tmp_path / "linked.jsonl"
    try:
        link.symlink_to(protected)
    except (NotImplementedError, OSError):
        pytest.skip("this filesystem does not permit symlinks")
    with pytest.raises(ValueError):
        entry._read_file(link, 1024)
    with pytest.raises(ValueError):
        entry._read_file(Path("relative-source.jsonl"), 1024)
    assert protected.read_text("utf8") == '{"secret":"never execute"}\n'


def test_preflight_cli_refuses_to_create_receipt_from_missing_complete_sources(tmp_path):
    target = tmp_path / "new-private-preflight.json"
    args = [
        "--arm", "reap128_v2_q8",
        "--machine", str(tmp_path / "missing-native.json"),
        "--trusted-full-audit", str(tmp_path / "missing-independent-audit.json"),
        "--humaneval", str(tmp_path / "missing-164.jsonl"),
        "--canonical-humaneval", str(tmp_path / "missing-original.json"),
        "--original-evaluator", str(tmp_path / "missing-reap-eval.py"),
        "--original-scorer", str(tmp_path / "missing-original-scorer.py"),
        "--out", str(target),
    ]
    assert entry.main(args) == 2
    assert not target.exists()


def test_original_private_source_pin_constants_and_no_code_execution_imports():
    assert entry.CANONICAL_HUMANEVAL_RAW_SHA == (
        "5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5"
    )
    import ast
    source = Path(entry.__file__).read_text(encoding="utf8")
    ast.parse(source)
    assert "import subprocess" not in source
    assert "import resource" not in source
    assert "subprocess.run(" not in source
    assert "candidate_code_executed\": False" in source
