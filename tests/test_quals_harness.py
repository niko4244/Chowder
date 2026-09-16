"""The quals harness's rules are the load-bearing part.

Every future prereg judge inherits its epistemics from ``quals_harness``:
UNKNOWN for anything missing or unreadable, exit 0 only on all-PASS, and a
read-only registry. These tests pin those rules so a harness change cannot
silently loosen the discipline every judge shares.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

QUALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "quals"
sys.path.insert(0, str(QUALS_DIR))

import quals_harness as harness  # noqa: E402


# ---- Verdict finalization ---------------------------------------------------


def test_an_all_pass_verdict_qualifies():
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    verdict.add("T2", "other", harness.PASS, "fine")
    assert "QUALIFIED" in verdict.finalize_status()


def test_any_fail_refuses():
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    verdict.add("T2", "broken", harness.FAIL, "over budget")
    assert "REFUSED" in verdict.finalize_status()


def test_any_unknown_refuses_to_certify():
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    verdict.add("T2", "unmeasured", harness.UNKNOWN, "artifact missing")
    status = verdict.finalize_status()
    assert "NOT CERTIFIED" in status
    assert "REFUSED" not in status


def test_info_rows_are_recorded_but_not_gating():
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    verdict.add(harness.INFO, "drift note", harness.UNKNOWN, "recorded, not gating")
    assert "QUALIFIED" in verdict.finalize_status()
    assert verdict.thresholds() == [("T1", "check", harness.PASS, "fine")]


def test_an_invalid_status_is_refused_at_add_time():
    verdict = harness.Verdict()
    with pytest.raises(ValueError):
        verdict.add("T1", "check", "ok", "not a status")


# ---- Strict epistemics ------------------------------------------------------


def test_load_json_returns_none_for_missing_file(tmp_path):
    assert harness.load_json(tmp_path / "absent.json") is None


def test_load_json_returns_none_for_unreadable_file(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    assert harness.load_json(bad) is None


def test_load_json_parses_present_files(tmp_path):
    good = tmp_path / "ok.json"
    good.write_text('{"measured": true}', encoding="utf-8")
    assert harness.load_json(good) == {"measured": True}


def test_load_json_safe_handles_none_and_garbage():
    assert harness.load_json_safe(None) is None
    assert harness.load_json_safe("") is None
    assert harness.load_json_safe("{bad") is None
    assert harness.load_json_safe('{"a": 1}') == {"a": 1}


def test_finite_number_rejects_bools_and_non_finite():
    assert harness.finite_number(1.5)
    assert harness.finite_number(0)
    assert not harness.finite_number(True)
    assert not harness.finite_number(False)
    assert not harness.finite_number(float("nan"))
    assert not harness.finite_number(float("inf"))
    assert not harness.finite_number("0.5")
    assert not harness.finite_number(None)


def test_phase_of_missing_ledger_is_empty_dict():
    assert harness.phase({}, "model_load") == {}
    result = {"lifecycle": {"phases": {"model_load": {"seconds": 1.0}}}}
    assert harness.phase(result, "model_load") == {"seconds": 1.0}


def test_sha256_file_hashes_streamed_content(tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"chowder")
    assert harness.sha256_file(blob) == hashlib.sha256(b"chowder").hexdigest()


# ---- Read-only discovery ----------------------------------------------------


def test_discover_on_empty_root_returns_empty(tmp_path):
    registry, train, evals = harness.discover(tmp_path)
    assert registry == tmp_path / "runs.db"
    assert train == [] and evals == []


def test_discover_sorts_and_skips_unreadable(tmp_path):
    run_a = tmp_path / ".chowder" / "runs" / "run-a"
    run_b = tmp_path / ".chowder" / "runs" / "run-b"
    eval_d = tmp_path / ".chowder" / "evals" / "eval-a"
    for directory in (run_a, run_b, eval_d):
        directory.mkdir(parents=True)
    (run_b / "worker-result.json").write_text('{"kind": "train-b"}', encoding="utf-8")
    (run_a / "worker-result.json").write_text("{broken", encoding="utf-8")
    (eval_d / "worker-result.json").write_text('{"kind": "eval"}', encoding="utf-8")
    _, train, evals = harness.discover(tmp_path)
    assert [payload["kind"] for _, payload in train] == ["train-b"]  # broken skipped
    assert [payload["kind"] for _, payload in evals] == ["eval"]


def test_open_registry_readonly_refuses_writes(tmp_path):
    db = tmp_path / "runs.db"
    writer = sqlite3.connect(db)
    writer.execute("CREATE TABLE results (gpu_hours REAL)")
    writer.execute("INSERT INTO results VALUES (0.5)")
    writer.commit()
    writer.close()

    ro = harness.open_registry_readonly(db)
    assert ro is not None
    assert ro.execute("SELECT gpu_hours FROM results").fetchall() == [(0.5,)]
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO results VALUES (0.9)")  # read-only URI refuses
    ro.close()


def test_open_registry_readonly_returns_none_when_absent(tmp_path):
    assert harness.open_registry_readonly(tmp_path / "absent.db") is None


# ---- Exit discipline --------------------------------------------------------


def _report(tmp_path, verdict):
    return harness.report(tmp_path, verdict, [], [], argv_len_ok=True)


def test_report_exit_zero_only_when_all_pass(tmp_path):
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    assert _report(tmp_path, verdict) == 0

    verdict.add("T2", "other", harness.FAIL, "nope")
    assert _report(tmp_path, verdict) == 1

    only_unknown = harness.Verdict()
    only_unknown.add("T1", "check", harness.UNKNOWN, "absent")
    assert _report(tmp_path, only_unknown) == 1


def test_report_bad_usage_returns_two(tmp_path):
    assert harness.report(tmp_path, harness.Verdict(), [], [], argv_len_ok=False) == 2


def test_report_missing_root_returns_two(tmp_path):
    verdict = harness.Verdict()
    verdict.add("T1", "check", harness.PASS, "fine")
    assert harness.report(tmp_path / "absent", verdict, [], [], argv_len_ok=True) == 2
