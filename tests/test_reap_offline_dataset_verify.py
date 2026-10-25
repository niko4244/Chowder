"""Adversarial CPU-only checks; actual cached datasets are separately read live."""

import json

import pytest

from chowder_batch import reap_offline_dataset_verify as verify


def _fixture(suite):
    if suite == "gsm8k":
        return [{"id": f"gsm8k-{i}", "prompt": f"Question {i}", "expected": "42"}
                for i in range(1319)]
    return [{"id": f"HumanEval/{i}", "prompt": f"Complete Python {i}",
             "header": "def solve(): pass", "test": "def check(fn): assert fn() == 42",
             "entry_point": "solve"} for i in range(164)]


def _frozen_synthetic_pins(monkeypatch):
    for suite in ("gsm8k", "humaneval"):
        monkeypatch.setitem(verify.reap_gates.FROZEN_SOURCE_SHA256, suite,
                            verify.reap_eval._digest(_fixture(suite)))


def test_full_frozen_offline_inputs_have_only_preflight_status(monkeypatch):
    _frozen_synthetic_pins(monkeypatch)
    result = verify.verify(loader=_fixture)
    assert result["status"] == verify.STATUS
    assert result["quality_gate"] == "NOT_EVALUATED"
    assert result["release_approved"] is False
    assert result["model_inference_performed"] is False
    assert result["cohorts"]["gsm8k"]["n"] == 1319
    assert result["cohorts"]["humaneval"]["n"] == 164


@pytest.mark.parametrize("fault", ["short_gsm", "swapped_ids", "changed_answer",
                                     "changed_test", "missing_test", "duplicate_id"])
def test_resealed_or_incomplete_offline_source_never_qualifies(monkeypatch, fault):
    _frozen_synthetic_pins(monkeypatch)
    copies = {suite: _fixture(suite) for suite in ("gsm8k", "humaneval")}
    if fault == "short_gsm":
        copies["gsm8k"].pop()
    elif fault == "swapped_ids":
        copies["humaneval"][0], copies["humaneval"][1] = (
            copies["humaneval"][1], copies["humaneval"][0])
    elif fault == "changed_answer":
        copies["gsm8k"][0]["expected"] = "another answer"
    elif fault == "changed_test":
        copies["humaneval"][0]["test"] += "\nassert False"
    elif fault == "missing_test":
        copies["humaneval"][0].pop("test")
    else:
        copies["gsm8k"][0]["id"] = copies["gsm8k"][1]["id"]
    with pytest.raises(ValueError, match="population|IDs|prompts|test|SHA256"):
        verify.verify(loader=copies.__getitem__)


def test_publication_is_complete_and_exclusive(tmp_path):
    dest = tmp_path / "benchmark-source.json"
    evidence = {"schema": verify.SCHEMA, "status": verify.STATUS}
    verify.publish_exclusive(dest, evidence)
    assert json.loads(dest.read_bytes()) == evidence
    prior = dest.read_bytes()
    with pytest.raises(FileExistsError):
        verify.publish_exclusive(dest, {"status": "FAKE_PASS"})
    assert dest.read_bytes() == prior
    assert not list(tmp_path.glob("*.pending"))


def test_interrupted_pending_write_never_becomes_authoritative(tmp_path, monkeypatch):
    dest = tmp_path / "benchmark-source.json"
    real_fsync = verify.os.fsync

    def stop_before_publish(fd):
        pending = list(tmp_path.glob(".*.pending"))
        assert len(pending) == 1
        assert not dest.exists()
        assert json.loads(pending[0].read_text(encoding="utf-8"))["status"] == verify.STATUS
        real_fsync(fd)
        raise OSError("injected fsync failure")

    monkeypatch.setattr(verify.os, "fsync", stop_before_publish)
    with pytest.raises(OSError, match="injected fsync failure"):
        verify.publish_exclusive(dest, {"schema": verify.SCHEMA, "status": verify.STATUS})
    assert not dest.exists()
    assert not list(tmp_path.glob("*.pending"))
