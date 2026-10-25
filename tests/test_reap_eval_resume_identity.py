"""Model/content-bound resume checks; synthetic CPU-only, no network or GPU."""

import hashlib
import json
import sys

import pytest

from chowder_batch import reap_eval


def _serve(out, model_sha="a" * 64):
    out.mkdir(parents=True, exist_ok=True)
    (out / "serve.json").write_text(json.dumps({
        "gguf": "C:/verified/reap160-q4.gguf",
        "gguf_sha256": model_sha,
        "flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
    }), encoding="utf-8")


def _run(monkeypatch, root, items, *, called=None, limit=0, suites="gsm8k",
         on_chat=None):
    called = called if called is not None else []
    monkeypatch.setattr(reap_eval, "_load", lambda suite: items[suite] if isinstance(items, dict) else items)

    def fake_chat(port, prompt, max_tokens):
        called.append(prompt)
        if on_chat:
            on_chat()
        return {"text": "42", "finish_reason": "stop", "completion_tokens": 1,
                "seconds": 0.1, "timings": {"predicted_per_second": 5}}

    monkeypatch.setattr(reap_eval, "_chat", fake_chat)
    monkeypatch.setattr(reap_eval, "_score", lambda suite, item, result: {
        "id": item["id"], "correct": True, "terminated": True,
        "trigram_ratio": 1.0, "completion_tokens": 1,
        "seconds": 0.1, "predicted_per_second": 5, "text": "42",
    })
    monkeypatch.setattr(sys, "argv", ["reap_eval.py", "--port", "8790", "--arm",
                                     "reap160_q4", "--suites", suites,
                                     "--out", str(root), "--limit", str(limit)])
    reap_eval.main()
    return called


ITEMS = [{"id": "gsm8k-0", "prompt": "what is one?", "expected": "1"},
         {"id": "gsm8k-1", "prompt": "what is two?", "expected": "2"}]
SUITES = {"gsm8k": ITEMS, "humaneval": [
    {"id": "HumanEval/0", "prompt": "write f", "header": "def f():\n",
     "entry_point": "f", "test": "def check(f): assert f() == 42"},
]}


def test_separate_suite_invocations_preserve_original_digest_and_both_gates(
    tmp_path, monkeypatch,
):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, SUITES, suites="gsm8k")
    before = json.loads((arm / "summary.json").read_text())
    gsm_bytes = (arm / "gsm8k.jsonl").read_bytes()
    assert _run(monkeypatch, tmp_path, SUITES, suites="humaneval") == ["write f"]
    summary = json.loads((arm / "summary.json").read_text())
    assert set(summary) == {"gsm8k", "humaneval"}
    assert summary["gsm8k"] == before["gsm8k"]
    assert (arm / "gsm8k.jsonl").read_bytes() == gsm_bytes
    assert _run(monkeypatch, tmp_path, SUITES, suites="gsm8k") == []
    assert json.loads((arm / "summary.json").read_text()) == summary


@pytest.mark.parametrize("mutation", [
    "result_bytes", "summary_sha", "no_witness", "wrong_model",
    "wrong_inputs", "wrong_decode", "duplicate_row", "unknown_id",
])
def test_second_suite_rejects_unverified_previous_suite_before_generation(
    tmp_path, monkeypatch, mutation,
):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, SUITES, suites="gsm8k")
    rows = arm / "gsm8k.jsonl"
    summary_file = arm / "summary.json"
    witness_file = arm / "gsm8k.inputs.json"
    if mutation == "result_bytes":
        rows.write_bytes(rows.read_bytes() + b"\n")
    elif mutation == "summary_sha":
        summary = json.loads(summary_file.read_text())
        summary["gsm8k"]["file_sha256"] = "b" * 64
        summary_file.write_text(json.dumps(summary))
    elif mutation == "no_witness":
        witness_file.unlink()
    elif mutation in ("wrong_model", "wrong_inputs", "wrong_decode"):
        witness = json.loads(witness_file.read_text())
        if mutation == "wrong_model":
            witness["model"]["gguf_sha256"] = "c" * 64
        elif mutation == "wrong_inputs":
            witness["source_items_sha256"] = "c" * 64
        else:
            witness["decode"]["max_tokens"] = 123
        witness_file.write_text(json.dumps(witness))
    elif mutation in ("duplicate_row", "unknown_id"):
        data = rows.read_text().splitlines()
        if mutation == "duplicate_row":
            data.append(data[0])
        else:
            data[0] = json.dumps({**json.loads(data[0]), "id": "unclaimed-id"})
        rows.write_text("\n".join(data) + "\n")
        summary = json.loads(summary_file.read_text())
        summary["gsm8k"]["file_sha256"] = hashlib.sha256(rows.read_bytes()).hexdigest()
        if mutation == "duplicate_row":
            summary["gsm8k"]["n"] += 1
        summary_file.write_text(json.dumps(summary))
    before = summary_file.read_bytes()
    called = []
    with pytest.raises(ValueError, match="SHA256|identity|witness|missing|duplicate|unexpected"):
        _run(monkeypatch, tmp_path, SUITES, suites="humaneval", called=called)
    assert called == []
    assert summary_file.read_bytes() == before
    assert not (arm / "humaneval.jsonl").exists()


@pytest.mark.parametrize("mutation", [
    "summary_changed", "prior_jsonl_changed", "prior_identity_changed",
    "new_jsonl_injected", "serving_model_changed",
])
def test_changed_suite_artifacts_during_generation_never_overwrite_prior_summary(
    tmp_path, monkeypatch, mutation,
):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, SUITES, suites="gsm8k")
    summary_path = arm / "summary.json"
    triggered = []

    def mutate():
        if triggered:
            return
        triggered.append(True)
        if mutation == "summary_changed":
            summary_path.write_bytes(summary_path.read_bytes() + b" ")
        elif mutation == "prior_jsonl_changed":
            rows = arm / "gsm8k.jsonl"
            rows.write_bytes(rows.read_bytes() + b"\n")
        elif mutation == "prior_identity_changed":
            witness = arm / "gsm8k.inputs.json"
            data = json.loads(witness.read_text())
            data["decode"]["max_tokens"] = 9
            witness.write_text(json.dumps(data))
        elif mutation == "new_jsonl_injected":
            with (arm / "humaneval.jsonl").open("a") as stream:
                stream.write(json.dumps({"id": "forged", "correct": True}) + "\n")
        else:
            _serve(arm, "e" * 64)

    with pytest.raises(ValueError, match="changed|SHA256|identity|unexpected"):
        _run(monkeypatch, tmp_path, SUITES, suites="humaneval", on_chat=mutate)
    assert triggered
    assert "humaneval" not in json.loads(summary_path.read_text())
    assert not (arm / ".summary.json.lock").exists()
    assert not (arm / ".summary.json.pending").exists()


def test_existing_exclusive_summary_lock_refuses_concurrent_reentry(tmp_path, monkeypatch):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    lock = arm / ".summary.json.lock"
    lock.write_text("another evaluation owns this arm")
    called = []
    with pytest.raises(FileExistsError):
        _run(monkeypatch, tmp_path, SUITES, called=called)
    assert called == []
    assert not (arm / "gsm8k.inputs.json").exists()
    assert lock.read_text() == "another evaluation owns this arm"


def test_existing_unclaimed_raw_refuses_before_any_generation(tmp_path, monkeypatch):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    (arm / "gsm8k.jsonl").write_text(json.dumps({
        "id": "gsm8k-0", "correct": True, "terminated": True,
        "trigram_ratio": 1.0}) + "\n")
    called = []
    with pytest.raises(ValueError, match="resume.*identity|unclaimed.*results"):
        _run(monkeypatch, tmp_path, ITEMS, called=called)
    assert not called


def test_resume_same_verified_model_and_frozen_inputs(tmp_path, monkeypatch):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, ITEMS, limit=1)
    assert (arm / "gsm8k.inputs.json").is_file()
    observed = _run(monkeypatch, tmp_path, ITEMS)
    assert observed == ["what is two?"]
    assert len((arm / "gsm8k.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("mutation", ["model", "prompt", "expected", "decode"])
def test_resume_refuses_changed_source_or_model(tmp_path, monkeypatch, mutation):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, ITEMS, limit=1)
    target = [dict(row) for row in ITEMS]
    if mutation == "model":
        _serve(arm, "b" * 64)
    elif mutation == "prompt":
        target[0]["prompt"] = "different prompt"
    elif mutation == "expected":
        target[0]["expected"] = "99"
    else:
        monkeypatch.setitem(reap_eval.MAX_TOKENS, "gsm8k", 512)
    called = []
    with pytest.raises(ValueError, match="resume.*identity|inputs.*changed"):
        _run(monkeypatch, tmp_path, target, called=called)
    assert not called


@pytest.mark.parametrize("corruption", ["duplicate", "unknown", "wrong_summary_sha"])
def test_resume_refuses_corrupt_or_unattested_scored_rows(tmp_path, monkeypatch, corruption):
    arm = tmp_path / "reap160_q4"
    _serve(arm)
    _run(monkeypatch, tmp_path, ITEMS, limit=1)
    data = arm / "gsm8k.jsonl"
    existing = json.loads(data.read_text().splitlines()[0])
    if corruption == "duplicate":
        data.write_text(data.read_text() + json.dumps(existing) + "\n")
    elif corruption == "unknown":
        data.write_text(data.read_text() + json.dumps({**existing, "id": "other"}) + "\n")
    else:
        summary = json.loads((arm / "summary.json").read_text())
        summary["gsm8k"]["file_sha256"] = "0" * 64
        (arm / "summary.json").write_text(json.dumps(summary))
    called = []
    with pytest.raises(ValueError, match="duplicate|unexpected|SHA256"):
        _run(monkeypatch, tmp_path, ITEMS, called=called)
    assert not called
