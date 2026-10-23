"""Model/content-bound resume checks; synthetic CPU-only, no network or GPU."""

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


def _run(monkeypatch, root, items, *, called=None, limit=0):
    called = called if called is not None else []
    monkeypatch.setattr(reap_eval, "_load", lambda suite: items)

    def fake_chat(port, prompt, max_tokens):
        called.append(prompt)
        return {"text": "42", "finish_reason": "stop", "completion_tokens": 1,
                "seconds": 0.1, "timings": {"predicted_per_second": 5}}

    monkeypatch.setattr(reap_eval, "_chat", fake_chat)
    monkeypatch.setattr(reap_eval, "_score", lambda suite, item, result: {
        "id": item["id"], "correct": True, "terminated": True,
        "trigram_ratio": 1.0, "completion_tokens": 1,
        "seconds": 0.1, "predicted_per_second": 5, "text": "42",
    })
    monkeypatch.setattr(sys, "argv", ["reap_eval.py", "--port", "8790", "--arm",
                                     "reap160_q4", "--suites", "gsm8k",
                                     "--out", str(root), "--limit", str(limit)])
    reap_eval.main()
    return called


ITEMS = [{"id": "gsm8k-0", "prompt": "what is one?", "expected": "1"},
         {"id": "gsm8k-1", "prompt": "what is two?", "expected": "2"}]


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
