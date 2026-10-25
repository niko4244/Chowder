"""Model/content-bound resume checks; synthetic CPU-only, no network or GPU."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

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


def test_source_witness_fsyncs_complete_json_before_exclusive_publication(
    tmp_path, monkeypatch,
):
    _serve(tmp_path)
    target = tmp_path / "gsm8k.inputs.json"
    observed = []
    real_fsync = reap_eval.os.fsync

    def trace_fsync(fd):
        pending = list(tmp_path.glob(".gsm8k.inputs.json.*.pending"))
        assert len(pending) == 1
        assert not target.exists()
        observed.append(json.loads(pending[0].read_text(encoding="utf-8")))
        return real_fsync(fd)

    monkeypatch.setattr(reap_eval.os, "fsync", trace_fsync)
    result = reap_eval._bind_evaluation_inputs(tmp_path, "gsm8k", ITEMS)
    assert observed == [result]
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob(".gsm8k.inputs.json.*.pending"))


def test_partial_source_witness_write_never_creates_authoritative_invalid_json(
    tmp_path, monkeypatch,
):
    _serve(tmp_path)
    real_open = Path.open
    wrote_partial = []

    class FailingWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            self.stream.write(data[:17])
            wrote_partial.append(True)
            raise OSError("injected interrupted source receipt write")

    def intercept(self, mode="r", *args, **kwargs):
        stream = real_open(self, mode, *args, **kwargs)
        if self.parent == tmp_path and mode == "xb" and self.name.endswith(".pending"):
            return FailingWrite(stream)
        return stream

    monkeypatch.setattr(Path, "open", intercept)
    with pytest.raises(OSError, match="injected interrupted source"):
        reap_eval._bind_evaluation_inputs(tmp_path, "gsm8k", ITEMS)
    assert wrote_partial == [True]
    assert not (tmp_path / "gsm8k.inputs.json").exists()
    assert not list(tmp_path.glob(".gsm8k.inputs.json.*.pending"))


def test_concurrent_source_witness_publication_never_replaces_other_identity(
    tmp_path, monkeypatch,
):
    _serve(tmp_path)
    target = tmp_path / "gsm8k.inputs.json"
    prior = b'{"another_immutable_identity":true}\n'
    real_rename = Path.rename
    real_link = reap_eval.os.link

    def race_rename(self, other):
        if self.name.endswith(".pending") and Path(other) == target:
            target.write_bytes(prior)
            raise FileExistsError("injected concurrent source receipt")
        return real_rename(self, other)

    def race_link(source, other, *args, **kwargs):
        if str(source).endswith(".pending") and Path(other) == target:
            target.write_bytes(prior)
            raise FileExistsError("injected concurrent source receipt")
        return real_link(source, other, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", race_rename)
    monkeypatch.setattr(reap_eval.os, "link", race_link)
    with pytest.raises(FileExistsError, match="concurrent source receipt"):
        reap_eval._bind_evaluation_inputs(tmp_path, "gsm8k", ITEMS)
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob(".gsm8k.inputs.json.*.pending"))


def test_stale_uncommitted_source_witness_pending_never_promoted(tmp_path):
    _serve(tmp_path)
    abandoned = tmp_path / ".gsm8k.inputs.json.previous.pending"
    abandoned.write_text('{"partial":', encoding="utf-8")
    current = reap_eval._bind_evaluation_inputs(tmp_path, "gsm8k", ITEMS)
    assert json.loads((tmp_path / "gsm8k.inputs.json").read_text()) == current
    assert abandoned.read_text(encoding="utf-8") == '{"partial":'


def test_new_v2_standalone_evaluation_refuses_before_any_generation(tmp_path, monkeypatch):
    root = tmp_path / "reap128_v2"
    _serve(root)
    monkeypatch.delenv("REAP_BOUND_EVALUATION_SESSION", raising=False)
    called = []
    monkeypatch.setattr(reap_eval, "_load", lambda _: called.append("dataset") or ITEMS)
    args = SimpleNamespace(arm="reap128_v2", port=8790, suites="gsm8k", limit=0, parallel=1)
    with pytest.raises(ValueError, match="serving.*attestation|bound serving session|serving owner not verifiable"):
        reap_eval._run_suites(args, root)
    assert called == []
    assert not (root / "gsm8k.jsonl").exists()
    assert not (root / "gsm8k.inputs.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="registered new-v2 listener owner is Windows-bound")
@pytest.mark.parametrize("mutation", ["none", "wrong_model", "wrong_pid", "wrong_port",
                                     "wrong_receipt", "wrong_flags", "foreign_listener"])
def test_live_v2_eval_requires_exact_model_process_and_listener(
    tmp_path, monkeypatch, mutation,
):
    import psutil

    root = tmp_path / "reap128_v2"
    _serve(root)
    sid = "a" * 32
    gguf = r"C:\verified\reap128-model.gguf"
    receipt = {"gguf": gguf, "gguf_sha256": "a" * 64,
               "eval_session_id": sid, "server_pid": 4242, "server_port": 8790,
               "flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]}
    (root / "serve.json").write_text(json.dumps(receipt), encoding="utf-8")
    claim = {"session_id": sid, "server_pid": 4242, "port": 8790,
             "model_sha256": "a" * 64, "receipt": "serve.json"}
    argv = [r"C:\llama.cpp\bin\llama-server.exe", *receipt["flags"],
            "--port", "8790", "-m", gguf]
    owner = 4242
    if mutation == "wrong_model":
        argv[-1] = r"C:\other\alien.gguf"
    if mutation == "wrong_pid":
        claim["server_pid"] = 9000
    if mutation == "wrong_port":
        claim["port"] = 8767
    if mutation == "wrong_receipt":
        claim["receipt"] = "../../foreign.json"
    if mutation == "wrong_flags":
        argv[2] = "off"
    if mutation == "foreign_listener":
        owner = 1234

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return argv

    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [SimpleNamespace(
        pid=owner, status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(port=8790, ip="127.0.0.1"))])
    monkeypatch.setenv("REAP_BOUND_EVALUATION_SESSION", json.dumps(claim))
    if mutation == "none":
        assert reap_eval._require_live_new_v2_server(root, arm="reap128_v2", port=8790) is None
    else:
        with pytest.raises(ValueError, match="serving|live|bound|session"):
            reap_eval._require_live_new_v2_server(root, arm="reap128_v2", port=8790)


def test_v2_live_server_loss_during_scoring_never_publishes_summary(tmp_path, monkeypatch):
    arm = tmp_path / "reap128_v2"
    _serve(arm)
    calls = []

    def owner_check(_root, *, arm, port):
        assert arm == "reap128_v2" and port == 8790
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("serving owner lost while scoring")

    monkeypatch.setattr(reap_eval, "_require_live_new_v2_server", owner_check)
    monkeypatch.setattr(reap_eval, "_load", lambda _: ITEMS)
    monkeypatch.setattr(reap_eval, "_chat", lambda *_a: {
        "text": "42", "finish_reason": "stop", "completion_tokens": 1,
        "seconds": 0.1, "timings": {"predicted_per_second": 5}})
    monkeypatch.setattr(reap_eval, "_score", lambda _suite, item, _response: {
        "id": item["id"], "correct": True, "terminated": True, "trigram_ratio": 1.0,
        "completion_tokens": 1, "seconds": 0.1, "predicted_per_second": 5,
        "text": "42"})
    args = SimpleNamespace(arm="reap128_v2", port=8790, suites="gsm8k", limit=0, parallel=1)
    with pytest.raises(ValueError, match="serving owner lost"):
        reap_eval._run_suites(args, arm)
    assert len(calls) == 2
    assert not (arm / "summary.json").exists()
    assert not (arm / ".summary.json.lock").exists()
    assert len((arm / "gsm8k.jsonl").read_text().splitlines()) == len(ITEMS)


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
