"""Fail-closed CPU inference orchestration without starting any real model."""

from __future__ import annotations

import json
from pathlib import Path

from chowder_batch import reap_streaming_smoke_supervised as guarded
from chowder_batch.reap_prune_supervised import HostState


def _fake_host(ram=20, commit=30, disk=30):
    return HostState(ram, commit, disk, 550)


def _inputs(tmp_path: Path):
    model = tmp_path / "candidate"
    model.mkdir()
    idx = model / "model.safetensors.index.json"
    record = model / "reap-prune.json"
    idx.write_text('{"weight_map":{}}')
    record.write_text('{"selection_sha256":"test"}')
    proof = tmp_path / "verified.json"
    proof.write_text(json.dumps({
        "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
        "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
        "candidate": str(model.resolve()),
        "shards": 26,
        "candidate_shard_sha256": {str(i): "0" * 64 for i in range(26)},
        "candidate_index_sha256": guarded._sha(idx),
        "candidate_prune_receipt_sha256": guarded._sha(record),
    }))
    return model, proof, tmp_path / "result.json"


def test_refuses_incomplete_full_proof_without_launch(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    doc = json.loads(proof.read_text())
    doc["status"] = "HEADER_AND_EXPERT_SAMPLE_PASS"
    proof.write_text(json.dumps(doc))
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host())
    assert guarded.main([
        "--model", str(model), "--verification", str(proof),
        "--out", str(output), "--execute", "--log", str(tmp_path / "log"),
    ]) == 2
    assert not output.exists()
    assert not (tmp_path / "log").exists()


def test_refuses_dynamic_commit_and_ram_without_launch(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host(ram=12, commit=15))
    monkeypatch.setattr(guarded, "_run_owned",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    assert guarded.main([
        "--model", str(model), "--verification", str(proof),
        "--out", str(output), "--execute", "--log", str(tmp_path / "log"),
    ]) == 2
    assert not (tmp_path / "log").exists()


def test_admitted_default_never_spawns_and_owned_path_is_cpu_only(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host())
    calls = []

    def fake_owned(command, *, out, log):
        calls.append(command)
        out.write_text('{"status":"CPU_BF16_END_TO_END_SMOKE_PASS_NOT_QUALITY_GATE"}')
        return 0

    monkeypatch.setattr(guarded, "_run_owned", fake_owned)
    args = ["--model", str(model), "--verification", str(proof), "--out", str(output)]
    assert guarded.main(args) == 0
    assert not calls
    assert guarded.main([*args, "--execute", "--log", str(tmp_path / "owned.log")]) == 0
    assert len(calls) == 1
    assert "-m" in calls[0] and "chowder_batch.reap_streaming_bf16_smoke" in calls[0]
    assert output.exists()


def test_owned_log_cannot_modify_original_source_tree(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    document = json.loads(proof.read_text())
    document["original_source"] = str(original.resolve())
    proof.write_text(json.dumps(document))
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host())
    monkeypatch.setattr(
        guarded, "_run_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launched")),
    )
    assert guarded.main([
        "--model", str(model), "--verification", str(proof),
        "--out", str(output), "--execute",
        "--log", str(original / "must-not-write.log"),
    ]) == 2
    assert not list(original.iterdir())


def test_resume_progress_is_bound_to_full_proof_and_separate_from_models(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    directory = tmp_path / "activation"
    directory.mkdir()
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host())
    launched = []

    def fake_owned(cmd, *, out, log):
        launched.append(cmd)
        out.write_text('{"status":"CPU_BF16_END_TO_END_SMOKE_PASS_NOT_QUALITY_GATE"}')
        return 0

    monkeypatch.setattr(guarded, "_run_owned", fake_owned)
    arguments = [
        "--model", str(model), "--verification", str(proof), "--out", str(output),
        "--progress-dir", str(directory), "--execute", "--log", str(tmp_path / "owned.log"),
    ]
    assert guarded.main(arguments) == 0
    assert launched and "--progress-dir" in launched[0]
    assert launched[0][launched[0].index("--full-verification-sha256") + 1] == guarded._sha(proof)


def test_activation_progress_cannot_modify_original_bf16_tree(tmp_path, monkeypatch):
    model, proof, output = _inputs(tmp_path)
    source = tmp_path / "original"
    source.mkdir()
    content = json.loads(proof.read_text())
    content["original_source"] = str(source.resolve())
    proof.write_text(json.dumps(content))
    monkeypatch.setattr(guarded, "host_state", lambda _: _fake_host())
    monkeypatch.setattr(guarded, "_run_owned",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    assert guarded.main([
        "--model", str(model), "--verification", str(proof), "--out", str(output),
        "--progress-dir", str(source), "--execute", "--log", str(tmp_path / "log"),
    ]) == 2
    assert not list(source.iterdir())
