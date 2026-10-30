"""CPU-only resumable equivalence contract; no production model weights read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder_batch import reap_verify_shards as verify
from chowder_batch.reap_prune_supervised import HostState


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, list[str]]:
    dirs = {name: tmp_path / name for name in
            ("source", "observed", "selection", "candidate", "proofs")}
    for directory in dirs.values():
        directory.mkdir()
    (dirs["selection"] / "selection.json").write_text('{"sealed":true}')
    (dirs["observed"] / "observation.json").write_text(json.dumps({
        "checkpoint_receipt": {"source_identity": {"artifacts": {}}}
    }))
    names = [f"model-{i:05d}-of-00026.safetensors" for i in range(1, 27)]
    index = {"weight_map": {f"tensor{i}": name for i, name in enumerate(names)},
             "metadata": {"total_size": 37_981_137_632}}
    (dirs["candidate"] / "model.safetensors.index.json").write_text(json.dumps(index))
    (dirs["candidate"] / "reap-prune.json").write_text('{"total_bytes":37981137632}')
    for name in names:
        blob = name.encode()
        (dirs["source"] / name).write_bytes(blob)
        (dirs["candidate"] / name).write_bytes(blob + b".pruned")
    observed = json.loads((dirs["observed"] / "observation.json").read_text())
    observed["checkpoint_receipt"]["source_identity"]["artifacts"] = {
        "weights/" + name: verify.sha256_file(dirs["source"] / name) for name in names
    }
    (dirs["observed"] / "observation.json").write_text(json.dumps(observed))
    monkeypatch.setattr(verify, "host_state", lambda path: HostState(18, 32, 50, 200))
    kwargs = [dirs[n] for n in ("source", "observed")]
    admission = verify._preflight(
        *kwargs, dirs["selection"] / "selection.json", dirs["candidate"], dirs["proofs"],
    )
    assert not admission["reasons"]
    return admission, names


def _write_fake_verified_shards(admission: dict, names: list[str]) -> None:
    for i, name in enumerate(names):
        rec = {
            "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
            "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_SHARD_PASS",
            "individual_shard": name, "candidate": admission["model"],
            "original_source": admission["source"], "shards": 1,
            "selection_sha256": verify.sha256_file(Path(admission["selection"])),
            "original_observation_sha256":
                verify.sha256_file(Path(admission["observed"]) / "observation.json"),
            "candidate_index_sha256":
                verify.sha256_file(Path(admission["model"]) / "model.safetensors.index.json"),
            "candidate_prune_receipt_sha256":
                verify.sha256_file(Path(admission["model"]) / "reap-prune.json"),
            "original_shard_sha256":
                verify.sha256_file(Path(admission["source"]) / name),
            "candidate_shard_sha256": {
                name: verify.sha256_file(Path(admission["model"]) / name)
            },
            "routed_tensors": 120 if i == 0 else 0,
            "row_or_tensor_checks": 15360 if i == 0 else 1,
            "source_non_mtp_tensor_bytes": 70_214_363_872 if i == 0 else 0,
            "pruned_tensor_bytes": 37_981_137_632 if i == 0 else 0,
        }
        (Path(admission["receipts"]) / verify._certificate_name(name)).write_text(
            json.dumps(rec)
        )


def test_shard_consolidation_requires_all_26_immutable_proofs(tmp_path, monkeypatch):
    admission, names = _fixture(tmp_path, monkeypatch)
    _write_fake_verified_shards(admission, names)
    target = tmp_path / "full.json"
    output = verify.consolidate(admission, target=target)
    assert output["status"] == "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
    assert output["shards"] == 26 and output["routed_tensors"] == 120
    assert len(output["partial_shard_proofs"]) == 26
    assert not target.exists()  # caller publishes only after complete validation
    (Path(admission["receipts"]) / verify._certificate_name(names[-1])).unlink()
    with pytest.raises(ValueError, match="missing per-shard"):
        verify.consolidate(admission, target=target)


def test_consolidation_detects_same_size_candidate_mutation(tmp_path, monkeypatch):
    admission, names = _fixture(tmp_path, monkeypatch)
    _write_fake_verified_shards(admission, names)
    file = Path(admission["model"]) / names[8]
    original = file.read_bytes()
    file.write_bytes(b"x" + original[1:])
    assert file.stat().st_size == len(original)
    with pytest.raises(ValueError, match="differs from attested bytes"):
        verify.consolidate(admission, target=tmp_path / "no-success.json")


def test_resumable_supervisor_dry_run_never_spawns_and_refuses_bad_log(tmp_path, monkeypatch):
    admission, names = _fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(verify, "_run_owned", lambda *args, **kwargs: calls.append(1) or 0)
    args = [
        "--source", admission["source"], "--observed", admission["observed"],
        "--selection", admission["selection"], "--candidate", admission["model"],
        "--receipts-dir", admission["receipts"],
    ]
    assert verify.main(args) == 0
    assert calls == []
    assert verify.main(args + ["--execute", "--log",
                               str(Path(admission["model"]) / "bad.log")]) == 2
    assert calls == [] and not (Path(admission["model"]) / "bad.log").exists()
    _write_fake_verified_shards(admission, names)
    assert verify.main(args) == 0  # no repeat of prior proofs
    assert calls == []


def test_host_admission_refuses_when_commit_falls(tmp_path, monkeypatch):
    admission, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(verify, "host_state", lambda _: HostState(17, 7, 30, 600))
    arguments = [
        "--source", admission["source"], "--observed", admission["observed"],
        "--selection", admission["selection"], "--candidate", admission["model"],
        "--receipts-dir", admission["receipts"], "--execute",
        "--log", str(tmp_path / "must-not-open.log"),
    ]
    assert verify.main(arguments) == 2
    assert not (tmp_path / "must-not-open.log").exists()


def test_mapped_comparison_start_reserve_keeps_runtime_floor(tmp_path, monkeypatch):
    admission, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(verify, "host_state", lambda _: HostState(14, 15.99, 30, 600))
    refused = verify._preflight(
        Path(admission["source"]), Path(admission["observed"]),
        Path(admission["selection"]), Path(admission["model"]),
        Path(admission["receipts"]),
    )
    assert refused["reasons"] == [
        "Windows commit below 16 GiB (8 GiB runtime floor + 8 GiB start reserve)"
    ]
    monkeypatch.setattr(verify, "host_state", lambda _: HostState(14, 16.0, 30, 600))
    admitted = verify._preflight(
        Path(admission["source"]), Path(admission["observed"]),
        Path(admission["selection"]), Path(admission["model"]),
        Path(admission["receipts"]),
    )
    assert admitted["status"] == "ADMITTED_NOT_EXECUTED"
