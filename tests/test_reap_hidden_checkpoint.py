"""CPU-only adversarial tests for committed BF16 layers; never load CUDA weights."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from chowder_batch import reap_hidden_checkpoint as cp


@pytest.fixture(autouse=True)
def tiny_disk_preflight(monkeypatch):
    # C: may not have the production 4 GiB reserve. Exercise full admission
    # separately while allocating only a few dozen bytes in these hermetic tests.
    monkeypatch.setattr(cp.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=100 * cp.GIB))


def _rows():
    return [
        torch.tensor([0x0000, 0x8000, 0x3F80, 0x7FC1, 0xFF80, 0x0001,
                      0x8001, 0x3F00], dtype=torch.uint16)
        .view(torch.bfloat16).reshape(1, 2, 4),
        torch.tensor([0x7F80, 0x1234, 0x0001, 0x8000], dtype=torch.uint16)
        .view(torch.bfloat16).reshape(1, 1, 4),
        torch.tensor([0x3F80, 0x4000, 0x3F00, 0xBF80, 0x0000, 0x8001,
                      0x3E00, 0x7FC2, 0x7FC1, 0xFF80, 0x8000, 0x0001],
                     dtype=torch.uint16)
        .view(torch.bfloat16).reshape(1, 3, 4),
    ]


def _bits(value):
    return value.contiguous().view(torch.uint16).flatten().tolist()


def _identity():
    return {"source_sha256": "a" * 64, "model_payload_sha256": "b" * 64,
            "token_ids_sha256": "c" * 64, "runtime": "frozen",
            "calibration_sha256": "d" * 64, "lengths": [2, 1, 3]}


def _open(tmp_path):
    return cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch", _identity())


def _embed(checkpoint):
    for i, tensor in enumerate(_rows()):
        checkpoint[i] = tensor
    checkpoint.finish_embeddings()


def _stats(layer):
    return {k: torch.full((3,), float(layer + j), dtype=torch.float64)
            for j, k in enumerate(("count", "gate_sum", "norm_sum", "reap_sum"))}


def _forward(value, layer):
    # Copy raw bits; do not canonicalize distinct NaN payloads or signed zeros.
    # Actual neural arithmetic is separately unchanged in reap_observe.py.
    bits = value.contiguous().view(torch.uint16)
    return bits.clone().view(torch.bfloat16).reshape(value.shape)


def _run_layer(checkpoint, layer, crash_row=None):
    checkpoint.begin_layer(layer)
    for i in range(len(checkpoint)):
        checkpoint[i] = _forward(checkpoint[i], layer)
        if i == crash_row:
            raise RuntimeError("injected crash in middle of layer")
    checkpoint.finish_layer(layer, _stats(layer), {"layer": layer, "routed": 9})


def _collect(checkpoint):
    checkpoint.begin_final_read()
    return [_bits(checkpoint[i]) for i in range(len(checkpoint))]


def test_three_layers_uninterrupted_vs_midlayer_resume_exact_bits(tmp_path):
    baseline = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "baseline", _identity())
    _embed(baseline)
    for layer in range(3):
        _run_layer(baseline, layer)
    expected = _collect(baseline)
    expected_stats = [_stats(i) for i in range(3)]
    baseline.close()

    interrupted = _open(tmp_path)
    _embed(interrupted)
    _run_layer(interrupted, 0)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run_layer(interrupted, 1, crash_row=1)
    interrupted.close()  # simulates process death releasing the OS writer lock

    restored = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                      _identity(), resume=True)
    assert restored.generation == 1
    assert restored.timings == [{"layer": 0, "routed": 9}]
    for layer in (1, 2):
        _run_layer(restored, layer)
    assert _collect(restored) == expected
    for layer, expected_stat in enumerate(expected_stats):
        actual = torch.load(tmp_path / "scratch" / f"layer-{layer:02d}.pt",
                            weights_only=True)
        assert all(torch.equal(actual[key], expected_stat[key]) for key in expected_stat)
    restored.close()


@pytest.mark.parametrize("crash_stage", ["before_marker", "after_marker"])
def test_crash_at_commit_publication_boundary(tmp_path, monkeypatch, crash_stage):
    obj = _open(tmp_path)
    _embed(obj)
    original = cp._atomic_bytes
    def broken(path, payload):
        if path.name == "CURRENT" and b'"generation":1' in payload:
            if crash_stage == "before_marker":
                raise OSError("injected fsync or replace failure")
            original(path, payload)
            raise OSError("injected process exit immediately after marker")
        return original(path, payload)
    monkeypatch.setattr(cp, "_atomic_bytes", broken)
    with pytest.raises(OSError, match="injected"):
        _run_layer(obj, 0)
    obj.close()
    monkeypatch.setattr(cp, "_atomic_bytes", original)
    recovered = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                        _identity(), resume=True)
    assert recovered.generation == (0 if crash_stage == "before_marker" else 1)
    if crash_stage == "before_marker":
        _run_layer(recovered, 0)
    _run_layer(recovered, 1)
    assert recovered.generation == 2
    recovered.close()


@pytest.mark.parametrize("failure_stage", [
    "slab_fsync", "state_checksum", "stats_save", "manifest_publish",
])
def test_commit_failures_before_current_preserve_prior_generation(
        tmp_path, monkeypatch, failure_stage):
    checkpoint = _open(tmp_path)
    _embed(checkpoint)
    if failure_stage == "slab_fsync":
        original = cp._sync_file
        def injected(path):
            if path.name == "state-B.bf16":
                raise OSError("injected failed slab fsync")
            return original(path)
        monkeypatch.setattr(cp, "_sync_file", injected)
    elif failure_stage == "state_checksum":
        original = cp.sha256_file
        def injected(path):
            if path.name == "state-B.bf16":
                raise OSError("injected failed state checksum")
            return original(path)
        monkeypatch.setattr(cp, "sha256_file", injected)
    elif failure_stage == "stats_save":
        def injected(*_):
            raise OSError("injected interrupted statistics")
        monkeypatch.setattr(cp, "_atomic_stats", injected)
    else:
        original = cp._atomic_bytes
        def injected(path, data):
            if path.name == "generation-001.json":
                raise OSError("injected interrupted manifest")
            return original(path, data)
        monkeypatch.setattr(cp, "_atomic_bytes", injected)
    with pytest.raises(OSError, match="injected"):
        _run_layer(checkpoint, 0)
    checkpoint.close()
    monkeypatch.undo()
    restored = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                       _identity(), resume=True)
    assert restored.generation == 0
    _run_layer(restored, 0)
    assert restored.generation == 1
    restored.close()


def test_missed_row_refuses_commit_and_cannot_advance(tmp_path):
    obj = _open(tmp_path)
    _embed(obj)
    obj.begin_layer(0)
    obj[0] = obj[0].clone()
    with pytest.raises(RuntimeError, match="incomplete"):
        obj.finish_layer(0, _stats(0), {"layer": 0})
    obj.close()
    restored = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                       _identity(), resume=True)
    assert restored.generation == 0
    with pytest.raises(RuntimeError, match="skip or replay"):
        restored.begin_layer(1)
    restored.close()


@pytest.mark.parametrize("tamper", [
    "slab", "stats", "CURRENT", "layout", "manifest", "truncated_slab",
])
def test_tampering_with_committed_checkpoint_fails_closed(tmp_path, tamper):
    obj = _open(tmp_path)
    _embed(obj)
    _run_layer(obj, 0)
    obj.close()
    folder = tmp_path / "scratch"
    file = {
        "slab": folder / "state-B.bf16",
        "stats": folder / "layer-00.pt",
        "CURRENT": folder / "CURRENT",
        "layout": folder / "layout.json",
        "manifest": folder / "generation-001.json",
        "truncated_slab": folder / "state-B.bf16",
    }[tamper]
    if tamper == "truncated_slab":
        with file.open("r+b") as handle:
            handle.truncate(4)
    else:
        value = bytearray(file.read_bytes())
        value[min(8, len(value) - 1)] ^= 0x01
        file.write_bytes(value)
    with pytest.raises((RuntimeError, ValueError, json.JSONDecodeError)):
        cp.BF16CheckpointHidden([2, 1, 3], 4, folder, _identity(), resume=True)


def test_changed_provenance_lengths_and_existing_dir_fail_closed(tmp_path):
    obj = _open(tmp_path)
    _embed(obj)
    with pytest.raises(RuntimeError, match="live writer"):
        cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                _identity(), resume=True)
    obj.close()
    different = dict(_identity(), source_sha256="changed")
    with pytest.raises(RuntimeError, match="provenance"):
        cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                different, resume=True)
    with pytest.raises(RuntimeError, match="layout"):
        cp.BF16CheckpointHidden([1, 2, 3], 4, tmp_path / "scratch",
                                _identity(), resume=True)
    with pytest.raises(FileExistsError):
        _open(tmp_path)


@pytest.mark.parametrize("fault", [
    "missing_stats", "tampered_embedded_manifest", "missing_generation",
])
def test_missing_or_broken_committed_chain_fails_closed(tmp_path, fault):
    checkpoint = _open(tmp_path)
    _embed(checkpoint)
    _run_layer(checkpoint, 0)
    _run_layer(checkpoint, 1)
    checkpoint.close()
    folder = tmp_path / "scratch"
    if fault == "missing_stats":
        (folder / "layer-00.pt").unlink()
    elif fault == "tampered_embedded_manifest":
        path = folder / "generation-000.json"
        data = json.loads(path.read_bytes())
        data["state_sha256"] = "a" * 64
        path.write_bytes(cp._canonical(data))
    else:
        (folder / "generation-001.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        cp.BF16CheckpointHidden([2, 1, 3], 4, folder, _identity(), resume=True)


def test_independent_embedding_generation_survives_restart(tmp_path):
    obj = _open(tmp_path)
    _embed(obj)
    obj.close()
    restored = cp.BF16CheckpointHidden([2, 1, 3], 4, tmp_path / "scratch",
                                       _identity(), resume=True)
    assert restored.generation == 0
    _run_layer(restored, 0)
    assert _collect(restored) == [_bits(r) for r in _rows()]
    restored.close()


def test_real_weight_payload_change_detected_even_if_same_size(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "model-00001.safetensors"},
    }))
    shard = model / "model-00001.safetensors"
    shard.write_bytes(b"BF16_A")
    (model / "config.json").write_text('{"layers":40}')
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(name="mock-GPU", major=8, minor=9))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90000)
    monkeypatch.setattr(cp.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="mock-driver, GPU-123"))
    params = dict(model_dir=model, weights_dir=model, calibration_sha256="a" * 64,
                  seqs=[torch.tensor([1, 2]), torch.tensor([9])], layer_ids=[0, 1],
                  out=tmp_path / "out")
    original = cp.fingerprint_inputs(**params)
    assert original["token_ids_sha256"]
    shard.write_bytes(b"BF16_B")
    changed = cp.fingerprint_inputs(**params)
    assert original["artifacts"] != changed["artifacts"]
    assert original["token_ids_sha256"] == changed["token_ids_sha256"]
    assert cp.fingerprint_inputs(**dict(params, seqs=[torch.tensor([2, 1]),
                                                     torch.tensor([9])]))["token_ids_sha256"] != original["token_ids_sha256"]


def test_fresh_disk_budget_requires_two_full_slabs_plus_four_gib(tmp_path, monkeypatch):
    monkeypatch.setattr(cp.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=2 * cp.GIB))
    with pytest.raises(RuntimeError, match="not enough disk"):
        _open(tmp_path)
    assert not (tmp_path / "scratch").exists()


def test_sealed_receipt_binds_final_state_all_stats_and_consumer(tmp_path):
    output = tmp_path / "observation"
    output.mkdir()
    checkpoint = _open(tmp_path)
    _embed(checkpoint)
    for layer in range(3):
        _run_layer(checkpoint, layer)
        checkpoint.mirror_layer_stats(output, layer)
    sealed = checkpoint.sealed_receipt()
    observed = {
        "calibration_sha256": _identity()["calibration_sha256"],
        "n_tokens": 6,
        "layers": checkpoint.timings,
        "hidden_storage": {
            "type": "dual file-backed CPU BF16 bits, atomic layer checkpoint",
            "last_committed_generation": 3,
        },
        "checkpoint_receipt": sealed,
    }
    cp.validate_observation_receipt(output, observed, expected_generation=3)
    # A copied layer file is not trusted just because observation.json passed.
    path = output / "layer-01.pt"
    original = path.read_bytes()
    path.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="not committed"):
        cp.validate_observation_receipt(output, observed, expected_generation=3)
    path.write_bytes(original)
    observed["checkpoint_receipt"]["root_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="root"):
        cp.validate_observation_receipt(output, observed, expected_generation=3)
    observed["checkpoint_receipt"] = sealed
    checkpoint.close()


def test_final_seal_catches_active_slab_corruption_after_commit(tmp_path):
    obj = _open(tmp_path)
    _embed(obj)
    _run_layer(obj, 0)
    original = obj.sealed_receipt()
    assert original["generation"] == 1
    slab = tmp_path / "scratch" / "state-B.bf16"
    with slab.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"\x11")
        stream.flush()
    with pytest.raises(RuntimeError, match="final hidden state changed"):
        obj.sealed_receipt()
    obj.close()


def test_consumer_rejects_dual_checkpoint_without_sealed_receipt(tmp_path):
    observed = {
        "hidden_storage": {
            "type": "dual file-backed CPU BF16 bits, atomic layer checkpoint",
            "last_committed_generation": 40,
        }
    }
    with pytest.raises(ValueError, match="missing/invalid sealed receipt"):
        cp.validate_observation_receipt(tmp_path, observed)


def test_full_d0_never_accepts_partial_or_altered_calibration():
    from chowder_batch.reap_observe import _require_frozen_d0_calibration
    from chowder_batch.reap_v2_source_identity import FROZEN_V2_BASE_SHA256

    _require_frozen_d0_calibration(FROZEN_V2_BASE_SHA256, 704, 0)
    for sha, rows, limit in (
        ("a" * 64, 704, 0),
        (FROZEN_V2_BASE_SHA256, 703, 0),
        (FROZEN_V2_BASE_SHA256, 704, 1),
    ):
        with pytest.raises(ValueError, match="frozen untruncated"):
            _require_frozen_d0_calibration(sha, rows, limit)


def test_full_d0_never_accepts_wrong_model_or_retokenized_census():
    from chowder_batch.reap_observe import _require_frozen_d0_model

    config = dict(num_hidden_layers=40, hidden_size=2048, num_experts=256,
                  num_experts_per_tok=8)
    _require_frozen_d0_model(SimpleNamespace(**config), 1_192_632)
    with pytest.raises(ValueError, match="704x40"):
        _require_frozen_d0_model(SimpleNamespace(**config), 1_192_636)
    for attribute, bad in (
        ("num_hidden_layers", 39), ("hidden_size", 1024),
        ("num_experts", 128), ("num_experts_per_tok", 4),
    ):
        with pytest.raises(ValueError, match="704x40"):
            _require_frozen_d0_model(
                SimpleNamespace(**dict(config, **{attribute: bad})), 1_192_632,
            )
