"""Regression tests for `chowder.qwen38_acquisition`.

No network access, no CUDA, no torch import, no real Hugging Face Hub
call anywhere in this file -- every network- or download-touching step is
a synthetic fake. `snapshot_download_fn` fakes write real bytes to a temp
directory so the manifest/verification/parameter-accounting machinery
(shared with parents A/B) exercises its real code paths.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from chowder.parent_eval import ParentTokenizerEvidence
from chowder.qwen38_acquisition import (
    PARENT_C_PIN,
    PARENT_D_PIN,
    AcquisitionError,
    RemoteFileInfo,
    acquire_parent,
    fetch_remote_listing,
    gate_tokenizer_compatibility,
    preflight_disk_capacity,
    refuse_gguf_only,
)
from chowder.qwen38_campaign import CampaignManifestError, ParentPin


def _write_safetensors_shard(path: Path) -> int:
    """One tiny, format-valid safetensors shard: a single F32 [2,2] tensor
    (16 bytes of tensor data). Returns the file's total byte size."""
    header = {
        "model.language_model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(b"\0" * 16)
    return path.stat().st_size


def _write_model_dir(root: Path) -> dict[str, int]:
    """Writes a minimal-but-real model directory; returns {relative_path: size_bytes}."""
    root.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sizes["config.json"] = config_path.stat().st_size
    shard_path = root / "model-00001-of-00001.safetensors"
    sizes["model-00001-of-00001.safetensors"] = _write_safetensors_shard(shard_path)
    return sizes


def test_pins_are_exact_and_reuse_parentpin_validation():
    assert PARENT_C_PIN.repo == "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
    assert PARENT_C_PIN.revision == "a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8"
    assert PARENT_D_PIN.repo == (
        "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
    )
    assert PARENT_D_PIN.revision == "81c73940f94023f7d64e3ae6abcc653fc837d415"
    with pytest.raises(CampaignManifestError):
        ParentPin(repo="OBLITERATUS/Qwen3.8-27B-OBLITERATED", revision="main", role="comparison")
    with pytest.raises(CampaignManifestError):
        ParentPin(repo="OBLITERATUS/Qwen3.8-27B-OBLITERATED", revision="latest", role="comparison")


def test_refuse_gguf_only_revision():
    pin = ParentPin(repo="some/repo", revision="a" * 40, role="comparison")
    listing = fetch_remote_listing(
        pin,
        list_files_fn=lambda repo, rev: [RemoteFileInfo(path="model.gguf", size_bytes=100)],
    )
    with pytest.raises(AcquisitionError, match="GGUF is not the training parent"):
        refuse_gguf_only(listing)


def test_refuse_revision_with_no_usable_weights():
    pin = ParentPin(repo="some/repo", revision="a" * 40, role="comparison")
    listing = fetch_remote_listing(
        pin, list_files_fn=lambda repo, rev: [RemoteFileInfo(path="README.md", size_bytes=10)]
    )
    with pytest.raises(AcquisitionError, match="not a checkpoint"):
        refuse_gguf_only(listing)


def test_refuse_gguf_only_passes_when_safetensors_present():
    pin = ParentPin(repo="some/repo", revision="a" * 40, role="comparison")
    listing = fetch_remote_listing(
        pin,
        list_files_fn=lambda repo, rev: [
            RemoteFileInfo(path="model.safetensors", size_bytes=100),
            RemoteFileInfo(path="config.json", size_bytes=10),
        ],
    )
    refuse_gguf_only(listing)  # must not raise


def test_preflight_disk_capacity_picks_first_fitting_root(tmp_path):
    pin = ParentPin(repo="some/repo", revision="a" * 40, role="comparison")
    listing = fetch_remote_listing(
        pin, list_files_fn=lambda repo, rev: [RemoteFileInfo(path="w.safetensors", size_bytes=100_000_000_000)]
    )
    small = tmp_path / "small"
    big = tmp_path / "big"
    chosen = preflight_disk_capacity(
        listing,
        [(small, 10_000_000_000), (big, 200_000_000_000)],
    )
    assert chosen == big


def test_preflight_disk_capacity_raises_when_nothing_fits(tmp_path):
    pin = ParentPin(repo="some/repo", revision="a" * 40, role="comparison")
    listing = fetch_remote_listing(
        pin, list_files_fn=lambda repo, rev: [RemoteFileInfo(path="w.safetensors", size_bytes=100_000_000_000)]
    )
    with pytest.raises(AcquisitionError, match="no candidate destination"):
        preflight_disk_capacity(listing, [(tmp_path / "only", 1_000_000_000)])


def test_acquire_parent_downloads_when_not_present(tmp_path):
    destination = tmp_path / "parent-c"
    calls = {"count": 0}

    def list_files_fn(repo, rev):
        return [
            RemoteFileInfo(path="config.json", size_bytes=1),
            RemoteFileInfo(path="model-00001-of-00001.safetensors", size_bytes=1),
        ]

    def snapshot_download_fn(*, repo_id, revision, local_dir):
        calls["count"] += 1
        _write_model_dir(Path(local_dir))
        return local_dir

    result = acquire_parent(
        PARENT_C_PIN,
        destination,
        list_files_fn=list_files_fn,
        snapshot_download_fn=snapshot_download_fn,
    )
    assert calls["count"] == 1
    assert result.already_present is False
    assert result.manifest.mode == "full"
    assert result.architecture["model_type"] == "qwen3_5"
    assert result.parameter_accounting is not None
    assert result.parameter_accounting["total_parameters"] == 4  # 2x2 F32 tensor
    assert (destination.parent / f"{destination.name}.manifest.json").is_file()


def test_acquire_parent_is_idempotent_and_skips_redownload(tmp_path):
    destination = tmp_path / "parent-c"
    calls = {"count": 0}

    def list_files_fn(repo, rev):
        return [
            RemoteFileInfo(path="config.json", size_bytes=1),
            RemoteFileInfo(path="model-00001-of-00001.safetensors", size_bytes=1),
        ]

    def snapshot_download_fn(*, repo_id, revision, local_dir):
        calls["count"] += 1
        _write_model_dir(Path(local_dir))
        return local_dir

    first = acquire_parent(
        PARENT_C_PIN, destination, list_files_fn=list_files_fn, snapshot_download_fn=snapshot_download_fn
    )
    second = acquire_parent(
        PARENT_C_PIN, destination, list_files_fn=list_files_fn, snapshot_download_fn=snapshot_download_fn
    )
    assert calls["count"] == 1  # second call never re-downloaded
    assert first.already_present is False
    assert second.already_present is True
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256


def test_acquire_parent_detects_partial_download(tmp_path):
    destination = tmp_path / "parent-c"

    def list_files_fn(repo, rev):
        return [
            RemoteFileInfo(path="config.json", size_bytes=1),
            RemoteFileInfo(path="model-00001-of-00001.safetensors", size_bytes=1),
            RemoteFileInfo(path="never-written.safetensors", size_bytes=1),
        ]

    def snapshot_download_fn(*, repo_id, revision, local_dir):
        _write_model_dir(Path(local_dir))  # never writes never-written.safetensors
        return local_dir

    with pytest.raises(AcquisitionError, match="acquisition incomplete"):
        acquire_parent(
            PARENT_C_PIN, destination, list_files_fn=list_files_fn, snapshot_download_fn=snapshot_download_fn
        )


def test_manifest_divergence_triggers_reacquisition_not_silent_accept(tmp_path):
    destination = tmp_path / "parent-c"
    calls = {"count": 0}

    def list_files_fn(repo, rev):
        return [
            RemoteFileInfo(path="config.json", size_bytes=1),
            RemoteFileInfo(path="model-00001-of-00001.safetensors", size_bytes=1),
        ]

    def snapshot_download_fn(*, repo_id, revision, local_dir):
        calls["count"] += 1
        _write_model_dir(Path(local_dir))
        return local_dir

    acquire_parent(
        PARENT_C_PIN, destination, list_files_fn=list_files_fn, snapshot_download_fn=snapshot_download_fn
    )
    assert calls["count"] == 1

    # Simulate content drift after the manifest was written: the checkpoint
    # bytes changed behind the manifest's back.
    (destination / "config.json").write_text(json.dumps({"architectures": ["Different"], "model_type": "qwen3_5"}))

    acquire_parent(
        PARENT_C_PIN, destination, list_files_fn=list_files_fn, snapshot_download_fn=snapshot_download_fn
    )
    assert calls["count"] == 2  # divergence was detected, not silently trusted; re-acquired


def test_tokenizer_gate_reuses_ensure_parent_tokenizer_compatible():
    reference = ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=151936, identity_sha256="a" * 64)
    same = ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=151936, identity_sha256="a" * 64)
    different = ParentTokenizerEvidence(tokenizer_class="TokenizersBackend", vocab_size=151936, identity_sha256="b" * 64)

    assert gate_tokenizer_compatibility(reference, same).compatible is True
    result = gate_tokenizer_compatibility(reference, different)
    assert result.compatible is False
    assert "tokenizer" in result.detail.lower()


def test_acquisition_result_round_trips_through_json(tmp_path):
    destination = tmp_path / "parent-d"

    def list_files_fn(repo, rev):
        return [
            RemoteFileInfo(path="config.json", size_bytes=1),
            RemoteFileInfo(path="model-00001-of-00001.safetensors", size_bytes=1),
        ]

    def snapshot_download_fn(*, repo_id, revision, local_dir):
        _write_model_dir(Path(local_dir))
        return local_dir

    reference = ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=151936, identity_sha256="a" * 64)
    result = acquire_parent(
        PARENT_D_PIN,
        destination,
        list_files_fn=list_files_fn,
        snapshot_download_fn=snapshot_download_fn,
        measure_tokenizer_fn=lambda dest: ParentTokenizerEvidence(
            tokenizer_class="TokenizersBackend", vocab_size=151936, identity_sha256="b" * 64
        ),
        reference_tokenizer=reference,
    )
    payload = json.dumps(result.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["pin"]["repo"] == PARENT_D_PIN.repo
    assert reloaded["tokenizer_gate"]["compatible"] is False
