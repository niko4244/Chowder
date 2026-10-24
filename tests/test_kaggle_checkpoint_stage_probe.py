"""CPU-only tests for bounded staging-to-scratch copy with integrity proof."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch.kaggle_checkpoint_stage_probe import (
    MAX_SAMPLE_BYTES,
    MAX_SAMPLE_SECONDS,
    copy_verify_prefix,
    main,
)


def _mount(tmp_path: Path, payload: bytes) -> tuple[Path, Path]:
    mount = tmp_path / "mount"
    scratch = tmp_path / "scratch"
    mount.mkdir()
    scratch.mkdir()
    shard_dir = mount / "datasets" / "account" / "model"
    shard_dir.mkdir(parents=True)
    (shard_dir / "model-00001-of-00026.safetensors").write_bytes(payload)
    return mount, scratch


def test_prefix_copy_verified_and_deleted(tmp_path):
    original = bytes(range(256)) * 500
    dataset, scratch = _mount(tmp_path, original)
    outcome = copy_verify_prefix(dataset, scratch, cap_bytes=1025, cap_seconds=60)
    assert outcome["status"] == "SAMPLE_COPY_VERIFIED"
    assert outcome["bounded_bytes_copied"] == 1025
    assert outcome["verified_bytes"] == 1025
    expected = hashlib.sha256(original[:1025]).hexdigest()
    assert outcome["source_prefix_sha256_not_checkpoint_hash"] == expected
    assert outcome["scratch_prefix_sha256"] == expected
    assert outcome["prefix_integrity_pass"] is True
    assert outcome["full_checkpoint_staged"] is False
    assert outcome["frozen_gate_d0"] == "NOT_MEASURED"
    assert not list(scratch.iterdir()), "temporary copy must not remain on disk"


def test_small_shard_still_never_claims_full_checkpoint(tmp_path):
    dataset, scratch = _mount(tmp_path, b"\x01\x02\x03")
    record = copy_verify_prefix(dataset, scratch)
    assert record["bounded_bytes_copied"] == 3
    assert record["status"] == "SAMPLE_COPY_VERIFIED"
    assert record["full_checkpoint_staged"] is False
    assert record["original_parent_staged"] is False


def test_empty_shard_refused_without_fake_success(tmp_path):
    dataset, scratch = _mount(tmp_path, b"")
    result = copy_verify_prefix(dataset, scratch)
    assert result == {"status": "EMPTY_SHARD_REFUSED", "bounded_bytes_copied": 0}
    assert not list(scratch.iterdir())


def test_no_mount_missing_scratch_and_no_shards(tmp_path):
    dataset, scratch = _mount(tmp_path, b"ok")
    assert copy_verify_prefix(tmp_path / "missing", scratch)["status"] == "DATASET_MOUNT_MISSING"
    assert copy_verify_prefix(dataset, tmp_path / "missing")["status"] == "SCRATCH_ROOT_MISSING"
    (dataset / "datasets/account/model/model-00001-of-00026.safetensors").unlink()
    assert copy_verify_prefix(dataset, scratch)["status"] == "MOUNTED_SHARDS_ABSENT"


@pytest.mark.parametrize("bytes_cap,seconds_cap", [
    (0, 1), (-1, 1), (MAX_SAMPLE_BYTES + 1, 1), (1, 0),
    (1, -1), (1, MAX_SAMPLE_SECONDS + 1), (True, 1), (1, True),
])
def test_invalid_read_limits_refuse_before_filesystem(tmp_path, bytes_cap, seconds_cap):
    with pytest.raises(ValueError, match="cap|timeout"):
        copy_verify_prefix(tmp_path / "missing", tmp_path / "missing",
                           cap_bytes=bytes_cap, cap_seconds=seconds_cap)


def test_dataset_cannot_be_scratch_parent_or_child(tmp_path):
    dataset, scratch = _mount(tmp_path, b"ok")
    with pytest.raises(ValueError, match="scratch|dataset"):
        copy_verify_prefix(dataset, dataset)
    with pytest.raises(ValueError, match="scratch"):
        copy_verify_prefix(dataset, dataset / "datasets")
    with pytest.raises(ValueError, match="dataset"):
        copy_verify_prefix(dataset, tmp_path)
    assert scratch.is_dir()


def test_reject_linked_shards(tmp_path):
    dataset, scratch = _mount(tmp_path, b"source")
    original = dataset / "datasets/account/model/model-00001-of-00026.safetensors"
    original.unlink()
    try:
        original.symlink_to(tmp_path / "foreign.safetensors")
    except OSError:
        pytest.skip("Windows symlink privileges unavailable on this host")
    with pytest.raises(ValueError, match="regular file"):
        copy_verify_prefix(dataset, scratch)


def test_report_exclusive_and_source_untouched(tmp_path):
    payload = b"real byte stream " * 400
    dataset, scratch = _mount(tmp_path, payload)
    out = tmp_path / "result.json"
    assert main(["--dataset", str(dataset), "--scratch", str(scratch),
                 "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["sample"]["status"] == "SAMPLE_COPY_VERIFIED"
    assert result["gpu_or_tpu_initialized"] is False
    assert result["model_loaded"] is False
    assert result["trained_steps"] == 0
    assert result["gate_d0"] == "NOT_MEASURED"
    assert (dataset / "datasets/account/model/model-00001-of-00026.safetensors").read_bytes() == payload
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--scratch", str(scratch),
              "--out", str(out)])
