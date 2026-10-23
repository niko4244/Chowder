"""CPU-only bounded Kaggle-mounted checkpoint read regression."""

import hashlib

import pytest

from chowder_batch.kaggle_dataset_io_probe import MAX_SAMPLE_BYTES, sample_mount


def test_missing_mount_and_missing_shards(tmp_path):
    assert sample_mount(tmp_path / "absent")["status"] == "MOUNT_MISSING"
    assert sample_mount(tmp_path)["status"] == "CHECKPOINT_SHARDS_ABSENT"


def test_bounded_stream_hashes_only_read_sample(tmp_path):
    source = tmp_path / "private-dataset" / "weights"
    source.mkdir(parents=True)
    content = bytes(range(256)) * 100
    (source / "model-00001-of-00026.safetensors").write_bytes(content)
    got = sample_mount(tmp_path, cap_bytes=257, cap_seconds=30.0)
    assert got["bounded_bytes_read"] == 257
    assert got["sample_sha256_not_full_weight_digest"] == hashlib.sha256(content[:257]).hexdigest()
    assert got["shard_size_bytes"] == len(content)
    assert got["full_checkpoint_staged"] is False
    assert got["registered_d0"] == "NOT_MEASURED"


@pytest.mark.parametrize("bad_bytes,bad_seconds", [
    (0, 1), (-1, 1), (MAX_SAMPLE_BYTES + 1, 1), (1024, 0), (1024, 121),
])
def test_read_budgets_fail_closed_before_file_io(tmp_path, bad_bytes, bad_seconds):
    with pytest.raises(ValueError, match="bounds"):
        sample_mount(tmp_path, cap_bytes=bad_bytes, cap_seconds=bad_seconds)
