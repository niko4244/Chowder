"""Large-checkpoint hashing must never buffer a whole shard in memory."""

import hashlib
from pathlib import Path

from chowder.growth.training_binding import _sha256_file, directory_digest


def test_artifact_sha256_streams_without_calling_read_bytes(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    shard = artifact / "model.safetensors"
    payload = (b"streaming-checkpoint-" * 43_001) + b"tail"
    shard.write_bytes(payload)

    original_read_bytes = Path.read_bytes

    def forbid_full_file_read(path):
        if path == shard:
            raise AssertionError("whole-shard read_bytes would allocate the full checkpoint")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_full_file_read)
    expected = hashlib.sha256(payload).hexdigest()
    assert _sha256_file(shard) == expected
    digest, entries = directory_digest(artifact)
    assert len(digest) == 64
    assert entries == ({"path": "model.safetensors", "sha256": expected},)
