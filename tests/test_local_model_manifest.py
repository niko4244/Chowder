"""Tests for local model content manifests (docs/LOCAL_MODELS.md provenance).

Offline and deterministic: synthetic HF-style model directories built in
tmp_path. Exercises the honesty rules the module claims: nothing invented,
absence recorded as evidence, drift reported rather than healed, full mode
being an explicit choice, and the signed manifest detecting post-hoc edits.
"""

import hashlib
import json

import pytest

from chowder.local_model_manifest import (
    FileRecord,
    LocalModelManifestError,
    build_local_model_manifest,
    manifest_summary,
    verify_local_model_manifest,
    write_manifest_file,
)


def _make_model_dir(root, *, shard_bytes=(100, 200), semantic=True, with_index=True):
    root.mkdir(parents=True, exist_ok=True)
    if semantic:
        (root / "config.json").write_text('{"model_type": "qwen3_5_text"}', encoding="utf-8")
        (root / "tokenizer_config.json").write_text('{"tokenizer_class": "Qwen2Tokenizer"}', encoding="utf-8")
    if with_index:
        (root / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    for index, size in enumerate(shard_bytes):
        (root / f"model-{index + 1:05d}-of-{len(shard_bytes):05d}.safetensors").write_bytes(
            bytes([index % 256]) * size
        )
    return root


# ---------------------------------------------------------------------------
# Building manifests
# ---------------------------------------------------------------------------


def test_fast_mode_inventories_weights_without_hashing(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="fast")

    assert manifest.mode == "fast"
    assert [f.path for f in manifest.weight_files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all(f.sha256 is None for f in manifest.weight_files)
    assert all(f.sha256 is not None for f in manifest.semantic_files if f.path == "config.json")
    assert manifest.total_weight_bytes == 300
    assert manifest.semantic_files[0].path == "config.json"  # fixed semantic order


def test_full_mode_hashes_every_weight_shard(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="full")

    assert all(f.sha256 is not None for f in manifest.weight_files)
    expected = hashlib.sha256(bytes([0]) * 100).hexdigest()
    assert manifest.weight_files[0].sha256 == expected


def test_absent_semantic_file_recorded_not_error(tmp_path):
    root = _make_model_dir(tmp_path / "model", semantic=False)
    manifest = build_local_model_manifest(root, mode="fast")
    by_name = {f.path: f for f in manifest.semantic_files}
    # chat_template was never written: recorded absent, not fabricated
    assert by_name["chat_template.jinja"].sha256 is None
    assert by_name["chat_template.jinja"].size_bytes == 0
    assert by_name["config.json"].sha256 is None  # not written in this fixture
    assert by_name["tokenizer_config.json"].sha256 is None


def test_manifest_requires_weight_shards(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LocalModelManifestError, match="no weight shards"):
        build_local_model_manifest(empty)


def test_manifest_refuses_missing_directory(tmp_path):
    with pytest.raises(LocalModelManifestError, match="not an existing directory"):
        build_local_model_manifest(tmp_path / "nope")


def test_manifest_digest_signs_the_content(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="fast")
    assert len(manifest.manifest_sha256) == 64
    # identical content in a different directory digests the same except
    # model_dir is part of the signed payload (the identity includes location)
    twin = _make_model_dir(tmp_path / "twin")
    twin_manifest = build_local_model_manifest(twin, mode="fast")
    assert manifest.manifest_sha256 != twin_manifest.manifest_sha256

    payload = json.loads(json.dumps(manifest.to_dict()))
    payload["weight_files"][0]["size_bytes"] += 1
    with pytest.raises(LocalModelManifestError, match="digest mismatch"):
        type(manifest).from_dict(payload)


def test_manifest_round_trip(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="full")
    rebuilt = type(manifest).from_dict(manifest.to_dict())
    assert rebuilt.manifest_sha256 == manifest.manifest_sha256
    assert rebuilt.weight_files == manifest.weight_files


# ---------------------------------------------------------------------------
# Verification: read-only, reports real divergences
# ---------------------------------------------------------------------------


def test_clean_verification(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="full")
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is True
    assert result.divergences == ()
    assert result.checked_files == len(manifest.semantic_files) + len(manifest.weight_files)


def test_missing_shard_is_reported_not_tolerated(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="full")
    (root / "model-00002-of-00002.safetensors").unlink()
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is False
    assert any("missing" in d for d in result.divergences)


def test_truncated_shard_is_caught_by_size(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="fast")  # fast: sizes only
    shard = root / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"x" * 50)  # same name, different size
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is False
    assert any("size drifted" in d for d in result.divergences)


def test_same_size_corruption_needs_full_mode_to_catch(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    # both manifests are taken BEFORE the corruption, as in real use:
    # a manifest describes the directory as it was measured
    fast_manifest = build_local_model_manifest(root, mode="fast")
    full_manifest = build_local_model_manifest(root, mode="full")
    shard = root / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"z" * 100)  # same size, different content

    # fast verification (sizes only) cannot see this — and does not lie
    # about it: its contract covers name, presence, and size
    assert verify_local_model_manifest(fast_manifest, root).clean is True

    # full mode hashes the bytes and catches the same-size replacement
    result = verify_local_model_manifest(full_manifest, root)
    assert result.clean is False
    assert any("sha256 mismatch" in d for d in result.divergences)


def test_forced_rehash_requires_recorded_digest(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    fast_manifest = build_local_model_manifest(root, mode="fast")
    # forcing rehash on a fast manifest verifies only what has digests;
    # it must neither crash nor invent a mismatch
    result = verify_local_model_manifest(fast_manifest, root, rehash_weights=True)
    assert result.clean is True


def test_unmanifested_extra_shard_reported(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="fast")
    (root / "sneaky.safetensors").write_bytes(b"surprise")
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is False
    assert any("unmanifested weight shard" in d for d in result.divergences)


def test_absent_then_present_semantic_file_is_divergence(tmp_path):
    root = _make_model_dir(tmp_path / "model", semantic=False, with_index=False)
    manifest = build_local_model_manifest(root, mode="fast")
    (root / "config.json").write_text("{}", encoding="utf-8")  # appeared later
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is False
    assert any("recorded it absent" in d for d in result.divergences)


def test_modified_semantic_file_is_divergence(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="fast")
    (root / "config.json").write_text('{"model_type": "something_else"}', encoding="utf-8")
    result = verify_local_model_manifest(manifest, root)
    assert result.clean is False
    assert any("config.json" in d and "mismatch" in d for d in result.divergences)


def test_verification_is_read_only(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    before = sorted((p.name, p.stat().st_mtime_ns) for p in root.iterdir())
    manifest = build_local_model_manifest(root, mode="full")
    verify_local_model_manifest(manifest, root)
    after = sorted((p.name, p.stat().st_mtime_ns) for p in root.iterdir())
    assert before == after


def test_clean_with_divergences_is_inconstructible():
    from chowder.local_model_manifest import ManifestVerification

    with pytest.raises(ValueError, match="clean"):
        ManifestVerification(clean=True, checked_files=1, divergences=("x",))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_file_record_validation():
    with pytest.raises(ValueError, match="path"):
        FileRecord(path="  ", size_bytes=1, sha256=None)
    with pytest.raises(ValueError, match="size_bytes"):
        FileRecord(path="a", size_bytes=-1, sha256=None)
    with pytest.raises(ValueError, match="sha256"):
        FileRecord(path="a", size_bytes=1, sha256="tooshort")


def test_manifest_summary_shape(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    summary = manifest_summary(build_local_model_manifest(root, mode="fast"))
    assert summary["mode"] == "fast"
    assert summary["weight_shards"] == 2
    assert summary["weight_shards_hashed"] == 0
    assert summary["total_weight_gib"] == round(300 / 2**30, 2)
    assert summary["manifest_sha256"]


def test_write_manifest_file_round_trips(tmp_path):
    root = _make_model_dir(tmp_path / "model")
    manifest = build_local_model_manifest(root, mode="full")
    out = tmp_path / "evidence" / "parent-a.manifest.json"
    file_digest = write_manifest_file(manifest, out)

    assert file_digest == hashlib.sha256(out.read_bytes()).hexdigest()
    reloaded = type(manifest).from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert reloaded.manifest_sha256 == manifest.manifest_sha256
