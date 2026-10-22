"""Pure-CPU index resolution and fail-closed checkpoint packaging fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.backends.moe_int8_experts import _shard_files


def _shards(tmp_path: Path, *, order: tuple[str, ...] = ("b.safetensors", "a.safetensors")) -> Path:
    for name in order:
        (tmp_path / name).write_bytes(b"fixture bytes")
    return tmp_path


def _index(root: Path, names: tuple[str, ...]) -> None:
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"model.layers.{i}.weight": name for i, name in enumerate(names)}}),
        encoding="utf-8",
    )


def test_index_first_seen_order_and_no_duplicates(tmp_path):
    root = _shards(tmp_path)
    _index(root, ("b.safetensors", "b.safetensors", "a.safetensors", "b.safetensors"))
    assert [p.name for p in _shard_files(root)] == ["b.safetensors", "a.safetensors"]


def test_unindexed_extra_shard_refuses_instead_of_silent_skip(tmp_path):
    root = _shards(tmp_path)
    _index(root, ("a.safetensors",))
    with pytest.raises(ValueError, match="unindexed.*b.safetensors"):
        _shard_files(root)


def test_missing_shard_refuses_instead_of_partial_load(tmp_path):
    root = _shards(tmp_path, order=("a.safetensors",))
    _index(root, ("a.safetensors", "missing.safetensors"))
    with pytest.raises(ValueError, match="missing.*missing.safetensors"):
        _shard_files(root)


@pytest.mark.parametrize("name", [
    "../outside.safetensors", "nested/part.safetensors", "..\\outside.safetensors",
    "C:\\weights.safetensors", "/absolute.safetensors", "foo.bin", "", None, 123,
])
def test_untrusted_index_cannot_escape_model_root(tmp_path, name):
    root = _shards(tmp_path, order=("a.safetensors",))
    _index(root, (name,))
    with pytest.raises(ValueError, match="unsafe or invalid safetensors shard name"):
        _shard_files(root)


@pytest.mark.parametrize("weight_map", [None, {}, [], "not a mapping"])
def test_empty_or_wrong_index_shape_is_not_a_valid_checkpoint(tmp_path, weight_map):
    root = _shards(tmp_path, order=("a.safetensors",))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid or empty"):
        _shard_files(root)


def test_unindexed_single_shard_fallback_is_deterministic(tmp_path):
    root = _shards(tmp_path)
    assert [p.name for p in _shard_files(root)] == ["a.safetensors", "b.safetensors"]


def test_symlink_shard_is_not_accepted(tmp_path):
    root = _shards(tmp_path, order=("a.safetensors",))
    linked = root / "b.safetensors"
    try:
        linked.symlink_to(root / "a.safetensors")
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks requires permissions on this platform")
    _index(root, ("a.safetensors", "b.safetensors"))
    with pytest.raises(ValueError, match="symlinks"):
        _shard_files(root)
