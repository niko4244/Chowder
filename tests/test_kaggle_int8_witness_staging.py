"""Hermetic regression for actual Kaggle int8 witness checkpoint staging."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "kaggle" / "int8_witness" / "run_step.py"
SPEC = importlib.util.spec_from_file_location("_kaggle_witness_staging_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _model(tmp_path: Path, content: bytes = b"correct BF16 or int8 checkpoint bytes"):
    source = tmp_path / "model"
    source.mkdir()
    (source / "model-00001-of-00026.safetensors").write_bytes(content)
    (source / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    return source, tmp_path / "scratch"


def test_refuses_existing_partial_shard_instead_of_reusing(tmp_path):
    source, dest = _model(tmp_path)
    dest.mkdir()
    part = dest / "model-00001-of-00026.safetensors"
    part.write_bytes(b"partial checkpoint")
    with pytest.raises(RuntimeError, match="refusing unverified reuse"):
        MODULE.stage_model(source, dest)
    assert part.read_bytes() == b"partial checkpoint"
    assert sorted(p.name for p in dest.iterdir()) == [part.name]


def test_refuses_existing_wrong_checkpoint_and_does_not_overwrite(tmp_path):
    source, dest = _model(tmp_path, b"correct")
    dest.mkdir()
    (dest / "model-00001-of-00026.safetensors").write_bytes(b"wrong!")
    with pytest.raises(RuntimeError, match="already contains data"):
        MODULE.stage_model(source, dest)
    assert (dest / "model-00001-of-00026.safetensors").read_bytes() == b"wrong!"


def test_successful_fresh_copy_checks_bytes_and_model_metadata(tmp_path):
    source, dest = _model(tmp_path, bytes(range(256)) * 100)
    receipt = MODULE.stage_model(source, dest)
    assert receipt["reused"] is False
    assert receipt["files"] == 2
    assert receipt["bytes"] == sum(p.stat().st_size for p in source.iterdir())
    assert receipt["staged"] == str(dest)
    for path in source.iterdir():
        target = dest / path.name
        assert target.read_bytes() == path.read_bytes()
        assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(RuntimeError, match="already contains data"):
        MODULE.stage_model(source, dest)


@pytest.mark.parametrize("direction", ("same", "child", "parent"))
def test_staging_never_overlaps_checkpoint_source(tmp_path, direction):
    source, dest = _model(tmp_path)
    if direction == "same":
        dest = source
    elif direction == "child":
        dest = source / "would-clobber-source"
    else:
        dest = tmp_path
    with pytest.raises(RuntimeError, match="overlap|already contains"):
        MODULE.stage_model(source, dest)


def test_requires_source_files(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    with pytest.raises(RuntimeError, match="no files"):
        MODULE.stage_model(source, tmp_path / "dest")


def test_fails_closed_without_scratch_capacity(tmp_path, monkeypatch):
    source, dest = _model(tmp_path)
    monkeypatch.setattr(MODULE.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="insufficient scratch"):
        MODULE.stage_model(source, dest)
    assert dest.is_dir()
    assert not list(dest.iterdir())


def test_staged_integrity_mismatch_fails_without_accepting_result(tmp_path, monkeypatch):
    source, dest = _model(tmp_path)
    monkeypatch.setattr(MODULE, "sha256", lambda _: "0" * 64)
    with pytest.raises(RuntimeError, match="failed SHA256"):
        MODULE.stage_model(source, dest)
    assert any(dest.iterdir())
    # Fail closed after interruption: partial files cannot be accepted later.
    with pytest.raises(RuntimeError, match="refusing unverified reuse"):
        MODULE.stage_model(source, dest)
