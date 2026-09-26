from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chowder.local_model_compat import (
    LocalModelCompatibilityError,
    patch_transformers5_custom_model,
    verify_local_custom_code,
)


def test_local_custom_code_requires_matching_digests(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    source = model / "modeling_spark.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert verify_local_custom_code(model, {source.name: digest}) == {
        source.name: digest
    }


def test_local_custom_code_rejects_tampering(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    source = model / "modeling_spark.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("class Model: changed\n", encoding="utf-8")

    with pytest.raises(LocalModelCompatibilityError, match="digest changed"):
        verify_local_custom_code(model, {source.name: digest})


def test_local_custom_code_rejects_missing_explicit_digest(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "modeling_spark.py").write_text("class Model: pass\n", encoding="utf-8")

    with pytest.raises(LocalModelCompatibilityError, match="explicit SHA-256"):
        verify_local_custom_code(model, {})


def test_patch_refuses_to_import_on_digest_mismatch(tmp_path: Path) -> None:
    """The import gate lives INSIDE patch_transformers5_custom_model: a caller
    that skips its own verification still cannot get unverified code imported."""
    model = tmp_path / "model"
    model.mkdir()
    source = model / "modeling_spark.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("class Model: changed\n", encoding="utf-8")

    with pytest.raises(LocalModelCompatibilityError, match="digest changed"):
        patch_transformers5_custom_model(model, {source.name: digest})


def test_patch_refuses_missing_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(LocalModelCompatibilityError, match="not found"):
        patch_transformers5_custom_model(model, {"modeling_spark.py": "0" * 64})
