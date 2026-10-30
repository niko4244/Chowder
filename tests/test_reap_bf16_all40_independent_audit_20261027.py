"""Adversarial source-shard and finite-numeric tests; no model/GPU access."""

import pytest
from pathlib import Path

from chowder_batch import reap_bf16_all40_independent_audit_20261027 as a


def test_exact_model_layer_source_sha_is_admitted(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model-00001-of-00026.safetensors").write_bytes(b"correct")
    name = "model-00001-of-00026.safetensors"
    sha = "a" * 64
    a.assert_exact_receipt_sources(model=model, claimed={name: sha},
                                  expected={name: sha}, required=[name])


@pytest.mark.parametrize("claimed,required", [
    ({}, ["x"]),
    ({"x": "a" * 64}, ["x", "x"]),
    ({"x": "b" * 64}, ["x"]),
    ({"x": "a" * 64, "y": "a" * 64}, ["x"]),
    ({"x": "a" * 63}, ["x"]),
    ({"x": None}, ["x"]),
])
def test_forged_missing_extra_and_resealed_layer_receipts_are_rejected(
        tmp_path, claimed, required):
    model = tmp_path / "model"
    model.mkdir()
    (model / "x").write_bytes(b"benign")
    with pytest.raises(ValueError):
        a.assert_exact_receipt_sources(model=model, claimed=claimed,
                                      expected={"x": "a" * 64}, required=required)


def test_symlinked_original_tensor_rejected(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    other = tmp_path / "other"
    other.write_bytes(b"benign")
    try:
        (model / "x").symlink_to(other)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        # Unprivileged Windows cannot create native symlinks, but production
        # must still reject an observed symlink at exactly the indexed path.
        (model / "x").write_bytes(b"benign")
        original = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink",
            lambda path: path == model / "x" or original(path),
        )
    with pytest.raises(ValueError, match="unsafe"):
        a.assert_exact_receipt_sources(
            model=model, claimed={"x": "a" * 64},
            expected={"x": "a" * 64}, required=["x"])
