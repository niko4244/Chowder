"""P4b: a scored base model must be bound by content, not by path.

The audit found the base model identified in run registries by a path plus
an unresolved revision: reloading "the same" base-plus-delta pair months
later reproduces whatever bytes happen to sit at that path now. This file
pins the binding seam for evaluators: `resolve_base_identity` produces an
immutable, path-independent content identity via
`build_local_model_manifest(mode="full")`, with the honest weaker modes
recorded as weaker (never silently upgraded to a full claim), and relocating
the identical directory does not change the identity.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from chowder.base_identity import (
    BaseIdentityError,
    base_identity_claim,
    describe_base_identity,
    resolve_base_identity,
)


def _tiny_model_dir(root: Path) -> Path:
    """The smallest thing the manifest accepts: one shard + a config."""
    d = root / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "test-tiny"}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 64)
    return d


def test_full_mode_binds_weight_contents(tmp_path):
    d = _tiny_model_dir(tmp_path)
    identity = resolve_base_identity(str(d))
    assert identity["mode"] == "full"
    assert identity["content_sha256"]
    assert identity["manifest_sha256"]
    assert identity["total_weight_bytes"] == 64
    shard = next(r for r in identity["weight_files"] if r["path"].endswith(".safetensors"))
    assert len(shard["sha256"]) == 64


def test_identity_is_path_independent(tmp_path):
    """The point of the seam: copy the identical directory elsewhere and the
    CONTENT identity must be unchanged (only the recorded path differs). The
    underlying manifest digest deliberately couples the path; the seam derives
    a path-free content digest above it."""
    src = _tiny_model_dir(tmp_path / "a")
    dst = tmp_path / "b" / "model"
    dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)
    one = resolve_base_identity(str(src))
    two = resolve_base_identity(str(dst))
    assert one["content_sha256"] == two["content_sha256"]
    assert one["model_dir"] != two["model_dir"]


def test_weaker_mode_is_labeled_weaker_not_upgraded(tmp_path):
    d = _tiny_model_dir(tmp_path)
    fast = resolve_base_identity(str(d), mode="fast")
    full = resolve_base_identity(str(d), mode="full")
    assert fast["mode"] == "fast"
    assert fast["weight_binding"] == "name-size-inventory"
    assert full["weight_binding"] == "per-shard-sha256"
    assert fast["content_sha256"] != full["content_sha256"]


def test_mismatch_is_reported_not_self_healed(tmp_path):
    """The module's own honesty rule, bound at the identity seam: a directory
    that no longer matches its recorded manifest is a divergence, not a pass."""
    d = _tiny_model_dir(tmp_path)
    identity = resolve_base_identity(str(d))
    claim = json.loads(base_identity_claim(identity))
    (d / "model.safetensors").write_bytes(b"\x01" * 64)
    from chowder.local_model_manifest import build_local_model_manifest

    rebuilt = build_local_model_manifest(d, mode="full")
    assert rebuilt.manifest_sha256 != claim["manifest_sha256"]


def test_missing_directory_refuses(tmp_path):
    with pytest.raises(BaseIdentityError):
        resolve_base_identity(str(tmp_path / "does-not-exist"))


def test_hub_reference_gets_revision_identity_not_a_fake_manifest(tmp_path):
    """A Hub model id is not a local directory: it must be recorded honestly
    as revision-bound provenance instead of failing the manifest or, worse,
    pretending to a content claim it cannot measure."""
    identity = describe_base_identity("Qwen/tiny-random", revision="abc123")
    assert identity["binding"] == "hub-revision"
    assert identity["revision"] == "abc123"
    assert "content_sha256" not in identity


def test_local_directory_gets_content_binding(tmp_path):
    d = _tiny_model_dir(tmp_path)
    identity = describe_base_identity(str(d), revision=None)
    assert identity["binding"] == "local-content"
    assert len(identity["content_sha256"]) == 64
