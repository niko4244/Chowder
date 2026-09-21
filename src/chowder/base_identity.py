"""Immutable content identity for the base model a run scores or trains.

The audit found the base model identified in run registries by a path plus an
unresolved revision: reloading "the same" base-plus-delta pair months later
reproduces whatever bytes happen to sit at that path now. P4 closes that by
binding a scored base to its content.

Design decisions, and the honesty rules they enforce
----------------------------------------------------
- A local directory gets `mode="full"` manifest treatment by default: per-shard
  sha256 over the actual weight bytes plus the semantic files. That is
  publication-grade and takes real minutes on a 50+ GiB checkpoint, so the
  caller may weaken it explicitly — but the weakening is *labeled*
  (`weight_binding: "name-size-inventory"`), never silently upgraded.
- The underlying `LocalModelManifest` deliberately couples its own digest to
  `model_dir` (relocation should be detectable at the manifest layer). The
  identity *seam* therefore derives a path-free `content_sha256` above it, so
  moving identical artifacts does not change their content identity while the
  recorded path still says where the bytes were found.
- A Hub model id is not measurable bytes; it is recorded as revision-bound
  provenance with no `content_sha256` at all. Pretending a revision string is
  a content claim is exactly the unresolved-revision defect this module
  exists to close — a missing claim is honest, a fake claim is not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .local_model_manifest import LocalModelManifestError, build_local_model_manifest


class BaseIdentityError(ValueError):
    """A base model identity cannot be established honestly."""


def _content_digest(manifest) -> str:
    """Path-free content digest derived above the manifest's path-coupled one."""
    payload = json.dumps(
        {
            "kind": "chowder.base_identity.v1",
            "mode": manifest.mode,
            "semantic_files": [f.to_dict() for f in manifest.semantic_files],
            "weight_files": [f.to_dict() for f in manifest.weight_files],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_base_identity(model_dir: str | Path, *, mode: str = "full") -> dict[str, Any]:
    """Content identity of a local model directory, independent of its path."""
    try:
        manifest = build_local_model_manifest(model_dir, mode=mode)
    except LocalModelManifestError as exc:
        # A directory that cannot be manifested honestly (absent, not a
        # directory, no weight shards) cannot be given a content identity:
        # refusing is the honest outcome, never a weaker label.
        raise BaseIdentityError(str(exc)) from exc
    return {
        "binding": "local-content",
        "mode": manifest.mode,
        "model_dir": manifest.model_dir,
        "manifest_sha256": manifest.manifest_sha256,
        "content_sha256": _content_digest(manifest),
        "weight_binding": "per-shard-sha256" if manifest.mode == "full" else "name-size-inventory",
        "total_weight_bytes": manifest.total_weight_bytes,
        "weight_files": [f.to_dict() for f in manifest.weight_files],
        "semantic_files": [f.to_dict() for f in manifest.semantic_files],
    }


def describe_base_identity(
    base_model: str, *, revision: str | None, local_dir: str | Path | None = None
) -> dict[str, Any]:
    """The identity an evaluation should record for `spec.base_model`.

    `local_dir` (or a `base_model` that names an existing directory) yields a
    content binding; anything else yields revision-bound provenance with no
    content claim.
    """
    candidate = Path(local_dir if local_dir is not None else base_model)
    if candidate.is_dir():
        identity = resolve_base_identity(candidate)
        if revision:
            identity["declared_revision"] = revision
        return identity
    if candidate.exists():
        raise BaseIdentityError(
            f"base model path exists but is not a directory: {candidate}"
        )
    provenance: dict[str, Any] = {
        "binding": "hub-revision",
        "base_model": base_model,
        "revision": revision,
    }
    if not revision:
        provenance["revision_warning"] = "no revision pinned; provenance is unresolved"
    return provenance


def base_identity_claim(identity: dict[str, Any]) -> str:
    """Canonical JSON for storing the identity beside a result."""
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
