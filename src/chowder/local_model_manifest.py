"""Content manifests for local model directories (docs/LOCAL_MODELS.md).

LOCAL_MODELS.md makes local directories first-class model sources and its
provenance section names exactly what this module delivers: "a manifest of
shard names/sizes plus selected config/tokenizer hashes, with an explicit
full-hash mode for publication-grade runs". A local directory bypasses the
Hub's integrity machinery — nothing re-checks a byte after the one-time
download — so the manifest is the content identity that keeps "download
once, hash once, pin once" honest.

Why selected hashes in fast mode, and which files are selected
--------------------------------------------------------------
A full hash of a 50+ GiB checkpoint takes real minutes and real disk
reads; making it the default would tempt callers to skip manifests
entirely, which is the worst outcome. The fast mode therefore hashes the
small files whose content decides semantics — config, generation config,
the safetensors index, the chat template, tokenizer assets, and
preprocessor configs — and records *names, sizes, and mtimes-resilient
metadata* for the weight shards without reading them. `mode="full"` adds
per-shard sha256 digests for publication-grade claims, and is always an
explicit caller choice.

What this module deliberately does NOT do
-----------------------------------------
- It never mutates the model directory. Verification is read-only.
- It never treats a mismatch as self-healing: `verify_manifest` reports
  the real divergence (missing file, size drift, digest mismatch) and it
  is the caller's decision to stop. A drifted shard must not be quietly
  accepted just because the rest matches.
- It does not invent architecture facts. Everything recorded is read
  from disk bytes or file metadata; nothing is defaulted from the Hub.
- It does not attempt to parse safetensors headers. The index JSON is
  hashed as a file; tensor-level audit belongs to the MoE instrumentation
  layer (`moe_instrumentation.py`), which reads real headers itself.

Honesty rule
------------
Every field in a `LocalModelManifest` is measured from the directory at
build time. There is no "expected size" fallback: if a listed file is
absent at verification time, that is a reported divergence, not a zero.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha256_file

#: Small semantic files hashed in every mode. Anything absent from the
#: directory is recorded as absent (many checkpoints legitimately lack,
#: e.g., a chat template) — absence is evidence, never an error.
_SEMANTIC_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

#: Weight-shard suffixes inventoried by name/size in fast mode and hashed
#: in full mode. Nothing else in the directory is treated as weights.
_WEIGHT_SUFFIXES: frozenset[str] = frozenset({".safetensors"})


class LocalModelManifestError(ValueError):
    """A local model directory cannot be manifested honestly."""


@dataclass(frozen=True)
class FileRecord:
    """One inventoried file: identity metadata plus its hash when taken."""

    path: str
    size_bytes: int
    sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("file record path must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("file record size_bytes must be a non-negative int")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or len(self.sha256) != 64
        ):
            raise ValueError("file record sha256 must be 64 hex chars or None")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class LocalModelManifest:
    """Measured content identity of one local model directory."""

    model_dir: str
    mode: str
    semantic_files: tuple[FileRecord, ...]
    weight_files: tuple[FileRecord, ...]
    manifest_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            raise ValueError("model_dir must be a non-empty string")
        if self.mode not in {"fast", "full"}:
            raise ValueError("manifest mode must be 'fast' or 'full'")
        if not self.weight_files:
            raise LocalModelManifestError(
                "no weight shards found in the directory; a manifest over a "
                "model directory with no .safetensors shards would look like "
                "evidence of a model it cannot back"
            )
        object.__setattr__(self, "manifest_sha256", self._digest())

    def _digest(self) -> str:
        payload = json.dumps(
            {
                "model_dir": self.model_dir,
                "mode": self.mode,
                "semantic_files": [f.to_dict() for f in self.semantic_files],
                "weight_files": [f.to_dict() for f in self.weight_files],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "mode": self.mode,
            "manifest_sha256": self.manifest_sha256,
            "semantic_files": [f.to_dict() for f in self.semantic_files],
            "weight_files": [f.to_dict() for f in self.weight_files],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalModelManifest":
        manifest = cls(
            model_dir=data["model_dir"],
            mode=data["mode"],
            semantic_files=tuple(
                FileRecord(**entry) for entry in data["semantic_files"]
            ),
            weight_files=tuple(FileRecord(**entry) for entry in data["weight_files"]),
        )
        recorded = data.get("manifest_sha256")
        if recorded is not None and recorded != manifest.manifest_sha256:
            raise LocalModelManifestError(
                "manifest digest mismatch: the serialized manifest was modified "
                f"after signing (recorded {str(recorded)[:12]}…, computed "
                f"{manifest.manifest_sha256[:12]}…)"
            )
        return manifest

    @property
    def total_weight_bytes(self) -> int:
        return sum(record.size_bytes for record in self.weight_files)


def build_local_model_manifest(model_dir: str | Path, *, mode: str = "fast") -> LocalModelManifest:
    """Measure a local model directory into a `LocalModelManifest`.

    `mode="fast"` hashes the semantic files and inventories weight shards
    by name and size. `mode="full"` additionally hashes every weight shard
    (publication-grade; minutes for a 50+ GiB checkpoint). The directory
    is never mutated.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise LocalModelManifestError(f"model dir is not an existing directory: {root}")

    semantic: list[FileRecord] = []
    for name in _SEMANTIC_FILES:
        path = root / name
        if not path.is_file():
            # absence is recorded as a record with size 0 and no digest;
            # `present=False` is derivable from sha256 is None + size 0
            semantic.append(FileRecord(path=name, size_bytes=0, sha256=None))
            continue
        digest = sha256_file(path)
        semantic.append(FileRecord(path=name, size_bytes=path.stat().st_size, sha256=digest))

    weights: list[FileRecord] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_file() or entry.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        if mode == "full":
            weights.append(
                FileRecord(
                    path=entry.name,
                    size_bytes=entry.stat().st_size,
                    sha256=sha256_file(entry),
                )
            )
        else:
            weights.append(FileRecord(path=entry.name, size_bytes=entry.stat().st_size, sha256=None))

    return LocalModelManifest(
        model_dir=str(root),
        mode=mode,
        semantic_files=tuple(semantic),
        weight_files=tuple(weights),
    )


@dataclass(frozen=True)
class ManifestVerification:
    """Result of checking a directory against a manifest.

    `clean` is True only when every semantic digest matches, every
    manifest-listed weight is present at its recorded size (and digest,
    in full mode), and no unmanifested weight shards appeared. A missing
    shard is a divergence, never silently tolerated.
    """

    clean: bool
    checked_files: int
    divergences: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.clean and self.divergences:
            raise ValueError("a clean verification cannot carry divergences")


def verify_local_model_manifest(
    manifest: LocalModelManifest,
    model_dir: str | Path,
    *,
    rehash_weights: bool | None = None,
) -> ManifestVerification:
    """Read-only verification of a directory against a manifest.

    Semantic files are always re-hashed. Weight shards are re-hashed when
    the manifest was built in full mode (or when `rehash_weights=True` is
    forced); otherwise their *size* is verified, which catches truncation
    and replacement-by-different-content in the overwhelmingly common
    cases at near-zero cost. Unmanifested extra weight shards are
    reported — they mean the directory changed identity behind the
    manifest's back.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise LocalModelManifestError(f"model dir is not an existing directory: {root}")

    do_weight_hashes = rehash_weights if rehash_weights is not None else (manifest.mode == "full")
    divergences: list[str] = []
    checked = 0

    present_names = {entry.name for entry in root.iterdir() if entry.is_file()}

    for record in manifest.semantic_files:
        checked += 1
        path = root / record.path
        if record.sha256 is None:
            # recorded absent: absence matches; presence is a divergence
            if record.path in present_names:
                divergences.append(
                    f"{record.path}: manifest recorded it absent but the file now exists"
                )
            continue
        if not path.is_file():
            divergences.append(f"{record.path}: manifest-listed file is missing")
            continue
        actual_size = path.stat().st_size
        if actual_size != record.size_bytes:
            divergences.append(
                f"{record.path}: size drifted (manifest {record.size_bytes} B, on disk {actual_size} B)"
            )
        actual_digest = sha256_file(path)
        if actual_digest != record.sha256:
            divergences.append(f"{record.path}: sha256 mismatch (content changed)")

    manifest_weight_names: set[str] = set()
    for record in manifest.weight_files:
        checked += 1
        manifest_weight_names.add(record.path)
        path = root / record.path
        if not path.is_file():
            divergences.append(f"{record.path}: weight shard is missing")
            continue
        actual_size = path.stat().st_size
        if actual_size != record.size_bytes:
            divergences.append(
                f"{record.path}: size drifted (manifest {record.size_bytes} B, on disk {actual_size} B)"
            )
            continue  # digest of a different-length file is uninteresting
        if do_weight_hashes and record.sha256 is not None:
            actual_digest = sha256_file(path)
            if actual_digest != record.sha256:
                divergences.append(f"{record.path}: sha256 mismatch (content changed)")

    for name in sorted(present_names - manifest_weight_names - set(_SEMANTIC_FILES)):
        if Path(name).suffix.lower() in _WEIGHT_SUFFIXES:
            divergences.append(f"{name}: unmanifested weight shard present in the directory")

    return ManifestVerification(
        clean=not divergences, checked_files=checked, divergences=tuple(divergences)
    )


def manifest_summary(manifest: LocalModelManifest) -> dict[str, Any]:
    """Compact, log-friendly summary (no digests beyond the manifest's own)."""
    hashed_semantic = sum(1 for f in manifest.semantic_files if f.sha256 is not None)
    hashed_weights = sum(1 for f in manifest.weight_files if f.sha256 is not None)
    return {
        "model_dir": manifest.model_dir,
        "mode": manifest.mode,
        "manifest_sha256": manifest.manifest_sha256,
        "semantic_files_recorded": len(manifest.semantic_files),
        "semantic_files_hashed": hashed_semantic,
        "weight_shards": len(manifest.weight_files),
        "weight_shards_hashed": hashed_weights,
        "total_weight_bytes": manifest.total_weight_bytes,
        "total_weight_gib": round(manifest.total_weight_bytes / 2**30, 2),
    }


def write_manifest_file(manifest: LocalModelManifest, output_path: str | Path) -> str:
    """Persist the signed manifest next to the model or in evidence storage.

    Returns the sha256 of the written file. The manifest carries its own
    `manifest_sha256` digest so `from_dict` can detect post-hoc edits of
    the serialized form.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
