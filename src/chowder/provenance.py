from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceManifest:
    experiment_id: str
    parent_id: str | None
    config: dict[str, Any]
    dataset_digest: str
    metrics: dict[str, float]
    seed: int

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash a directory tree by relative path and file content.

    Absolute paths, mtimes, permissions, and directory-entry ordering do not
    affect the digest. Symlinks are rejected so an artifact digest cannot
    silently depend on content outside the artifact root.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"artifact directory contains unsupported symlink: {entry}")
        if not entry.is_file():
            continue
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
