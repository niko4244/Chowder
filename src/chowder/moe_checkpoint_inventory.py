"""One-time SHA256 checkpoint enrollment and cheap, fail-closed integrity checks.

The enrollment hashes every payload byte once. Later checks pin that verified
inventory to the gate table and inspect every file's identity/metadata, hashing
only the small semantic files. Changed shard metadata requires a new full
enrollment; this deliberately does not claim that filesystem metadata can
detect a malicious in-place rewrite that restores every timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .local_model_manifest import _payload_file_names, _WEIGHT_SUFFIXES
from .provenance import sha256_file

SCHEMA = "chowder.moe-checkpoint-sha256.v1"
DEFAULT_FILE = ".chowder-moe-sha256-inventory.json"
ENV_FILE = "CHOWDER_MOE_SHA_INVENTORY"
_HEX = re.compile(r"^[0-9a-f]{64}$")


class CheckpointInventoryError(ValueError):
    """An inventory is missing, incomplete, or no longer describes its model."""


def inventory_path(model_dir: str | Path) -> Path:
    return Path(os.environ.get(ENV_FILE) or Path(model_dir) / DEFAULT_FILE)


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "dev": stat.st_dev,
        "ino": stat.st_ino,
    }


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _payload_digest(files: list[dict[str, Any]]) -> str:
    lines = [f"{row['name']} {row['stat']['size']} {row['sha256']}" for row in files]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _file_names(root: Path) -> list[str]:
    names = _payload_file_names(root)
    if not any(Path(name).suffix.lower() in _WEIGHT_SUFFIXES for name in names):
        raise CheckpointInventoryError(f"no safetensors checkpoint weights under {root}")
    return names


def enroll(model_dir: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Explicitly hash all checkpoint bytes once; create a new inventory only."""
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise CheckpointInventoryError(f"checkpoint path is not a directory: {root}")
    target = Path(output) if output is not None else inventory_path(root)
    if target.exists():
        raise CheckpointInventoryError(f"inventory already exists (refusing overwrite): {target}")
    files = []
    names = _file_names(root)
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise CheckpointInventoryError(f"refusing linked or missing payload: {path}")
        before = _signature(path)
        digest = sha256_file(path)
        after = _signature(path)
        if after != before:
            raise CheckpointInventoryError(f"payload changed while being hashed: {path}")
        files.append({"name": name, "sha256": digest, "stat": after})
    if _file_names(root) != names:
        raise CheckpointInventoryError("checkpoint payload names changed during enrollment")
    payload = {
        "schema": SCHEMA,
        "root": str(root),
        "content_sha256": _payload_digest(files),
        "files": files,
    }
    payload["inventory_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def verified_identity(model_dir: str | Path) -> dict[str, Any]:
    """Cheap check against an already *content-hashed* and pinned inventory.

    Does not read large weight payloads. Each weight's size, modification and
    creation/change time, device and file identity must be unchanged. Semantic
    files (config, index, tokenizer, etc.) have their SHA checked every time.
    """
    root = Path(model_dir).resolve(strict=True)
    path = inventory_path(root)
    if not path.is_file():
        raise CheckpointInventoryError(
            f"verified SHA inventory missing at {path}; enroll once before qualification"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        digest = doc.pop("inventory_sha256")
        if doc["schema"] != SCHEMA or doc["root"] != str(root):
            raise CheckpointInventoryError("inventory schema or source directory mismatch")
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise CheckpointInventoryError("inventory pin is absent or malformed")
        if digest != hashlib.sha256(_canonical(doc)).hexdigest():
            raise CheckpointInventoryError("inventory digest mismatches its recorded contents")
        rows = doc["files"]
        if not isinstance(rows, list) or [r["name"] for r in rows] != _file_names(root):
            raise CheckpointInventoryError("checkpoint payload inventory is incomplete or changed")
        if doc["content_sha256"] != _payload_digest(rows):
            raise CheckpointInventoryError("recorded checkpoint content digest mismatches SHA inventory")
        if not _HEX.fullmatch(doc["content_sha256"]):
            raise CheckpointInventoryError("invalid checkpoint content digest")
        for row in rows:
            name = row["name"]
            if not _HEX.fullmatch(row["sha256"]):
                raise CheckpointInventoryError(f"missing or malformed SHA256: {name}")
            file = root / name
            if file.is_symlink() or not file.is_file() or _signature(file) != row["stat"]:
                raise CheckpointInventoryError(f"checkpoint payload changed since verified SHA inventory: {name}")
            if file.suffix.lower() not in _WEIGHT_SUFFIXES and sha256_file(file) != row["sha256"]:
                raise CheckpointInventoryError(f"semantic payload SHA256 mismatch: {name}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        if isinstance(error, CheckpointInventoryError):
            raise
        raise CheckpointInventoryError(f"invalid or unreadable verified SHA inventory: {path}: {error}") from error
    return {"content_sha256": doc["content_sha256"], "inventory_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll SHA256 inventory once (full checkpoint read)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = enroll(args.model, args.out)
    print(json.dumps({"content_sha256": result["content_sha256"], "inventory_sha256": result["inventory_sha256"]}))


if __name__ == "__main__":
    main()
