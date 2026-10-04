"""Formal adapter-bundle identity and layout.

A repair-only artifact is a bundle, not an undocumented PEFT directory: the
root ``default`` adapter is the frozen parent and ``repair`` is the learned
module.  The manifest makes that contract machine-readable and hash-bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .provenance import sha256_directory

MANIFEST_NAME = "chowder-adapter-bundle.json"
FORMAT = "chowder-adapter-bundle-v1"
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}


def _root_adapter_sha256(root: Path) -> str:
    """Hash the parent adapter files, excluding bundle/tokenizer additions."""
    digest = hashlib.sha256()
    adapter_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
    }
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.name in adapter_files
        ),
        key=lambda path: path.name,
    )
    for path in paths:
        relative = path.name.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_adapter_bundle_manifest(
    bundle_dir: str | Path,
    *,
    parent_sha256: str,
    repair_sha256: str,
    active_adapters: tuple[str, ...] = ("default", "repair"),
) -> Path:
    root = Path(bundle_dir).resolve()
    if not (root / "adapter_config.json").is_file():
        raise FileNotFoundError(f"bundle root adapter is missing: {root}")
    repair = root / "repair"
    if not (repair / "adapter_config.json").is_file():
        raise FileNotFoundError(f"bundle repair adapter is missing: {repair}")
    payload = {
        "format": FORMAT,
        "parent": {
            "adapter": "default",
            "sha256": parent_sha256,
            "root_sha256": _root_adapter_sha256(root),
        },
        "modules": [
            {"name": "repair", "adapter": "repair", "sha256": repair_sha256}
        ],
        "active_adapters": list(active_adapters),
        "combination": "linear",
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_adapter_bundle_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"adapter bundle manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("unsupported adapter bundle manifest")
    if payload.get("active_adapters") != ["default", "repair"]:
        raise ValueError("adapter bundle must activate default and repair")
    if payload.get("combination") != "linear":
        raise ValueError("unsupported adapter bundle combination")
    parent = payload.get("parent")
    modules = payload.get("modules")
    if not isinstance(parent, dict) or not isinstance(modules, list) or len(modules) != 1:
        raise ValueError("adapter bundle manifest has invalid module records")
    repair = modules[0]
    if not isinstance(repair, dict) or repair.get("name") != "repair":
        raise ValueError("adapter bundle manifest has no repair module")
    root_hash = parent.get("root_sha256")
    if not isinstance(root_hash, str) or len(root_hash) != 64:
        raise ValueError("adapter bundle parent root SHA is invalid")
    if _root_adapter_sha256(root) != root_hash:
        raise ValueError("parent adapter root content does not match bundle manifest")
    actual_repair = sha256_directory(root / "repair")
    if actual_repair != repair.get("sha256"):
        raise ValueError("repair adapter content does not match bundle manifest")
    return payload


def bundle_parent_sha256(bundle_dir: str | Path) -> str:
    payload = read_adapter_bundle_manifest(bundle_dir)
    value = payload["parent"].get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("adapter bundle parent SHA is invalid")
    return value


def bundle_root_sha256(bundle_dir: str | Path) -> str:
    payload = read_adapter_bundle_manifest(bundle_dir)
    value = payload["parent"].get("root_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("adapter bundle parent root SHA is invalid")
    return value
