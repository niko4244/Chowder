"""Registered original BF16 source identity, pinned to a PUBLIC upstream commit.

The versioned JSON in reference/ is sourced from the revision-specific
Hugging Face tree API. It records upstream LFS SHA-256 OIDs (not local cache
ETags or observer-produced self hashes). Small metadata SHA-256 hashes were
computed only after validating their Git blob OIDs or upstream LFS SHA-256.

Metadata admission reads no model tensors; full admission compares SHA-256
from the ordinary observer's streaming shard hash pass against the pin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MANIFEST_SHA256 = "54e7e0d121cab5fd7e331be1ee38c5b4deddd7eee7f0cc7b112a4322f8418dca"
PIN_PATH = Path(__file__).resolve().parent / "reference" / (
    "reap_v2_original_qwen36_bf16_995ad96e.json"
)


class OriginalBF16PinError(ValueError):
    """Original registered source did not match pinned Hugging Face revision."""


def _sha256(path: Path) -> str:
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OriginalBF16PinError(f"source changed during digest: {path.name}")
    return h.hexdigest()


def load_original_pin() -> dict:
    raw = PIN_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise OriginalBF16PinError("versioned original BF16 pin bytes changed")
    pin = json.loads(raw)
    if (pin.get("schema") != "chowder.reap-original-bf16-upstream-pin.v1"
            or pin.get("repository") != REPO
            or pin.get("revision") != REVISION
            or len(pin.get("weight_shards", {})) != 26
            or len(pin.get("model_metadata", {})) != 11):
        raise OriginalBF16PinError("original BF16 pin identity/count changed")
    frozen = pin.get("frozen_retokens", {})
    lengths = frozen.get("lengths", [])
    if (pin.get("frozen_calibration_sha256") !=
            "f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929"
            or len(lengths) != 704 or frozen.get("count") != 704
            or frozen.get("sum") != 1_192_632
            or len(frozen.get("sha256", "")) != 64
            or sum(lengths) != 1_192_632):
        raise OriginalBF16PinError("registered calibration retokenization pin changed")
    return pin


def _safe_file(path: Path, parent: Path, expected_size: int) -> None:
    if (path.is_symlink() or not path.is_file()
            or path.resolve().parent != parent.resolve()
            or path.stat().st_size != expected_size):
        raise OriginalBF16PinError(f"missing/unsafe/wrong-size upstream source {path.name}")


def validate_registered_source_metadata(model_dir: Path, weights_dir: Path) -> dict:
    """Cheap CPU-only gate: 26 names/sizes, 11 small content hashes and index."""
    model_dir, weights_dir = Path(model_dir), Path(weights_dir)
    pin = load_original_pin()
    for name, expected in pin["model_metadata"].items():
        target = model_dir / name
        _safe_file(target, model_dir, expected["size"])
        if _sha256(target) != expected["sha256"]:
            raise OriginalBF16PinError(f"upstream model metadata payload mismatch: {name}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_bytes())
    mapping = index.get("weight_map")
    if (not isinstance(mapping, dict) or not mapping
            or set(mapping.values()) != set(pin["weight_shards"])):
        raise OriginalBF16PinError("original BF16 shard index references unexpected files")
    for name, expected in pin["weight_shards"].items():
        _safe_file(weights_dir / name, weights_dir, expected["size"])
    return {
        "repo": REPO,
        "revision": REVISION,
        "upstream_manifest_sha256": MANIFEST_SHA256,
    }


def validate_registered_source_artifacts(artifacts: dict) -> dict:
    """Content-bound comparison usable both BEFORE load and by D0 consumers."""
    pin = load_original_pin()
    expected = {
        **{f"weights/{name}": data["sha256"]
           for name, data in pin["weight_shards"].items()},
        **{f"model/{name}": data["sha256"]
           for name, data in pin["model_metadata"].items()},
    }
    if not isinstance(artifacts, dict) or artifacts != expected:
        raise OriginalBF16PinError(
            "observer artifacts differ from independently pinned original BF16 revision"
        )
    return {
        "repo": REPO,
        "revision": REVISION,
        "upstream_manifest_sha256": MANIFEST_SHA256,
    }


def validate_registered_token_identity(identity: dict, *, n_seqs: int, n_tokens: int) -> None:
    """Bind ordered tokenized rows to the exact frozen 704 calibration texts."""
    pin = load_original_pin()
    expected = pin["frozen_retokens"]
    if (n_seqs != 704 or n_tokens != 1_192_632
            or identity.get("calibration_sha256") != pin["frozen_calibration_sha256"]
            or identity.get("lengths") != expected["lengths"]
            or identity.get("token_ids_sha256") != expected["sha256"]):
        raise OriginalBF16PinError(
            "frozen 704 ordered token IDs/lengths do not match registered original"
        )
