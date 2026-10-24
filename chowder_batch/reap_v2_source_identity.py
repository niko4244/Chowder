"""Frozen Amendment-4 v2 calibration identity, shared by scoring and pruning.

This module must not import reap_select or the build preflight: both entry
points need identical source-admission semantics, including direct-script use.
The existing admitted Ornith file was measured under the registered tokenizer:
all 39 bare-rendered turns are shorter than the frozen 256-token minimum. It
therefore contributes *zero* rows, not the 13 unregistered worked-render rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

FROZEN_V2_BASE_ROWS = 704
FROZEN_V2_BASE_SHA256 = "f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929"
FROZEN_ORNITH_ADMITTED_SHA256 = "92c35137658ca212fb5783eb1b1087929a889f1ae5a6eaf9a584866a3373d01a"
FROZEN_ORNITH_POOL = 39
ORNITH_QUOTA = 64
MAX_ADMITTED_BYTES = 16 * 1024 * 1024
ABSENT_ORNITH = "absent: no admitted Ornith curriculum supplied"


class V2SourceIdentityError(ValueError):
    """The calibration is not the registered frozen v2 source population."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise V2SourceIdentityError(reason)


def frozen_v2_source_identity(calibration: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the exact frozen 704 rows and the optional bare slot.

    The frozen admitted Ornith file has no eligible bare rows at the >=256
    token floor. In particular, a fabricated 13-row suffix must not pass just
    because its manifest and observer were rehashed consistently.
    """
    raw = (calibration / "calibration.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    base_sha = hashlib.sha256(b"".join(lines[:FROZEN_V2_BASE_ROWS])).hexdigest()
    _require(
        len(lines) >= FROZEN_V2_BASE_ROWS and base_sha == FROZEN_V2_BASE_SHA256,
        "frozen v2 base identity mismatch: requires the preregistered "
        f"{FROZEN_V2_BASE_ROWS}-row SHA256 {FROZEN_V2_BASE_SHA256}",
    )
    tail = lines[FROZEN_V2_BASE_ROWS:]
    _require(not tail, "frozen v2 bare Ornith admitted source has zero eligible rows; "
             "nonempty appended suffix is not preregistered")
    ornith = manifest.get("ornith")
    if isinstance(ornith, dict):
        _require(ornith.get("render") == "bare",
                 "frozen v2 Ornith rendering must be bare")
        pinned = ornith.get("sha256")
        _require(pinned == FROZEN_ORNITH_ADMITTED_SHA256,
                 "frozen v2 admitted Ornith SHA256 differs from registered source")
        stats = (manifest.get("sources") or {}).get("ornith_curriculum")
        _require(isinstance(stats, dict) and type(stats.get("kept")) is int
                 and stats["kept"] == 0 and stats.get("quota") == ORNITH_QUOTA
                 and stats.get("pool") == FROZEN_ORNITH_POOL,
                 "frozen v2 bare Ornith manifest must declare pool 39 and kept 0")
        admitted = ornith.get("path")
        _require(isinstance(admitted, str) and bool(admitted),
                 "frozen v2 Ornith admitted source path missing")
        admitted_path = Path(admitted)
        _require(admitted_path.is_file(),
                 "frozen v2 Ornith admitted source file missing")
        _require(admitted_path.stat().st_size <= MAX_ADMITTED_BYTES,
                 "frozen v2 Ornith admitted source exceeds safe metadata bound")
        _require(hashlib.sha256(admitted_path.read_bytes()).hexdigest() == pinned,
                 "frozen v2 admitted Ornith bytes disagree with pinned SHA256")
        admission = "bare: pinned admitted source, zero eligible"
    else:
        _require(ornith == ABSENT_ORNITH
                 and "ornith_curriculum" not in (manifest.get("sources") or {}),
                 "frozen v2 absent-Ornith declaration inconsistent with sources")
        admission = "absent: no admitted source provided"
    return {
        "frozen_base_sha256": base_sha,
        "frozen_base_rows": FROZEN_V2_BASE_ROWS,
        "ornith_suffix_rows": 0,
        "ornith_admission": admission,
    }
