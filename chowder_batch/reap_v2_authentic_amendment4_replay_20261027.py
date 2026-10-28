"""Source-pinned authentic historical v1 Amendment4 selector replay.

This is an exact SHA-bound reproduction of the existing frozen selection
SNAPSHOT, not a newly qualified dense/Q8/REAP160 experiment or reconstruction
of missing original row receipts. New gates must go through the source-pinned
bridge and original deterministic selector separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chowder_batch import reap_select

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs/quals/reap128-v2-frozen-gate-snapshot-prequality-20261025.json"
SOURCE_SHA256 = "7043980b1a5551131c348d720704382eeee42b63001afcccd9f031b7528bb42e"
HISTORICAL_V1_GATE1_LB = -0.03780741197234255


class FrozenReplayError(ValueError):
    """Historical registered result changed or is being misrepresented."""


def replay(source: Path = SOURCE) -> dict:
    if source.is_symlink() or not source.is_file():
        raise FrozenReplayError("historical frozen snapshot not a regular file")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise FrozenReplayError("immutable historical original selector snapshot SHA drift")
    try:
        snap = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise FrozenReplayError("historical original frozen snapshot invalid") from exc
    if (not isinstance(snap, dict)
            or snap.get("schema") != "reap-frozen-amendment4-selection-v1"
            or snap.get("release_approved") is not False
            or not isinstance(snap.get("gates"), dict)
            or set(snap["gates"]) != {"reap128", "reap160"}
            or snap["gates"]["reap128"]["gates"]["gate1"].get("status") != "FAIL"
            or snap["gates"]["reap128"]["gates"]["gate1"].get("statistic")
                != HISTORICAL_V1_GATE1_LB
            or snap["gates"]["reap128"]["gates"]["gate1"].get("threshold") != -0.03
            or snap["gates"]["reap128"]["gates"]["d0"].get("status") != "PASS"
            or snap["gates"]["reap128"]["gates"]["gate0"].get("status") != "PASS"):
        raise FrozenReplayError("registered v1 strict Gate1 FAIL/source data invalid")
    recalculated = reap_select.select(snap["gates"], snap.get("paired_v2_minus_v1"))
    if (recalculated != snap.get("decision")
            or recalculated.get("status") != "BLOCKED"
            or recalculated.get("chosen") is not None
            or "both REAP128_v2 and REAP160" not in recalculated.get("reason", "")):
        raise FrozenReplayError("authentic Amendment4 frozen fallback selector drift")
    return {"schema": "chowder.reap-v2-historical-v1-amendment4-replay.v1",
            "source_sha256": digest,
            "status": "HISTORICAL_ORIGINAL_V1_REPLAY_ONLY",
            "historical_v1_gate1": {"status": "FAIL", "lower_bound": HISTORICAL_V1_GATE1_LB,
                                     "strict_margin": -0.03},
            "current_v2_gate0": "UNKNOWN", "current_v2_gate1": "UNKNOWN",
            "current_v2_gate2": "UNKNOWN", "current_v2_control": "UNKNOWN",
            "current_v2_speed": "UNKNOWN", "reap160_native_speed": "UNKNOWN",
            "amendment4_optional_nonzero_margin_paired_mcnemar_ci": "BLOCKED_UNRESOLVED_PROSPECTIVE_METHOD",
            "authentic_historical_selector": recalculated,
            "full_new_source_population_scored_here": False,
            "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}


def main() -> None:
    print(json.dumps(replay(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
