"""Read-only original AutomationBench guard.v1 scorer, loaded from historical Git.

Do not require the live guard/runner working-tree SHA to equal a retired v1.
Load the exact historical v1 original scored runner in isolated module memory;
preserve active scripts and the old 43/600 owner's files byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import types
from pathlib import Path

from chowder_batch import reap_original600_live_independent_audit_20261027 as scorer

COMMIT = "ea453b2f9ea85e8729673e401b0d6caec49986b7"
ROOT = Path("F:/reap-v2-q4-original-automationbench600-checkpointed-20261027-0953")
OWNER_SHA = "849309859a5c1ef8736ee67668b33d8fbbf1dc3334b246451af4c4e5d46f220a"
RUNNER = "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py"
GUARD = "chowder_batch/reap_context_guard_proxy_20261026.py"
SOURCE_SHA = {
    GUARD: "7b4fc9799c581a76c9a84d764871e5ff47cb286759addf12c1231885f44d0fc8",
    RUNNER: "38e7646ad259afdeaadc308d0e6f0da7d44dd3d93ccf59a43e90fccbdff0ede7",
    "chowder_batch/reap_original600_live_independent_audit_20261027.py":
        "89b06b2c908f80f2f9f8ed5802fbb32fd8888d1a7e4113e454b334b63e422539",
    "chowder_batch/reap_q4_automationbench_ctx64k_20261026.py":
        "5d8b9cb66ca5ac5824f443a917b3efa9ff38d7f26749c3340be8bd98edac755b",
    "chowder_batch/reap_q8_automationbench_kaggle_20261026.py":
        "1ba5b0ffa9d1edd0d82f3e374c8dabb3db859935610c56988b555539830e8eb9",
}


def sources() -> dict[str, bytes]:
    repo = Path(__file__).resolve().parent.parent
    selected = {}
    for name, expected in SOURCE_SHA.items():
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"],
                                      cwd=repo, timeout=20)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("historical original guard.v1 pinned Git source drift: " + name)
        if name not in (GUARD, RUNNER):
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
                raise ValueError("historical v1 scoring dependency drift: " + name)
        selected[name] = raw
    if b'POLICY_NAME = "guard.v1"' not in selected[GUARD]:
        raise ValueError("historical guard.v1 policy source cannot be authenticated")
    return selected


def historical_runner(source: bytes) -> types.ModuleType:
    """Load original runner code (definitions only), never call model inference."""
    name = "chowder_batch._guardv1_original_git_scoring_runner"
    module = types.ModuleType(name)
    module.__package__ = "chowder_batch"
    module.__file__ = f"{COMMIT}:{RUNNER}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    if (module.q4 is not scorer.run.q4
            or module.source_ids is not scorer.run.source_ids
            or module.SCHEMA != scorer.run.SCHEMA
            or not callable(module.verify_item)):
        raise ValueError("historical v1 original grader/runner dependency identity drift")
    return module


def inspect() -> dict:
    raw = sources()
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise ValueError("original historical guard.v1 source root missing or linked")
    owner_file = ROOT / "owner.json"
    if (owner_file.is_symlink() or not owner_file.is_file()
            or hashlib.sha256(owner_file.read_bytes()).hexdigest() != OWNER_SHA):
        raise ValueError("original guard.v1 immutable owner SHA drift")
    frozen = historical_runner(raw[RUNNER])
    before_run = scorer.run
    before_policy = frozen.q4.POLICY_NAME
    try:
        scorer.run = frozen
        frozen.q4.POLICY_NAME = "guard.v1"
        result = scorer.inspect(ROOT)
    finally:
        scorer.run = before_run
        frozen.q4.POLICY_NAME = before_policy
    if (result["owner_sha256"] != OWNER_SHA
            or result["completed_scored_tasks"] != 43
            or result["totals"]["strict_passes"] != 0
            or result["registered_population"] != 600
            or result["official_complete_600_claim_allowed"] is not False
            or result["noncomparable_to_frozen_four_slot"] is not True
            or result["frozen_gen0_status"] != "BLOCKED"
            or result["release_approved"] is not False):
        raise ValueError("historical v1 original source population/grade invalid")
    return {
        **result,
        "source_lock": "chowder.original600-guardv1-inmemory-git-historical-audit.v1",
        "historic_git_commit": COMMIT,
        "historic_guard_source_sha256": SOURCE_SHA[GUARD],
        "historic_runner_source_sha256": SOURCE_SHA[RUNNER],
        "frozen_four_slot_comparable": False,
        "model_executed_for_this_audit": False,
    }


def main() -> int:
    try:
        data = inspect()
        print(json.dumps({k: v for k, v in data.items() if k != "rows"},
                         sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "REFUSED_ORIGINAL_GUARDV1_HISTORICAL_SOURCE",
                          "reason": str(error)[:380], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
