"""Read-only historical guard.v3 scorer with the original runner loaded from Git.

Unlike the legacy historical reader, this never relies on the current working
tree's runner implementation. The original 600-task checkpoint is immutable;
the only output is an exclusive, separately hashed audit snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types

from chowder_batch import reap_original600_live_independent_audit_20261027 as audit

ROOT = Path("F:/reap-v2-q4-original-automationbench600-guardv3-20261027")
GIT_COMMIT = "84ff9703d6e2aba68eda8585ee461e3d84abec0c"
OWNER_SHA = "b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529"
PINNED = {
    "chowder_batch/reap_context_guard_proxy_20261026.py":
        "b9eb375519a0eb7018f7cbc007e697478b77df7d734facfc20d239005bd2ee3a",
    "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py":
        "38e7646ad259afdeaadc308d0e6f0da7d44dd3d93ccf59a43e90fccbdff0ede7",
    "chowder_batch/reap_original600_live_independent_audit_20261027.py":
        "89b06b2c908f80f2f9f8ed5802fbb32fd8888d1a7e4113e454b334b63e422539",
    "chowder_batch/reap_q4_automationbench_ctx64k_20261026.py":
        "5d8b9cb66ca5ac5824f443a917b3efa9ff38d7f26749c3340be8bd98edac755b",
    "chowder_batch/reap_q8_automationbench_kaggle_20261026.py":
        "1ba5b0ffa9d1edd0d82f3e374c8dabb3db859935610c56988b555539830e8eb9",
}
RUNNER = "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py"
GUARD = "chowder_batch/reap_context_guard_proxy_20261026.py"
SCHEMA = "chowder.original600-guardv3-inmemory-git-runner-audit.v1"


def pinned_sources() -> dict[str, bytes]:
    repo = Path(__file__).resolve().parent.parent
    result = {}
    for name, expected in PINNED.items():
        raw = subprocess.check_output(["git", "show", f"{GIT_COMMIT}:{name}"],
                                      cwd=repo, timeout=20)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("original Git source identity drift: " + name)
        # The current runner and guard may have advanced to newer versions.
        # The runner is loaded from Git; the guard's callbacks are never
        # invoked by this strictly read-only scorer. All scoring dependencies
        # must remain identical to their immutable original source.
        if name not in (RUNNER, GUARD) and hashlib.sha256(
                (repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError("original scorer/dependency working-tree drift: " + name)
        result[name] = raw
    if b'POLICY_NAME = "guard.v3"' not in result[GUARD]:
        raise ValueError("historical guard policy is not original v3")
    return result


def frozen_runner(raw: bytes) -> types.ModuleType:
    """Load immutable historical runner in this reader only; never execute run()."""
    name = "chowder_batch._reap_original600_guardv3_pinned_readonly_worker11"
    module = types.ModuleType(name)
    module.__package__ = "chowder_batch"
    module.__file__ = f"{GIT_COMMIT}:{RUNNER}"
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    if module.q4 is not audit.run.q4 or module.source_ids is not audit.run.source_ids:
        raise ValueError("historical runner dependency identity drift")
    if module.SCHEMA != audit.run.SCHEMA or not callable(module.verify_item):
        raise ValueError("historical runner checkpoint API drift")
    return module


def inspect() -> dict:
    sources = pinned_sources()
    original = audit.run
    frozen = frozen_runner(sources[RUNNER])
    previous_policy = frozen.q4.POLICY_NAME
    try:
        frozen.q4.POLICY_NAME = "guard.v3"
        audit.run = frozen
        result = audit.inspect(ROOT)
    finally:
        audit.run = original
        frozen.q4.POLICY_NAME = previous_policy
    if (result["owner_sha256"] != OWNER_SHA
            or result["registered_population"] != 600
            or result["noncomparable_to_frozen_four_slot"] is not True
            or result["frozen_gen0_status"] != "BLOCKED"
            or result["release_approved"] is not False
            or result["official_complete_600_claim_allowed"] !=
            (result["completed_scored_tasks"] == 600)):
        raise ValueError("historical original source/owner/science admission drift")
    return {
        **result,
        "historical_git_runner_source_sha256": PINNED[RUNNER],
        "historical_git_commit": GIT_COMMIT,
        "historical_guard_policy": "guard.v3",
        "source_lock": SCHEMA,
        "original_source_executed_for_scoring_in_memory_only": True,
        "frozen_four_slot_comparable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = inspect()
        if args.snapshot:
            target = args.snapshot
            if (not target.is_absolute() or target.exists() or target.is_symlink()
                    or target.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("exclusive new F: snapshot required")
            with target.open("xb") as stream:
                stream.write((json.dumps(result, sort_keys=True, indent=2,
                                         allow_nan=False) + "\n").encode())
        print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                         sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as err:
        print(json.dumps({"status": "REFUSED_HISTORICAL_GUARDV3_GIT_SCORING",
                          "reason": str(err)[:400], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
