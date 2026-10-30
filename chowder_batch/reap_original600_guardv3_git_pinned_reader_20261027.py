"""Read-only historical guard.v3 scorer isolated from later runner commits.

The original checkpoint is research-only and may contain separate native
serving sessions. Loading exact, SHA-pinned Git sources into this fresh Python
interpreter prevents a later guard/runner edit from changing the old grader.
No native server, evaluator, container, or checkpoint is started or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parent.parent
ROOT = Path("F:/reap-v2-q4-original-automationbench600-guardv3-20261027")
COMMIT = "84ff9703d6e2aba68eda8585ee461e3d84abec0c"
OWNER_SHA = "b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529"
PINNED = {
    "chowder_batch/reap_context_guard_proxy_20261026.py":
        "b9eb375519a0eb7018f7cbc007e697478b77df7d734facfc20d239005bd2ee3a",
    "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py":
        "38e7646ad259afdeaadc308d0e6f0da7d44dd3d93ccf59a43e90fccbdff0ede7",
    "chowder_batch/reap_original600_live_independent_audit_20261027.py":
        "89b06b2c908f80f2f9f8ed5802fbb32fd8888d1a7e4113e454b334b63e422539",
}
IMPORTED_AUXILIARY = (
    "chowder_batch/reap_q4_automationbench_ctx64k_20261026.py",
    "chowder_batch/reap_q8_automationbench_kaggle_20261026.py",
    # ctx64k imports the budget calculator at module import time. Omitting
    # this transitive executable dependency permits a worktree replacement
    # to affect the supposedly Git-pinned original guard.v3 scorer/runner.
    "chowder_batch/reap_context_budget_20261026.py",
)
# These are the exact immutable auxiliary source bytes in original commit
# 84ff970, not hashes learned from today's worktree or a Git replace object.
AUXILIARY_SHA256 = {
    "chowder_batch/reap_q4_automationbench_ctx64k_20261026.py":
        "5d8b9cb66ca5ac5824f443a917b3efa9ff38d7f26749c3340be8bd98edac755b",
    "chowder_batch/reap_q8_automationbench_kaggle_20261026.py":
        "1ba5b0ffa9d1edd0d82f3e374c8dabb3db859935610c56988b555539830e8eb9",
    "chowder_batch/reap_context_budget_20261026.py":
        "0286d498a35227074ba5b4aa32576967790ccbe6dd987e43b5ff3f01fd2e4632",
}
SCHEMA = "chowder.reap-original600-guardv3-git-pinned-independent.v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_assertions_enabled() -> None:
    """Original pinned scorer uses assert for evidence-type/timing validation.

    Python -O/-OO and PYTHONOPTIMIZE remove those original checks at compile
    time. Immutable source hashes do not make optimized bytecode equivalent.
    Refuse before loading the original scorer or inspecting evidence.
    """
    if sys.flags.optimize != 0 or not __debug__:
        raise ValueError("historical guard.v3 scorer requires Python assertions enabled")


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{COMMIT}:{path}"], cwd=REPO, timeout=20,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )


def source_bytes() -> dict[str, bytes]:
    result = {}
    if set(IMPORTED_AUXILIARY) != set(AUXILIARY_SHA256):
        raise ValueError("historical auxiliary source SHA inventory incomplete")
    for path, expected in PINNED.items():
        raw = git_bytes(path)
        if sha(raw) != expected:
            raise ValueError("historical Git source SHA mismatch: " + path)
        result[path] = raw
    for path in IMPORTED_AUXILIARY:
        original = git_bytes(path)
        if sha(original) != AUXILIARY_SHA256[path]:
            raise ValueError("historical scorer auxiliary dependency drift: " + path)
        result[path] = original
    if b'POLICY_NAME = "guard.v3"' not in result[next(iter(PINNED))]:
        raise ValueError("historical source does not implement guard.v3")
    return result


def _load(name: str, raw: bytes) -> types.ModuleType:
    if name in sys.modules:
        raise ValueError("fresh interpreter required for historical module: " + name)
    package = sys.modules["chowder_batch"]
    short = name.rsplit(".", 1)[-1]
    if hasattr(package, short):
        raise ValueError("historical package already contains module: " + short)
    module = types.ModuleType(name)
    module.__package__ = "chowder_batch"
    module.__file__ = f"{COMMIT}:{name.replace('.', '/')}.py"
    sys.modules[name] = module
    try:
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    setattr(package, short, module)
    return module


def historical_modules():
    _require_assertions_enabled()
    import chowder_batch  # noqa: F401 - initializes only the namespace package

    package_paths = list(chowder_batch.__path__)
    if (len(package_paths) != 1
            or Path(package_paths[0]).resolve() != (REPO / "chowder_batch").resolve()):
        raise ValueError("historical namespace source path is not uniquely original")
    raw = source_bytes()
    prefix = "chowder_batch/"
    names = list(PINNED)
    # Source SHA equality is insufficient if importlib executes a preloaded
    # module or timestamp-valid but substituted .pyc. Execute every transitive
    # original Python dependency directly from the authenticated Git bytes.
    # Strict import order: context budget -> original guard -> ctx64k/Q8 ->
    # historical runner -> scorer. A preimported auxiliary fails closed.
    def historical_name(path: str) -> str:
        return "chowder_batch." + path.removeprefix(prefix).removesuffix(".py")

    budget_path = "chowder_batch/reap_context_budget_20261026.py"
    q4_path = "chowder_batch/reap_q4_automationbench_ctx64k_20261026.py"
    q8_path = "chowder_batch/reap_q8_automationbench_kaggle_20261026.py"
    budget = _load(historical_name(budget_path), raw[budget_path])
    guard = _load("chowder_batch." + names[0].removeprefix(prefix).removesuffix(".py"), raw[names[0]])
    q4 = _load(historical_name(q4_path), raw[q4_path])
    q8 = _load(historical_name(q8_path), raw[q8_path])
    runner = _load("chowder_batch." + names[1].removeprefix(prefix).removesuffix(".py"), raw[names[1]])
    audit = _load("chowder_batch." + names[2].removeprefix(prefix).removesuffix(".py"), raw[names[2]])
    if (guard.POLICY_NAME != "guard.v3" or runner.q4.POLICY_NAME != "guard.v3"
            or runner.start_guard is not guard.start_guard
            or runner.q4 is not q4 or runner.source_ids is not q8
            or q4.assess is not budget.assess
            or q4.trajectory_from_native_log is not budget.trajectory_from_native_log
            or audit.run is not runner):
        raise ValueError("historical guard/runner/scorer identities are mixed")
    return runner, audit


def _assert_ordinary_tree(root: Path) -> None:
    """Reject redirected evidence at *every* depth before original scoring.

    On Windows, an NTFS junction/mount point can be a directory with
    ``is_symlink() == False``. Checking only the owner and native logs leaves
    nested task evidence reachable through an unreviewed reparse target.
    ``lstat`` inspects the link itself rather than following it; the Windows
    reparse attribute covers junctions and other non-symlink redirects.
    """
    pending = [root]
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    while pending:
        path = pending.pop()
        try:
            metadata = path.lstat()
            if (stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & reparse_flag):
                raise ValueError("original guard.v3 evidence reparse point: " + str(path))
            if stat.S_ISDIR(metadata.st_mode):
                pending.extend(path.iterdir())
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError("original guard.v3 evidence nonregular entry: " + str(path))
        except OSError as exc:
            raise ValueError("original guard.v3 evidence tree unreadable: " + str(path)) from exc


def inspect(root: Path = ROOT) -> dict:
    _require_assertions_enabled()
    if root != ROOT or not root.is_dir() or root.is_symlink():
        raise ValueError("only the original ordinary guard.v3 evidence root is allowed")
    _assert_ordinary_tree(root)
    owner = root / "owner.json"
    if owner.is_symlink() or not owner.is_file() or sha(owner.read_bytes()) != OWNER_SHA:
        raise ValueError("original guard.v3 owner SHA drift")
    from chowder_batch.reap_original600_guardv3_stable_snapshot_20261029 import stable_snapshot

    # The historical scorer opens the same files repeatedly for hashing,
    # JSON parsing and WorldState reconstruction. Never let those reads race
    # with changes to F:; run them against a handle-locked, byte-identical
    # private copy. The context manager rechecks F: after scoring, and always
    # removes the temporary copy. It does not write into original evidence.
    with stable_snapshot(root, REPO) as stable:
        runner, audit = historical_modules()
        result = audit.inspect(stable)
        if (result["owner_sha256"] != OWNER_SHA
                or result["official_complete_600_claim_allowed"] !=
                   (result["completed_scored_tasks"] == 600)
                or result["noncomparable_to_frozen_four_slot"] is not True
                or result["frozen_gen0_status"] != "BLOCKED"
                or result["chosen"] is not None
                or result["release_approved"] is not False):
            raise ValueError("historical result has invalid provenance or release state")
        sessions = []
        for path in sorted(stable.iterdir()):
            if re.fullmatch(r"(native-session-\d+\.private\.log|guard-session-\d+\.private\.jsonl)", path.name):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("linked/nonfile original native session")
                sessions.append({"name": path.name, "sha256": runner._digest(path)})
    # A contributor may add a junction while the original scorer is reading.
    # Recheck the complete tree before reporting any authenticated result.
    _assert_ordinary_tree(root)
    return {**result, "schema": SCHEMA, "historical_source_commit": COMMIT,
            "historical_source_sha256": PINNED,
            "historical_guard_policy": "guard.v3", "session_evidence": sessions,
            "checkpoint_may_span_distinct_native_sessions": True,
            "native_sessions_pooled_for_frozen_comparison": False,
            "scored_from_handle_locked_original_evidence_snapshot": True,
            "research_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = inspect()
        if args.snapshot:
            if (not args.snapshot.is_absolute() or args.snapshot.exists()
                    or args.snapshot.is_symlink()
                    or args.snapshot.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive F: snapshot path required")
            with args.snapshot.open("xb") as stream:
                stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "session_evidence")},
                         sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "REFUSED_HISTORICAL_SOURCE_OR_EVIDENCE",
                          "reason": str(exc)[:380], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
