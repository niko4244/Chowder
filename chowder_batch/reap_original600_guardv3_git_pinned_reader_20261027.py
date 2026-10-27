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
from pathlib import Path
import re
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
)
SCHEMA = "chowder.reap-original600-guardv3-git-pinned-independent.v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{COMMIT}:{path}"], cwd=REPO, timeout=20
    )


def source_bytes() -> dict[str, bytes]:
    result = {}
    for path, expected in PINNED.items():
        raw = git_bytes(path)
        if sha(raw) != expected:
            raise ValueError("historical Git source SHA mismatch: " + path)
        result[path] = raw
    for path in IMPORTED_AUXILIARY:
        if sha(git_bytes(path)) != sha((REPO / path).read_bytes()):
            raise ValueError("historical scorer auxiliary dependency drift: " + path)
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
    import chowder_batch  # noqa: F401 - initializes only the namespace package

    raw = source_bytes()
    prefix = "chowder_batch/"
    names = list(PINNED)
    # Strict import order: the historical runner must import historical guard.
    guard = _load("chowder_batch." + names[0].removeprefix(prefix).removesuffix(".py"), raw[names[0]])
    runner = _load("chowder_batch." + names[1].removeprefix(prefix).removesuffix(".py"), raw[names[1]])
    audit = _load("chowder_batch." + names[2].removeprefix(prefix).removesuffix(".py"), raw[names[2]])
    if (guard.POLICY_NAME != "guard.v3" or runner.q4.POLICY_NAME != "guard.v3"
            or runner.start_guard is not guard.start_guard
            or audit.run is not runner):
        raise ValueError("historical guard/runner/scorer identities are mixed")
    return runner, audit


def inspect(root: Path = ROOT) -> dict:
    if root != ROOT or not root.is_dir() or root.is_symlink():
        raise ValueError("only the original ordinary guard.v3 evidence root is allowed")
    owner = root / "owner.json"
    if owner.is_symlink() or not owner.is_file() or sha(owner.read_bytes()) != OWNER_SHA:
        raise ValueError("original guard.v3 owner SHA drift")
    runner, audit = historical_modules()
    result = audit.inspect(root)
    if (result["owner_sha256"] != OWNER_SHA
            or result["official_complete_600_claim_allowed"] !=
               (result["completed_scored_tasks"] == 600)
            or result["noncomparable_to_frozen_four_slot"] is not True
            or result["frozen_gen0_status"] != "BLOCKED"
            or result["chosen"] is not None
            or result["release_approved"] is not False):
        raise ValueError("historical result has invalid provenance or release state")
    sessions = []
    for path in sorted(root.iterdir()):
        if re.fullmatch(r"(native-session-\d+\.private\.log|guard-session-\d+\.private\.jsonl)", path.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError("linked/nonfile original native session")
            sessions.append({"name": path.name, "sha256": runner._digest(path)})
    return {**result, "schema": SCHEMA, "historical_source_commit": COMMIT,
            "historical_source_sha256": PINNED,
            "historical_guard_policy": "guard.v3", "session_evidence": sessions,
            "checkpoint_may_span_distinct_native_sessions": True,
            "native_sessions_pooled_for_frozen_comparison": False,
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
