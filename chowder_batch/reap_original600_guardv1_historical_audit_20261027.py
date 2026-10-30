"""Reaudit the ACTUAL original guard.v1 AutomationBench600 receipts after guard.v2.

The running new guard.v2 source has a different policy identity. It must not
silently relabel the 43 original guard.v1 task receipts as guard.v2 or refuse
their immutable, independently verifiable history. This read-only wrapper
pins the original source code to immutable Git and the exact source owner,
then temporarily passes the historical policy name solely to the original
independent JSON receipt scorer. It never serves a model or resumes a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from chowder_batch import reap_original600_live_independent_audit_20261027 as independent

SCHEMA = "chowder.original600-guard-v1-frozen-source-history.v1"
ORIGINAL_GIT = "ea453b2f9ea85e8729673e401b0d6caec49986b7"
OLD_OWNER_SHA = "849309859a5c1ef8736ee67668b33d8fbbf1dc3334b246451af4c4e5d46f220a"
ORIGINAL_RUN = Path("F:/reap-v2-q4-original-automationbench600-checkpointed-20261027-0953")
SOURCE_HASHES = {
    "chowder_batch/reap_context_guard_proxy_20261026.py":
        "7b4fc9799c581a76c9a84d764871e5ff47cb286759addf12c1231885f44d0fc8",
    "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py":
        "38e7646ad259afdeaadc308d0e6f0da7d44dd3d93ccf59a43e90fccbdff0ede7",
    "chowder_batch/reap_original600_live_independent_audit_20261027.py":
        "89b06b2c908f80f2f9f8ed5802fbb32fd8888d1a7e4113e454b334b63e422539",
}


def _source_pin(path: str, *, require_worktree_match: bool) -> None:
    repo = Path(__file__).resolve().parent.parent
    original = subprocess.run(["git", "show", ORIGINAL_GIT + ":" + path],
                              cwd=repo, capture_output=True, timeout=12, check=True).stdout
    frozen_hash = hashlib.sha256(original).hexdigest()
    if frozen_hash != SOURCE_HASHES[path]:
        raise ValueError("immutable original Git source pin does not match registered source")
    if require_worktree_match and hashlib.sha256((repo / path).read_bytes()).hexdigest() != frozen_hash:
        raise ValueError("read-only independent original scorer/runner source drift")
    if path.endswith("reap_context_guard_proxy_20261026.py"):
        if b'POLICY_NAME = "guard.v1"' not in original:
            raise ValueError("source Git commit did not contain historical guard.v1 policy")


def audit_original(root: Path = ORIGINAL_RUN) -> dict:
    if root != ORIGINAL_RUN or not root.is_dir() or root.is_symlink():
        raise ValueError("not the registered genuine original guard.v1 run")
    for path in SOURCE_HASHES:
        _source_pin(path, require_worktree_match=not path.endswith(
            "reap_context_guard_proxy_20261026.py"))
    owner_file = root / "owner.json"
    if owner_file.is_symlink() or hashlib.sha256(owner_file.read_bytes()).hexdigest() != OLD_OWNER_SHA:
        raise ValueError("immutable historical original owner SHA changed")
    owner = independent.run._json(owner_file)
    if owner.get("context_guard_policy") != "guard.v1":
        raise ValueError("historical original 600 guard policy changed")
    before = independent.run.q4.POLICY_NAME
    try:
        independent.run.q4.POLICY_NAME = "guard.v1"
        result = independent.inspect(root)
    finally:
        independent.run.q4.POLICY_NAME = before
    if result["owner_sha256"] != OLD_OWNER_SHA or not result["noncomparable_to_frozen_four_slot"]:
        raise ValueError("original independent scored-source contract changed")
    return {
        **result,
        "historical_source_lock": SCHEMA,
        "historical_guard_policy": "guard.v1",
        "new_guard_v2_compatible_for_pooling": False,
        "reproduced_original_source_git_commit": ORIGINAL_GIT,
        "original_guard_source_sha256": SOURCE_HASHES[
            "chowder_batch/reap_context_guard_proxy_20261026.py"],
    }


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, help="Exclusive new F: receipt; no original file edited")
    args = cli.parse_args()
    try:
        result = audit_original()
        if args.snapshot is not None:
            if (not args.snapshot.is_absolute() or args.snapshot.exists()
                    or args.snapshot.is_symlink()
                    or args.snapshot.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("exclusive new F: evidence path required")
            with args.snapshot.open("xb") as stream:
                stream.write(independent.run._canonical(result))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                         sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "REFUSED_UNTRUSTED_GUARD_V1_HISTORICAL_AUDIT",
                          "reason": str(error)[:400],
                          "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
