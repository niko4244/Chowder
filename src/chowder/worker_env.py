"""Make subprocess workers run the same Chowder as the process that launched them.

Every Chowder worker runs in a fresh interpreter (`python -m chowder.<worker>`, or
the Unsloth worker by file path). A fresh interpreter resolves `import chowder`
through whatever the environment says, and with an editable install that is a
`.pth` file pointing at exactly ONE checkout. So a worker launched from any other
checkout or worktree silently imports that other checkout's code: parent and child
run different Chowders.

This was found, not hypothesised. On the development machine the editable install
(`__editable__.chowder_ai-0.3.0.pth`) points at `C:\\Users\\nikma\\Chowder\\src`,
the main checkout on an older branch. Run from a worktree, the gated real-ML smoke
test trained successfully and then the evaluator worker crashed with
`EvalSuiteSpec.__init__() got an unexpected keyword argument 'canonical_rendering'`:
the parent serialised a spec field only the worktree's code defines, and the child,
running main's code, could not read it.

The crash was the lucky case. Where the versions happened to be wire-compatible --
the training worker was -- nothing failed, and a worktree run went through
Chowder's lifecycle training and evaluating with a *different code version than the
one under review*, recording results against the wrong code.

`worker_env()` prepends the source root of the `chowder` package THIS process
imported to `PYTHONPATH`, which Python searches before site-packages and `.pth`
entries, so the child resolves `chowder` to the same code. Nothing else changes:
the interpreter, site-packages and every other variable are inherited as before.

That guarantee is still environmental: it holds only while the launch is wired
up. P4 adds the *evidence* half — `chowder_source_identity()` records a content
digest of the package this process actually imported, and a worker's `main()`
calls `verify_source_identity()` with the identity its controller declared
BEFORE reading its spec. If the two disagree (env broken, wrong checkout, a
competing editable install), the worker refuses instead of running against the
wrong code and recording results against it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class WorkerIdentityError(RuntimeError):
    """A worker's imported source does not match the identity its parent declared."""


def identity_from_tree(package_root: Path) -> dict[str, Any]:
    """Content identity of a package tree: hash of paths + bytes, pycache excluded.

    The digest is over *content*: the same files under a different root hash
    identically (the root is recorded, not mixed into the digest), while one
    changed byte in one module changes everything. `__pycache__` and `.pyc`
    artifacts are excluded — compiled caches are derivable, not source.
    """
    root = Path(package_root).resolve()
    if not root.is_dir():
        raise WorkerIdentityError(f"package root is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or rel.endswith((".pyc", ".pyo")):
            continue
        entries.append({"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not entries:
        raise WorkerIdentityError(f"package tree contains no source files: {root}")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"source_root": str(root), "source_sha256": digest, "files": len(entries)}


def chowder_source_identity() -> dict[str, Any]:
    """Identity of the `chowder` package THIS process actually imported."""
    import chowder

    return identity_from_tree(Path(chowder.__file__).resolve().parent)


def verify_source_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Child-side check: refuse unless the declared identity matches reality.

    Called first in every worker `main()`, before the worker reads its spec —
    the plan's "fail before training when they disagree".
    """
    if not isinstance(expected, Mapping):
        raise WorkerIdentityError("invalid chowder identity: expected a mapping")
    missing = {"source_root", "source_sha256"} - set(expected)
    if missing:
        raise WorkerIdentityError(
            f"invalid chowder identity: missing fields {sorted(missing)}"
        )
    sha = expected["source_sha256"]
    if not isinstance(sha, str) or len(sha) != 64:
        raise WorkerIdentityError("invalid chowder identity: source_sha256 must be 64 hex chars")
    actual = chowder_source_identity()
    if actual["source_sha256"] != sha:
        raise WorkerIdentityError(
            "source identity mismatch: the worker imported different chowder code "
            f"than its controller declared (expected {sha[:12]}…, actual "
            f"{actual['source_sha256'][:12]}…) — refusing to run against the wrong code"
        )
    if Path(str(expected["source_root"])).resolve() != Path(actual["source_root"]).resolve():
        raise WorkerIdentityError(
            "source identity mismatch: the worker's chowder package lives at "
            f"{actual['source_root']}, not the declared {expected['source_root']}"
        )
    return actual


def chowder_source_root() -> Path:
    """The directory containing the `chowder` package this process imported."""
    import chowder

    return Path(chowder.__file__).resolve().parent.parent


def worker_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a Chowder worker subprocess: the parent's code, first.

    `extra` is applied last, for per-launch variables such as
    `PYTHONUNBUFFERED`. It must not be used to set `PYTHONPATH` -- that would
    silently undo the one thing this function exists to guarantee -- so doing so
    raises rather than being quietly honoured.
    """
    if extra and "PYTHONPATH" in extra:
        raise ValueError(
            "worker_env(extra=...) must not set PYTHONPATH; it would override the "
            "guarantee that the worker imports the same chowder as its parent"
        )
    env = dict(os.environ)
    root = str(chowder_source_root())
    inherited = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *inherited])
    if extra:
        env.update(extra)
    return env
