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
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


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
