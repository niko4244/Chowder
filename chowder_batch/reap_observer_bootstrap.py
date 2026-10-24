"""Internal no-model startup barrier for the Windows supervised REAP observer.

The parent first assigns this very lightweight process to a kill-on-job-close
Windows Job Object, THEN creates a ready marker. It is not a standalone
observer entry point and must never self-start any background workload.
"""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path


def _ensure_package_root_on_path(script: Path) -> Path:
    """Make `chowder_batch.*` importable for the script we are about to run.

    The supervisor starts this bootstrap by *script path*, so CPython puts only
    ``<repo>/chowder_batch`` on ``sys.path`` -- not the repository root. The
    observer's cross-module imports are package-qualified
    (``from chowder_batch.reap_v2_source_identity import ...``), so without the
    repository root every supervised launch died at startup with
    ``ModuleNotFoundError: No module named 'chowder_batch'`` before any layer ran.

    An explicit ``python -m chowder_batch.reap_observe`` invocation happens to
    work because module form puts the CWD on the path; that is why this only
    broke on the supervised path. Restore the same importability here rather
    than rewriting the observer's imports.
    """
    package_root = script.resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return package_root


def main(argv: list[str] | None = None) -> int:
    parts = list(sys.argv[1:] if argv is None else argv)
    if len(parts) < 3:
        raise SystemExit("internal bootstrap requires ready marker and observer command")
    ready = Path(parts[0])
    script = Path(parts[1])
    observer_args = parts[2:]
    if script.name != "reap_observe.py" or script.resolve().parent != Path(__file__).resolve().parent:
        raise SystemExit("refusing non-observer command in internal bootstrap")
    deadline = time.monotonic() + 30
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise SystemExit("observer owner never completed job-object admission")
        time.sleep(0.1)
    _ensure_package_root_on_path(script)
    # The process PID is retained; the parent's Job Object will kill this
    # exact worker on unexpected supervisor exit. No shell or grandchildren.
    sys.argv = [str(script), *observer_args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
