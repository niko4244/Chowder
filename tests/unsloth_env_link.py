"""Reuse one persistent Unsloth environment from a FRESH per-run work directory.

`UnslothPeftExecutor` resolves its interpreter as
`<project work_dir>/.chowder/envs/unsloth` (`_isolated_python`), so the multi-GB
Unsloth environment is bound to the project directory. That coupling made both
real Unsloth tests effectively dead:

* `test_unsloth_peft_real.py` looked for the environment in pytest's throwaway
  `tmp_path`, which never contains one, so it always skipped;
* `test_project_runner_repair_unsloth.py` worked around it by using the
  persistent environment root AS its work directory. It passed once (PR #135) and
  could never pass again: its registry, runs and repairs were written into that
  shared root, so every later run failed in 0.5s with
  `RegistryInvariantError: duplicate persisted experiment id: real-unsloth-sft`.

The executor only ever *reads* through that path -- it looks up the interpreter
and hashes `chowder-unsloth-manifest.json` -- so linking the persistent
environment into a fresh work directory is safe and needs no product change. Each
run gets its own registry; the environment is shared; earlier runs' artifacts in
the persistent root are left untouched.

A directory junction is used on Windows because it needs no administrator
privilege, unlike `os.symlink`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from chowder.unsloth_env import unsloth_env_dir, unsloth_python

ENV_ROOT_VAR = "CHOWDER_REAL_UNSLOTH_ENV_ROOT"


def link_persistent_unsloth_env(work_dir: Path) -> Path | None:
    """Make `work_dir`'s Unsloth env dir point at the persistent one.

    Returns the linked env dir, or None when `CHOWDER_REAL_UNSLOTH_ENV_ROOT` is
    unset or holds no provisioned environment -- callers skip in that case.
    """
    raw = os.environ.get(ENV_ROOT_VAR)
    if not raw:
        return None
    persistent = unsloth_env_dir(Path(raw).expanduser().resolve())
    if not unsloth_python(persistent).is_file():
        return None
    target = unsloth_env_dir(work_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(persistent), str(target))
    else:
        os.symlink(persistent, target, target_is_directory=True)
    return target
