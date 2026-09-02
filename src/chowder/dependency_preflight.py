from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


class MissingDependencyError(RuntimeError):
    """A package required for this run is not importable in this
    environment. Raised during the cheap preflight/profile step, before any
    GPU-hours are reserved or a worker subprocess is spawned -- discovering
    a missing dependency there is a config-time mistake, not a training
    failure, and should look like one rather than being reported deep
    inside a spawned subprocess after paying for the reservation and the
    process startup.
    """


class InsufficientDiskSpaceError(RuntimeError):
    """Free disk space at the run's work_dir is below the configured
    minimum. Raised at the same preflight point as MissingDependencyError,
    before any GPU-hours are reserved -- an out-of-space failure partway
    through a model download or checkpoint write is far more expensive to
    discover than a cheap `shutil.disk_usage` check up front.
    """


_QLORA_PACKAGES = ("bitsandbytes",)


def _missing(packages: tuple[str, ...]) -> list[str]:
    return [name for name in packages if importlib.util.find_spec(name) is None]


def check_dependencies(
    *, packages: tuple[str, ...], quantization: str, label: str
) -> None:
    """Raise MissingDependencyError if any package `label` needs isn't
    importable. `packages` is the base set this workload always needs;
    bitsandbytes is checked in addition only when quantization == "4bit",
    since it's an optional extra ([qlora]) not required otherwise.
    """
    missing = _missing(packages)
    if quantization == "4bit":
        missing += _missing(_QLORA_PACKAGES)
    if not missing:
        return
    extras = "chowder-ai[train]"
    if quantization == "4bit":
        extras += " and chowder-ai[qlora]"
    raise MissingDependencyError(
        f"{label} is missing required package(s): {', '.join(missing)}; install {extras}"
    )


def check_disk_space(*, path: str | Path, minimum_free_gb: float, label: str) -> None:
    """Raise InsufficientDiskSpaceError if `path`'s filesystem has less than
    `minimum_free_gb` free. `path` need not exist yet -- the nearest existing
    ancestor directory is measured, matching how work_dir/registry_path are
    created lazily elsewhere in this codebase.
    """
    if minimum_free_gb <= 0:
        return
    target = Path(path)
    while not target.exists():
        parent = target.parent
        if parent == target:
            break
        target = parent
    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < minimum_free_gb:
        raise InsufficientDiskSpaceError(
            f"{label} requires at least {minimum_free_gb:.2f} GB free at "
            f"{target}, but only {free_gb:.2f} GB is available"
        )
