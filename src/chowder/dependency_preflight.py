from __future__ import annotations

import importlib.util


class MissingDependencyError(RuntimeError):
    """A package required for this run is not importable in this
    environment. Raised during the cheap preflight/profile step, before any
    GPU-hours are reserved or a worker subprocess is spawned -- discovering
    a missing dependency there is a config-time mistake, not a training
    failure, and should look like one rather than being reported deep
    inside a spawned subprocess after paying for the reservation and the
    process startup.
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
