"""The kernel-side half: verify, run, and record one dispatched attempt.

``chowder.growth.kaggle_cli_transport`` stages a generated entry script into
the kernel; after installing the pinned Chowder commit, that script calls
:func:`run_kernel_job` with the job document the transport wrote. Everything
in this module also runs on a laptop, so each guarantee is unit-tested
without a GPU, a token or Kaggle:

* the installed source is the commit the job declared, read back from the
  package's own ``direct_url.json`` -- never assumed from the install
  command (the cross-check ``kaggle/bootstrap_environment.py`` established);
* every declared input is re-hashed at its declared kernel path and echoed in
  ``input_verification``; a missing or moved input is an error record, never
  a quiet run. A declaration may name several candidate paths -- a Kaggle
  dataset mounts at ``/kaggle/input/<slug>/`` or, on a slug conflict, at
  ``/kaggle/input/<owner>/<slug>/`` -- and the input is the candidate whose
  bytes match the declared digest;
* the payload command runs against exactly those inputs, and its exit status
  becomes the record's state;
* the artifact manifest is every file under the output directory except the
  record itself and bytecode caches, each with a sha256 and byte count;
* the environment (python version, resolved packages, model commit) is the
  kernel's own reading, never reconstructed by the dispatcher;
* a declared resume reports the search's vocabulary -- a marker file written
  by the command may say ``resumed``; absent or malformed, the honest answer
  is ``not-a-resume``, and the backend fails the attempt on it.

The record is written to ``<output>/job-record.json`` even on failure: a
failed attempt without evidence is a dropped failure.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "JOB_RECORD_NAME",
    "KERNEL_JOB_SPEC_NAME",
    "RESUME_STATE_NAME",
    "RESUME_STATES",
    "install_spec",
    "installed_commit",
    "package_versions",
    "run_kernel_job",
    "sha256_file",
]

#: Where the kernel writes its record; the transport consumes it and never
#: copies it into the artifact directory (a record cannot hash itself).
JOB_RECORD_NAME = "job-record.json"
#: The job document the transport stages beside the entry script.
KERNEL_JOB_SPEC_NAME = "chowder-job-spec.json"
#: A command that really loaded a declared checkpoint writes this marker; its
#: absence is reported as ``not-a-resume``, never as a silent restart.
RESUME_STATE_NAME = "resume-state.json"
RESUME_STATES = ("resumed", "not-a-resume")

CommandRunner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[str]"]

_SHA256_HEX = frozenset("0123456789abcdef")


def sha256_file(path: str | Path) -> tuple[str, int]:
    """The sha256 and byte count of a file (the one hashing convention)."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_commit(commit_sha: str) -> str:
    commit = str(commit_sha).strip().lower()
    if len(commit) != 40 or any(character not in _SHA256_HEX for character in commit):
        raise ValueError(
            f"commit_sha must be a full 40-character commit sha, got {commit_sha!r}"
        )
    return commit


def install_spec(
    repository: str, commit_sha: str, *, extras: Sequence[str] = ("train",)
) -> str:
    """The pip requirement that installs Chowder at exactly this commit."""
    commit = _validate_commit(commit_sha)
    url = str(repository).strip()
    if not url:
        raise ValueError("repository must be non-empty")
    if not url.endswith(".git"):
        url = f"{url}.git"
    if "://" not in url and not url.startswith("git@"):
        raise ValueError(f"repository must be a git URL, got {repository!r}")
    extra = f"[{','.join(str(item) for item in extras)}]" if extras else ""
    return f"chowder-ai{extra} @ git+{url}@{commit}"


def installed_commit() -> str | None:
    """The VCS commit the installed ``chowder-ai`` actually resolved to."""
    try:
        distribution = importlib.metadata.distribution("chowder-ai")
    except importlib.metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        direct_url = json.loads(text)
    except json.JSONDecodeError:
        return None
    return (direct_url.get("vcs_info") or {}).get("commit_id")


def package_versions() -> dict[str, str]:
    """Every installed distribution's resolved version, sorted by name."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = metadata["Name"] if metadata else None
        if name and distribution.version:
            versions[str(name)] = str(distribution.version)
    return dict(sorted(versions.items()))


def _default_command_runner(
    command: Sequence[str], working_dir: Path
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(command), cwd=str(working_dir), capture_output=True, text=True
    )


def _tail(text: str | None, limit: int = 2000) -> str:
    return (text or "").strip()[-limit:]


def _candidate_paths(location: Any) -> tuple[Path, ...]:
    """The declared kernel path(s) an input may sit at.

    The dispatcher cannot know which mount path Kaggle chooses before the
    mount happens, so a declaration may name several candidates; the input is
    the candidate whose bytes match the declared digest, never merely the
    first one that exists.
    """
    if isinstance(location, str):
        return (Path(location),) if location.strip() else ()
    if isinstance(location, (list, tuple)):
        return tuple(
            Path(str(item)) for item in location if str(item).strip()
        )
    return ()


def _resume_state(output_root: Path, declared_resume_from: str | None) -> str | None:
    if declared_resume_from is None:
        return None
    marker = output_root / RESUME_STATE_NAME
    if marker.is_file():
        try:
            document = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = None
        if isinstance(document, Mapping) and document.get("resume_state") in RESUME_STATES:
            return str(document["resume_state"])
    # No proof the declared checkpoint loaded: the honest answer is that this
    # was not a resume, which the search reads as a lineage stop.
    return "not-a-resume"


def run_kernel_job(
    document: Mapping[str, Any],
    *,
    working_dir: str | Path = "/kaggle/working",
    installed_commit_fn: Callable[[], str | None] = installed_commit,
    package_versions_fn: Callable[[], Mapping[str, str]] = package_versions,
    command_runner: CommandRunner = _default_command_runner,
) -> dict[str, Any]:
    """Verify, run and record one attempt; returns the record it wrote."""
    if not isinstance(document, Mapping):
        raise ValueError("the kernel job document must be a mapping")
    spec = document.get("spec")
    kernel = document.get("kernel")
    if not isinstance(spec, Mapping) or not isinstance(kernel, Mapping):
        raise ValueError("the kernel job document must carry 'spec' and 'kernel' mappings")

    output_root = Path(kernel.get("output_dir") or working_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    declared_commit = str((spec.get("source") or {}).get("commit_sha", ""))
    observed_commit = installed_commit_fn()
    if observed_commit is None:
        errors.append(
            "chowder-ai is not installed in this kernel, so the running source "
            "cannot be identified"
        )
    elif observed_commit != declared_commit:
        errors.append(
            f"the installed chowder commit {observed_commit!r} does not equal "
            f"the declared {declared_commit!r}"
        )

    input_paths = kernel.get("input_paths") or {}
    if not isinstance(input_paths, Mapping):
        input_paths = {}
        errors.append("kernel.input_paths must be a mapping of declared name to kernel path")
    input_verification: list[list[str]] = []
    input_locations: dict[str, str] = {}
    for entry in spec.get("inputs") or []:
        if not isinstance(entry, Mapping):
            errors.append(f"declared input entry is not an object: {entry!r}")
            continue
        name = str(entry.get("name", ""))
        declared_sha = str(entry.get("sha256", ""))
        declared_bytes = entry.get("bytes")
        candidates = _candidate_paths(input_paths.get(name))
        if not candidates:
            errors.append(f"declared input {name!r} has no kernel path in input_paths")
            continue
        matched: Path | None = None
        differences: list[str] = []
        for path in candidates:
            if not path.is_file():
                differences.append(f"{path} (absent)")
                continue
            sha256, size = sha256_file(path)
            if sha256 == declared_sha and size == declared_bytes:
                matched = path
                break
            differences.append(f"{path} ({sha256}, {size} bytes)")
        if matched is None:
            errors.append(
                f"declared input {name!r} moved: no declared kernel path "
                f"carries its declared bytes ({declared_sha}, "
                f"{declared_bytes!r} bytes); observed: {'; '.join(differences)}"
            )
            continue
        input_verification.append([name, declared_sha])
        input_locations[name] = str(matched)

    resume_from = spec.get("resume_from")
    resume_state = _resume_state(output_root, resume_from)

    command = kernel.get("command") or []
    command_record: dict[str, Any] | None = None
    if command:
        if not isinstance(command, (list, tuple)) or not all(
            isinstance(part, str) and part for part in command
        ):
            errors.append("kernel.command must be a list of non-empty strings")
        else:
            completed = command_runner(command, output_root)
            command_record = {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }
            if completed.returncode != 0:
                errors.append(
                    f"the payload command exited {completed.returncode}: "
                    f"{_tail(completed.stderr) or _tail(completed.stdout)}"
                )

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if (
            relative == JOB_RECORD_NAME
            or "__pycache__" in path.parts
            or relative.endswith((".pyc", ".pyo"))
        ):
            continue
        sha256, size = sha256_file(path)
        artifacts.append({"path": relative, "sha256": sha256, "bytes": size})

    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "packages": dict(sorted(package_versions_fn().items())),
        "recorded_from": "kernel",
    }
    if kernel.get("model_commit"):
        environment["model_commit"] = str(kernel["model_commit"])

    record: dict[str, Any] = {
        "spec_id": str(spec.get("spec_id", "")),
        "state": "complete" if not errors else "error",
        "source_commit_sha": observed_commit or "",
        "input_verification": input_verification,
        "input_locations": input_locations,
        "artifact_manifest": artifacts,
        "environment": environment,
        "resume_state": resume_state,
        "resume_from": resume_from,
        "mounts": list(spec.get("mounts") or []),
        "command": command_record,
        "error": "; ".join(errors),
    }
    (output_root / JOB_RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
