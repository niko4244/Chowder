from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .backends.transformers_peft import (
    _CHECKPOINT_MANIFEST_NAME,
    TransformersPeftExecutor,
    TransformersPeftRunSpec,
)
from .executors import ExecutionContext
from .provenance import sha256_directory, sha256_file


@dataclass(frozen=True)
class DiscoveredCheckpoint:
    """One Trainer checkpoint found under a project's work_dir, checked
    against the project's CURRENT config -- the same bound-input check
    TransformersPeftExecutor.run() itself performs before trusting a
    resume, just run ahead of time so a caller (the TUI, in particular)
    can show which checkpoints are actually resumable before launching
    anything, rather than the user guessing a path and finding out only
    after a run starts.
    """

    checkpoint_dir: Path
    step: int | None
    mtime: float
    valid: bool
    mismatches: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        if self.valid:
            return "compatible with the current project configuration"
        return "incompatible: " + ", ".join(sorted(self.mismatches))


def _parse_step(name: str) -> int | None:
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def _current_bound_inputs(
    *, resolved_config: Mapping[str, Any], context: ExecutionContext
) -> tuple[dict[str, Any] | None, str | None]:
    """The bound inputs a NEW run under this exact config would record,
    computed the same way a real run does -- including hashing the actual
    dataset/replay/parent-adapter files on disk when the config doesn't
    declare an expected hash, since a TUI-generated project never does.
    Comparing against a config's merely-declared (often absent) hash
    instead of the real file content would make every checkpoint look
    incompatible for exactly the projects most likely to need discovery.

    Returns (bound_inputs, None) on success, or (None, error) if the
    current config can't even be resolved into a runnable spec.
    """
    try:
        spec = TransformersPeftRunSpec.from_resolved_config(
            resolved_config,
            work_dir=context.work_dir,
            output_dir=Path(context.work_dir) / ".chowder" / "_discovery_scratch",
            seed=context.seed,
            hardware=context.hardware,
        )
        if spec.dataset_sha256 is None:
            spec = replace(spec, dataset_sha256=sha256_file(spec.dataset))
        if spec.replay_dataset is not None and spec.replay_sha256 is None:
            spec = replace(spec, replay_sha256=sha256_file(spec.replay_dataset))
        if spec.parent_adapter is not None and spec.parent_adapter_sha256 is None:
            spec = replace(
                spec, parent_adapter_sha256=sha256_directory(spec.parent_adapter)
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return TransformersPeftExecutor._bound_inputs(spec), None


def discover_checkpoints(
    *, work_dir: str | Path, resolved_config: Mapping[str, Any], context: ExecutionContext
) -> tuple[DiscoveredCheckpoint, ...]:
    """Find every Trainer checkpoint under work_dir/.chowder/runs/*/adapter/
    trainer and validate each against the current config's bound inputs.

    Never raises for a malformed or incompatible checkpoint -- those are
    reported with valid=False and a reason, not silently skipped or fatal
    to the whole discovery. Sorted with valid checkpoints first, most
    recent (by Trainer step, then by directory mtime) within each group.
    """
    root = Path(work_dir).expanduser() / ".chowder" / "runs"
    if not root.is_dir():
        return ()

    wanted, config_error = _current_bound_inputs(
        resolved_config=resolved_config, context=context
    )

    discovered: list[DiscoveredCheckpoint] = []
    for trainer_dir in sorted(root.glob("*/adapter/trainer")):
        manifest_path = trainer_dir / _CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, Mapping):
            continue

        for checkpoint_dir in sorted(trainer_dir.glob("checkpoint-*")):
            if not checkpoint_dir.is_dir():
                continue
            if wanted is None:
                mismatches = {
                    "config": {
                        "checkpoint": None,
                        "requested": config_error or "current config is invalid",
                    }
                }
            else:
                mismatches = {
                    key: {"checkpoint": manifest.get(key), "requested": value}
                    for key, value in wanted.items()
                    if manifest.get(key) != value
                }
            discovered.append(
                DiscoveredCheckpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=_parse_step(checkpoint_dir.name),
                    mtime=checkpoint_dir.stat().st_mtime,
                    valid=not mismatches,
                    mismatches=mismatches,
                )
            )

    discovered.sort(key=lambda c: (not c.valid, -(c.step or 0), -c.mtime))
    return tuple(discovered)
