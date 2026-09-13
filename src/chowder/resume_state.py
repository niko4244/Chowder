"""Does this checkpoint actually contain resumable training state?

The completed GSM8K rerun left an **empty** trainer-state directory, so the run
that produced an adapter could not be continued -- and its artifacts could not
say whether the state had ever been written. The same gap exists inside
``Trainer.train(resume_from_checkpoint=...)``: when ``optimizer.pt`` is missing,
Transformers restores the model weights, silently initialises a fresh optimizer,
and continues. Momentum, the LR-scheduler position, and the RNG stream all start
over while the run looks resumed.

So every checkpoint is *inventoried* before it is trusted, and an inventory that
is not complete is refused rather than treated as an implicit fresh start. Two
rules keep that honest:

* An empty file or one that fails a read probe is **not** a present file. A
  ``trainer_state.json`` without a usable ``global_step`` is treated as missing,
  with the reason recorded -- otherwise a corrupt checkpoint would pass as
  complete and the resume point would be guessed.
* A resume is only accepted if it can be **witnessed**: the restore point has to
  be known and the run has to advance past it. "It ran" is not "it resumed".

This metadata inventory probes one byte per file before a model load. Tensor
payloads are never deserialized here: nonempty corrupt contents remain unverified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Trainer's checkpoint state pieces, by the name this module reports them under.
CHECKPOINT_STATE_FILES: Mapping[str, str] = {
    "optimizer": "optimizer.pt",
    "scheduler": "scheduler.pt",
    "scaler": "scaler.pt",
    "trainer_state": "trainer_state.json",
    "training_args": "training_args.bin",
    "rng_state": "rng_state.pth",
}

#: Pieces without which continuing optimization would be a fresh start wearing
#: a resumed run's name: the optimizer moments, the scheduler position, and the
#: recorded step.
REQUIRED_STATE_PIECES: tuple[str, ...] = ("optimizer", "scheduler", "trainer_state")

#: Accelerate writes one of these per process; they carry the dataloader/RNG
#: stream position, so they are inventoried (and can be required) but never
#: guessed at.
RANDOM_STATE_GLOB = "random_states_*.pkl"


class IncompleteCheckpointError(RuntimeError):
    """A checkpoint cannot be resumed from without silently starting over."""


@dataclass(frozen=True)
class CheckpointInventory:
    """What a checkpoint directory really contains, measured.

    ``state`` is ``complete`` (every required piece is nonempty and passes a read probe),
    ``partial`` (some are), or ``unreadable`` (none are, or the path is not a
    directory). ``global_step`` is ``None`` when it cannot be read -- an unknown
    resume point, never 0. Completeness does not verify tensor payload integrity.
    """

    directory: str
    state: str
    files: Mapping[str, int]
    present: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    global_step: int | None
    random_state_files: tuple[str, ...] = ()
    #: The total step horizon the checkpoint was produced under, when recorded.
    #: A resumed run declaring a different horizon is a deliberate continuation
    #: with a recomputed remaining LR trajectory -- allowed, but never silently
    #: presented as an exact resume.
    max_steps: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.state == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "state": self.state,
            "files": dict(self.files),
            "present": list(self.present),
            "missing_required": list(self.missing_required),
            "missing_optional": list(self.missing_optional),
            "global_step": self.global_step,
            "max_steps": self.max_steps,
            "random_state_files": list(self.random_state_files),
            "notes": list(self.notes),
        }


def _read_trainer_state(
    path: Path, notes: list[str]
) -> tuple[int | None, int | None]:
    """The checkpoint's (global_step, max_steps), each None if unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        notes.append(f"trainer_state.json is unparseable: {type(exc).__name__}: {exc}")
        return None, None
    if not isinstance(payload, Mapping):
        notes.append("trainer_state.json is not a JSON object")
        return None, None
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int):
        notes.append(
            f"trainer_state.json has no usable global_step (got {step!r}), so the "
            "resume point is unknown"
        )
        step = None
    horizon = payload.get("max_steps")
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        horizon = None
    return (
        None if step is None else int(step),
        None if horizon is None else int(horizon),
    )


def inventory_checkpoint(directory: str | Path) -> CheckpointInventory:
    """Measure a checkpoint's state files. Never raises; never guesses."""
    path = Path(directory).expanduser()
    notes: list[str] = []
    try:
        resolved = str(path.resolve())
    except OSError:  # pragma: no cover - only on an unrepresentable path
        resolved = str(path)

    if not path.is_dir():
        return CheckpointInventory(
            directory=resolved,
            state="unreadable",
            files={},
            present=(),
            missing_required=tuple(REQUIRED_STATE_PIECES),
            missing_optional=tuple(
                piece for piece in CHECKPOINT_STATE_FILES if piece not in REQUIRED_STATE_PIECES
            ),
            global_step=None,
            notes=(f"not a directory: {resolved}",),
        )

    files: dict[str, int] = {}
    for piece, name in CHECKPOINT_STATE_FILES.items():
        candidate = path / name
        if candidate.is_file():
            try:
                with candidate.open("rb") as state_file:
                    if not state_file.read(1):
                        notes.append(f"{name} is empty")
                        continue
                files[piece] = int(candidate.stat().st_size)
            except OSError as exc:
                notes.append(f"{name} is unreadable: {exc}")
                continue

    global_step: int | None = None
    max_steps: int | None = None
    if "trainer_state" in files:
        global_step, max_steps = _read_trainer_state(
            path / CHECKPOINT_STATE_FILES["trainer_state"], notes
        )
        if global_step is None:
            # Present but unusable is missing: a corrupt step file must not let a
            # checkpoint pass as complete with a guessed resume point.
            files.pop("trainer_state", None)

    random_state_files = tuple(sorted(p.name for p in path.glob(RANDOM_STATE_GLOB)))

    present = tuple(sorted(files))
    missing_required = tuple(
        piece for piece in REQUIRED_STATE_PIECES if piece not in present
    )
    missing_optional = tuple(
        piece
        for piece in CHECKPOINT_STATE_FILES
        if piece not in present and piece not in REQUIRED_STATE_PIECES
    )
    if not present:
        state = "unreadable"
    elif missing_required:
        state = "partial"
    else:
        state = "complete"

    return CheckpointInventory(
        directory=resolved,
        state=state,
        files=dict(files),
        present=present,
        missing_required=missing_required,
        missing_optional=missing_optional,
        global_step=global_step,
        random_state_files=random_state_files,
        max_steps=max_steps,
        notes=tuple(notes),
    )


def assert_resumable(
    inventory: CheckpointInventory, *, require_rng: bool = False
) -> None:
    """Refuse a checkpoint that would silently start optimization over.

    ``require_rng`` is for exact-resume equivalence work: the optimizer state
    keeps the continuation meaningful, while the RNG stream is what makes the
    *data order* reproducible.
    """
    if inventory.state == "unreadable":
        raise IncompleteCheckpointError(
            f"cannot resume from {inventory.directory}: unreadable checkpoint "
            "directory (no optimizer, scheduler, or trainer-state file). "
            "Refusing rather than treating it as a fresh start -- 'no state found' "
            "and 'start over' are different runs, and only one of them is what the "
            f"caller asked for. Notes: {list(inventory.notes)}"
        )
    if inventory.missing_required:
        raise IncompleteCheckpointError(
            f"cannot resume from {inventory.directory}: required training state is "
            f"missing ({list(inventory.missing_required)}), present is "
            f"{list(inventory.present)}. Resuming would restore the weights and "
            "silently initialise a FRESH optimizer, scheduler, and RNG stream -- the "
            "run would look resumed while restarting. Notes: "
            f"{list(inventory.notes)}"
        )
    if require_rng and "rng_state" not in inventory.present:
        raise IncompleteCheckpointError(
            f"cannot resume from {inventory.directory}: rng_state.pth is missing, so "
            "the RNG stream position is unknown and the data order would silently "
            "restart. Required for an exact-resume comparison."
        )


def resume_witness(
    inventory: CheckpointInventory,
    *,
    final_global_step: int | None,
    declared_max_steps: int | None = None,
) -> dict[str, Any]:
    """Did the run actually continue from the checkpoint, measured after the fact?

    ``matched`` means the measurements do **not contradict** a real resume: the
    state is complete, the restore point is known, and the run did not end
    *behind* it. It deliberately does not require progress, because resuming a
    checkpoint that already reached its horizon is a legitimate no-op -- Trainer
    restores it, finds nothing left to do, and reports the same step. Refusing
    that would break a real supported workflow (the activation-offload resume
    path resumes the newest checkpoint under an unchanged epoch horizon), so the
    zero-progress cases are *recorded* in ``progress_state`` instead: an accepted
    resume that merely sat there is visible in the evidence.

    What actually catches a silent fresh start is the completeness rule: with
    ``optimizer.pt`` missing, Trainer restores the weights and re-initialises the
    optimizer, and that is refused before training begins.
    """
    restored = inventory.global_step
    final = None if final_global_step is None else int(final_global_step)
    steps_executed: int | None = None
    if restored is not None and final is not None:
        steps_executed = final - restored

    reason: str | None = None
    if not inventory.is_complete:
        unreadable = (
            ", and its resume point is unknown" if restored is None else ""
        )
        reason = (
            "the checkpoint was not complete, so this run resumed without all of the "
            f"state the caller asked for (missing {list(inventory.missing_required)}){unreadable}"
        )
    elif restored is None:
        reason = (
            "the checkpoint's resume point is unknown, so no continuation can be "
            "witnessed from it"
        )
    elif final is None:
        reason = "the run reported no final global step, so no continuation can be witnessed"
    elif steps_executed is not None and steps_executed < 0:
        reason = (
            f"the run ended at step {final}, BEHIND its restore point {restored} -- the "
            "optimizer state it restored cannot belong to a shorter run"
        )

    progress_state: str
    if reason is not None or steps_executed is None:
        progress_state = "unknown"
    elif steps_executed > 0:
        progress_state = "advanced"
    elif (
        declared_max_steps is not None
        and restored is not None
        and restored >= int(declared_max_steps)
    ):
        progress_state = "already_at_horizon"
    else:
        progress_state = "no_further_steps"

    # A different declared horizon is a legitimate longer continuation, but the
    # remaining LR trajectory is recomputed for the new total, so the run is not
    # an exact resume and its evidence has to say so.
    horizon_changed: bool | None = None
    if inventory.max_steps is not None and declared_max_steps is not None:
        horizon_changed = int(declared_max_steps) != int(inventory.max_steps)

    return {
        "requested_checkpoint": inventory.directory,
        "restored_global_step": restored,
        "final_global_step": final,
        "steps_executed": steps_executed,
        "complete": inventory.is_complete,
        "missing_required": list(inventory.missing_required),
        "optimizer_state_present": "optimizer" in inventory.present,
        "rng_state_present": "rng_state" in inventory.present,
        "restored_max_steps": inventory.max_steps,
        "declared_max_steps": None if declared_max_steps is None else int(declared_max_steps),
        "horizon_changed": horizon_changed,
        "progress_state": progress_state,
        "matched": reason is None,
        "reason": reason,
    }
