"""P7: a checkpoint either contains resumable training state or it does not.

The hazard this closes is silence, not a crash. `Trainer.train(resume_from_checkpoint=...)`
restores the model weights and then, when `optimizer.pt` is absent, continues
with freshly initialised optimizer state and no complaint: the run *looks*
resumed while momentum, the LR scheduler position, and the RNG stream all
silently start over. The completed rerun's empty trainer-state directory is the
same shape of gap in the evidence.

So the rule is: an incomplete checkpoint is refused, never treated as an
implicit fresh start -- and a resume that cannot be *witnessed* after the fact is
refused too, because "it ran" is not "it resumed".
"""

from __future__ import annotations

import json

import pytest

from chowder.resume_state import (
    REQUIRED_STATE_PIECES,
    CheckpointInventory,
    IncompleteCheckpointError,
    assert_resumable,
    inventory_checkpoint,
    resume_witness,
)


def _write_state(checkpoint_dir, *, step=50, pieces=None, trainer_state=None):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for piece in REQUIRED_STATE_PIECES if pieces is None else pieces:
        name = {
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "trainer_state": "trainer_state.json",
            "scaler": "scaler.pt",
            "rng_state": "rng_state.pth",
            "training_args": "training_args.bin",
        }[piece]
        if piece == "trainer_state":
            payload = (
                {"global_step": step} if trainer_state is None else trainer_state
            )
            (checkpoint_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        else:
            (checkpoint_dir / name).write_bytes(b"state")
    return checkpoint_dir


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_a_complete_checkpoint_reports_the_step_it_stopped_at(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-50", step=50)

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.state == "complete"
    assert inventory.is_complete is True
    assert inventory.global_step == 50
    assert inventory.missing_required == ()
    assert "optimizer" in inventory.present
    assert inventory.files["optimizer"] == len(b"state")


def test_missing_optimizer_state_is_partial_not_a_fresh_start(tmp_path):
    checkpoint = _write_state(
        tmp_path / "checkpoint-50", pieces=["scheduler", "trainer_state"]
    )

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.state == "partial"
    assert inventory.missing_required == ("optimizer",)
    assert inventory.global_step == 50
    with pytest.raises(IncompleteCheckpointError) as excinfo:
        assert_resumable(inventory)
    message = str(excinfo.value)
    assert "optimizer" in message
    # The message has to name the consequence, because this is the failure that
    # otherwise happens in silence inside Trainer.
    assert "fresh" in message.lower()


def test_a_corrupt_trainer_state_is_not_a_present_trainer_state(tmp_path):
    checkpoint = _write_state(
        tmp_path / "checkpoint-50", trainer_state={"global_step": "fifty"}
    )

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.global_step is None
    assert "trainer_state" in inventory.missing_required
    assert inventory.state == "partial"
    assert any("global_step" in note for note in inventory.notes)


def test_an_unparseable_trainer_state_is_reported_by_reason(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-50")
    (checkpoint / "trainer_state.json").write_text("{not json", encoding="utf-8")

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.global_step is None
    assert "trainer_state" in inventory.missing_required
    assert any("unparseable" in note for note in inventory.notes)


def test_an_empty_directory_is_unreadable_not_complete(tmp_path):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.state == "unreadable"
    assert set(inventory.missing_required) == set(REQUIRED_STATE_PIECES)
    with pytest.raises(IncompleteCheckpointError, match="unreadable"):
        assert_resumable(inventory)


def test_a_missing_directory_is_unreadable_not_a_fresh_start(tmp_path):
    inventory = inventory_checkpoint(tmp_path / "nope")
    assert inventory.state == "unreadable"
    assert inventory.global_step is None
    with pytest.raises(IncompleteCheckpointError, match="unreadable"):
        assert_resumable(inventory)


def test_optional_state_is_inventoried_without_blocking_the_verdict(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-50")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "random_states_0.pkl").write_bytes(b"sampler")

    inventory = inventory_checkpoint(checkpoint)

    assert inventory.state == "complete"
    assert inventory.is_complete is True
    assert "rng_state" in inventory.present
    assert "scaler" in inventory.missing_optional
    assert inventory.random_state_files == ("random_states_0.pkl",)


def test_exact_resume_can_require_the_rng_stream(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-50")

    inventory = inventory_checkpoint(checkpoint)
    assert_resumable(inventory)  # optimizer state is what continuation needs
    with pytest.raises(IncompleteCheckpointError, match="rng"):
        assert_resumable(inventory, require_rng=True)

    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    assert_resumable(inventory_checkpoint(checkpoint), require_rng=True)


def test_inventory_serializes_for_durable_evidence(tmp_path):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-50"))
    payload = inventory.to_dict()

    assert payload["state"] == "complete"
    assert payload["global_step"] == 50
    assert isinstance(payload["files"], dict)
    json.dumps(payload)  # must be JSON-serializable evidence


# ---------------------------------------------------------------------------
# the witness: it ran, but did it resume?
# ---------------------------------------------------------------------------


def test_witness_confirms_a_resume_that_actually_restored_state(tmp_path):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-4", step=4))

    witness = resume_witness(inventory, final_global_step=8)

    assert witness["restored_global_step"] == 4
    assert witness["final_global_step"] == 8
    assert witness["steps_executed"] == 4
    assert witness["matched"] is True
    assert witness["reason"] is None
    assert witness["progress_state"] == "advanced"
    assert witness["optimizer_state_present"] is True


def test_a_resume_with_nothing_left_to_run_is_recorded_not_refused(tmp_path):
    """A checkpoint that already reached its horizon resumes legitimately and
    does nothing further. Treating that as a failure would refuse a real
    workflow, so it is recorded distinctly instead."""
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-4", step=4))

    witness = resume_witness(inventory, final_global_step=4, declared_max_steps=4)

    assert witness["matched"] is True
    assert witness["reason"] is None
    assert witness["steps_executed"] == 0
    assert witness["progress_state"] == "already_at_horizon"


def test_a_resume_that_stopped_at_its_restore_point_below_the_horizon_is_visible(
    tmp_path,
):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-4", step=4))

    witness = resume_witness(inventory, final_global_step=4, declared_max_steps=8)

    # Not refused (a cancellation can legitimately look like this), but named:
    # there were steps left to run and none were taken.
    assert witness["matched"] is True
    assert witness["progress_state"] == "no_further_steps"


def test_a_run_that_ends_behind_its_restore_point_is_refused(tmp_path):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-8", step=8))

    witness = resume_witness(inventory, final_global_step=4, declared_max_steps=8)

    assert witness["matched"] is False
    assert "BEHIND" in (witness["reason"] or "")
    assert witness["steps_executed"] == -4
    assert witness["progress_state"] == "unknown"


def test_witness_refuses_when_the_restore_point_is_unknown(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-4", step=4)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    inventory = inventory_checkpoint(checkpoint)

    witness = resume_witness(inventory, final_global_step=8)

    assert witness["matched"] is False
    assert "unknown" in (witness["reason"] or "")


def test_witness_records_the_incomplete_checkpoint_rather_than_hiding_it(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-4", pieces=["trainer_state"])
    inventory = inventory_checkpoint(checkpoint)

    witness = resume_witness(inventory, final_global_step=8)

    assert witness["complete"] is False
    assert witness["optimizer_state_present"] is False
    assert list(witness["missing_required"]) == ["optimizer", "scheduler"]


def test_the_checkpoint_records_the_horizon_it_was_produced_under(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-4", step=4)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 4, "max_steps": 8}), encoding="utf-8"
    )

    inventory = inventory_checkpoint(checkpoint)
    assert inventory.max_steps == 8

    witness = resume_witness(inventory, final_global_step=8, declared_max_steps=8)
    assert witness["horizon_changed"] is False


def test_a_longer_declared_horizon_is_recorded_not_hidden(tmp_path):
    """Deliberate continuation is allowed, but it is not an exact resume:
    Trainer recomputes the remaining LR trajectory for the new total."""
    checkpoint = _write_state(tmp_path / "checkpoint-4", step=4)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 4, "max_steps": 8}), encoding="utf-8"
    )
    inventory = inventory_checkpoint(checkpoint)

    witness = resume_witness(inventory, final_global_step=16, declared_max_steps=16)

    assert witness["matched"] is True  # still a real continuation
    assert witness["horizon_changed"] is True
    assert witness["restored_max_steps"] == 8
    assert witness["declared_max_steps"] == 16


def test_a_checkpoint_with_no_recorded_horizon_says_unknown(tmp_path):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-4", step=4))
    witness = resume_witness(inventory, final_global_step=8, declared_max_steps=8)
    assert inventory.max_steps is None
    assert witness["horizon_changed"] is None  # unknown, not "unchanged"


def test_inventory_accepts_a_string_path_and_reports_it_resolved(tmp_path):
    checkpoint = _write_state(tmp_path / "checkpoint-50")
    inventory = inventory_checkpoint(str(checkpoint))
    assert inventory.directory == str(checkpoint.resolve())


def test_inventory_is_frozen_evidence_not_a_live_handle(tmp_path):
    inventory = inventory_checkpoint(_write_state(tmp_path / "checkpoint-50"))
    assert isinstance(inventory, CheckpointInventory)
    with pytest.raises(Exception):
        inventory.state = "complete"  # type: ignore[misc]
