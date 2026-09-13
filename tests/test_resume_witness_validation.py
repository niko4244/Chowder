"""A worker assertion is not an independently consistent resume record."""
from dataclasses import replace

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.resume_state import CheckpointInventory, resume_witness


@pytest.fixture
def inventory(tmp_path):
    return CheckpointInventory(
        directory=str((tmp_path / "checkpoint-4").resolve()),
        state="complete", files={"optimizer": 12, "scheduler": 12, "trainer_state": 40},
        present=("optimizer", "scheduler", "trainer_state"),
        missing_required=(), missing_optional=("rng_state",), global_step=4, max_steps=8,
    )


def _telemetry(inventory):
    return {"global_step": 8, "resume": resume_witness(
        inventory, final_global_step=8, declared_max_steps=8,
    )}


def test_bare_true_is_not_a_witness(inventory):
    with pytest.raises(ValueError, match="resume witness"):
        TransformersPeftExecutor._summarize_resume({"resume": {"matched": True}}, inventory)


@pytest.mark.parametrize("field,value", [
    ("requested_checkpoint", "wrong-checkpoint"),
    ("requested_checkpoint", None),
    ("restored_global_step", 3),
    ("restored_global_step", True),
    ("final_global_step", 9),
    ("final_global_step", 8.0),
    ("steps_executed", 8),
    ("steps_executed", "4"),
    ("complete", False),
    ("complete", 1),
    ("optimizer_state_present", False),
    ("rng_state_present", True),
    ("missing_required", ["optimizer"]),
    ("matched", "true"),
])
def test_conflicting_or_wrongly_typed_worker_claim_is_refused(inventory, field, value):
    telemetry = _telemetry(inventory)
    telemetry["resume"][field] = value
    with pytest.raises(ValueError):
        TransformersPeftExecutor._summarize_resume(telemetry, inventory)


@pytest.mark.parametrize("field", [
    "requested_checkpoint", "restored_global_step", "final_global_step",
    "steps_executed", "complete", "optimizer_state_present", "rng_state_present",
    "missing_required",
])
def test_incomplete_positive_witness_is_refused(inventory, field):
    telemetry = _telemetry(inventory)
    del telemetry["resume"][field]
    with pytest.raises(ValueError, match="resume witness"):
        TransformersPeftExecutor._summarize_resume(telemetry, inventory)


def test_missing_independent_final_counter_is_refused(inventory):
    telemetry = _telemetry(inventory)
    del telemetry["global_step"]
    with pytest.raises(ValueError, match="resume witness"):
        TransformersPeftExecutor._summarize_resume(telemetry, inventory)


def test_worker_cannot_override_incomplete_parent_inventory(inventory):
    telemetry = _telemetry(inventory)
    partial = replace(inventory, state="partial", missing_required=("optimizer",))
    with pytest.raises(ValueError, match="resume witness"):
        TransformersPeftExecutor._summarize_resume(telemetry, partial)


def test_consistent_record_states_its_verification_boundary(inventory):
    result = TransformersPeftExecutor._summarize_resume(_telemetry(inventory), inventory)
    assert result["state"] == "witnessed"
    assert result["verification"] == "source-metadata-and-worker-report"


def test_absent_legacy_witness_remains_unknown(inventory):
    result = TransformersPeftExecutor._summarize_resume({"global_step": 8}, inventory)
    assert result["state"] == "unknown"
