"""CPU-only adversarial proof that old optimizer state cannot be relabeled.

The parent manifest binds *previously written* checkpoints to training inputs.
Creating/upgrading that witness after a checkpoint exists makes the provenance
claim unverifiable even if the checkpoint contains apparently resumable state.
"""

import json
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor


def _state(trainer: Path, *, name: str = "checkpoint-50") -> Path:
    checkpoint = trainer / name
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_bytes(b"state from unrelated old model")
    (checkpoint / "scheduler.pt").write_bytes(b"previous scheduler")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 50, "max_steps": 100}), encoding="utf-8"
    )
    (checkpoint / "rng_state.pth").write_bytes(b"old RNG")
    return checkpoint


@pytest.mark.parametrize("state_file", ["complete", "partial"])
def test_orphan_checkpoint_must_not_acquire_forged_parent_manifest(
    tmp_path: Path, state_file: str
) -> None:
    trainer = tmp_path / "trainer"
    checkpoint = _state(trainer)
    if state_file == "partial":
        (checkpoint / "optimizer.pt").unlink()
    manifest = trainer / "chowder-checkpoint-manifest.json"
    requested = {"checkpoint_recipe_sha256": "new recipe", "expert_numeric_mode": "unquantized"}

    with pytest.raises(RuntimeError, match="pre-existing checkpoint"):
        TransformersPeftExecutor._write_checkpoint_manifest(trainer, requested)

    assert not manifest.exists()
    assert (checkpoint / "scheduler.pt").read_bytes() == b"previous scheduler"


def test_subset_manifest_cannot_gain_missing_input_identity_after_checkpoint(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "trainer"
    _state(trainer)
    manifest = trainer / "chowder-checkpoint-manifest.json"
    original = {"revision": None, "expert_numeric_mode": "unquantized"}
    manifest.write_text(json.dumps(original), encoding="utf-8")
    requested = {**original, "dataset_sha256": "b" * 64}

    with pytest.raises(RuntimeError, match="retroactively certify"):
        TransformersPeftExecutor._write_checkpoint_manifest(trainer, requested)

    assert json.loads(manifest.read_text(encoding="utf-8")) == original


def test_matching_parent_manifest_and_existing_checkpoint_still_permitted(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer"
    _state(trainer)
    manifest = trainer / "chowder-checkpoint-manifest.json"
    bound = {"dataset_sha256": "d" * 64, "expert_numeric_mode": "rowwise-int8-dequant"}
    manifest.write_text(json.dumps(bound), encoding="utf-8")

    TransformersPeftExecutor._write_checkpoint_manifest(trainer, bound)
    assert json.loads(manifest.read_text(encoding="utf-8")) == bound


def test_new_and_precheckpoint_subset_manifests_remain_writable(tmp_path: Path) -> None:
    bound = {"dataset_sha256": "e" * 64, "expert_numeric_mode": "unquantized"}
    for name in ("fresh", "precheckpoint"):
        trainer = tmp_path / name
        trainer.mkdir()
        manifest = trainer / "chowder-checkpoint-manifest.json"
        if name == "precheckpoint":
            manifest.write_text(json.dumps({"dataset_sha256": bound["dataset_sha256"]}))
        TransformersPeftExecutor._write_checkpoint_manifest(trainer, bound)
        assert json.loads(manifest.read_text(encoding="utf-8")) == bound
