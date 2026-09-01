import pytest

from chowder.contamination import RepairExample, write_holdout_fingerprint_index, write_verified_repair_dataset
from chowder.failures import RepairPlan
from chowder.provenance import sha256_directory
from chowder.repair_candidates import (
    RepairVariant,
    VerifiedParentAdapter,
    VerifiedRepairDataset,
    build_repair_candidate,
)


def _plan():
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="reasoning failure",
        suspected_cause="answer-selection weakness",
        intervention="independent repair examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _dataset(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    write_holdout_fingerprint_index([("hidden", "answer")], holdout)
    repair = tmp_path / "repair.jsonl"
    digest, audit = write_verified_repair_dataset(
        [RepairExample("independent", "response", "source-1")],
        [holdout],
        repair,
    )
    return VerifiedRepairDataset(str(repair), digest, audit)


def _parent_adapter(tmp_path, name="parent-adapter"):
    path = tmp_path / name
    path.mkdir()
    (path / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"parent-weights")
    return VerifiedParentAdapter(str(path), sha256_directory(path))


def test_continuation_candidate_binds_parent_adapter_into_identity_and_config(tmp_path):
    dataset = _dataset(tmp_path)
    parent = _parent_adapter(tmp_path)
    variant = RepairVariant("lr-low", 0.2, training_patch={"learning_rate": 5e-5})
    candidate = build_repair_candidate(
        parent_id="parent",
        plan=_plan(),
        dataset=dataset,
        variant=variant,
        parent_adapter=parent,
    )
    assert candidate.config_patch["backend"]["parent_adapter"] == {
        "path": str((tmp_path / "parent-adapter").resolve()),
        "sha256": parent.sha256,
    }
    assert candidate.config_patch["repair"]["parent_adapter_sha256"] == parent.sha256
    assert candidate.config_patch["repair"]["continuation"] is True

    other = _parent_adapter(tmp_path, "other-parent")
    other_path = tmp_path / "other-parent" / "adapter_model.safetensors"
    other_path.write_bytes(b"different-parent-weights")
    other = VerifiedParentAdapter(str(other_path.parent), sha256_directory(other_path.parent))
    other_candidate = build_repair_candidate(
        parent_id="parent",
        plan=_plan(),
        dataset=dataset,
        variant=variant,
        parent_adapter=other,
    )
    assert candidate.experiment_id != other_candidate.experiment_id


def test_continuation_candidate_refuses_lora_topology_patch(tmp_path):
    with pytest.raises(ValueError, match="cannot change LoRA topology"):
        build_repair_candidate(
            parent_id="parent",
            plan=_plan(),
            dataset=_dataset(tmp_path),
            variant=RepairVariant("rank-change", 0.2, lora_patch={"r": 8}),
            parent_adapter=_parent_adapter(tmp_path),
        )


def test_continuation_candidate_refuses_mutated_parent_adapter(tmp_path):
    parent = _parent_adapter(tmp_path)
    (tmp_path / "parent-adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="parent adapter content changed"):
        build_repair_candidate(
            parent_id="parent",
            plan=_plan(),
            dataset=_dataset(tmp_path),
            variant=RepairVariant("default", 0.2),
            parent_adapter=parent,
        )
