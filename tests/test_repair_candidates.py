import json

import pytest

from chowder.contamination import (
    RepairExample,
    audit_repair_examples,
    example_fingerprints,
    write_holdout_fingerprint_index,
    write_verified_repair_dataset,
)
from chowder.failures import RepairPlan
from chowder.graph import ExperimentGraph
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_file
from chowder.repair_candidates import (
    RepairVariant,
    VerifiedRepairDataset,
    VerifiedReplayDataset,
    build_repair_candidate,
    build_repair_population,
)


def _plan():
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="reasoning failures",
        suspected_cause="answer-selection weakness",
        intervention="add independently sourced near-neighbor examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _verified_dataset(tmp_path):
    holdout = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], holdout)
    repair_path = tmp_path / "repair.jsonl"
    digest, audit = write_verified_repair_dataset(
        [RepairExample("3+3?", "6", "independent-1")],
        [holdout],
        repair_path,
    )
    return VerifiedRepairDataset(str(repair_path), digest, audit)


def _replay_dataset(tmp_path, *, name="base.jsonl", ratio=1.0):
    path = tmp_path / name
    path.write_text('{"text":"parent training example"}\n', encoding="utf-8")
    return VerifiedReplayDataset(str(path), sha256_file(path), ratio)


def _root():
    return Experiment(
        experiment_id="root",
        parent_id=None,
        hypothesis=Hypothesis("baseline", "none", "none"),
        config_patch={
            "backend": {
                "base_model": "example/model",
                "revision": "abc123",
                "precision": "bf16",
                "quantization": "4bit",
                "dataset": "base.jsonl",
                "text_field": "content",
                "training": {"learning_rate": 1e-4, "epochs": 2},
                "lora": {"r": 16, "alpha": 32},
            },
            "evaluation": {
                "type": "transformers-text",
                "precision": "inherit",
                "quantization": "inherit",
                "suites": [{"name": "quality", "dataset": "holdout.jsonl"}],
            },
        },
        estimated_gpu_hours=1.0,
    )


def test_repair_candidate_is_deterministic_and_does_not_patch_evaluation(tmp_path):
    dataset = _verified_dataset(tmp_path)
    variant = RepairVariant(
        "lr-low",
        0.5,
        training_patch={"learning_rate": 5e-5, "epochs": 1},
        lora_patch={"r": 8},
        expected_deltas={"quality": 0.03},
    )
    first = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=dataset, variant=variant
    )
    second = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=dataset, variant=variant
    )
    assert first.experiment_id == second.experiment_id
    assert "evaluation" not in first.config_patch
    assert set(first.config_patch["backend"]) == {
        "dataset",
        "dataset_sha256",
        "text_field",
        "training",
        "lora",
    }
    assert first.config_patch["backend"]["text_field"] == "text"
    assert first.config_patch["backend"]["dataset_sha256"] == dataset.sha256
    assert first.config_patch["repair"]["repair_dataset_sha256"] == dataset.sha256
    assert (
        first.config_patch["repair"]["repair_index_sha256"]
        == dataset.contamination_audit.repair_index_sha256
    )
    assert first.hypothesis.expected_deltas == {"quality": 0.03}


def test_resolved_repair_config_inherits_protocol_defining_settings(tmp_path):
    graph = ExperimentGraph()
    root = _root()
    graph.add(root)
    child = build_repair_candidate(
        parent_id="root",
        plan=_plan(),
        dataset=_verified_dataset(tmp_path),
        variant=RepairVariant(
            "rank8",
            0.4,
            training_patch={"learning_rate": 2e-4},
            lora_patch={"r": 8},
        ),
    )
    graph.add(child)
    resolved = graph.resolve_config(child.experiment_id)
    assert resolved["backend"]["base_model"] == "example/model"
    assert resolved["backend"]["revision"] == "abc123"
    assert resolved["backend"]["precision"] == "bf16"
    assert resolved["backend"]["quantization"] == "4bit"
    assert resolved["backend"]["dataset"] == dataset_path(child)
    assert resolved["backend"]["dataset_sha256"] == child.config_patch["backend"]["dataset_sha256"]
    assert resolved["backend"]["text_field"] == "text"
    assert resolved["backend"]["training"] == {"learning_rate": 2e-4, "epochs": 2}
    assert resolved["backend"]["lora"] == {"r": 8, "alpha": 32}
    assert resolved["evaluation"] == root.config_patch["evaluation"]


def dataset_path(candidate):
    return candidate.config_patch["backend"]["dataset"]


def test_repair_population_produces_distinct_candidate_branches(tmp_path):
    dataset = _verified_dataset(tmp_path)
    candidates = build_repair_population(
        parent_id="root",
        plan=_plan(),
        dataset=dataset,
        variants=(
            RepairVariant("lr-low", 0.3, training_patch={"learning_rate": 5e-5}),
            RepairVariant("lr-high", 0.3, training_patch={"learning_rate": 2e-4}),
            RepairVariant("rank-low", 0.3, lora_patch={"r": 8}),
        ),
    )
    assert len(candidates) == 3
    assert len({candidate.experiment_id for candidate in candidates}) == 3
    assert {candidate.tags[-1] for candidate in candidates} == {
        "variant:lr-low",
        "variant:lr-high",
        "variant:rank-low",
    }


def test_replay_is_verified_and_part_of_candidate_identity(tmp_path):
    dataset = _verified_dataset(tmp_path)
    replay_a = _replay_dataset(tmp_path, name="a.jsonl", ratio=0.5)
    replay_b = _replay_dataset(tmp_path, name="b.jsonl", ratio=1.0)
    variant = RepairVariant("same", 0.2)
    a = build_repair_candidate(
        parent_id="root",
        plan=_plan(),
        dataset=dataset,
        variant=variant,
        replay=replay_a,
    )
    b = build_repair_candidate(
        parent_id="root",
        plan=_plan(),
        dataset=dataset,
        variant=variant,
        replay=replay_b,
    )
    assert a.experiment_id != b.experiment_id
    assert a.config_patch["backend"]["replay"] == {
        "dataset": str((tmp_path / "a.jsonl").resolve()),
        "sha256": replay_a.sha256,
        "ratio": 0.5,
        "manifest": None,
        "manifest_sha256": None,
    }
    assert a.config_patch["repair"]["replay_dataset_sha256"] == replay_a.sha256
    assert a.config_patch["repair"]["replay_manifest_sha256"] is None
    assert a.config_patch["repair"]["replay_ratio"] == pytest.approx(0.5)


def test_candidate_refuses_replay_mutated_after_binding(tmp_path):
    dataset = _verified_dataset(tmp_path)
    replay = _replay_dataset(tmp_path)
    with open(replay.path, "a", encoding="utf-8") as handle:
        handle.write('{"text":"tampered"}\n')
    with pytest.raises(ValueError, match="replay dataset content changed"):
        build_repair_candidate(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variant=RepairVariant("default", 0.2),
            replay=replay,
        )


def test_candidate_refuses_same_file_for_repair_and_replay(tmp_path):
    dataset = _verified_dataset(tmp_path)
    replay = VerifiedReplayDataset(dataset.path, dataset.sha256, 1.0)
    with pytest.raises(ValueError, match="must be different files"):
        build_repair_candidate(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variant=RepairVariant("default", 0.2),
            replay=replay,
        )


def test_candidate_refuses_dataset_mutated_after_clean_audit(tmp_path):
    dataset = _verified_dataset(tmp_path)
    with open(dataset.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "tampered"}) + "\n")
    with pytest.raises(ValueError, match="content changed"):
        build_repair_candidate(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variant=RepairVariant("default", 0.2),
        )


def test_candidate_refuses_new_dataset_digest_reusing_old_clean_audit(tmp_path):
    dataset = _verified_dataset(tmp_path)
    path = dataset.path
    row = json.loads(open(path, encoding="utf-8").read())
    row["prompt"] = "100+100?"
    row["expected"] = "200"
    row["text"] = "User: 100+100?\nAssistant: 200"
    row["prompt_sha256"], row["pair_sha256"] = example_fingerprints(
        row["prompt"], row["expected"]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    forged = VerifiedRepairDataset(path, sha256_file(path), dataset.contamination_audit)
    with pytest.raises(ValueError, match="do not match contamination audit"):
        build_repair_candidate(
            parent_id="root",
            plan=_plan(),
            dataset=forged,
            variant=RepairVariant("default", 0.2),
        )


def test_candidate_identity_includes_holdout_audit_identity(tmp_path):
    repair_path = tmp_path / "repair.jsonl"
    repair_examples = [RepairExample("3+3?", "6", "independent-1")]
    holdout_a = tmp_path / "holdout-a.jsonl"
    holdout_b = tmp_path / "holdout-b.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], holdout_a)
    write_holdout_fingerprint_index([("Capital of France?", "Paris")], holdout_b)
    digest, audit_a = write_verified_repair_dataset(
        repair_examples, [holdout_a], repair_path
    )
    audit_b = audit_repair_examples(repair_examples, [holdout_b])
    assert audit_a.clean and audit_b.clean
    dataset_a = VerifiedRepairDataset(str(repair_path), digest, audit_a)
    dataset_b = VerifiedRepairDataset(str(repair_path), digest, audit_b)
    variant = RepairVariant("same", 0.2)
    candidate_a = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=dataset_a, variant=variant
    )
    candidate_b = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=dataset_b, variant=variant
    )
    assert candidate_a.experiment_id != candidate_b.experiment_id


def test_population_requires_unique_variant_names(tmp_path):
    dataset = _verified_dataset(tmp_path)
    with pytest.raises(ValueError, match="names must be unique"):
        build_repair_population(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variants=(
                RepairVariant("same", 0.2),
                RepairVariant("same", 0.3),
            ),
        )
