import json

import pytest

from chowder.contamination import RepairExample, write_holdout_fingerprint_index, write_verified_repair_dataset
from chowder.failures import RepairPlan
from chowder.repair_candidates import (
    RepairVariant,
    VerifiedRepairDataset,
    build_autonomous_repair_population,
    build_repair_candidate,
)
from chowder.repair_sources import (
    RepairSource,
    SourcedRepairExample,
    write_provenanced_repair_dataset,
)


def _plan():
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="reasoning failure cluster",
        suspected_cause="answer-selection weakness",
        intervention="source independent near-neighbor examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _holdout(tmp_path):
    path = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], path)
    return path


def _provenanced_dataset(tmp_path, *, ref="corpus://math-v1", content_sha="a" * 64, suffix=""):
    dataset_path = tmp_path / f"repair{suffix}.jsonl"
    manifest_path = tmp_path / f"repair-sources{suffix}.json"
    dataset_sha, audit, manifest_sha = write_provenanced_repair_dataset(
        examples=[
            SourcedRepairExample(
                example_id="ex-1",
                source_id="source-1",
                prompt="3+3?",
                expected="6",
            )
        ],
        sources=[RepairSource("source-1", ref, content_sha)],
        holdout_fingerprint_files=[_holdout(tmp_path)],
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )
    return VerifiedRepairDataset(
        str(dataset_path),
        dataset_sha,
        audit,
        source_manifest_path=str(manifest_path),
        source_manifest_sha256=manifest_sha,
    )


def test_autonomous_repair_requires_and_accepts_valid_source_manifest(tmp_path):
    dataset = _provenanced_dataset(tmp_path)
    candidates = build_autonomous_repair_population(
        parent_id="root",
        plan=_plan(),
        dataset=dataset,
        variants=(RepairVariant("lr-low", 0.2, training_patch={"learning_rate": 5e-5}),),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.config_patch["repair"]["source_manifest_sha256"] == dataset.source_manifest_sha256


def test_autonomous_repair_rejects_clean_dataset_without_source_manifest(tmp_path):
    holdout = _holdout(tmp_path)
    repair_path = tmp_path / "manual-repair.jsonl"
    dataset_sha, audit = write_verified_repair_dataset(
        [RepairExample("3+3?", "6", "manual")],
        [holdout],
        repair_path,
    )
    dataset = VerifiedRepairDataset(str(repair_path), dataset_sha, audit)
    with pytest.raises(ValueError, match="requires an independent-source provenance manifest"):
        build_autonomous_repair_population(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variants=(RepairVariant("default", 0.2),),
        )


def test_source_writer_rejects_example_referencing_undeclared_source(tmp_path):
    with pytest.raises(ValueError, match="undeclared sources"):
        write_provenanced_repair_dataset(
            examples=[SourcedRepairExample("ex-1", "missing", "3+3?", "6")],
            sources=[RepairSource("source-1", "corpus://math", "a" * 64)],
            holdout_fingerprint_files=[_holdout(tmp_path)],
            dataset_path=tmp_path / "repair.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )


def test_candidate_rejects_tampered_source_manifest(tmp_path):
    dataset = _provenanced_dataset(tmp_path)
    manifest_path = dataset.source_manifest_path
    assert manifest_path is not None
    payload = json.loads(open(manifest_path, encoding="utf-8").read())
    payload["sources"][0]["ref"] = "corpus://tampered"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    with pytest.raises(ValueError, match="source manifest content changed"):
        build_repair_candidate(
            parent_id="root",
            plan=_plan(),
            dataset=dataset,
            variant=RepairVariant("default", 0.2),
            require_source_manifest=True,
        )


def test_candidate_identity_changes_when_independent_source_manifest_changes(tmp_path):
    first = _provenanced_dataset(
        tmp_path,
        ref="corpus://math-v1",
        content_sha="a" * 64,
        suffix="-a",
    )
    second = _provenanced_dataset(
        tmp_path,
        ref="corpus://math-v2",
        content_sha="b" * 64,
        suffix="-b",
    )
    assert first.sha256 == second.sha256
    assert first.contamination_audit.repair_index_sha256 == second.contamination_audit.repair_index_sha256
    assert first.source_manifest_sha256 != second.source_manifest_sha256
    variant = RepairVariant("same", 0.2)
    candidate_a = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=first, variant=variant, require_source_manifest=True
    )
    candidate_b = build_repair_candidate(
        parent_id="root", plan=_plan(), dataset=second, variant=variant, require_source_manifest=True
    )
    assert candidate_a.experiment_id != candidate_b.experiment_id
