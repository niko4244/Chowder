from __future__ import annotations

import json

import pytest

from chowder.hardware import AcceleratorProfile, HardwareSnapshot
from chowder.project import ProjectValidationError, load_project, project_from_mapping, write_project
from chowder.project_runner import hardware_profile_from_snapshot


def _payload(tmp_path):
    return {
        "schema_version": 1,
        "name": "test project",
        "work_dir": str(tmp_path),
        "registry_path": ".chowder/runs.db",
        "seed": 7,
        "goal": {
            "metrics": [
                {
                    "name": "quality",
                    "minimum": 0.8,
                    "direction": "maximize",
                }
            ],
            "gpu_hour_budget": 1.0,
            "max_parallel_candidates": 1,
        },
        "baseline": {
            "experiment_id": "baseline",
            "metrics": {"quality": 0.2},
            "gpu_hours": 0.0,
        },
        "experiment": {
            "experiment_id": "initial-sft",
            "estimated_gpu_hours": 0.25,
            "config_patch": {},
        },
        "config": {
            "seed": 7,
            "backend": {
                "schema_version": 1,
                "type": "transformers-peft",
                "base_model": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
                "dataset": "train.jsonl",
                "text_field": "text",
                "max_length": 64,
                "precision": "fp32",
                "quantization": "none",
                "training": {
                    "epochs": 1.0,
                    "learning_rate": 0.0002,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "logging_steps": 1,
                    "gradient_checkpointing": False,
                },
                "lora": {
                    "r": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                    "use_rslora": False,
                },
                "runtime": {"active_accelerator_count": 0},
            },
            "evaluation": {
                "type": "transformers-text",
                "estimated_gpu_hours": 0.05,
                "precision": "fp32",
                "quantization": "none",
                "device": "cpu",
                "suites": [
                    {
                        "name": "quality",
                        "dataset": "eval.jsonl",
                        "prompt_field": "prompt",
                        "expected_field": "expected",
                        "scoring": "normalized_exact_match",
                        "max_new_tokens": 4,
                        "use_chat_template": False,
                    }
                ],
            },
        },
    }


def test_project_round_trip_and_relative_files(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "Question: 1+1? Answer: 2"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text(
        json.dumps({"prompt": "Question: 1+1? Answer:", "expected": "2"}) + "\n",
        encoding="utf-8",
    )
    path = write_project(tmp_path / "project.json", _payload(tmp_path))
    project = load_project(path)
    assert project.name == "test project"
    assert project.work_dir == tmp_path.resolve()
    assert project.registry_path == (tmp_path / ".chowder" / "runs.db").resolve()
    assert project.goal.metrics[0].name == "quality"
    assert project.experiment.experiment_id == "initial-sft"


def test_project_rejects_metric_suite_mismatch(tmp_path):
    payload = _payload(tmp_path)
    payload["config"]["evaluation"]["suites"][0]["name"] = "different"
    with pytest.raises(ProjectValidationError, match="suite names"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_project_rejects_unknown_training_key(tmp_path):
    payload = _payload(tmp_path)
    payload["config"]["backend"]["training"]["learning_rtae"] = 1e-3
    with pytest.raises(ValueError, match="unsupported key"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_project_rejects_future_schema(tmp_path):
    payload = _payload(tmp_path)
    payload["schema_version"] = 999
    with pytest.raises(ProjectValidationError, match="unsupported project.schema_version"):
        project_from_mapping(payload, source_dir=tmp_path)


def _repair_section(tmp_path, **overrides):
    section = {
        "corpus_files": ["repair_corpus.jsonl"],
        "variants": [
            {
                "name": "more-epochs",
                "estimated_gpu_hours": 0.3,
                "training_patch": {"epochs": 2.0},
                "expected_deltas": {"quality": 0.05},
            }
        ],
    }
    section.update(overrides)
    return section


def test_repair_defaults_to_none_when_absent(tmp_path):
    payload = _payload(tmp_path)
    project = project_from_mapping(payload, source_dir=tmp_path)
    assert project.repair is None


def test_repair_section_parses_into_repair_spec(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path)
    project = project_from_mapping(payload, source_dir=tmp_path)
    assert project.repair is not None
    assert project.repair.corpus_files == (str((tmp_path / "repair_corpus.jsonl").resolve()),)
    assert len(project.repair.variants) == 1
    variant = project.repair.variants[0]
    assert variant.name == "more-epochs"
    assert variant.estimated_gpu_hours == 0.3
    assert dict(variant.training_patch) == {"epochs": 2.0}
    assert dict(variant.lora_patch) == {}
    assert dict(variant.expected_deltas) == {"quality": 0.05}
    assert project.repair.provider_max_examples == 32
    assert project.repair.provider_min_examples == 1
    assert project.repair.provider_examples_per_failure == 2
    assert project.repair.policy.max_depth == 3
    assert project.repair.policy.min_score_improvement == 1e-4
    assert project.repair.policy.max_failure_signature_occurrences == 1
    assert project.repair.policy.replay_ratio == 1.0


def test_repair_corpus_files_resolved_relative_to_work_dir(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path, corpus_files=["nested/repair.jsonl"])
    project = project_from_mapping(payload, source_dir=tmp_path)
    assert project.repair.corpus_files == (
        str((tmp_path / "nested" / "repair.jsonl").resolve()),
    )


def test_repair_rejects_empty_corpus_files(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path, corpus_files=[])
    with pytest.raises(ProjectValidationError, match="corpus_files"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_repair_rejects_empty_variants(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path, variants=[])
    with pytest.raises(ProjectValidationError, match="variants"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_repair_rejects_lora_patch_in_variant(tmp_path):
    payload = _payload(tmp_path)
    section = _repair_section(tmp_path)
    section["variants"][0]["lora_patch"] = {"r": 8}
    payload["repair"] = section
    with pytest.raises(ProjectValidationError, match="lora_patch"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_repair_rejects_invalid_variant_name(tmp_path):
    payload = _payload(tmp_path)
    section = _repair_section(tmp_path)
    section["variants"][0]["name"] = "   "
    payload["repair"] = section
    with pytest.raises(ProjectValidationError, match="name"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_repair_policy_accepts_overrides(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(
        tmp_path,
        policy={
            "max_depth": 5,
            "min_score_improvement": 0.01,
            "max_failure_signature_occurrences": 2,
            "replay_ratio": 0.5,
        },
        max_examples=16,
        min_examples=2,
        examples_per_failure=3,
    )
    project = project_from_mapping(payload, source_dir=tmp_path)
    assert project.repair.policy.max_depth == 5
    assert project.repair.policy.min_score_improvement == 0.01
    assert project.repair.policy.max_failure_signature_occurrences == 2
    assert project.repair.policy.replay_ratio == 0.5
    assert project.repair.provider_max_examples == 16
    assert project.repair.provider_min_examples == 2
    assert project.repair.provider_examples_per_failure == 3


def test_repair_policy_rejects_invalid_max_depth(tmp_path):
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path, policy={"max_depth": 0})
    with pytest.raises(ProjectValidationError, match="max_depth"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_validate_files_rejects_missing_repair_corpus(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "Question: 1+1? Answer: 2"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text(
        json.dumps({"prompt": "Question: 1+1? Answer:", "expected": "2"}) + "\n",
        encoding="utf-8",
    )
    payload = _payload(tmp_path)
    payload["repair"] = _repair_section(tmp_path)
    project = project_from_mapping(payload, source_dir=tmp_path)
    with pytest.raises(ProjectValidationError, match="repair.corpus_files"):
        project.validate_files()

    (tmp_path / "repair_corpus.jsonl").write_text(
        json.dumps({"prompt": "Question: 1+1? Answer:", "expected": "2", "suite": "quality"})
        + "\n",
        encoding="utf-8",
    )
    project.validate_files()


def test_kaggle_t4_pair_remains_two_discrete_vram_pools():
    snapshot = HardwareSnapshot(
        platform="Linux",
        cpu_count=4,
        ram_gb=30.0,
        storage_total_gb=100.0,
        storage_free_gb=80.0,
        accelerators=(
            AcceleratorProfile("nvidia", "Tesla T4", 15.0, index=0),
            AcceleratorProfile("nvidia", "Tesla T4", 15.0, index=1),
        ),
    )
    profile = hardware_profile_from_snapshot(snapshot)
    assert profile.accelerator_vram_gb == (15.0, 15.0)
    assert profile.vram_gb == 15.0
    assert profile.total_accelerator_vram_gb == 30.0
