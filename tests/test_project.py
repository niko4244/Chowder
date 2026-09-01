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
