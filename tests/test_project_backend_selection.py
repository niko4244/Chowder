from __future__ import annotations

import pytest

from chowder.project import ProjectValidationError, project_from_mapping


def _payload(tmp_path, *, backend_type="peft", engine="transformers"):
    backend = {
        "schema_version": 1,
        "type": backend_type,
        "base_model": "example/model",
        "dataset": "train.jsonl",
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
    }
    if engine is not None:
        backend["engine"] = engine
    return {
        "schema_version": 1,
        "name": "engine selection",
        "work_dir": str(tmp_path),
        "registry_path": ".chowder/runs.db",
        "seed": 7,
        "goal": {
            "metrics": [{"name": "quality", "minimum": 0.8}],
            "gpu_hour_budget": 1.0,
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
            "backend": backend,
            "evaluation": {
                "type": "transformers-text",
                "precision": "fp32",
                "quantization": "none",
                "device": "cpu",
                "suites": [
                    {
                        "name": "quality",
                        "dataset": "eval.jsonl",
                        "max_new_tokens": 4,
                        "use_chat_template": False,
                    }
                ],
            },
        },
    }


def test_project_accepts_canonical_peft_transformers_without_mutating_config(tmp_path):
    project = project_from_mapping(_payload(tmp_path), source_dir=tmp_path)
    assert project.config["backend"]["type"] == "peft"
    assert project.config["backend"]["engine"] == "transformers"


def test_project_keeps_legacy_transformers_peft_backward_compatible(tmp_path):
    project = project_from_mapping(
        _payload(tmp_path, backend_type="transformers-peft", engine=None),
        source_dir=tmp_path,
    )
    assert project.config["backend"]["type"] == "transformers-peft"
    assert "engine" not in project.config["backend"]


def test_project_recognizes_unsloth_but_refuses_to_fake_availability(tmp_path):
    with pytest.raises(ProjectValidationError, match="isolated executor"):
        project_from_mapping(
            _payload(tmp_path, backend_type="peft", engine="unsloth"),
            source_dir=tmp_path,
        )


def test_project_requires_explicit_engine_for_canonical_peft(tmp_path):
    with pytest.raises(ProjectValidationError, match="backend.engine is required"):
        project_from_mapping(
            _payload(tmp_path, backend_type="peft", engine=None),
            source_dir=tmp_path,
        )
