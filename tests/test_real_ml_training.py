from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.project import write_project
from chowder.project_runner import run_project
from chowder.registry import RunRegistry


pytestmark = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)


def test_real_tiny_llama_train_evaluate_and_persist(tmp_path: Path):
    """Prove the user project path performs actual PEFT training and evaluation."""

    train_path = tmp_path / "train.jsonl"
    train_rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "prompt": "Question: What token comes after alpha? Answer:",
                "expected": "beta",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    project_path = tmp_path / "project.json"
    write_project(
        project_path,
        {
            "schema_version": 1,
            "name": "real tiny llama smoke",
            "work_dir": str(tmp_path),
            "registry_path": ".chowder/runs.db",
            "seed": 123,
            "goal": {
                "metrics": [
                    {
                        "name": "quality",
                        "minimum": 0.0,
                        "direction": "maximize",
                        "regression_tolerance": 1.0,
                    }
                ],
                "gpu_hour_budget": 2.0,
                "max_parallel_candidates": 1,
                # Promotion is not the smoke criterion; successful real training,
                # artifact persistence, and independent evaluation are.
                "minimum_promotion_gain": 1.0,
                "require_protocol_match": False,
            },
            "baseline": {
                "experiment_id": "baseline",
                "metrics": {"quality": 0.0},
                "gpu_hours": 0.0,
            },
            "experiment": {
                "experiment_id": "real-sft",
                "estimated_gpu_hours": 0.25,
                "hypothesis": {
                    "observation": "tiny model is unadapted",
                    "suspected_cause": "target examples are unseen",
                    "intervention": "one small LoRA SFT run",
                    "expected_deltas": {"quality": 0.0},
                },
                "config_patch": {},
                "tags": ["integration", "real-ml"],
            },
            "config": {
                "seed": 123,
                "backend": {
                    "schema_version": 1,
                    "type": "transformers-peft",
                    "base_model": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
                    "dataset": "train.jsonl",
                    "text_field": "text",
                    "max_length": 64,
                    "precision": "fp32",
                    "quantization": "none",
                    "trust_remote_code": False,
                    "training": {
                        "epochs": 1.0,
                        "learning_rate": 0.001,
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
                    "runtime": {
                        "active_accelerator_count": 0,
                        "timeout_seconds": 180.0,
                    },
                },
                "evaluation": {
                    "type": "transformers-text",
                    "estimated_gpu_hours": 0.05,
                    "precision": "fp32",
                    "quantization": "none",
                    "device": "cpu",
                    "trust_remote_code": False,
                    "runtime": {"timeout_seconds": 180.0},
                    "suites": [
                        {
                            "name": "quality",
                            "dataset": "eval.jsonl",
                            "prompt_field": "prompt",
                            "expected_field": "expected",
                            "scoring": "normalized_exact_match",
                            "max_new_tokens": 2,
                            "use_chat_template": False,
                        }
                    ],
                },
            },
        },
    )

    events = []
    outcome = run_project(project_path, on_event=events.append)
    assert len(outcome.generation.candidates) == 1
    candidate = outcome.generation.candidates[0]
    assert candidate.error is None, candidate.error
    assert candidate.artifact is not None
    assert candidate.evaluation is not None
    assert candidate.result is not None

    adapter_dir = Path(candidate.artifact.artifact_ref)
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert any(
        path.name.startswith("adapter_model") for path in adapter_dir.iterdir()
    )
    assert candidate.artifact.evidence["backend"] == "transformers-peft"
    assert len(candidate.artifact.evidence["artifact_sha256"]) == 64
    assert candidate.evaluation.evidence["evaluator"] == "transformers-text"
    assert len(candidate.evaluation.evidence["protocol_sha256"]) == 64
    assert set(candidate.result.metrics) == {"quality"}
    assert candidate.result.gpu_hours >= 0.0

    registry_path = tmp_path / ".chowder" / "runs.db"
    with RunRegistry(registry_path) as registry:
        artifacts = registry.list_training_artifacts()
        evaluations = registry.list_evaluation_outcomes()
        results = registry.list_results()
        assert len(artifacts) == 1
        assert len(evaluations) == 1
        assert len(results) == 1
        assert artifacts[0].artifact_ref == str(adapter_dir)
        assert evaluations[0].source_artifact_ref == str(adapter_dir)
        assert results[0].artifact_ref == str(adapter_dir)
