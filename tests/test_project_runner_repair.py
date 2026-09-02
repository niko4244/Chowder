from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.project import write_project
from chowder.project_runner import ProjectRunEvent, run_project
from chowder.recursive_repair import RecursiveRepairStopReason


pytestmark = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)


def test_real_run_project_autonomously_repairs_a_rejected_candidate(tmp_path: Path):
    """Prove the bounded autonomous-repair loop is actually wired into
    run_project(): a candidate that cannot possibly clear the promotion gate
    is trained and evaluated for real, its rejection is harvested into a real
    failure/repair plan, and a real second training+evaluation hop is run
    from a config-driven repair variant -- with no Python-level orchestration
    beyond calling run_project() once."""

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "Question: What token comes after alpha? Answer: beta"},
                {"text": "Question: What token comes after red? Answer: blue"},
            ]
        ),
        encoding="utf-8",
    )

    # A prompt with no relationship to the training data -- an essentially
    # untrained tiny model has no way to produce an exact match, so this
    # deterministically harvests a real failure row regardless of what the
    # model happens to output.
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "prompt": "Question: What token comes after gamma? Answer:",
                "expected": "delta",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Genuinely independent of the eval/holdout prompt above -- the
    # contamination audit refuses repair data that overlaps holdout prompts,
    # so these must teach something disjoint, not the held-out answer itself.
    repair_corpus_path = tmp_path / "repair_corpus.jsonl"
    repair_corpus_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "prompt": "Question: What token comes after one? Answer:",
                    "expected": "two",
                    "suite": "quality",
                },
                {
                    "prompt": "Question: What token comes after up? Answer:",
                    "expected": "down",
                    "suite": "quality",
                },
            ]
        ),
        encoding="utf-8",
    )

    project_path = tmp_path / "project.json"
    write_project(
        project_path,
        {
            "schema_version": 1,
            "name": "real tiny llama repair smoke",
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
                "gpu_hour_budget": 3.0,
                "max_parallel_candidates": 1,
                # A [0, 1]-bounded metric can gain at most 1.0 over a 0.0
                # baseline -- requiring 2.0 makes promotion provably
                # impossible, so the initial candidate is deterministically
                # rejected and the repair loop deterministically stops at
                # MAX_DEPTH rather than PROMOTED.
                "minimum_promotion_gain": 2.0,
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
                "tags": ["integration", "real-ml", "repair"],
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
            "repair": {
                "corpus_files": ["repair_corpus.jsonl"],
                "variants": [
                    {
                        "name": "more-epochs",
                        "estimated_gpu_hours": 0.3,
                        "training_patch": {"epochs": 2.0},
                        "expected_deltas": {"quality": 0.05},
                    }
                ],
                "policy": {"max_depth": 1},
            },
        },
    )

    events: list[ProjectRunEvent] = []
    outcome = run_project(project_path, on_event=events.append)

    # Neither the initial candidate nor the repair hop could possibly clear
    # the gate (minimum_promotion_gain=2.0 exceeds what a [0, 1]-bounded
    # metric can ever gain) -- but both trained and evaluated for real
    # without crashing.
    assert outcome.promoted_experiment_id is None
    assert outcome.succeeded is True

    repair_events = [event for event in events if event.stage == "repair"]
    assert len(repair_events) == 2, "expected a repair-start and repair-stop event"

    assert outcome.repair is not None
    assert outcome.repair.new_hops == 1
    assert outcome.repair.stop_reason == RecursiveRepairStopReason.MAX_DEPTH

    hop = outcome.repair.hops[0]
    assert hop.target_experiment_id == "real-sft"
    repair_outcome = hop.outcome
    assert repair_outcome.target.plan.requires_independent_source is True
    assert len(repair_outcome.population.proposed_candidates) >= 1

    repair_generation = repair_outcome.repair_generation
    assert len(repair_generation.candidates) >= 1
    repair_candidate = repair_generation.candidates[0]
    assert repair_candidate.error is None, repair_candidate.error
    assert repair_candidate.artifact is not None
    assert repair_candidate.evaluation is not None
    assert repair_candidate.result is not None

    adapter_dir = Path(repair_candidate.artifact.artifact_ref)
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert repair_candidate.evaluation.evidence["evaluator"] == "transformers-text"

    # outcome.generation is reassigned to the repair loop's final generation,
    # so a caller reading only .generation (as the CLI does) sees the latest
    # real attempt, not the stale initial one.
    assert outcome.generation is repair_outcome.repair_generation
    assert outcome.generation.candidates[0].experiment_id == repair_candidate.experiment_id
