"""Track E real-hardware acceptance: the recursive autonomous-repair loop
proven for the Transformers engine in
tests/test_project_runner_repair.py::test_real_run_project_autonomously_repairs_a_rejected_candidate
runs unmodified through the real Unsloth backend on real CUDA.

Gated behind CHOWDER_REAL_UNSLOTH_SMOKE=1 (matching test_unsloth_peft_real.py,
not CHOWDER_REAL_ML_SMOKE) since this needs a real, already-set-up isolated
Unsloth environment (`chowder setup unsloth`, see docs/UNSLOTH.md) that
ordinary CI cannot provision. This is deliberately not a custom one-off
script: it calls run_project() exactly once, with backend.engine='unsloth'
in the project config as the only difference from the Transformers version --
proving the existing recursive-repair orchestration (baseline -> train ->
evaluate -> reject -> harvest -> cluster -> repair request -> independent
corrective examples -> contamination audit -> real second Unsloth training
hop from the exact parent adapter -> independent re-evaluation) needs no
Unsloth-specific orchestration code, per the Qwen3.8 program's Track E
requirement for real CUDA acceptance on a small model before the 27B parent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.project import write_project
from chowder.project_runner import run_project
from chowder.recursive_repair import RecursiveRepairStopReason
from chowder.run_events import FailureEvent, RepairEvent, RunEventPayload
from chowder.unsloth_env import unsloth_env_dir, unsloth_python
from unsloth_env_link import link_persistent_unsloth_env

_REAL_UNSLOTH_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_UNSLOTH_SMOKE") != "1",
    reason="real Unsloth smoke requires CHOWDER_REAL_UNSLOTH_SMOKE=1 and a "
    "real isolated Unsloth environment (see docs/UNSLOTH.md)",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
_ENV_ROOT_VAR = "CHOWDER_REAL_UNSLOTH_ENV_ROOT"


@_REAL_UNSLOTH_SMOKE
def test_real_run_project_autonomously_repairs_a_rejected_candidate_through_unsloth(
    tmp_path: Path,
):
    # A FRESH work dir every run, with the persistent environment linked in.
    # Using the persistent root itself as work_dir (the previous approach) wrote
    # this run's registry into it, so the test passed once and then failed on
    # every rerun with a duplicate-experiment-id error. See unsloth_env_link.
    work_dir = tmp_path
    env_dir = link_persistent_unsloth_env(work_dir) or unsloth_env_dir(work_dir)
    python_executable = unsloth_python(env_dir)
    if not python_executable.is_file():
        pytest.skip(
            f"no isolated Unsloth environment at {env_dir}; run "
            "`chowder setup unsloth --root <dir>` once and point "
            f"{_ENV_ROOT_VAR} at that dir"
        )

    train_path = work_dir / "train.jsonl"
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
    eval_path = work_dir / "eval.jsonl"
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
    repair_corpus_path = work_dir / "repair_corpus.jsonl"
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

    project_path = work_dir / "project.json"
    write_project(
        project_path,
        {
            "schema_version": 1,
            "name": "real tiny llama unsloth repair smoke",
            "work_dir": str(work_dir),
            "registry_path": ".chowder/unsloth-repair-smoke-runs.db",
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
                "experiment_id": "real-unsloth-sft",
                "estimated_gpu_hours": 0.25,
                "hypothesis": {
                    "observation": "tiny model is unadapted",
                    "suspected_cause": "target examples are unseen",
                    "intervention": "one small LoRA SFT run through Unsloth",
                    "expected_deltas": {"quality": 0.0},
                },
                "config_patch": {},
                "tags": ["integration", "real-ml", "repair", "unsloth"],
            },
            "config": {
                "seed": 123,
                "backend": {
                    "schema_version": 1,
                    "type": "peft",
                    "engine": "unsloth",
                    "base_model": _TINY_MODEL,
                    "dataset": "train.jsonl",
                    "text_field": "text",
                    "max_length": 64,
                    "quantization": "none",
                    "lora": {"r": 4, "alpha": 8},
                    "training": {
                        "epochs": 1.0,
                        "max_steps": 4,
                        "learning_rate": 1e-3,
                        "batch_size": 2,
                        "gradient_accumulation_steps": 1,
                        "logging_steps": 1,
                    },
                    "runtime": {"timeout_seconds": 300.0},
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
                        "name": "more-steps",
                        "estimated_gpu_hours": 0.3,
                        "training_patch": {"max_steps": 8},
                        "expected_deltas": {"quality": 0.05},
                    }
                ],
                "policy": {"max_depth": 1},
            },
        },
    )

    events: list[RunEventPayload] = []
    outcome = run_project(project_path, on_event=events.append)

    # Neither the initial candidate nor the repair hop could possibly clear
    # the gate (minimum_promotion_gain=2.0 exceeds what a [0, 1]-bounded
    # metric can ever gain) -- but both trained for real through Unsloth and
    # evaluated for real without crashing.
    assert outcome.promoted_experiment_id is None
    assert outcome.succeeded is True

    repair_events = [event for event in events if isinstance(event, RepairEvent)]
    assert len(repair_events) == 2, "expected a repair-start and repair-stop event"
    assert repair_events[0].stop_reason is None  # the starting event
    assert repair_events[1].stop_reason == RecursiveRepairStopReason.MAX_DEPTH.value

    failure_events = [event for event in events if isinstance(event, FailureEvent)]
    assert len(failure_events) >= 1
    assert failure_events[0].failure_count >= 1

    assert outcome.repair is not None
    assert outcome.repair.new_hops == 1
    assert outcome.repair.stop_reason == RecursiveRepairStopReason.MAX_DEPTH

    hop = outcome.repair.hops[0]
    assert hop.target_experiment_id == "real-unsloth-sft"
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

    # The repair candidate's own training evidence proves it actually ran
    # through the Unsloth engine (not silently through Transformers) and
    # actually continued from the rejected initial candidate's exact adapter
    # weights, not a fresh-initialized one.
    training_evidence = repair_candidate.artifact.evidence
    assert training_evidence.get("continued_from_parent_adapter") is True
    assert training_evidence.get("parent_adapter_sha256")

    adapter_dir = Path(repair_candidate.artifact.artifact_ref)
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert repair_candidate.evaluation.evidence["evaluator"] == "transformers-text"

    assert outcome.generation is repair_outcome.repair_generation
    assert outcome.generation.candidates[0].experiment_id == repair_candidate.experiment_id
