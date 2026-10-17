"""Recipes are projected from the campaign's own measured model, not a synthetic step.

Gen-2 (2026-09-20): the device probe timed a synthetic step at ~0.002 s, so each
recipe projected ~0.00085 GPU-h, while the real 9B attempt measured 1,318 s over
344 steps plus a 3,688 s in-run evaluation. Settlement refuses any attempt whose
actual wall exceeds its projection by more than 25%, so with the synthetic
projection no real attempt could ever settle. These tests pin the calibrated
path: the measured lifecycle is read with its sources digested, the planner
projects training *and* the in-run evaluation from it, and the executor template
reserves the measured evaluation cost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder.growth.campaign_prepare import (
    MODEL_LIFECYCLE_BASIS,
    CampaignPrepareRefusal,
    calibrate_model_lifecycle,
)
from chowder.growth.recipe_planner import HardwareBudget


def _training_result(tmp_path: Path, *, steps: int = 344, seconds: float = 1318.27) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run-spec.json").write_text(
        json.dumps({"max_length": 2048, "batch_size": 1, "gradient_accumulation_steps": 1,
                    "precision": "bf16", "quantization": "none", "gradient_checkpointing": True}),
        encoding="utf-8",
    )
    path = run / "worker-result.json"
    path.write_text(
        json.dumps({"telemetry": {"global_step": steps, "lifecycle": {"measured_seconds": seconds}}}),
        encoding="utf-8",
    )
    return path


def _evaluation_result(tmp_path: Path, *, seconds: float = 3687.94) -> Path:
    path = tmp_path / "eval-result.json"
    path.write_text(
        json.dumps({"runtime": {"lifecycle": {"measured_seconds": seconds}}}), encoding="utf-8"
    )
    return path


def _budget(lifecycle=None) -> HardwareBudget:
    return HardwareBudget(
        gpu_name="test",
        vram_gb=16.0,
        measured_step_seconds_at_seq={512: 0.015, 1024: 0.0019, 2048: 0.0023},
        measured_load_seconds=0.001,
        wall_multiplier=3.5,
        model_lifecycle=lifecycle,
    )


def test_the_calibration_reads_the_measured_lifecycle_and_digests_its_sources(tmp_path: Path) -> None:
    training, evaluation = _training_result(tmp_path), _evaluation_result(tmp_path)
    lifecycle = calibrate_model_lifecycle(training, evaluation)
    assert lifecycle["basis"] == MODEL_LIFECYCLE_BASIS
    assert lifecycle["train_seconds_per_step"] == pytest.approx(1318.27 / 344, abs=1e-6)
    assert lifecycle["measured_seq_len"] == 2048
    assert lifecycle["in_run_eval_seconds"] == pytest.approx(3687.94)
    assert lifecycle["conditions"]["batch_size"] == 1
    digests = {source["role"]: source["sha256"] for source in lifecycle["sources"]}
    assert digests["training"] == hashlib.sha256(training.read_bytes()).hexdigest()
    assert digests["evaluation"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "broken",
    ["training_steps", "evaluation_seconds", "run_spec"],
)
def test_the_calibration_refuses_an_incomplete_measurement(tmp_path: Path, broken: str) -> None:
    training = _training_result(tmp_path, steps=0 if broken == "training_steps" else 344)
    evaluation = _evaluation_result(tmp_path, seconds=0.0 if broken == "evaluation_seconds" else 3687.94)
    if broken == "run_spec":
        (training.parent / "run-spec.json").unlink()
    with pytest.raises(CampaignPrepareRefusal):
        calibrate_model_lifecycle(training, evaluation)


def test_a_calibrated_projection_charges_measured_training_and_the_in_run_evaluation(tmp_path: Path) -> None:
    from chowder.growth.recipe_planner import RecipePlanner

    lifecycle = calibrate_model_lifecycle(_training_result(tmp_path), _evaluation_result(tmp_path))
    planner = RecipePlanner(budget=_budget(lifecycle), max_device_gpu_hours=2.0, max_wall_gpu_hours=2.0)
    device, wall = planner.project_cost(seq_len=2048, max_steps=344)
    expected = (344 * lifecycle["train_seconds_per_step"] + lifecycle["in_run_eval_seconds"]) / 3600.0
    assert wall == pytest.approx(expected)
    assert device == pytest.approx(expected)
    # The measured attempt: within the 25% settlement tolerance of its projection.
    actual = (1318.27 + 3687.94) / 3600.0
    assert actual <= wall * 1.25


def test_a_calibrated_projection_scales_with_the_recipes_effective_batch(tmp_path: Path) -> None:
    """Gen-2 a15 trained 2 x 4 examples per step against a 1 x 1 calibration and
    measured 33.3 s/step where the unscaled projection said 3.83: the recipe's
    effective batch must scale the measured step."""
    from chowder.growth.recipe_planner import RecipePlanner

    lifecycle = calibrate_model_lifecycle(_training_result(tmp_path), _evaluation_result(tmp_path))
    planner = RecipePlanner(budget=_budget(lifecycle), max_device_gpu_hours=9.0, max_wall_gpu_hours=9.0)
    _, one = planner.project_cost(seq_len=2048, max_steps=100, examples_per_step=1)
    _, eight = planner.project_cost(seq_len=2048, max_steps=100, examples_per_step=8)
    eval_hours = lifecycle["in_run_eval_seconds"] / 3600.0
    assert (eight - eval_hours) == pytest.approx(8 * (one - eval_hours))
    # a15's measured 100 steps (3,332 s) sit within the 25% settlement tolerance.
    assert 3332.32 / 3600.0 <= (eight - eval_hours) * 1.25


def test_the_planner_fits_its_steps_to_the_ceiling_at_the_effective_batch(tmp_path: Path) -> None:
    from chowder.growth.recipe_planner import RecipePlanner

    lifecycle = calibrate_model_lifecycle(_training_result(tmp_path), _evaluation_result(tmp_path))
    planner = RecipePlanner(budget=_budget(lifecycle), max_device_gpu_hours=2.0, max_wall_gpu_hours=2.0)

    from chowder.growth.curriculum import CurriculumItem

    item = CurriculumItem(
        item_id="cur-1", skill="math.arithmetic", role="TARGET", priority=0.8, confidence=0.7,
        weakness_evidence="measured", desired_improvement=0.1, preservation_risks=(),
        source_strategy="verified_synthesis", example_count=10340, token_target=80_000,
        difficulty_band="medium", verification_method="exact_match", training_type="sft",
        evaluation_set="tier1", protected_regression_set=(),
    )
    recipes = planner.propose([item], count=1)
    recipe = recipes[0]
    assert recipe.batch_size * recipe.gradient_accumulation == 8
    assert recipe.projected_wall_gpu_hours <= 2.0
    assert recipe.max_steps < 344  # shrunk to fit, not silently overrun


def test_without_a_calibration_the_synthetic_projection_is_unchanged() -> None:
    from chowder.growth.recipe_planner import RecipePlanner

    planner = RecipePlanner(budget=_budget(), max_device_gpu_hours=2.0, max_wall_gpu_hours=2.0)
    device, wall = planner.project_cost(seq_len=2048, max_steps=344)
    assert device == pytest.approx((344 * 0.0023 + 0.001) / 3600.0)
    assert wall == pytest.approx(device * 3.5)


def test_the_template_reserves_the_measured_in_run_evaluation(tmp_path: Path) -> None:
    from chowder.growth.campaign import CampaignManifest
    from chowder.growth.campaign_prepare import MEASURED_PARENT, _write_project_template

    document = {
        "cycle_id": "lifecycle-fixture",
        "parent_version": "gen1",
        "base_model_path": str(tmp_path / "base"),
        "base_model_digest": "a" * 64,
        "state_root": str(tmp_path / "state"),
        "target_benchmarks": ["generation-diagnostics@gen1-eval-protocol-v1"],
        "protected_benchmarks": ["math500@2024-04"],
        "broad_benchmarks": [],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": {
            "device_gpu_hours_ceiling_per_recipe": 2.0,
            "wall_gpu_hours_ceiling_per_recipe": 2.0,
            "device_gpu_hours_ceiling_campaign": 5.0,
            "wall_gpu_hours_ceiling_campaign": 5.0,
            "device_time_measured": False,
        },
        "recipes": ["recipe-a"],
        "candidate_selection_policy": "first_successful",
        "stopping_rules": ["stop before compute on admission refusal"],
        "protection": {
            "trusted_ancestor_version": "gen0", "slice_regression_max": 0.0625, "n_samples": 16,
            "seed": 1234, "shuffle": False,
            "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 8},
            "prompt_policy": "chat_template",
        },
    }
    manifest = CampaignManifest.from_mapping(document, source="test")
    parent_eval = tmp_path / "parent.json"
    parent_eval.write_text(json.dumps({"runs": [{"measurement_origin": MEASURED_PARENT,
                                                 "benchmark_qualified_id": "math500@2024-04",
                                                 "score": 0.0}]}), encoding="utf-8")
    material = tmp_path / "material.json"
    material.write_text(json.dumps({"suites": [{"name": "math500", "dataset": "m.jsonl"}]}), encoding="utf-8")
    path = _write_project_template(
        manifest, tmp_path, evaluation_material_path=material, parent_eval_path=parent_eval,
        evaluation_gpu_hours=3687.944 / 3600.0,  # the measured in-run evaluation
    )
    template = json.loads(path.read_text(encoding="utf-8"))
    evaluation = template["config"]["evaluation"]["estimated_gpu_hours"]
    training = template["experiment"]["estimated_gpu_hours"]
    assert evaluation == pytest.approx(3687.944 / 3600.0)
    assert training == pytest.approx(2.0 - 3687.944 / 3600.0)


def test_the_two_reservations_fit_the_engine_budget_exactly(tmp_path: Path) -> None:
    """Gen-2 attempt-13: a 6-decimal rounded evaluation reserve made training +
    evaluation sum to 2.0000001 against a 2.0 budget, and the engine refused the
    lifecycle before a single step. Reserve and engine must agree to the bit."""
    test_the_template_reserves_the_measured_in_run_evaluation(tmp_path)
    template = json.loads((tmp_path / "project-template.json").read_text(encoding="utf-8"))
    training = template["experiment"]["estimated_gpu_hours"]
    evaluation = template["config"]["evaluation"]["estimated_gpu_hours"]
    budget = template["goal"]["gpu_hour_budget"]
    # The engine's own rule (engine.resize_reservation): delta over the old
    # reservation may not exceed the remaining budget by more than 1e-12.
    remaining = budget - training
    assert (training + evaluation) - training <= remaining + 1e-12


def test_the_runner_refuses_a_malformed_lifecycle(tmp_path: Path) -> None:
    from chowder.growth.campaign_runner import CampaignRunRefusal, _model_lifecycle

    assert _model_lifecycle(None, tmp_path) is None
    with pytest.raises(CampaignRunRefusal):
        _model_lifecycle({"train_seconds_per_step": 3.8, "measured_seq_len": 2048}, tmp_path)
    with pytest.raises(CampaignRunRefusal):
        _model_lifecycle(
            {"train_seconds_per_step": 0.0, "measured_seq_len": 2048, "in_run_eval_seconds": 10.0},
            tmp_path,
        )
