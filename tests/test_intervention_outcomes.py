import pytest

from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.intervention_outcomes import (
    build_intervention_outcomes,
    filter_outcomes,
    group_by_arm,
)
from chowder.memory import HardwareProfile
from chowder.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Goal,
    Hypothesis,
    MetricTarget,
)
from chowder.registry import RunRegistry


def _goal(minimum=0.0):
    return Goal((MetricTarget("quality", minimum=minimum),), gpu_hour_budget=10.0)


def _baseline(quality=0.70):
    return ExperimentResult("base", {"quality": quality}, 0.0)


def _experiment(experiment_id, config_patch, *, parent_id=None, status=None, hours=1.0):
    experiment = Experiment(
        experiment_id,
        parent_id,
        Hypothesis("obs", "cause", f"intervention for {experiment_id}"),
        config_patch,
        hours,
    )
    if status is not None:
        experiment.status = status
    return experiment


# The exact evidence/telemetry shape the real transformers-peft executor
# writes (see backends/transformers_peft.py::run and
# backends/transformers_worker.py::train), trimmed to the keys this view
# actually reads. A hand-written double is used rather than real training
# because this module reads persisted evidence -- it never needs a GPU.
def _production_shaped_evidence():
    return {
        "recipe_sha256": "r" * 64,
        "model_provenance": {"requested_base_model": "sshleifer/tiny-gpt2"},
        "hardware_aware_defaults": {
            "min_device_vram_gb": 16.0,
            "resolved_activation_offload": True,
            "resolved_optimizer_tiering": False,
            "resolved_frozen_layer_streaming": True,
        },
        "resource_usage": {"active_accelerator_count": 1, "visible_accelerator_count": 2},
    }


def _production_shaped_telemetry():
    return {
        "train_loss": 0.4,
        "global_step": 24,
        "train_runtime_seconds": 12.5,
        "peak_vram_gb": 3.25,
    }


class _Trainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        return TrainingArtifact(
            f"train-{experiment.experiment_id}",
            experiment.experiment_id,
            "/artifact",
            0.4,
            telemetry=_production_shaped_telemetry(),
            evidence=_production_shaped_evidence(),
        )

    def cancel(self, run_id):
        pass


class _Evaluator:
    name = "fake-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            f"eval-{experiment.experiment_id}",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": 0.85},
            0.1,
            {"suite": "q"},
        )

    def cancel(self, run_id):
        pass


def test_build_intervention_outcomes_from_cycle_populated_registry(tmp_path):
    """Populate the registry the way production does -- through
    ExperimentCycleRunner, which records the experiment, the training
    artifact, the evaluation outcome, the result, and the post-gate status
    -- then read the whole normalized row back out of it."""
    registry = RunRegistry(tmp_path / "runs.db")
    engine = EvolutionEngine(_goal(minimum=0.8), _baseline())
    experiment = _experiment("e1", {"backend": {"training": {"learning_rate": 1e-3}}})
    assert engine.propose([experiment]) == (experiment,)
    registry.record_experiment(experiment)
    runner = ExperimentCycleRunner(
        engine,
        _Trainer(),
        _Evaluator(),
        ExecutionContext(HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7),
        registry=registry,
    )
    outcome = runner.run_generation([experiment])
    assert outcome.promoted is not None

    rows = build_intervention_outcomes(registry, goal=_goal(minimum=0.8), baseline=_baseline())
    registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.experiment_id == "e1"
    assert row.parent_id is None
    assert row.config_patch == {"backend": {"training": {"learning_rate": 1e-3}}}
    assert row.arm == frozenset({"backend.training.learning_rate"})
    assert row.intervention == "intervention for e1"
    assert row.training_run_id == "train-e1"
    assert row.base_model == "sshleifer/tiny-gpt2"
    assert row.recipe_sha256 == "r" * 64
    assert row.min_device_vram_gb == 16.0
    assert row.active_accelerator_count == 1
    assert row.memory_fabric_mechanisms == frozenset(
        {"activation_offload", "frozen_layer_streaming"}
    )
    assert row.gpu_hours == pytest.approx(0.5)
    assert row.training_gpu_hours == pytest.approx(0.4)
    assert row.metrics == {"quality": 0.85}
    assert row.gate_accepted is True
    assert row.gate_score_vs_baseline == pytest.approx(0.15)
    assert row.gate_score_vs_parent is None
    assert row.regressions_vs_baseline == {}
    assert row.peak_vram_gb == pytest.approx(3.25)
    assert row.train_runtime_seconds == pytest.approx(12.5)
    assert row.global_step == 24


def test_build_intervention_outcomes_reports_none_when_evidence_is_absent(tmp_path):
    """A result recorded with no training artifact and no evidence must
    surface every evidence-derived field as None -- never a default, a
    zero, or an imputed value."""
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("bare", {"backend": {"lora": {"r": 8}}}))
    registry.record_result(ExperimentResult("bare", {"quality": 0.9}, 0.6))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.training_run_id is None
    assert row.base_model is None
    assert row.recipe_sha256 is None
    assert row.min_device_vram_gb is None
    assert row.active_accelerator_count is None
    assert row.memory_fabric_mechanisms is None
    assert row.training_gpu_hours is None
    assert row.peak_vram_gb is None
    assert row.train_runtime_seconds is None
    assert row.global_step is None
    assert row.gate_score_vs_parent is None
    # Status was never adjudicated, so the gate verdict is not on record.
    assert row.gate_accepted is None
    # What IS stored is still populated for real.
    assert row.gpu_hours == pytest.approx(0.6)
    assert row.metrics == {"quality": 0.9}
    assert row.arm == frozenset({"backend.lora.r"})


def test_partial_memory_fabric_evidence_is_none_rather_than_assumed_off(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("partial", {}))
    registry.record_training_artifact(
        TrainingArtifact(
            "train-partial",
            "partial",
            "/artifact",
            0.2,
            evidence={"hardware_aware_defaults": {"resolved_activation_offload": True}},
        )
    )
    registry.record_result(ExperimentResult("partial", {"quality": 0.9}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert rows[0].memory_fabric_mechanisms is None


def test_no_enabled_mechanisms_is_an_empty_set_not_none(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("resident", {}))
    registry.record_training_artifact(
        TrainingArtifact(
            "train-resident",
            "resident",
            "/artifact",
            0.2,
            evidence={
                "hardware_aware_defaults": {
                    "resolved_activation_offload": False,
                    "resolved_optimizer_tiering": False,
                    "resolved_frozen_layer_streaming": False,
                }
            },
        )
    )
    registry.record_result(ExperimentResult("resident", {"quality": 0.9}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert rows[0].memory_fabric_mechanisms == frozenset()


def test_experiments_without_a_persisted_result_are_skipped(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiments(
        (
            _experiment("ran", {}),
            _experiment("never-ran", {}, status=ExperimentStatus.REJECTED),
        )
    )
    registry.record_result(ExperimentResult("ran", {"quality": 0.9}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert [row.experiment_id for row in rows] == ["ran"]


def test_score_vs_parent_uses_the_parents_own_persisted_result(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiments(
        (
            _experiment("root", {"backend": {"training": {"learning_rate": 1e-3}}}),
            _experiment("child", {"backend": {"lora": {"r": 8}}}, parent_id="root"),
        )
    )
    registry.record_result(ExperimentResult("root", {"quality": 0.80}, 0.3))
    registry.record_result(ExperimentResult("child", {"quality": 0.88}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline(0.70))
    registry.close()

    root, child = rows
    assert root.gate_score_vs_parent is None
    assert child.gate_score_vs_baseline == pytest.approx(0.18)
    assert child.gate_score_vs_parent == pytest.approx(0.08)


def test_ambiguous_training_artifacts_join_nothing_rather_than_guessing(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("retried", {}))
    for run_id in ("train-a", "train-b"):
        registry.record_training_artifact(
            TrainingArtifact(run_id, "retried", "/artifact", 0.2, telemetry={"peak_vram_gb": 1.0})
        )
    # No evidence["training_run_id"], so which artifact produced this
    # result is genuinely not stored.
    registry.record_result(ExperimentResult("retried", {"quality": 0.9}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert rows[0].training_run_id is None
    assert rows[0].peak_vram_gb is None


def test_recorded_training_run_id_disambiguates_multiple_artifacts(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("retried", {}))
    registry.record_training_artifact(
        TrainingArtifact("train-a", "retried", "/a", 0.2, telemetry={"peak_vram_gb": 1.0})
    )
    registry.record_training_artifact(
        TrainingArtifact("train-b", "retried", "/b", 0.7, telemetry={"peak_vram_gb": 9.5})
    )
    registry.record_result(
        ExperimentResult(
            "retried", {"quality": 0.9}, 0.8, evidence={"training_run_id": "train-b"}
        )
    )

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert rows[0].training_run_id == "train-b"
    assert rows[0].peak_vram_gb == pytest.approx(9.5)
    assert rows[0].training_gpu_hours == pytest.approx(0.7)


def test_non_numeric_telemetry_values_are_none_not_coerced(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    registry.record_experiment(_experiment("odd", {}))
    registry.record_training_artifact(
        TrainingArtifact(
            "train-odd",
            "odd",
            "/artifact",
            0.2,
            telemetry={"peak_vram_gb": "n/a", "global_step": 3.5, "train_runtime_seconds": True},
        )
    )
    registry.record_result(ExperimentResult("odd", {"quality": 0.9}, 0.3))

    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())
    registry.close()

    assert rows[0].peak_vram_gb is None
    # A float is not a step count, and a bool is not a measurement.
    assert rows[0].global_step is None
    assert rows[0].train_runtime_seconds is None


def _rows_for_queries(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    specs = (
        ("lr-pass", {"backend": {"training": {"learning_rate": 1e-3}}}, ExperimentStatus.PASSED, 0.90, "tiny-gpt2"),
        ("lr-fail", {"backend": {"training": {"learning_rate": 9e-1}}}, ExperimentStatus.REJECTED, 0.60, "tiny-gpt2"),
        ("lora-pass", {"backend": {"lora": {"r": 8}}}, ExperimentStatus.PASSED, 0.75, "tiny-llama"),
        ("unadjudicated", {"backend": {"lora": {"r": 16}}}, None, 0.72, "tiny-llama"),
    )
    for experiment_id, patch, status, quality, base_model in specs:
        registry.record_experiment(_experiment(experiment_id, patch, status=status))
        registry.record_training_artifact(
            TrainingArtifact(
                f"train-{experiment_id}",
                experiment_id,
                "/artifact",
                0.2,
                evidence={"model_provenance": {"requested_base_model": base_model}},
            )
        )
        registry.record_result(ExperimentResult(experiment_id, {"quality": quality}, 0.3))
    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline(0.70))
    registry.close()
    return rows


def test_filter_outcomes_by_base_model(tmp_path):
    rows = _rows_for_queries(tmp_path)
    selected = filter_outcomes(rows, base_model="tiny-llama")
    assert [row.experiment_id for row in selected] == ["lora-pass", "unadjudicated"]


def test_filter_outcomes_by_intervention_key_path(tmp_path):
    rows = _rows_for_queries(tmp_path)
    selected = filter_outcomes(rows, touches_key_path="backend.training.learning_rate")
    assert [row.experiment_id for row in selected] == ["lr-pass", "lr-fail"]


def test_filter_outcomes_by_gate_acceptance_excludes_unadjudicated_rows(tmp_path):
    rows = _rows_for_queries(tmp_path)
    accepted = filter_outcomes(rows, gate_accepted=True)
    rejected = filter_outcomes(rows, gate_accepted=False)
    assert [row.experiment_id for row in accepted] == ["lr-pass", "lora-pass"]
    assert [row.experiment_id for row in rejected] == ["lr-fail"]
    # "not on record" is never treated as a rejection.
    assert "unadjudicated" not in {row.experiment_id for row in accepted + rejected}


def test_filter_outcomes_by_score_delta_threshold(tmp_path):
    rows = _rows_for_queries(tmp_path)
    # Scores vs. the 0.70 baseline: +0.20, -0.10, +0.05, +0.02.
    assert [row.experiment_id for row in filter_outcomes(rows, min_score_vs_baseline=0.05)] == [
        "lr-pass",
        "lora-pass",
    ]
    assert [row.experiment_id for row in filter_outcomes(rows, min_score_vs_baseline=0.01)] == [
        "lr-pass",
        "lora-pass",
        "unadjudicated",
    ]
    assert filter_outcomes(rows, min_score_vs_baseline=0.5) == ()


def test_filter_outcomes_criteria_are_anded_together(tmp_path):
    rows = _rows_for_queries(tmp_path)
    selected = filter_outcomes(
        rows, base_model="tiny-llama", gate_accepted=True, min_score_vs_baseline=0.05
    )
    assert [row.experiment_id for row in selected] == ["lora-pass"]


def test_filter_outcomes_without_criteria_returns_every_row(tmp_path):
    rows = _rows_for_queries(tmp_path)
    assert filter_outcomes(rows) == rows


def test_group_by_arm_groups_same_key_paths_regardless_of_value(tmp_path):
    rows = _rows_for_queries(tmp_path)
    grouped = group_by_arm(rows)

    learning_rate_arm = frozenset({"backend.training.learning_rate"})
    lora_arm = frozenset({"backend.lora.r"})
    assert set(grouped) == {learning_rate_arm, lora_arm}
    assert [row.experiment_id for row in grouped[learning_rate_arm]] == ["lr-pass", "lr-fail"]
    assert [row.experiment_id for row in grouped[lora_arm]] == ["lora-pass", "unadjudicated"]


def test_group_by_arm_of_no_outcomes_is_empty(tmp_path):
    assert group_by_arm(()) == {}
