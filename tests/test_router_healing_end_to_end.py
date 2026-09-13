"""P10: a router-training experiment through the *normal* project interface.

The plan's milestone is not "a router backend exists" -- it is that a bounded
router-training experiment completes through the interface a user actually has:

    chowder project-validate <project.json>
    chowder train <project.json>

So this module tests the project seam, not the executor again:

* the **contract** layer validates a router project (and refuses the ones that
  could not be measured honestly) without loading anything;
* the **baseline** layer proves the untouched base is scored by the *router*
  evaluator rather than the PEFT text evaluator -- a baseline measured by a
  different scorer than the one that will score the candidate is not a baseline;
* the **real** layer builds a genuine tiny Qwen3 MoE and drives the whole
  generation (baseline -> train -> evaluate -> gate -> registry) through
  :func:`chowder.project_runner.run_project`, which is what the CLI calls.

The real layer is gated on torch/transformers, so it executes in the real-ML
job where a silent skip would show up in the skip count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.backend_selection import ROUTER_HEALING_ENGINE, resolve_training_engine
from chowder.executors import ExecutionContext, EvaluationOutcome
from chowder.lifecycle import (
    PHASE_BASELINE_GENERATION,
    PHASE_CANDIDATE_GENERATION,
    PHASE_MODEL_LOAD,
    ledger_from_payload,
)
from chowder.memory import HardwareProfile
from chowder.project import ProjectValidationError, load_project, project_from_mapping
from chowder.registry import RunRegistry


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        vram_gb=0.0,
        ram_gb=64.0,
        nvme_gb=1000.0,
        pcie_gbps=16.0,
        ram_gbps=50.0,
        nvme_gbps=3.0,
    )


def _require_real_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")


def _write_corpora(root: Path) -> tuple[Path, Path, Path]:
    """A training corpus, a distinct holdout corpus, and a base directory."""
    train = root / "corpus.txt"
    train.write_text(
        "\n".join(
            f"the router selects expert {index % 4} for token number {index} here"
            for index in range(400)
        ),
        encoding="utf-8",
    )
    holdout = root / "holdout.txt"
    holdout.write_text(
        "\n".join(
            f"held out sentence {index} asks which expert answers token {index * 3}"
            for index in range(400)
        ),
        encoding="utf-8",
    )
    base = root / "base"
    base.mkdir(exist_ok=True)
    return base, train, holdout


def _project_payload(root: Path, *, base: Path, train: Path, holdout: Path, **knob_overrides):
    knobs = {
        "base_model_dir": str(base),
        "corpus_path": str(train),
        "holdout_corpus_path": str(holdout),
        "max_steps": 4,
        "learning_rate": 0.05,
        "seq_len": 16,
        "batch_size": 2,
        "seed": 1,
    }
    knobs.update(knob_overrides)
    return {
        "schema_version": 1,
        "name": "router-healing-tiny",
        "work_dir": str(root / "work"),
        "registry_path": str(root / "work" / "runs.db"),
        "seed": 1,
        "goal": {
            "metrics": [{"name": "holdout_loss", "direction": "minimize"}],
            "gpu_hour_budget": 1.0,
            "max_parallel_candidates": 1,
            "minimum_promotion_gain": 0.0,
        },
        "baseline": {"mode": "auto"},
        "experiment": {
            "experiment_id": "router-pilot",
            "estimated_gpu_hours": 0.1,
            "hypothesis": {
                "observation": "the router gates are untrained",
                "suspected_cause": "the base was never adapted to this corpus",
                "intervention": "train mlp.gate.weight with the experts frozen",
            },
        },
        "config": {
            "backend": {
                "type": ROUTER_HEALING_ENGINE,
                "router_healing": knobs,
            },
            "evaluation": {"type": ROUTER_HEALING_ENGINE, "eval_batches": 2},
        },
    }


# --- the contract layer ------------------------------------------------------


def test_a_router_project_loads_and_reports_its_engine(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    project = load_project(path, validate_files=True)

    assert resolve_training_engine(project.config) == ROUTER_HEALING_ENGINE
    # The router project must not be validated against the PEFT-text schema:
    # it has no suite table and no chat template, and inventing them to satisfy
    # a loader would put fake evidence in the project file.
    assert "suites" not in project.config["evaluation"]
    assert project.goal.metrics[0].name == "holdout_loss"


def test_a_router_project_refuses_a_metric_nothing_measures(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    payload["goal"]["metrics"].append({"name": "gsm8k", "direction": "maximize"})

    with pytest.raises(ProjectValidationError, match="not measured by the router-healing"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_a_router_project_refuses_a_goal_without_the_capability_metric(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    payload["goal"]["metrics"] = [{"name": "dead_experts", "direction": "minimize"}]

    with pytest.raises(ProjectValidationError, match="holdout_loss"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_a_router_project_refuses_the_training_corpus_as_its_holdout(tmp_path):
    base, train, _holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=train)

    with pytest.raises(ProjectValidationError, match="measures fit, not capability"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_a_router_project_refuses_a_missing_router_knob(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    del payload["config"]["backend"]["router_healing"]["holdout_corpus_path"]

    with pytest.raises(ProjectValidationError, match="holdout_corpus_path"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_a_router_project_refuses_an_accelerator_device(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(
        tmp_path, base=base, train=train, holdout=holdout, device="cuda"
    )

    with pytest.raises(ProjectValidationError, match="not qualified"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_a_router_project_refuses_the_peft_text_evaluation_type(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    payload["config"]["evaluation"] = {
        "type": "transformers-text",
        "suites": [{"name": "holdout_loss", "dataset": "holdout.txt"}],
    }

    with pytest.raises(ProjectValidationError, match="evaluation.type must be 'router-healing'"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_validation_refuses_files_that_are_not_there(tmp_path):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    holdout.unlink()
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="holdout corpus not found"):
        load_project(path, validate_files=True)


# --- the baseline arm is the router evaluator's, not the text evaluator's ----


def test_the_automatic_baseline_is_measured_by_the_router_evaluator(tmp_path, monkeypatch):
    """A baseline from a different scorer is not a baseline.

    The PEFT text evaluator is replaced with a poison object: if the router
    baseline path ever falls back to it, this test fails loudly instead of
    comparing a router holdout loss against a text scorer's number.
    """
    from chowder import project_runner
    from chowder.backends.router_healing import RouterHealingEvaluator

    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    project = project_from_mapping(payload, source_dir=tmp_path)
    project.work_dir.mkdir(parents=True, exist_ok=True)

    class _PoisonTextEvaluator:
        def evaluate(self, **_kwargs):  # pragma: no cover - the point is not to run
            raise AssertionError("the PEFT text evaluator must not score a router baseline")

    seen: dict[str, object] = {}

    def _fake_evaluate_base(self, *, config, context, experiment_id="baseline"):
        seen["config"] = config
        seen["work_dir"] = context.work_dir
        seen["is_router"] = isinstance(self, RouterHealingEvaluator)
        return EvaluationOutcome(
            run_id="baseline-eval",
            experiment_id=experiment_id,
            source_artifact_ref="base-model:test",
            metrics={"holdout_loss": 3.5, "experts_per_token": 2.0, "dead_experts": 0.0},
            gpu_hours=0.0,
            evidence={
                "backend": "transformers-router-healing-evaluator",
                "base_holdout_loss": 3.5,
                "payload_applied": False,
            },
        )

    monkeypatch.setattr(project_runner, "BaseModelTextEvaluator", _PoisonTextEvaluator)
    monkeypatch.setattr(RouterHealingEvaluator, "evaluate_base", _fake_evaluate_base)

    with RunRegistry(project.registry_path) as registry:
        result, resolved_revision = project_runner._run_automatic_baseline(
            project,
            ExecutionContext(_hardware(), str(project.work_dir), project.seed),
            registry,
            None,
        )
        experiments = list(registry.list_experiments())
        results = list(registry.list_results())

    assert seen["is_router"] is True
    assert seen["config"] is project.config
    assert resolved_revision is None
    assert result.metrics["holdout_loss"] == 3.5
    assert result.artifact_ref is None
    assert [experiment.experiment_id for experiment in experiments] == ["baseline"]
    assert [stored.experiment_id for stored in results] == ["baseline"]


def test_the_automatic_baseline_row_reaches_a_terminal_status(tmp_path, monkeypatch):
    """A measured baseline row must not remain ``planned``.

    ``planned`` means "has not run yet". A row that already carries a scored
    result but stays ``planned`` fabricates a pending experiment that never
    resolves -- the durable record then disagrees with the evidence it
    stores. The convention ``parent_tournament`` already applies to a
    measured base-model row applies here too: measured and trustworthy
    becomes ``passed``. This is a *measurement* verdict, not a gate verdict;
    the gate's own accept/reject lives on the candidate's row.
    """
    from chowder import project_runner
    from chowder.backends.router_healing import RouterHealingEvaluator
    from chowder.models import ExperimentStatus

    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    project = project_from_mapping(payload, source_dir=tmp_path)
    project.work_dir.mkdir(parents=True, exist_ok=True)

    def _fake_evaluate_base(self, *, config, context, experiment_id="baseline"):
        return EvaluationOutcome(
            run_id="baseline-eval",
            experiment_id=experiment_id,
            source_artifact_ref="base-model:test",
            metrics={"holdout_loss": 3.5, "experts_per_token": 2.0, "dead_experts": 0.0},
            gpu_hours=0.0,
            evidence={
                "backend": "transformers-router-healing-evaluator",
                "base_holdout_loss": 3.5,
                "payload_applied": False,
            },
        )

    monkeypatch.setattr(RouterHealingEvaluator, "evaluate_base", _fake_evaluate_base)

    with RunRegistry(project.registry_path) as registry:
        project_runner._run_automatic_baseline(
            project,
            ExecutionContext(_hardware(), str(project.work_dir), project.seed),
            registry,
            None,
        )
        experiments = {
            experiment.experiment_id: experiment for experiment in registry.list_experiments()
        }

    assert experiments["baseline"].status is ExperimentStatus.PASSED


def test_a_failed_baseline_measurement_settles_the_row_as_failed(tmp_path, monkeypatch):
    """A baseline evaluation that raises must still settle its row.

    When the base measurement fails, the exception aborts the project before
    any candidate runs -- but the row exists, and it must not be stranded in
    ``planned`` either. The honest record is ``failed`` with no result row,
    which is exactly what censored-outcome accounting expects of an attempt
    that never produced a scored outcome.
    """
    from chowder import project_runner
    from chowder.backends.router_healing import RouterHealingEvaluator
    from chowder.models import ExperimentStatus

    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    project = project_from_mapping(payload, source_dir=tmp_path)
    project.work_dir.mkdir(parents=True, exist_ok=True)

    def _boom(self, *, config, context, experiment_id="baseline"):
        raise RuntimeError("baseline measurement exploded")

    monkeypatch.setattr(RouterHealingEvaluator, "evaluate_base", _boom)

    with RunRegistry(project.registry_path) as registry:
        with pytest.raises(RuntimeError, match="baseline measurement exploded"):
            project_runner._run_automatic_baseline(
                project,
                ExecutionContext(_hardware(), str(project.work_dir), project.seed),
                registry,
                None,
            )
        experiments = {
            experiment.experiment_id: experiment for experiment in registry.list_experiments()
        }
        result_ids = [result.experiment_id for result in registry.list_results()]

    assert experiments["baseline"].status is ExperimentStatus.FAILED
    assert result_ids == []


# --- the real layer ----------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_router_project(tmp_path_factory):
    """A real tiny Qwen3 MoE (E=4, k=2) and a real router project file."""
    _require_real_model()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    root = tmp_path_factory.mktemp("router-healing-project")
    lines = [
        f"the router selects expert {index % 4} for token number {index} in this sentence"
        for index in range(400)
    ]
    train = root / "corpus.txt"
    train.write_text("\n".join(lines), encoding="utf-8")
    holdout = root / "holdout.txt"
    holdout.write_text(
        "\n".join(
            f"held out sentence {index} asks which expert answers token {index * 3}"
            for index in range(400)
        ),
        encoding="utf-8",
    )

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        lines, trainers.WordLevelTrainer(vocab_size=64, special_tokens=["[UNK]", "[PAD]"])
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]"
    )

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    base = root / "base"
    Qwen3MoeForCausalLM(config).float().save_pretrained(base)
    fast.save_pretrained(base)

    payload = _project_payload(root, base=base, train=train, holdout=holdout)
    project_path = root / "project.json"
    project_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": project_path, "root": root, "project": payload}


def test_a_router_project_validates_through_the_cli_loader(tiny_router_project):
    project = load_project(tiny_router_project["path"], validate_files=True)
    assert project.name == "router-healing-tiny"
    assert resolve_training_engine(project.config) == ROUTER_HEALING_ENGINE


def test_a_router_project_runs_end_to_end_through_the_normal_runner(tiny_router_project):
    """baseline -> train -> evaluate -> gate -> registry, all through run_project.

    This is the plan's P10/P11 CPU leg: one bounded router-training experiment
    through the interface a user has, producing a registry row, a gate verdict,
    a phase ledger and a measured cost.
    """
    from chowder.project_runner import run_project

    outcome = run_project(tiny_router_project["path"])

    candidate = outcome.generation.candidates[0]
    assert candidate.error is None, candidate.error
    assert candidate.succeeded
    assert candidate.artifact is not None
    assert candidate.evaluation is not None

    # The trained artifact is a router payload published next to the run.
    payload_dir = Path(candidate.artifact.artifact_ref)
    assert (payload_dir / "router_payload.safetensors").is_file()
    assert (payload_dir / "router_payload.json").is_file()

    # Training evidence: the exact intended set, and the checks that gate it.
    evidence = candidate.artifact.evidence
    freeze = evidence["freeze_summary"]
    assert freeze["trainable_param_names"], freeze
    assert all(name.endswith("mlp.gate.weight") for name in freeze["trainable_param_names"])
    assert evidence["trainability"]["ok"] is True
    assert evidence["coverage"]["ok"] is True
    assert freeze["layers_with_trainable_shared_expert_gate"] == 0

    # The frozen half of the recipe is verified, and the strategy that verified
    # it is named: a sampled digest is weaker evidence than a full one, so which
    # one ran has to be readable rather than implied.
    frozen = evidence["frozen"]
    assert frozen["ok"] is True
    assert frozen["changed"] == {}
    assert frozen["frozen_parameters"] > 0
    assert set(frozen["digest_strategy"]) <= {"full", "sampled"}
    assert evidence["coverage"]["expected_count"] == len(freeze["trainable_param_names"])
    assert evidence["coverage"]["missing"] == []

    # The evaluation is the router evaluator's, and it applied the payload.
    evaluation_evidence = candidate.evaluation.evidence
    assert evaluation_evidence["backend"] == "transformers-router-healing-evaluator"
    assert evaluation_evidence["payload_applied"] is True
    assert evaluation_evidence["application_control"]["parameters_changed"] is True

    # Cost is measured, not asserted: CPU-only means zero accelerator hours, and
    # the wall time is a real number from the ledger rather than a zero.
    result = candidate.result
    assert result is not None
    assert result.gpu_hours == 0.0
    ledger = ledger_from_payload(evaluation_evidence["phase_ledger"])
    assert ledger.to_dict()["phases"][PHASE_CANDIDATE_GENERATION]["seconds"] > 0.0

    # Durable accounting: the registry can reconstruct the whole generation.
    with RunRegistry(outcome.project.registry_path) as registry:
        experiments = {
            experiment.experiment_id: experiment for experiment in registry.list_experiments()
        }
        results = {stored.experiment_id: stored for stored in registry.list_results()}
        evaluations = list(registry.list_evaluation_outcomes())
        lineage = registry.lineage("router-pilot")

    assert "baseline" in experiments
    assert "router-pilot" in experiments
    assert experiments["router-pilot"].status.value in {"passed", "failed"}
    # The baseline row is a completed measurement: `passed`, not a stranded
    # `planned` beside its own scored result.
    assert experiments["baseline"].status.value == "passed"
    assert results["router-pilot"].metrics["holdout_loss"] > 0.0
    # The project path records the automatic baseline as its own experiment row
    # compared by value, not as the candidate's parent: `parent_id` stays None for
    # a project-ranked candidate, exactly as it already does for PEFT projects.
    # The durable link is instead the shared measurement -- asserted below.
    assert lineage == ()

    # Both arms were measured by the router evaluator, under one protocol.
    base_evaluations = [entry for entry in evaluations if entry.evidence.get("arm") == "base"]
    candidate_evaluations = [
        entry for entry in evaluations if entry.evidence.get("arm") == "candidate"
    ]
    assert len(base_evaluations) == 1
    assert len(candidate_evaluations) == 1
    assert base_evaluations[0].evidence["payload_applied"] is False
    assert candidate_evaluations[0].evidence["payload_applied"] is True
    # The gate's baseline and the base arm measured inside the candidate's own
    # evaluation are the same number: that is the durable link between the two
    # registry rows, not a parent_id the project path does not set.
    assert results["baseline"].metrics["holdout_loss"] == pytest.approx(
        candidate_evaluations[0].evidence["base_holdout_loss"]
    )
    # The base arm's own ledger cannot claim a candidate comparison it never ran,
    # so its required set is smaller -- but it is still *required*, not skipped.
    base_ledger = ledger_from_payload(base_evaluations[0].evidence["phase_ledger"])
    base_ledger.require([PHASE_MODEL_LOAD, PHASE_BASELINE_GENERATION], purpose="test")
    assert base_ledger.to_dict()["unmeasured"][PHASE_CANDIDATE_GENERATION]
