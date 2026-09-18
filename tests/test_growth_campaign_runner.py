"""A preregistered manifest drives one real generation cycle.

The mission's reusable Gen-2+ path: ``chowder growth campaign plan|run``.
These tests exercise the production assembly end to end -- identity, the
declared inputs, curriculum planning, recipe selection, admission, execution,
settlement, candidate selection, adjudication and the lineage record -- with
the recording runner from ``test_growth_training_binding`` standing in for the
trainer subprocess. That is the seam the binding tests use, so the only thing
faked is the GPU.

Every field the manifest declares is proven to change behavior or to refuse
the run: ``FIELD_ENFORCEMENT`` names the behavior each one drives and
``assert_every_field_enforced`` fails if the schema and that table diverge, and
the tests below exercise the load-bearing ones -- ceilings, stopping rules,
declared inputs, the parent digest, the recipe set and the promotion sets.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from chowder.cli import main as chowder_main
from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth import campaign_runner
from chowder.growth.campaign import (
    STOPPING_RULE_ENFORCEMENT,
    STOPPING_RULE_ON_ADMISSION_REFUSAL,
    STOPPING_RULE_ON_CAMPAIGN_OVERRUN,
    STOPPING_RULES,
    CampaignManifest,
    CampaignManifestError,
    stops_on_admission_refusal,
    stops_on_campaign_overrun,
)
from chowder.growth.campaign_runner import (
    FIELD_ENFORCEMENT,
    NON_BEHAVIORAL_FIELDS,
    CampaignRunRefusal,
    assert_every_field_enforced,
    plan_campaign,
    run_campaign,
)
from chowder.growth.capability import ALL_SKILLS, CapabilityProfile, SkillEstimate
from chowder.growth.lineage import GenerationLedger
from chowder.growth.training_binding import directory_digest

from test_growth_training_binding import _RecordingRunner, _source, _template

TARGET_ID = "generation-diagnostics@gen1-eval-protocol-v1"
PROTECTED_ID = "math500@2024-04"
BROAD_ID = "mgsm@2022-11"
CALIBRATION_ID = "simpleqa_verified@2025-09"
TARGET_METRIC = "eos_termination_rate"

PARENT_VERSION = "gen1"
CANDIDATE_VERSION = "gen2"

DEVICE_PER_RECIPE = 0.30
WALL_PER_RECIPE = 0.10
#: Wall-charged cost one attempt reports. Deliberately far above what the
#: planner *projects* for this hardware, so the difference between an admitted
#: plan and an overrunning actual is a real signal in these tests.
ATTEMPT_WALL_GPU_HOURS = 0.05

HARDWARE: Mapping[str, Any] = {
    "gpu_name": "test-gpu",
    "vram_gb": 24.0,
    "measured_step_seconds_at_seq": {"512": 0.001, "1024": 0.002, "2048": 0.004},
    "measured_load_seconds": 5.0,
    "wall_multiplier": 3.5,
}

GOAL: Mapping[str, Any] = {
    "metrics": [{"name": "holdout_loss", "direction": "minimize"}],
    "gpu_hour_budget": WALL_PER_RECIPE,
    "max_parallel_candidates": 1,
    "minimum_promotion_gain": 0.0,
}

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFEST = ROOT / "docs" / "gen2_campaign_manifest.example.json"


# --------------------------------------------------------------------------
# declared-input fixture: everything the manifest points at is written here
# --------------------------------------------------------------------------


def _row(
    qualified_id: str,
    version: str,
    origin: str,
    samples: Sequence[float],
    *,
    metric: str = "accuracy",
) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=version,
        score=sum(samples) / len(samples),
        n_samples=len(samples),
        per_sample_scores=tuple(float(value) for value in samples),
        metric=metric,
        measurement_origin=origin,
    )


def _write_reports(inputs: Path) -> dict[str, str]:
    """Parent-measured and candidate-measured rows, written by the producer."""
    parent_runs = (
        _row(TARGET_ID, PARENT_VERSION, MEASURED_PARENT, (0, 0, 0, 0, 1, 0, 0, 0), metric=TARGET_METRIC),
        _row(PROTECTED_ID, PARENT_VERSION, MEASURED_PARENT, (0, 1, 0, 1, 0, 1, 0, 1)),
        _row(BROAD_ID, PARENT_VERSION, MEASURED_PARENT, (0, 1, 1, 0, 0, 1, 1, 0)),
    )
    candidate_runs = (
        _row(TARGET_ID, CANDIDATE_VERSION, MEASURED_THIS_GENERATION, (1, 1, 1, 1, 1, 1, 1, 1), metric=TARGET_METRIC),
        _row(PROTECTED_ID, CANDIDATE_VERSION, MEASURED_THIS_GENERATION, (0, 1, 0, 1, 0, 1, 0, 1)),
        _row(BROAD_ID, CANDIDATE_VERSION, MEASURED_THIS_GENERATION, (0, 1, 1, 0, 0, 1, 1, 0)),
    )
    EvalReport(generation_version=PARENT_VERSION, runs=parent_runs).save(
        inputs / "parent-eval-report.json"
    )
    EvalReport(generation_version=CANDIDATE_VERSION, runs=candidate_runs).save(
        inputs / "candidate-eval-report.json"
    )
    return {
        "parent_eval_report_path": str(inputs / "parent-eval-report.json"),
        "candidate_eval_report_path": str(inputs / "candidate-eval-report.json"),
    }


def _declaration(tmp_path: Path, *, recipe_count: int = 2, **overrides: Any) -> dict[str, Any]:
    """The manifest document, with every declared input it names on disk."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "parent-model"
    parent.mkdir(exist_ok=True)
    (parent / "config.json").write_text("{}", encoding="utf-8")
    digest, _entries = directory_digest(parent)

    profile = CapabilityProfile(
        model_version=PARENT_VERSION,
        raw_scores={TARGET_ID: 0.125, PROTECTED_ID: 0.5, BROAD_ID: 0.5},
        skills=tuple(
            SkillEstimate(skill=skill, estimate=0.25, confidence=0.9, evidence=(TARGET_ID,))
            for skill in ALL_SKILLS
        ),
    )
    (inputs / "parent-profile.json").write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    (inputs / "hardware-budget.json").write_text(json.dumps(HARDWARE), encoding="utf-8")
    (inputs / "data-registry.json").write_text(
        json.dumps({"sources": [asdict(_source("src-1", inclusion_decision="included"))]}),
        encoding="utf-8",
    )
    (inputs / "project-template.json").write_text(
        json.dumps(_template(tmp_path, goal=dict(GOAL))), encoding="utf-8"
    )
    (inputs / "contamination.json").write_text(
        json.dumps(
            {
                # The firewall's own section shape: each benchmark maps to a
                # verdict object, exactly as ``ContaminationFirewall.manifest``
                # writes it. A bare string here would fail to bind at all.
                "benchmarks": {
                    qid: {"status": "CLEAN", "reason": "fixture"}
                    for qid in (TARGET_ID, PROTECTED_ID, BROAD_ID)
                },
                "policy": {},
                "training_sources": {},
            }
        ),
        encoding="utf-8",
    )

    document: dict[str, Any] = {
        "cycle_id": "gen2-campaign",
        "parent_version": PARENT_VERSION,
        "base_model_path": str(parent),
        "base_model_digest": digest,
        "state_root": str(tmp_path / "state"),
        "target_benchmarks": [TARGET_ID],
        "protected_benchmarks": [PROTECTED_ID],
        "broad_benchmarks": [BROAD_ID],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": {
            "device_gpu_hours_ceiling_per_recipe": DEVICE_PER_RECIPE,
            "wall_gpu_hours_ceiling_per_recipe": WALL_PER_RECIPE,
            "device_gpu_hours_ceiling_campaign": 0.60,
            "wall_gpu_hours_ceiling_campaign": WALL_PER_RECIPE * recipe_count,
        },
        "recipes": [f"recipe-placeholder-{index:02d}" for index in range(recipe_count)],
        "candidate_selection_policy": "first_successful",
        "stopping_rules": [
            STOPPING_RULE_ON_ADMISSION_REFUSAL,
            STOPPING_RULE_ON_CAMPAIGN_OVERRUN,
        ],
        "contamination_manifest_path": str(inputs / "contamination.json"),
        "project_template_path": str(inputs / "project-template.json"),
        "data_registry_path": str(inputs / "data-registry.json"),
        "hardware_budget_path": str(inputs / "hardware-budget.json"),
        "parent_profile_path": str(inputs / "parent-profile.json"),
        "notes": "campaign-runner fixture",
    }
    document.update(_write_reports(inputs))
    document.update(overrides)
    return document


def _campaign(
    tmp_path: Path,
    *,
    runner: Any = None,
    planned: bool = True,
    **overrides: Any,
) -> tuple[CampaignManifest, Any, dict[str, Any]]:
    """Write the declaration, plan it with production, declare its recipes.

    ``planned=False`` leaves the declared recipe ids alone, so a manifest that
    names recipes the planner never proposed can be exercised.
    """
    runner = runner if runner is not None else _RecordingRunner(gpu_hours=ATTEMPT_WALL_GPU_HOURS)
    document = _declaration(tmp_path, **overrides)
    inputs = tmp_path / "inputs"
    manifest_path = inputs / "campaign.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    plan = plan_campaign(CampaignManifest.from_file(manifest_path))
    item_ids = [item.item_id for item in plan.items]
    document["training_material_path"] = str(inputs / "training-material.json")
    (inputs / "training-material.json").write_text(
        json.dumps(
            {
                "sources": {item_id: "src-1" for item_id in item_ids},
                "material": {
                    item_id: [f"synthetic line {index} for {item_id}" for index in range(40)]
                    for item_id in item_ids
                },
            }
        ),
        encoding="utf-8",
    )
    if planned:
        document["recipes"] = [recipe.recipe_id for recipe in plan.recipes]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return CampaignManifest.from_file(manifest_path), runner, document


def _verbs(runner: Any) -> list[str]:
    return [command[-2] for command in runner.commands]


def _phase(run: Any, name: str) -> Mapping[str, Any]:
    for phase in run.phases:
        if phase["phase"] == name:
            return phase
    raise AssertionError(f"the run recorded no {name!r} phase: {[p['phase'] for p in run.phases]}")


def _redeclare(document: Mapping[str, Any], **changes: Any) -> CampaignManifest:
    return CampaignManifest.from_mapping({**document, **changes})


# --------------------------------------------------------------------------
# a manifest-driven campaign reaches a verdict
# --------------------------------------------------------------------------


def test_a_declared_campaign_reaches_a_recorded_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Plan -> admission -> train -> settle -> adjudicate -> ledger, all driven
    by the declaration and nothing else."""
    manifest, runner, _document = _campaign(tmp_path)
    monkeypatch.setattr(campaign_runner, "default_runner", runner)

    run = run_campaign(manifest)

    assert run.verdict == "PROMOTED"
    assert (run.parent_version, run.candidate_version) == (PARENT_VERSION, CANDIDATE_VERSION)
    # Admission is the executor's decision, asked once per declared recipe.
    assert [entry["admitted"] for entry in run.admission] == [True, True]
    assert all(entry["projected_wall_gpu_hours"] > 0 for entry in run.admission)
    assert _phase(run, "plan")["verdict"] == "ok"
    assert _phase(run, "admission")["verdict"] == "ok"
    assert _phase(run, "campaign_projection")["verdict"] == "ok"
    # Exactly the declared recipe set ran, through the production executor.
    assert _verbs(runner).count("project-validate") == 2
    assert _verbs(runner).count("train") == 2
    # Settlement is post-run, and it reports the unit it actually measured.
    assert run.settlement["budget_compliant"] is True
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)
    assert run.cost["device_measured"] is False
    assert run.ceiling_enforcement["device"] == "admission:projected_plan"
    assert run.ceiling_enforcement["wall"] == "settlement:measured"
    assert _phase(run, "promotion")["verdict"] == "PROMOTED"

    # The verdict is durable under the derived candidate version, with the
    # cycle accounting artifact beside it.
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert ledger.effective_verdict(CANDIDATE_VERSION) == "PROMOTED"
    record = json.loads(Path(run.record_path).read_text(encoding="utf-8"))
    assert record["verdict"] == "PROMOTED"
    assert record["cycle_outcome"]["verdict"] == "PROMOTED"
    assert Path(record["cost"]["accounting_path"]).exists()
    assert record["cost"]["accounting_digest"]


def test_the_candidate_version_is_declared_or_derived_never_invented(tmp_path: Path):
    manifest, _runner, _document = _campaign(tmp_path)
    assert manifest.candidate_version == ""
    assert manifest.resolved_candidate_version() == CANDIDATE_VERSION

    declared, _runner2, _document2 = _campaign(
        tmp_path / "declared", candidate_version="custom-label"
    )
    assert declared.resolved_candidate_version() == "custom-label"


# --------------------------------------------------------------------------
# budgets and stopping rules change behavior
# --------------------------------------------------------------------------


def _tight_campaign_budget(**overrides: Any) -> dict[str, float]:
    """A campaign envelope a plan fits but the actual spend does not."""
    budget = {
        "device_gpu_hours_ceiling_per_recipe": DEVICE_PER_RECIPE,
        "wall_gpu_hours_ceiling_per_recipe": WALL_PER_RECIPE,
        "device_gpu_hours_ceiling_campaign": 0.60,
        "wall_gpu_hours_ceiling_campaign": WALL_PER_RECIPE * 0.4,
    }
    budget.update(overrides)
    return budget


def test_a_campaign_ceiling_breach_stops_the_remaining_recipes_and_vetoes_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The declared stopping rule changes how much compute runs, and a
    campaign that blew its own envelope does not promote."""
    manifest, runner, _document = _campaign(tmp_path, budget=_tight_campaign_budget())
    monkeypatch.setattr(campaign_runner, "default_runner", runner)

    run = run_campaign(manifest)

    assert _verbs(runner).count("train") == 1
    assert run.verdict == "REJECTED"
    assert _phase(run, "stopping")["verdict"] == "stopped"
    assert _phase(run, "resource_veto")["verdict"] == "REJECTED"
    assert run.settlement["budget_compliant"] is False
    assert any("WALL" in reason for reason in run.settlement["budget_failure_reasons"])
    # The artifact, the measurements and the honest verdict all survive, but
    # the overrun campaign records no promoted generation: the resource veto
    # is authoritative over lineage, not just over the report.
    record = json.loads(Path(run.record_path).read_text(encoding="utf-8"))
    assert record["verdict"] == "REJECTED"
    assert record["cycle_outcome"]["verdict"] == "REJECTED"
    assert Path(record["cost"]["accounting_path"]).exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


def test_without_the_overrun_rule_every_recipe_runs_and_the_resource_gate_still_vetoes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest, runner, _document = _campaign(
        tmp_path,
        budget=_tight_campaign_budget(),
        stopping_rules=[STOPPING_RULE_ON_ADMISSION_REFUSAL],
    )
    monkeypatch.setattr(campaign_runner, "default_runner", runner)

    run = run_campaign(manifest)

    assert _verbs(runner).count("train") == 2
    assert run.verdict == "REJECTED"
    assert _phase(run, "settlement")["verdict"] == "violated"
    assert "stopping" not in [phase["phase"] for phase in run.phases]


def test_a_plan_that_does_not_fit_the_campaign_envelope_refuses_before_compute(
    tmp_path: Path,
):
    """Admission and settlement are different controls; neither substitutes
    for the other, and the campaign-level device ceiling has real teeth even
    when the executor reports wall only."""
    _manifest, runner, document = _campaign(tmp_path)
    for field_name, code in (
        ("wall_gpu_hours_ceiling_campaign", "PROJECTED_WALL_GPU_HOURS_EXCEEDED"),
        ("device_gpu_hours_ceiling_campaign", "PROJECTED_DEVICE_GPU_HOURS_EXCEEDED"),
    ):
        refused = run_campaign(
            _redeclare(document, budget={**_tight_campaign_budget(), field_name: 1e-6})
        )
        assert refused.verdict == "REFUSED"
        assert code in _phase(refused, "campaign_projection")["detail"]
    assert runner.commands == []


def test_an_unknown_stopping_rule_refuses_at_load(tmp_path: Path):
    document = _declaration(tmp_path, stopping_rules=["stop whenever it feels slow"])
    path = tmp_path / "inputs" / "campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CampaignManifestError) as error:
        CampaignManifest.from_file(path)
    assert "stopping rule" in str(error.value)


def test_every_recognized_stopping_rule_names_what_enforces_it():
    """The alias table is the contract: no accepted rule is decoration."""
    assert set(STOPPING_RULES) == set(STOPPING_RULE_ENFORCEMENT)
    assert all(text.strip() for text in STOPPING_RULE_ENFORCEMENT.values())
    for spelling in ("stop on admission refusal", "stop before compute on admission refusal"):
        assert stops_on_admission_refusal((spelling,))
    for spelling in (
        "stop on campaign overrun",
        "stop on campaign settlement overrun",
        "stop on campaign settlement overrun (artifact preserved)",
    ):
        assert stops_on_campaign_overrun((spelling,))
    # A rule for a different behavior does not trigger either branch.
    assert not stops_on_campaign_overrun(("stop on admission refusal",))
    assert not stops_on_admission_refusal(("stop on campaign overrun",))


def test_the_committed_gen2_preregistration_manifest_still_loads():
    """``docs/quals/GEN2_PREREG`` claims this manifest validates; the alias
    table above is what keeps that claim true as the rule spellings vary."""
    manifest = CampaignManifest.from_file(
        ROOT / "docs" / "gen2" / "gen2_campaign.json"
    )
    assert manifest.cycle_id == "gen2-response-surface-compliance"
    assert manifest.recipe_ids
    unknown = [rule for rule in manifest.stopping_rules if rule not in STOPPING_RULES]
    assert unknown == []

    # The trusted-ancestor arm is declared, so the judge's T16 gate is
    # decidable: an undeclared input would produce no arm and leave every gen2
    # candidate permanently INCONCLUSIVE (GEN2_PREREG_AMENDMENT2_2026-09-18).
    ancestor = Path(manifest.baseline_eval_report_path)
    assert ancestor.name == "gen0-baseline-evaluation.json"
    # The declared input must not be the judged artifact itself: the runner
    # copies it into the run root, it does not write it in place.
    assert ancestor.name != "baseline_evaluation.json"
    assert ancestor.parent != Path(manifest.state_root)

    amendment2 = ROOT / "docs" / "quals" / "GEN2_PREREG_AMENDMENT2_2026-09-18.md"
    assert amendment2.is_file()
    prereg = (ROOT / "docs" / "quals" / "GEN2_PREREG_2026-09-17.md").read_text(
        encoding="utf-8"
    )
    assert amendment2.name in prereg


# --------------------------------------------------------------------------
# declared inputs, identity and the recipe set
# --------------------------------------------------------------------------


def test_a_missing_declared_input_refuses_before_any_subprocess(tmp_path: Path):
    manifest, runner, document = _campaign(tmp_path)
    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(_redeclare(document, project_template_path=""))
    assert "project_template_path" in str(error.value)
    assert runner.commands == []


def _bind_row_artifact(
    report_path: Path, qualified_id: str, *, reference: str, payload: str
) -> None:
    """Rebind one row of a written report to an artifact, digest included."""
    report = EvalReport.load(report_path)
    runs = tuple(
        replace(
            run,
            raw_artifact_ref=reference,
            metadata={
                **(run.metadata or {}),
                "artifact_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
        )
        if run.benchmark_qualified_id == qualified_id
        else run
        for run in report.runs
    )
    EvalReport(generation_version=report.generation_version, runs=runs).save(report_path)


def test_the_run_carries_the_measurements_its_rows_name(tmp_path: Path, monkeypatch):
    """A row bound to an artifact gets those bytes in the run root.

    The judge recomputes the digest a row declares over the file it names, so the
    evidence the run writes has to include the measurement, not only the report
    that points at it.
    """
    manifest, runner, _document = _campaign(tmp_path)
    monkeypatch.setattr(campaign_runner, "default_runner", runner)
    inputs = tmp_path / "inputs"
    reference = "raw/candidate-target.json"
    payload = '{"target": "measured"}'
    artifact = inputs / reference
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(payload, encoding="utf-8")
    _bind_row_artifact(
        inputs / "candidate-eval-report.json", TARGET_ID, reference=reference, payload=payload
    )

    run = run_campaign(manifest)

    assert run.verdict == "PROMOTED"
    carried = Path(manifest.state_root) / reference
    assert carried.read_bytes() == artifact.read_bytes()
    written = [
        entry["detail"] for entry in run.phases if entry["phase"] == "certification_evidence"
    ][0]
    assert Path(reference).name in written


def test_a_row_that_names_a_missing_artifact_refuses_the_run(tmp_path: Path, monkeypatch):
    """Evidence is not assembled around a measurement nobody can read."""
    manifest, runner, _document = _campaign(tmp_path)
    monkeypatch.setattr(campaign_runner, "default_runner", runner)
    inputs = tmp_path / "inputs"
    _bind_row_artifact(
        inputs / "candidate-eval-report.json",
        TARGET_ID,
        reference="raw/never-written.json",
        payload="unused",
    )

    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(manifest)

    assert "raw/never-written.json" in str(error.value)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()


def test_two_arms_naming_one_artifact_differently_refuse(tmp_path: Path, monkeypatch):
    """One run root cannot carry two different measurements under one name."""
    manifest, runner, _document = _campaign(tmp_path)
    monkeypatch.setattr(campaign_runner, "default_runner", runner)
    inputs = tmp_path / "inputs"
    reference = "raw/shared.json"
    # Two arms, in different directories, each naming the same relative path: the
    # refs are relative to the report that declares them, so these are two
    # different measurements that would land on one run-root path.
    (inputs / "parent").mkdir(parents=True, exist_ok=True)
    (inputs / "parent-eval-report.json").replace(
        inputs / "parent" / "parent-eval-report.json"
    )
    for report_path, qualified_id, payload in (
        (inputs / "candidate-eval-report.json", TARGET_ID, '{"arm": "candidate"}'),
        (inputs / "parent" / "parent-eval-report.json", PROTECTED_ID, '{"arm": "parent"}'),
    ):
        artifact = report_path.parent / reference
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(payload, encoding="utf-8")
        _bind_row_artifact(
            report_path, qualified_id, reference=reference, payload=payload
        )
    document = json.loads((inputs / "campaign.json").read_text(encoding="utf-8"))
    document["parent_eval_report_path"] = str(
        inputs / "parent" / "parent-eval-report.json"
    )
    (inputs / "campaign.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(CampaignManifest.from_file(inputs / "campaign.json"))

    assert "different content" in str(error.value)


def test_a_parent_digest_that_does_not_match_the_tree_refuses_before_compute(tmp_path: Path):
    _manifest, runner, document = _campaign(tmp_path)
    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(_redeclare(document, base_model_digest="b" * 64))
    assert "does not match" in str(error.value)
    assert runner.commands == []


def test_a_parent_adapter_is_digest_verified_separately_from_the_base(tmp_path: Path):
    """A base digest must never be accepted as proof of an adapter tree."""
    _manifest, runner, document = _campaign(tmp_path)
    adapter = tmp_path / "gen1-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    adapter_digest, _entries = directory_digest(adapter)

    # Correct adapter digest: verified, and the phase says which object it was.
    _manifest2, runner2, document2 = _campaign(tmp_path / "ok")
    ok_adapter = tmp_path / "ok" / "gen1-adapter"
    ok_adapter.mkdir()
    (ok_adapter / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    ok_digest, _entries2 = directory_digest(ok_adapter)
    accepted = run_campaign(
        _redeclare(
            document2,
            parent_adapter_path=str(ok_adapter),
            parent_adapter_digest=ok_digest,
        )
    )
    assert _phase(accepted, "identity")["verdict"] == "ok"
    assert ok_digest[:12] in _phase(accepted, "identity")["detail"]

    # The base's digest is not an adapter digest: the run refuses before compute.
    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(
            _redeclare(
                document,
                parent_adapter_path=str(adapter),
                parent_adapter_digest=document["base_model_digest"],
            )
        )
    assert "adapter" in str(error.value)
    assert adapter_digest not in str(error.value) or "does not match" in str(error.value)
    assert runner.commands == []


def test_a_recipe_the_planner_did_not_propose_refuses_rather_than_substituting(
    tmp_path: Path,
):
    manifest, runner, _document = _campaign(
        tmp_path, planned=False, recipes=["recipe-a", "recipe-b"]
    )
    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(manifest)
    assert "did not propose" in str(error.value)
    assert "recipe-a" in str(error.value)
    assert runner.commands == []


def test_the_plan_prints_the_ids_a_run_will_honor_without_compute(tmp_path: Path):
    manifest, runner, _document = _campaign(tmp_path, planned=False)
    plan = plan_campaign(manifest)
    proposed = [recipe.recipe_id for recipe in plan.recipes]
    assert proposed and all(recipe_id.startswith("recipe-") for recipe_id in proposed)
    assert all(recipe.projected_wall_gpu_hours > 0 for recipe in plan.recipes)

    # What the plan prints is accepted back verbatim as the declaration.
    accepted, _runner2, _document2 = _campaign(tmp_path / "accepted", planned=False, recipes=proposed)
    assert list(accepted.recipe_ids) == proposed
    assert [recipe.recipe_id for recipe in plan_campaign(accepted).recipes] == proposed
    assert runner.commands == []


# --------------------------------------------------------------------------
# the declared promotion sets are the ones judged
# --------------------------------------------------------------------------


def test_a_declared_calibration_set_is_not_promoted_without_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest, runner, _document = _campaign(tmp_path, calibration_benchmarks=[CALIBRATION_ID])
    monkeypatch.setattr(campaign_runner, "default_runner", runner)

    run = run_campaign(manifest)

    assert run.verdict == "INCONCLUSIVE"


def test_every_declared_field_is_enforced_or_the_run_refuses():
    assert_every_field_enforced()
    assert set(FIELD_ENFORCEMENT) == set(CampaignManifest.__dataclass_fields__)
    assert NON_BEHAVIORAL_FIELDS == {"notes"}
    assert all(behavior.strip() for behavior in FIELD_ENFORCEMENT.values())
    with pytest.raises(CampaignManifestError):
        assert_every_field_enforced(_DriftedManifest)


class _DriftedManifest(CampaignManifest):
    """A schema carrying a field the enforcement table does not name."""

    __dataclass_fields__ = {  # type: ignore[assignment]
        **CampaignManifest.__dataclass_fields__,
        "an_unenforced_field": object(),
    }


# --------------------------------------------------------------------------
# the committed example, planned and run through the real CLI
# --------------------------------------------------------------------------


def test_the_example_manifest_is_planned_and_run_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The example the docs point at validates, plans and runs end to end.

    Only environment-specific values are substituted (declared input paths,
    the parent tree and its digest, the state root): the declaration -- sets,
    ceilings, how many recipes -- is the committed example's, and the recipe
    ids it runs are the ones the CLI's own ``plan`` printed.
    """
    manifest, runner, document = _campaign(tmp_path)
    monkeypatch.setattr(campaign_runner, "default_runner", runner)

    example = json.loads(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    for field_name, value in document.items():
        if field_name.endswith("_path"):
            example[field_name] = value
    example["base_model_path"] = document["base_model_path"]
    example["base_model_digest"] = document["base_model_digest"]
    example["state_root"] = str(tmp_path / "example-state")
    # The committed example demonstrates the adapter identity fields; this
    # fixture has no adapter on disk, so the declared pair is dropped rather
    # than pointed at machine-specific bytes a test must not depend on.
    example.pop("parent_adapter_path", None)
    example.pop("parent_adapter_digest", None)
    example_path = tmp_path / "example-campaign.json"
    example_path.write_text(json.dumps(example), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["chowder", "growth", "campaign", "plan", str(example_path)])
    assert chowder_main() == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "PLANNED"
    assert planned["recipes"] and planned["recipes_declared"] is False

    example["recipes"] = list(planned["recipes"])
    example_path.write_text(json.dumps(example), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["chowder", "growth", "campaign", "run", str(example_path)])
    assert chowder_main() == 0
    outcome = json.loads(capsys.readouterr().out)

    assert outcome["verdict"] == "PROMOTED"
    assert outcome["cycle_id"] == example["cycle_id"]
    assert outcome["candidate_version"] == CANDIDATE_VERSION
    assert Path(outcome["record_path"]).exists()
    assert _verbs(runner).count("train") == len(example["recipes"])
    assert manifest.cycle_id == "gen2-campaign"


def test_the_cli_refuses_a_malformed_manifest_without_touching_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    example = json.loads(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    example["stop_after"] = "when it is good enough"
    path = tmp_path / "typo-campaign.json"
    path.write_text(json.dumps(example), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["chowder", "growth", "campaign", "run", str(path)])

    with pytest.raises(CampaignManifestError) as error:
        chowder_main()
    assert "unknown manifest fields" in str(error.value)
    assert capsys.readouterr().out == ""
