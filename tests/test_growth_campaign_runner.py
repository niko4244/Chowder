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
    CANDIDATE_ARTIFACT_DIGEST_STALE,
    FIELD_ENFORCEMENT,
    NON_BEHAVIORAL_FIELDS,
    CampaignRunRefusal,
    assert_every_field_enforced,
    plan_campaign,
    run_campaign,
    undeclared_inputs,
)
from chowder.growth.candidate_evaluation import (
    CANDIDATE_EVALUATION_COST_UNMEASURED,
    CANDIDATE_EVALUATION_COST_UNREPORTED,
    CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK,
    CANDIDATE_EVALUATION_IDENTITY_UNBOUND,
    CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED,
    CANDIDATE_EVALUATION_NOT_PRODUCED,
    CandidateEvaluation,
    CandidateEvaluationRefusal,
    EvaluationRequest,
)
from chowder.growth.target_selection import build_skill_profile
from chowder.growth.compute_cost import ComputeCost
from chowder.growth.lineage import GenerationLedger
from chowder.growth.training_binding import directory_digest
from chowder.local_model_manifest import model_content_digest

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


#: The declared protection policy this fixture campaigns with: the frozen
#: 16-item mini-slice protocol, a gen0 trusted ancestor, and the frozen tolerance.
PROTECTION: dict[str, Any] = {
    "trusted_ancestor_version": "gen0",
    "slice_regression_max": 0.0625,
    "n_samples": 16,
    "seed": 1234,
    "shuffle": False,
    "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512},
    "prompt_policy": "chat_template",
}


#: Sixteen per-sample values, the frozen mini-slice size, so the protected row a
#: campaign certifies against is the shape certification requires.
PROTECTED_SAMPLES: tuple[float, ...] = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)


def _protected_artifact_ref(version: str) -> str:
    """Version-scoped, so two arms never claim one run-root path with two sets of
    bytes (which the evidence writer refuses, correctly)."""
    return f"raw/protected-{version}-slice.json"


def _row(
    qualified_id: str,
    version: str,
    origin: str,
    samples: Sequence[float],
    *,
    metric: str = "accuracy",
    protocol: bool = False,
    artifact: tuple[str, str] | None = None,
) -> BenchmarkRun:
    metadata: dict[str, Any] = {}
    if protocol:
        metadata.update(
            {
                "sample_indices": list(range(len(samples))),
                "seed": 1234,
                "shuffle": False,
                "decoding": dict(PROTECTION["decoding"]),
                "prompt_policy": "chat_template",
            }
        )
    reference = ""
    if artifact is not None:
        reference, digest = artifact
        metadata["artifact_sha256"] = digest
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=version,
        score=sum(samples) / len(samples),
        n_samples=len(samples),
        per_sample_scores=tuple(float(value) for value in samples),
        metric=metric,
        measurement_origin=origin,
        raw_artifact_ref=reference,
        metadata=metadata,
    )


def _slice_artifact(report_path: Path, *, version: str, qualified_id: str) -> tuple[str, str]:
    """The raw bytes one protected row names, written beside the report."""
    payload = json.dumps(
        {"generation_version": version, "benchmark": qualified_id, "n": 16}, sort_keys=True
    )
    reference = _protected_artifact_ref(f"{version}-{qualified_id.replace('@', '-')}")
    artifact = report_path.parent / reference
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(payload, encoding="utf-8")
    return reference, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol_rows(
    report_path: Path,
    *,
    version: str,
    origin: str,
    samples: Sequence[float],
) -> tuple[BenchmarkRun, ...]:
    """One declared-protocol measurement per protected/broad id in the arm."""
    return tuple(
        _row(
            qualified_id,
            version,
            origin,
            samples,
            protocol=True,
            artifact=_slice_artifact(report_path, version=version, qualified_id=qualified_id),
        )
        for qualified_id in (PROTECTED_ID, BROAD_ID)
    )


def _write_reports(
    inputs: Path,
    *,
    adapter_digest: str,
    candidate_digest: str = "",
    with_ancestor: bool = False,
    base_digest: str = "",
) -> dict[str, str]:
    """The measured arms, written the way a producer writes them.

    The protected rows carry the declared protocol, the per-sample values, a
    real artifact beside the report and its digest -- the evidence certification
    verifies. Each arm's ``model_identity`` names the bytes it measured: the
    parent arm the frozen parent adapter, the ancestor arm the dense base, and
    the candidate arm whatever digest the caller passes (absent by default,
    because a candidate arm bound to nothing is exactly the state that must not
    promote).
    """
    parent_path = inputs / "parent-eval-report.json"
    EvalReport(
        generation_version=PARENT_VERSION,
        runs=(
            # Sixteen items, the size the declared mini-slice protocol pins: the
            # campaign measures every declared set under one protocol, so the
            # target arm is a slice too. Score is 2/16 = 0.125, as before.
            _row(
                TARGET_ID,
                PARENT_VERSION,
                MEASURED_PARENT,
                (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                metric=TARGET_METRIC,
            ),
            *_protocol_rows(
                parent_path, version=PARENT_VERSION, origin=MEASURED_PARENT,
                samples=PROTECTED_SAMPLES,
            ),
        ),
        model_identity={"adapter_digest": adapter_digest} if adapter_digest else {},
    ).save(parent_path)

    candidate_path = inputs / "candidate-eval-report.json"
    EvalReport(
        generation_version=CANDIDATE_VERSION,
        runs=(
            _row(TARGET_ID, CANDIDATE_VERSION, MEASURED_THIS_GENERATION, (1,) * 16, metric=TARGET_METRIC),
            *_protocol_rows(
                candidate_path, version=CANDIDATE_VERSION,
                origin=MEASURED_THIS_GENERATION, samples=PROTECTED_SAMPLES,
            ),
        ),
        model_identity={"adapter_digest": candidate_digest} if candidate_digest else {},
    ).save(candidate_path)

    declared = {"parent_eval_report_path": str(parent_path)}
    if with_ancestor:
        ancestor_path = inputs / "baseline-eval-report.json"
        EvalReport(
            generation_version="gen0",
            runs=_protocol_rows(
                ancestor_path, version="gen0", origin=MEASURED_PARENT,
                samples=PROTECTED_SAMPLES,
            ),
            model_identity={"base_model_digest": base_digest} if base_digest else {},
        ).save(ancestor_path)
        declared["baseline_eval_report_path"] = str(ancestor_path)
    return declared


def _declaration(
    tmp_path: Path,
    *,
    recipe_count: int = 2,
    candidate_role_digest: str = "",
    with_ancestor: bool = False,
    with_parent_adapter: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """The manifest document, with every declared input it names on disk."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "parent-model"
    parent.mkdir(exist_ok=True)
    (parent / "config.json").write_text("{}", encoding="utf-8")
    # The base is pinned to its *model-content* digest (payload files only), not
    # its whole-tree directory digest: a real base tree grows a HuggingFace
    # download cache after it is fetched, and a whole-tree digest would move with
    # the cache while the model itself was unchanged.
    digest = model_content_digest(parent).digest
    adapter = tmp_path / "parent-adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_model.safetensors").write_text("gen1-parent-weights", encoding="utf-8")
    adapter_digest, _adapter_entries = directory_digest(adapter)

    # The prepared profile is the *attributed* one production writes: each skill
    # is estimated only from the benchmarks that measure it. A fixture that
    # carried the old flat mean would prove the planner works on a document
    # nothing produces.
    profile = build_skill_profile(
        generation=PARENT_VERSION,
        runs=(
            _parent_row(TARGET_ID, 0.125),
            _parent_row(PROTECTED_ID, 0.5),
            _parent_row(BROAD_ID, 0.5),
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
        # The declared branch-protection policy: what the campaign must show the
        # candidate held against, and within what tolerance. Certification runs
        # before any lineage record, so a declaration without it cannot promote.
        "protection": dict(PROTECTION),
        "notes": "campaign-runner fixture",
    }
    if with_parent_adapter:
        document["parent_adapter_path"] = str(adapter)
        document["parent_adapter_digest"] = adapter_digest
    document.update(
        _write_reports(
            inputs,
            adapter_digest=adapter_digest if with_parent_adapter else "",
            candidate_digest=candidate_role_digest,
            with_ancestor=with_ancestor,
            base_digest=digest,
        )
    )
    document.update(overrides)
    document.setdefault(
        "evaluation_material_path", str(_write_evaluation_material(inputs, document))
    )
    return document


#: How many items each fixture evaluation dataset holds. The declared protection
#: protocol pins the mini-slice size; the fixture data must be at least that big
#: or the production evaluator refuses a slice shorter than the protocol.
EVAL_ITEMS = 16


def _write_evaluation_material(inputs: Path, document: Mapping[str, Any]) -> Path:
    """The evaluation material for every benchmark this declaration measures.

    The candidate arm is a *run output*, but the data it is measured on is an
    input, so a campaign fixture declares it exactly as it declares the training
    corpus: one dataset per declared benchmark, with the fields the item scorer
    reads. Each file is written where the declaration points, not generated by
    the evaluator, so a test can also point at a dataset that is missing or too
    short and get the refusal.
    """
    inputs.mkdir(parents=True, exist_ok=True)
    declared: list[str] = []
    for set_name in (
        "target_benchmarks",
        "protected_benchmarks",
        "broad_benchmarks",
        "calibration_benchmarks",
        "reliability_benchmarks",
    ):
        for qualified_id in document.get(set_name, ()):
            if qualified_id not in declared:
                declared.append(qualified_id)
    suites = []
    for index, qualified_id in enumerate(declared):
        dataset = inputs / f"eval-{index:02d}-{qualified_id.replace('@', '_').replace('/', '_')}.jsonl"
        dataset.write_text(
            "".join(
                json.dumps({"prompt": f"prompt {item} for {qualified_id}", "expected": f"{item}"})
                + "\n"
                for item in range(EVAL_ITEMS)
            ),
            encoding="utf-8",
        )
        suites.append(
            {
                "benchmark_qualified_id": qualified_id,
                "name": qualified_id.split("@", 1)[0],
                "dataset": str(dataset),
                "scoring": "normalized_exact_match",
                # The registry declares each benchmark's primary metric, and the
                # binder refuses a row that reports another one, so the material
                # says which metric it is producing.
                "metric": TARGET_METRIC if qualified_id == TARGET_ID else "accuracy",
            }
        )
    path = inputs / "evaluation-material.json"
    path.write_text(
        json.dumps({"suites": suites}, indent=2), encoding="utf-8"
    )
    return path


def _parent_row(qualified_id: str, score: float) -> BenchmarkRun:
    """One real parent-side measurement, as preparation would write it."""
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=PARENT_VERSION,
        score=score,
        n_samples=16,
        measurement_origin=MEASURED_PARENT,
    )


def _campaign(
    tmp_path: Path,
    *,
    runner: Any = None,
    planned: bool = True,
    candidate_role_digest: str = "",
    with_ancestor: bool = True,
    with_parent_adapter: bool = True,
    **overrides: Any,
) -> tuple[CampaignManifest, Any, dict[str, Any]]:
    """Write the declaration, plan it with production, declare its recipes.

    ``planned=False`` leaves the declared recipe ids alone, so a manifest that
    names recipes the planner never proposed can be exercised.

    The trusted-ancestor arm is declared by default: the run's readiness phase
    refuses a campaign that cannot be certified before it trains, so a manifest
    without one never reaches the code these fixtures exercise. Pass
    ``with_ancestor=False`` to exercise that refusal.
    """
    runner = runner if runner is not None else _RecordingRunner(gpu_hours=ATTEMPT_WALL_GPU_HOURS)
    document = _declaration(
        tmp_path,
        candidate_role_digest=candidate_role_digest,
        with_ancestor=with_ancestor,
        with_parent_adapter=with_parent_adapter,
        **overrides,
    )
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


def _refusal(run: Any) -> str:
    """Every phase detail of a run, joined: a refused run records its reason.

    A run that fails after training returns a REFUSED record rather than raising,
    because the compute really happened and must stay accounted for. Its reason
    lives in the phase that refused.
    """
    return " | ".join(str(phase.get("detail")) for phase in run.phases)


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


#: A protected slice the candidate holds at the floor: gen1 regressed on it, and a
#: gen2 that merely matches that gen1 inherits the regression.
PARENT_SAMPLES: tuple[float, ...] = (0,) * 16


def _rewrite_protected(
    report_path: Path, version: str, samples: Sequence[float]
) -> None:
    """Re-measure an arm's protected slices, artifacts and digests included."""
    report = EvalReport.load(report_path)
    runs = tuple(
        _row(
            run.benchmark_qualified_id,
            run.generation_version,
            run.measurement_origin,
            samples,
            metric=run.metric,
            protocol=True,
            artifact=_slice_artifact(
                report_path,
                version=f"{version}-remade",
                qualified_id=run.benchmark_qualified_id,
            ),
        )
        if run.benchmark_qualified_id in (PROTECTED_ID, BROAD_ID)
        else run
        for run in report.runs
    )
    EvalReport(
        generation_version=report.generation_version,
        runs=runs,
        hardware=report.hardware,
        date=report.date,
        model_identity=report.model_identity,
    ).save(report_path)


#: A *reported* zero: an evaluation leg that really held no accelerator, said so,
#: and named how it was measured. This is the only shape in which a zero charge
#: is legal -- an unreported cost is refused rather than defaulted to this.
FREE_EVALUATION = ComputeCost.zero(
    source="candidate evaluation",
    measurement_method="recording evaluator: no accelerator held",
)


class _RecordingEvaluator:
    """The evaluation seam production supplies: it measures the artifact the run
    selected and reports what it measured, and what that cost.

    ``bind`` is on by default because that is what a real evaluator does -- it is
    handed an artifact and a request, and its report names those bytes. With
    ``bind=False`` it returns a report that names other bytes, which is the state
    the run must refuse. ``into_run_root``/
    ``absolute_refs`` model production by default: the measurements are written
    into the run root and named relatively to it. Passing ``absolute_refs=True``
    models an evaluator whose evidence lives outside the run it measured for,
    which the run now refuses. ``cost`` defaults to an explicit measured zero
    (see :data:`FREE_EVALUATION`); passing ``cost=None`` models an evaluator that
    reports no cost at all, which the run also refuses.
    """

    def __init__(
        self,
        report_path: Path,
        *,
        bind: bool = True,
        absolute_refs: bool = False,
        into_run_root: bool = True,
        cost: ComputeCost | None = FREE_EVALUATION,
        origin: str | None = None,
    ) -> None:
        self.report_path = Path(report_path)
        self.bind = bind
        self.absolute_refs = absolute_refs
        self.into_run_root = into_run_root
        self.cost = cost
        self.origin = origin
        self.requests: list[EvaluationRequest] = []

    def __call__(self, request: EvaluationRequest) -> CandidateEvaluation:
        self.requests.append(request)
        if not self.report_path.is_file():
            raise CandidateEvaluationRefusal(
                f"the evaluation report {self.report_path} does not exist, so "
                "there is no measurement of the selected artifact"
            )
        if self.into_run_root:
            # A production instrument writes its measurements into the run it was
            # asked to measure for, and names them relatively to that root.
            root = Path(request.output_root)
            source = EvalReport.load(self.report_path)
            for run in source.runs:
                reference = str(run.raw_artifact_ref or "")
                if not reference or Path(reference).is_absolute():
                    continue
                origin = self.report_path.parent / reference
                destination = root / reference
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(origin.read_bytes())
        report = EvalReport.load(self.report_path)
        runs = tuple(
            replace(
                run,
                measurement_origin=self.origin or run.measurement_origin,
                raw_artifact_ref=(
                    str((self.report_path.parent / run.raw_artifact_ref).resolve())
                    if self.absolute_refs
                    and run.raw_artifact_ref
                    and not Path(run.raw_artifact_ref).is_absolute()
                    else run.raw_artifact_ref
                ),
            )
            for run in report.runs
        )
        identity = dict(report.model_identity)
        if self.bind:
            identity["adapter_digest"] = request.artifact_sha256
            identity["base_model_digest"] = request.base_model_digest
        return CandidateEvaluation(
            report=EvalReport(
                generation_version=report.generation_version,
                runs=runs,
                hardware=report.hardware,
                date=report.date,
                model_identity=identity,
            ),
            cost=self.cost,
        )


def _candidate_report(tmp_path: Path) -> Path:
    return tmp_path / "inputs" / "candidate-eval-report.json"


def _patch_runner(monkeypatch: pytest.MonkeyPatch, runner: Any) -> None:
    """Install the executor's process seam (the recording trainer subprocess)."""
    monkeypatch.setattr(campaign_runner, "default_runner", runner)


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    *,
    bind: bool = True,
    evaluator: Any = None,
    cost: ComputeCost | None = FREE_EVALUATION,
    wired: bool = True,
) -> None:
    """Install the two seams a run needs: the executor and the evaluator.

    ``wired=False`` leaves the production evaluator unwired, which is how a
    build without an instrument behaves and what the refusal is asserted against.
    ``cost=None`` models a seam that reports no cost, which the run refuses.
    """
    _patch_runner(monkeypatch, runner)
    if not wired:
        monkeypatch.setattr(campaign_runner, "default_evaluator_factory", None)
        return
    monkeypatch.setattr(
        campaign_runner,
        "default_evaluator_factory",
        lambda manifest, *, state_root=None: evaluator
        if evaluator is not None
        else _RecordingEvaluator(_candidate_report(Path(manifest.state_root).parent), bind=bind, cost=cost),
    )


def test_an_unbound_candidate_arm_cannot_be_recorded_as_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The order that matters: the run refuses a candidate arm that names other
    bytes, before any verdict or lineage record.

    The rule would promote (candidate target 1.0 vs parent 0.125, no protected
    regression) and the envelope is compliant, but the evaluated report does not
    name the artifact the run selected -- so the campaign refuses at the seam and
    the ledger has no generation for it.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner, bind=False)

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_IDENTITY_UNBOUND in _refusal(run)
    assert "adapter_digest" in _refusal(run)
    # The training spend is closed out even though the run refused.
    assert Path(run.cost["accounting_path"]).is_file()
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)
    assert [attempt["status"] for attempt in run.attempts]
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    with pytest.raises(KeyError):
        ledger.effective_verdict(CANDIDATE_VERSION)


def test_a_candidate_that_matches_a_regressed_parent_cannot_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The branch-protection case, adjudicated before the ledger is written.

    gen2 merely matches gen1, which itself regressed against the trusted ancestor:
    the generic parent-vs-candidate rule sees no regression and would promote; the
    trusted-ancestor gate fails first and no promoted generation is recorded.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)

    # The parent arm joins the candidate at the floor while the ancestor holds
    # the protected slice: gen1 regressed, gen2 inherited it.
    inputs = tmp_path / "inputs"
    _rewrite_protected(inputs / "parent-eval-report.json", PARENT_VERSION, PARENT_SAMPLES)
    _rewrite_protected(inputs / "candidate-eval-report.json", CANDIDATE_VERSION, PARENT_SAMPLES)

    run = run_campaign(manifest)

    assert run.verdict == "REJECTED"
    assert "trusted-ancestor protection" in _phase(run, "certification_veto")["detail"]
    assert _phase(run, "promotion")["verdict"] == "PROMOTED"
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    with pytest.raises(KeyError):
        ledger.effective_verdict(CANDIDATE_VERSION)


def test_a_certified_campaign_reaches_a_recorded_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Plan -> admission -> train -> evaluate -> settle -> certify -> adjudicate ->
    ledger, all driven by the declaration and nothing else."""
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)

    run = run_campaign(manifest)

    assert run.verdict == "PROMOTED"
    # The candidate arm is the run's own measurement of the artifact it selected:
    # the evaluator was asked about that artifact, and the arm it wrote names it.
    request = campaign_runner.build_evaluator(manifest, state_root=Path(manifest.state_root))
    assert request is not None
    candidate_arm = EvalReport.load(
        Path(manifest.state_root) / "candidate_evaluation.json"
    )
    assert candidate_arm.model_identity["adapter_digest"] == run.selection["artifact_sha256"]
    assert _phase(run, "certification")["verdict"] == "PASS"
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
    _patch_seams(monkeypatch, runner)

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
    _patch_seams(monkeypatch, runner)

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


def test_the_committed_gen2_declaration_is_not_runnable_and_says_so():
    """The checked-in gen2 declaration is pinned to its real readiness.

    Seven of the eight inputs a run reads from disk, none of them present as an
    artifact: the historical Gen-1 driver composed four of them in process, the
    parent profile was never measured from gen1, the evaluation material the
    production evaluator measures with does not exist yet, and there is no
    measured parent arm. (The eighth, the contamination manifest, is declared
    but produced by the run into its own state root.) Both entry points refuse
    before compute and name *every* missing input at once, and the declaration's
    own notes carry the same statement -- documentation is not allowed to run
    ahead of what exists.
    """
    manifest = CampaignManifest.from_file(ROOT / "docs" / "gen2" / "gen2_campaign.json")

    assert undeclared_inputs(manifest, phase="run") == (
        "project_template_path",
        "training_material_path",
        "data_registry_path",
        "hardware_budget_path",
        "parent_profile_path",
        "evaluation_material_path",
        "parent_eval_report_path",
    )
    assert undeclared_inputs(manifest, phase="plan") == (
        "parent_profile_path",
        "hardware_budget_path",
    )
    with pytest.raises(CampaignRunRefusal) as error:
        plan_campaign(manifest)
    for field_name in ("parent_profile_path", "hardware_budget_path"):
        assert field_name in str(error.value)
    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(manifest)
    for field_name in undeclared_inputs(manifest, phase="run"):
        assert field_name in str(error.value)
    assert "GEN2_PREREG_AMENDMENT5" in manifest.notes
    assert "run output" in manifest.notes


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
    EvalReport(
        generation_version=report.generation_version,
        runs=runs,
        hardware=report.hardware,
        date=report.date,
        model_identity=report.model_identity,
    ).save(report_path)


def test_the_run_carries_the_measurements_its_rows_name(tmp_path: Path, monkeypatch):
    """A declared arm's row bound to an artifact gets those bytes in the run root.

    The judge recomputes the digest a row declares over the file it names, so the
    evidence the run writes has to include the measurement, not only the report
    that points at it.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    inputs = tmp_path / "inputs"
    reference = "raw/parent-target.json"
    payload = '{"target": "measured"}'
    artifact = inputs / reference
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(payload, encoding="utf-8")
    _bind_row_artifact(
        inputs / "parent-eval-report.json", TARGET_ID, reference=reference, payload=payload
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
    _patch_seams(monkeypatch, runner)
    inputs = tmp_path / "inputs"
    _bind_row_artifact(
        inputs / "parent-eval-report.json",
        TARGET_ID,
        reference="raw/never-written.json",
        payload="unused",
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert "raw/never-written.json" in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    # The evidence failure happens after training, so the spend it produced is
    # still recorded rather than lost with the exception.
    assert Path(run.cost["accounting_path"]).is_file()
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)


def test_a_candidate_row_the_run_never_wrote_refuses(tmp_path: Path, monkeypatch):
    """The candidate arm's relative refs resolve against the run root.

    An evaluation that names a relative measurement the run root does not hold
    cannot be verified, so the run refuses instead of writing an arm pointing at
    bytes nobody can read.
    """
    manifest, runner, _document = _campaign(tmp_path)
    _patch_seams(
        monkeypatch,
        runner,
        # A relative ref the evaluator never wrote into the run root: it names
        # bytes nobody can read, so the run cannot carry them as evidence.
        evaluator=_RecordingEvaluator(
            _candidate_report(tmp_path), absolute_refs=False, into_run_root=False
        ),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert "candidate arm" in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()


def test_two_arms_naming_one_artifact_differently_refuse(tmp_path: Path, monkeypatch):
    """One run root cannot carry two different measurements under one name."""
    manifest, runner, _document = _campaign(tmp_path)
    inputs = tmp_path / "inputs"
    reference = "raw/shared.json"
    # Two arms naming the same relative path: the declared arm's ref is relative
    # to its own report, the candidate arm's to the run root, so these are two
    # different measurements that would land on one run-root path.
    parent_artifact = inputs / reference
    parent_artifact.parent.mkdir(parents=True, exist_ok=True)
    parent_artifact.write_text('{"arm": "parent"}', encoding="utf-8")
    _bind_row_artifact(
        inputs / "parent-eval-report.json",
        PROTECTED_ID,
        reference=reference,
        payload='{"arm": "parent"}',
    )
    run_root_artifact = Path(manifest.state_root) / reference
    run_root_artifact.parent.mkdir(parents=True, exist_ok=True)
    run_root_artifact.write_text('{"arm": "candidate"}', encoding="utf-8")
    _bind_row_artifact(
        _candidate_report(tmp_path),
        TARGET_ID,
        reference=reference,
        payload='{"arm": "candidate"}',
    )
    _patch_seams(
        monkeypatch,
        runner,
        evaluator=_RecordingEvaluator(
            _candidate_report(tmp_path), absolute_refs=False, into_run_root=False
        ),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert "different content" in _refusal(run)


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
    """A declared set the evaluator could not measure stays unmeasured, and an
    unmeasured hard gate is not a pass: the run reaches no promotion."""
    manifest, runner, _document = _campaign(tmp_path, calibration_benchmarks=[CALIBRATION_ID])
    _patch_seams(monkeypatch, runner)
    # The evaluator was asked for the calibration set and could not measure it.
    # That is an honest non-measurement, not a missing row.
    report = EvalReport.load(_candidate_report(tmp_path))
    EvalReport(
        generation_version=report.generation_version,
        runs=(
            *report.runs,
            BenchmarkRun(
                benchmark_qualified_id=CALIBRATION_ID,
                adapter="none",
                generation_version=CANDIDATE_VERSION,
                score=None,
                support="UNSUPPORTED_HARNESS",
                measurement_origin="UNMEASURED",
                notes="the evaluator could not measure this declared set",
            ),
        ),
        hardware=report.hardware,
        model_identity=report.model_identity,
    ).save(_candidate_report(tmp_path))

    run = run_campaign(manifest)

    assert run.verdict == "INCONCLUSIVE"
    assert _phase(run, "candidate_evaluation")["verdict"] == "measured"


def test_a_declared_set_the_evaluation_never_mentions_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unevaluated declaration is a missing measurement, not an omission."""
    manifest, runner, _document = _campaign(tmp_path, calibration_benchmarks=[CALIBRATION_ID])
    _patch_seams(monkeypatch, runner)

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert "CANDIDATE_EVALUATION_INCOMPLETE" in _refusal(run)
    assert CALIBRATION_ID in _refusal(run)


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
# the candidate arm is a run output, never a declared input
# --------------------------------------------------------------------------


def test_the_retired_candidate_report_input_is_rejected_with_its_reason(tmp_path: Path):
    """A campaign cannot hand the run a pre-existing candidate report.

    The declaration is refused at load with the reason, rather than being read as
    a typo: the candidate arm is measured by the run, so a report prepared outside
    it is not the candidate side of promotion.
    """
    _manifest, _runner, document = _campaign(tmp_path)
    document["candidate_eval_report_path"] = str(_candidate_report(tmp_path))

    with pytest.raises(CampaignManifestError) as error:
        CampaignManifest.from_mapping(document)

    assert "candidate_eval_report_path" in str(error.value)
    assert "run output" in str(error.value)


def test_a_run_that_cannot_measure_its_candidate_refuses_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No evaluation material means no instrument, and no candidate arm.

    The refusal names what is missing and happens before any compute: a campaign
    that cannot measure the artifact it is about to train must not spend a
    training budget discovering that afterwards.
    """
    manifest, runner, document = _campaign(tmp_path, with_ancestor=True)
    _patch_runner(monkeypatch, runner)
    unprepared = _redeclare(document, evaluation_material_path="")

    with pytest.raises(CampaignRunRefusal) as error:
        run_campaign(unprepared)

    assert "evaluation_material_path" in str(error.value)
    assert not (Path(manifest.state_root) / "cycle_compute_accounting.json").exists()
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


def test_an_evaluator_that_cannot_cover_a_declared_benchmark_refuses_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The evaluator's admission seam runs before the trainer's.

    Material that names no dataset for a declared benchmark is knowable without
    a GPU, so the run refuses at its readiness phase with nothing spent -- the
    mirror of the executor's own projected-cost admission.
    """
    manifest, runner, document = _campaign(tmp_path)
    _patch_runner(monkeypatch, runner)
    material = json.loads(Path(manifest.evaluation_material_path).read_text(encoding="utf-8"))
    material["suites"] = [
        suite
        for suite in material["suites"]
        if suite["benchmark_qualified_id"] != PROTECTED_ID
    ]
    Path(manifest.evaluation_material_path).write_text(
        json.dumps(material), encoding="utf-8"
    )
    monkeypatch.setattr(campaign_runner, "default_evaluator_factory", None)

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert _phase(run, "readiness")["verdict"] == "refused"
    assert PROTECTED_ID in _phase(run, "readiness")["detail"]
    assert "selection" not in {phase["phase"] for phase in run.phases}
    assert run.attempts == ()
    assert document["state_root"] == str(Path(manifest.state_root))


def test_two_rows_for_one_benchmark_refuse_even_when_both_are_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """One benchmark carries one row, measured or not.

    The honest non-measurement path used to skip the duplicate check, so two
    ``UNMEASURED`` rows under one id were accepted. They cannot manufacture a
    promotion, but they are ambiguous evidence about coverage -- and the binder
    pairs on one row per benchmark -- so the arm refuses instead.
    """
    manifest, runner, _document = _campaign(tmp_path, calibration_benchmarks=[CALIBRATION_ID])
    _patch_seams(monkeypatch, runner)
    report = EvalReport.load(_candidate_report(tmp_path))
    unmeasured = BenchmarkRun(
        benchmark_qualified_id=CALIBRATION_ID,
        adapter="none",
        generation_version=CANDIDATE_VERSION,
        score=None,
        support="UNSUPPORTED_HARNESS",
        measurement_origin="UNMEASURED",
        notes="the evaluator could not measure this declared set",
    )
    EvalReport(
        generation_version=report.generation_version,
        runs=(*report.runs, unmeasured, replace(unmeasured, notes="reported twice")),
        hardware=report.hardware,
        model_identity=report.model_identity,
    ).save(_candidate_report(tmp_path))

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK in _refusal(run)
    assert CALIBRATION_ID in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()


def test_an_artifact_that_changed_after_training_refuses_before_it_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A digest recorded at training time is a claim about bytes that can move.

    The run re-derives the selected artifact's digest from disk before it is
    measured, so a mutated adapter refuses instead of being certified under a
    digest the frozen judge would later recompute and reject. Nothing is
    recorded as promoted, and the training spend is still closed out.
    """
    manifest, runner, _document = _campaign(tmp_path)
    _patch_seams(monkeypatch, runner)

    class _MutatingEvaluator(_RecordingEvaluator):
        """Rewrites the artifact it was asked to measure, as a race would."""

        def __call__(self, request: EvaluationRequest) -> CandidateEvaluation:
            artifact = Path(request.artifact_ref)
            if artifact.is_dir():
                (artifact / "adapter_model.safetensors").write_text(
                    "bytes that replaced the selected artifact", encoding="utf-8"
                )
            return super().__call__(request)

    _patch_seams(
        monkeypatch,
        runner,
        evaluator=_MutatingEvaluator(_candidate_report(tmp_path)),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_ARTIFACT_DIGEST_STALE in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()
    # The compute really happened, so it is still accounted for.
    assert Path(run.cost["accounting_path"]).is_file()
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)
    assert run.attempts and all(attempt["recipe_id"] for attempt in run.attempts)


def test_an_artifact_that_changed_after_being_measured_refuses_before_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The second boundary: bytes that move between measurement and verdict.

    The digest is checked again immediately before the evidence set is written
    and a verdict is bound to it. An evaluation that measured the artifact
    honestly and a mutation that landed afterwards must not become a certified
    promotion -- the frozen judge would recompute the digest of the bytes on
    disk and reject it.
    """
    manifest, runner, _document = _campaign(tmp_path)

    class _MutatingAfterMeasurement(_RecordingEvaluator):
        """Measures honestly, then lets the artifact change underneath."""

        def __call__(self, request: EvaluationRequest) -> CandidateEvaluation:
            evaluation = super().__call__(request)
            artifact = Path(request.artifact_ref)
            if artifact.is_dir():
                (artifact / "adapter_model.safetensors").write_text(
                    "mutated after the measurement was taken", encoding="utf-8"
                )
            return evaluation

    _patch_seams(
        monkeypatch,
        runner,
        evaluator=_MutatingAfterMeasurement(_candidate_report(tmp_path)),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_ARTIFACT_DIGEST_STALE in _refusal(run)
    # The evaluation ran and was charged; the verdict was never reached.
    assert _phase(run, "candidate_evaluation")["verdict"] == "measured"
    assert "certification" not in {phase["phase"] for phase in run.phases}
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()
    assert Path(run.cost["accounting_path"]).is_file()


def test_a_candidate_arm_whose_evidence_lives_outside_the_run_root_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A run's own measurement must live inside the run it measured for.

    An absolute ref would make a certified verdict depend on a directory
    somebody else can move or delete, so the evidence writer refuses it rather
    than recording a pointer the judge hashes in place.
    """
    manifest, runner, _document = _campaign(tmp_path)
    _patch_seams(
        monkeypatch,
        runner,
        evaluator=_RecordingEvaluator(
            _candidate_report(tmp_path), absolute_refs=True, into_run_root=False
        ),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert "absolute artifact" in _refusal(run)
    assert "run root" in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


def test_parent_rows_returned_as_the_candidate_arm_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A provenance substitution is refused by name, never rebound."""
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(
        monkeypatch,
        runner,
        evaluator=_RecordingEvaluator(_candidate_report(tmp_path), origin=MEASURED_PARENT),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED in _refusal(run)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


def test_a_selected_candidate_with_no_artifact_refuses_with_its_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A cycle that trained nothing evaluable records a refusal, not a promotion.

    The compute was really spent, so the accounting is written and carried by the
    refusal record: the refusal is evidence, not a silent no-op.
    """
    manifest, runner, _document = _campaign(tmp_path)
    _patch_seams(monkeypatch, runner)

    class _FailingExecutor:
        firewall = campaign_runner.ContaminationFirewall()

        def admit(self, recipe: Any) -> None:
            return None

        def __call__(self, recipe: Any, items: Any) -> Mapping[str, Any]:
            return {
                "recipe_id": recipe.recipe_id,
                "attempt": "attempt-01",
                "status": "FAILED",
                "artifact_ref": None,
                "measured_gpu_hours": 0.01,
            }

    run = run_campaign(manifest, train_fn=_FailingExecutor())

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_NOT_PRODUCED in _phase(run, "candidate_evaluation")["detail"]
    accounting = Path(run.cost["accounting_path"])
    assert accounting.is_file()
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * 0.01)
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()


def test_the_evaluations_measured_cost_is_charged_and_can_veto_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Measuring the candidate is compute, and it counts.

    Two attempts at 0.05 wall fit the declared 0.12 campaign ceiling; the 0.05
    the evaluation cost does not. The same run promotes when the evaluation
    reports an explicit, measured zero -- which is what makes this a statement
    about accounting rather than about the rule.
    """
    budget = {
        "device_gpu_hours_ceiling_per_recipe": DEVICE_PER_RECIPE,
        "wall_gpu_hours_ceiling_per_recipe": WALL_PER_RECIPE,
        "device_gpu_hours_ceiling_campaign": 0.60,
        "wall_gpu_hours_ceiling_campaign": 0.12,
    }
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True, budget=budget)

    _patch_runner(monkeypatch, runner)
    monkeypatch.setattr(
        campaign_runner,
        "default_evaluator_factory",
        lambda manifest, *, state_root=None: _RecordingEvaluator(
            _candidate_report(Path(manifest.state_root).parent), cost=FREE_EVALUATION
        ),
    )
    free = run_campaign(manifest)
    assert free.verdict == "PROMOTED"
    assert free.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)

    manifest2, runner2, _document2 = _campaign(tmp_path / "charged", with_ancestor=True, budget=budget)
    _patch_runner(monkeypatch, runner2)
    monkeypatch.setattr(
        campaign_runner,
        "default_evaluator_factory",
        lambda manifest, *, state_root=None: _RecordingEvaluator(
            _candidate_report(Path(manifest.state_root).parent),
            cost=ComputeCost.from_wall_only(0.05, source="candidate evaluation"),
        ),
    )
    charged = run_campaign(manifest2)

    assert charged.verdict == "REJECTED"
    assert charged.settlement["budget_compliant"] is False
    assert charged.cost["wall_gpu_hours"] == pytest.approx(0.15)
    assert any(
        "WALL" in reason for reason in charged.settlement["budget_failure_reasons"]
    )
    ledger = GenerationLedger(Path(manifest2.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()
    # The evaluation leg is in the durable accounting artifact, not only in the
    # campaign's own summary of itself.
    accounting = json.loads(
        Path(charged.cost["accounting_path"]).read_text(encoding="utf-8")
    )
    legs = json.dumps(accounting)
    assert "candidate evaluation" in legs


def test_an_evaluation_that_reports_no_cost_is_refused_rather_than_charged_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unreported cost would settle as zero, so it is a refusal.

    The run that reaches this seam is otherwise a clean promotion: certified
    target improvement, no protected regression. It is refused anyway, and no
    generation is recorded, because a campaign may not spend compute it cannot
    account for. The training spend is still written out -- that compute really
    happened -- which is what makes this a recording failure and not a lost run.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner, cost=None)

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_COST_UNREPORTED in _phase(run, "candidate_evaluation")[
        "detail"
    ]
    # The compute that really happened is durably closed out anyway.
    accounting = Path(run.cost["accounting_path"])
    assert accounting.is_file()
    assert run.cost["wall_gpu_hours"] == pytest.approx(2 * ATTEMPT_WALL_GPU_HOURS)
    assert json.dumps(json.loads(accounting.read_text(encoding="utf-8"))) .count("attempt") >= 2
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


def test_a_zero_cost_that_names_no_measurement_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A bare zero is indistinguishable from an unreported cost.

    ``ComputeCost.zero(source=...)`` is what a leg that reports nothing looks
    like, and charging it would make a real evaluation leg vanish from the
    campaign's accounting. Only a zero that names how it was measured is
    accepted.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(
        monkeypatch,
        runner,
        cost=ComputeCost.zero(source="candidate evaluation"),
    )

    run = run_campaign(manifest)

    assert run.verdict == "REFUSED"
    assert CANDIDATE_EVALUATION_COST_UNMEASURED in _phase(run, "candidate_evaluation")[
        "detail"
    ]
    assert not (Path(manifest.state_root) / "candidate_evaluation.json").exists()
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert CANDIDATE_VERSION not in ledger.versions()


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
    manifest, runner, document = _campaign(tmp_path, with_ancestor=True)
    # The run's two seams: the trainer subprocess and the candidate evaluator.
    # The evaluator measures the artifact the run selected, which is what makes
    # the candidate arm a run output rather than a declared input.
    _patch_seams(monkeypatch, runner)

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
    # A fresh executor and evaluator for the run below: the command counts
    # asserted here describe exactly that run.
    runner = _RecordingRunner(gpu_hours=ATTEMPT_WALL_GPU_HOURS)
    _patch_seams(monkeypatch, runner)
    monkeypatch.setattr(sys, "argv", ["chowder", "growth", "campaign", "run", str(example_path)])
    assert chowder_main() == 0
    outcome = json.loads(capsys.readouterr().out)

    assert outcome["verdict"] == "PROMOTED"
    assert outcome["certification"]["status"] == "PASS"
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
