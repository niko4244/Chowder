"""The adversarial audit of every settlement and compute-cost path.

Companion to the promotion matrix: the same one-mutation-at-a-time attack, over
the half of the verdict that matrix does not touch -- the frozen resource
envelope and the accounting that settles it. Each test flips exactly one
artifact and asserts the verdict moves, and the ones that pin a fixed defect say
which revert breaks them. An envelope matrix is only worth having if it fails
when the control is weakened.

Three layers, in increasing cost:

1. **The settlement rule** (``settle_cost``). One ceiling, one unit, one
   measurement flag or one projection mutated at a time over a clean envelope.
   Every declared ceiling is a hard gate, each is settled in its own unit, and
   an unmeasured figure is never read as a measured zero.
2. **The campaign contract and its ledger** (``settle_campaign``,
   ``settle_campaign_projection``, ``CycleCostLedger``, ``settlement_refusal``).
   What a declaration lets settlement certify, what the total an auditor reads
   is made of, and the one predicate every advancement decision reads.
3. **The run and the second opinion.** A whole campaign through the fixture
   seams, then the frozen Gen-2 judge over the root the run wrote: T13 settles
   the accounting artifact, and T23 requires that artifact to be the one the
   run pinned with the settlement the record claims. The attack this layer
   exists for: a run REFUSED on its own frozen envelope must not become
   certifiable by editing the accounting artifact.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import pytest

from chowder.growth.attempt_failure import FailureClass, classify_failure
from chowder.growth.campaign import (
    CampaignManifest,
    settle_campaign,
    settle_campaign_projection,
)
from chowder.growth.campaign_runner import _ceiling_enforcement, run_campaign
from chowder.growth.candidate_search import advanced
from chowder.growth.compute_cost import (
    ACTUAL_DEVICE_GPU_HOURS_EXCEEDED,
    ACTUAL_DEVICE_GPU_HOURS_UNMEASURED,
    ACTUAL_EXCEEDS_PROJECTION,
    ACTUAL_WALL_GPU_HOURS_EXCEEDED,
    PROJECTED_DEVICE_GPU_HOURS_EXCEEDED,
    PROJECTED_WALL_GPU_HOURS_EXCEEDED,
    RESOURCE_OVERRUN,
    ComputeCost,
    CycleCostLedger,
    ledger_digest,
    settle_cost,
    settlement_refusal,
)
from chowder.growth.cycle import select_candidate

from test_growth_campaign_runner import (
    ATTEMPT_WALL_GPU_HOURS,
    _campaign,
    _patch_seams,
)
from test_growth_training_binding import _RecordingRunner

# ==========================================================================
# Layer 1 -- the settlement rule, one artifact at a time
# ==========================================================================


def _settle(
    *,
    device: float = 0.0,
    wall: float = 0.0,
    device_measured: bool = True,
    device_ceiling: float | None = 1.0,
    wall_ceiling: float | None = 2.0,
    project_budget: float | None = None,
    projected_wall: float | None = None,
    tolerance: float = 0.25,
):
    """One envelope, settled against one actual cost.

    ``device_measured`` is the whole distinction this file turns on: a
    wall-charged summary leaves device time unseparated, and that placeholder
    must never settle a declared device ceiling.
    """
    actual = (
        ComputeCost.measured(device_gpu_hours=device, wall_gpu_hours=wall, source="audit:actual")
        if device_measured
        else ComputeCost.from_wall_only(wall, source="audit:actual")
    )
    projected = (
        None
        if projected_wall is None
        else ComputeCost(0.0, projected_wall, source="audit:projection")
    )
    return settle_cost(
        actual=actual,
        projected=projected,
        device_ceiling=device_ceiling,
        wall_ceiling=wall_ceiling,
        project_budget_wall_gpu_hours=project_budget,
        projection_tolerance=tolerance,
    )


def _codes(verdict: Any) -> list[str]:
    return [str(reason).split(":", 1)[0].strip() for reason in verdict.failure_reasons]


def _codes_settlement(run: Any) -> list[str]:
    """The unit codes of a run record's own settlement verdict."""
    return [
        str(reason).split(":", 1)[0].strip()
        for reason in run.settlement["budget_failure_reasons"]
    ]


def test_an_unmeasured_zero_is_not_a_measured_zero() -> None:
    """The producer's placeholder and a real zero settle differently.

    ``ComputeCost.from_wall_only`` writes ``device_gpu_hours=0.0`` because the
    trainer never separated device time. Reading that zero as "device free"
    certifies compliance the run never demonstrated, so a declared device
    ceiling refuses it while an *observed* zero passes. This is the original
    unit-confusion defect of attempts 07/08, one layer down.
    """
    observed = _settle(device=0.0, device_measured=True)
    assert observed.compliant is True, observed.failure_reasons

    placeholder = _settle(device=0.0, device_measured=False)
    assert placeholder.compliant is False
    assert _codes(placeholder) == [ACTUAL_DEVICE_GPU_HOURS_UNMEASURED]
    assert "unmeasured is not compliance" in placeholder.failure_reasons[0]


def test_each_ceiling_is_settled_in_its_own_unit_and_only_its_own() -> None:
    """A device overrun is not a wall overrun, and neither substitutes for the other."""
    device_only = _settle(device=1.5, wall=0.0)
    assert _codes(device_only) == [ACTUAL_DEVICE_GPU_HOURS_EXCEEDED]

    wall_only = _settle(device=0.0, wall=2.5)
    assert _codes(wall_only) == [ACTUAL_WALL_GPU_HOURS_EXCEEDED]


def test_the_ceiling_is_the_boundary_and_a_tolerance_scales_with_it() -> None:
    """At the ceiling passes; past it fails. No cushion is declared, so none exists."""
    assert _settle(wall=2.0).compliant is True
    assert _settle(wall=2.0 + 1e-9).compliant is False
    assert _settle(device=1.0, wall=2.0).compliant is True
    assert _settle(device=1.0 + 1e-9).compliant is False


def test_every_declared_breach_is_reported_and_none_is_downgraded() -> None:
    """Four controls are violated at once; all four identifiers come back.

    A settlement that reported only the first breach would let a caller fix one
    number and re-run, discovering the next one only after another GPU pass.
    """
    verdict = _settle(
        device=5.0,
        wall=5.0,
        device_ceiling=1.0,
        wall_ceiling=2.0,
        project_budget=3.0,
        projected_wall=1.0,
    )
    assert verdict.compliant is False
    assert _codes(verdict) == [
        ACTUAL_DEVICE_GPU_HOURS_EXCEEDED,
        ACTUAL_WALL_GPU_HOURS_EXCEEDED,
        RESOURCE_OVERRUN,
        ACTUAL_EXCEEDS_PROJECTION,
    ]


def test_a_zero_ceiling_is_a_ceiling_and_only_none_removes_the_control() -> None:
    """``0.0`` is a declared bound; ``None`` is the absence of one.

    An implementation that tested ceilings for truthiness would silently
    upgrade a zero budget to an unlimited one -- the shape a caller reaches for
    when it means "no accelerator may be spent".
    """
    assert _settle(device=0.0, device_measured=True, device_ceiling=0.0).compliant is True
    assert _settle(device=0.0, device_measured=False, device_ceiling=0.0).compliant is False
    assert _settle(device=99.0, device_ceiling=None).compliant is True
    assert _settle(wall=99.0, wall_ceiling=None).compliant is True


def test_the_project_budget_is_charged_in_wall_units_and_is_its_own_control() -> None:
    """The project engine charges wall, so a device figure cannot trip it."""
    device_figure = _settle(device=9.0, wall=0.5, project_budget=1.0)
    assert RESOURCE_OVERRUN not in _codes(device_figure)

    wall_figure = _settle(device=0.0, wall=1.5, project_budget=1.0)
    assert _codes(wall_figure) == [RESOURCE_OVERRUN]


def test_the_projection_check_needs_a_real_projection_to_check() -> None:
    """The tolerance is directional, and a zero projection is an absent basis.

    The campaign settles with a zero projection on purpose: the campaign-level
    projection control is admission, and this check is the per-recipe one. What
    must not happen is a zero projection reading as "anything fits" *past a
    declared ceiling* -- the ceilings are a separate clause and still gate.
    """
    at_the_edge = _settle(wall=1.25, projected_wall=1.0, tolerance=0.25)
    assert at_the_edge.compliant is True, at_the_edge.failure_reasons

    past_it = _settle(wall=1.25 + 1e-6, projected_wall=1.0, tolerance=0.25)
    assert _codes(past_it) == [ACTUAL_EXCEEDS_PROJECTION]

    zero_basis = _settle(wall=0.5, projected_wall=0.0)
    assert ACTUAL_EXCEEDS_PROJECTION not in _codes(zero_basis)
    over_ceiling = _settle(wall=3.0, projected_wall=0.0)
    assert _codes(over_ceiling) == [ACTUAL_WALL_GPU_HOURS_EXCEEDED]


def test_a_negative_projection_tolerance_is_refused_not_ignored() -> None:
    """A tolerance is a declared allowance; a negative one refuses at the call."""
    with pytest.raises(ValueError):
        _settle(wall=1.0, projected_wall=1.0, tolerance=-0.25)


# ==========================================================================
# Layer 2 -- the campaign contract and the ledger that feeds it
# ==========================================================================


#: The shipped shape of the campaign budget: two unit-named ceilings per level.
def _budget(**overrides: Any) -> dict[str, float | bool]:
    budget: dict[str, float | bool] = {
        "device_gpu_hours_ceiling_per_recipe": 0.30,
        "wall_gpu_hours_ceiling_per_recipe": 0.75,
        "device_gpu_hours_ceiling_campaign": 0.60,
        "wall_gpu_hours_ceiling_campaign": 1.50,
    }
    budget.update(overrides)
    return budget


def _manifest(**overrides: Any) -> CampaignManifest:
    document: dict[str, Any] = {
        "cycle_id": "gen2-campaign",
        "parent_version": "gen1",
        "base_model_path": "F:/llm-models/base",
        "base_model_digest": "a" * 64,
        "state_root": "C:/runs/gen2",
        "target_benchmarks": ["generation-diagnostics@gen2-response-surface-v1"],
        "protected_benchmarks": ["math500@2024-04"],
        "broad_benchmarks": ["mgsm@2022-11"],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": _budget(),
        "recipes": ["recipe-a", "recipe-b"],
        "candidate_selection_policy": "first_successful",
        "stopping_rules": ["stop before compute on admission refusal"],
    }
    document.update(overrides)
    return CampaignManifest.from_mapping(document)


def test_the_declared_device_ceiling_settles_only_when_the_ledger_measured_it() -> None:
    """``device_time_measured`` is a claim about the executor, and it has teeth.

    Declaring False keeps the device ceilings as admission constraints on the
    projected plan and settlement makes no claim about them. Declaring True
    makes the campaign settle the device ceiling against a measured figure --
    and then a ledger whose device figure was never separated refuses, because
    the alternative is settling it against a placeholder zero.
    """
    measured = _manifest(budget=_budget(device_time_measured=True))
    over = settle_campaign(
        measured,
        total=ComputeCost.measured(device_gpu_hours=0.75, wall_gpu_hours=0.1, source="ledger"),
    )
    assert over.compliant is False
    assert _codes(over) == [ACTUAL_DEVICE_GPU_HOURS_EXCEEDED]

    observed_zero = settle_campaign(
        measured,
        total=ComputeCost.measured(device_gpu_hours=0.0, wall_gpu_hours=0.1, source="ledger"),
    )
    assert observed_zero.compliant is True, observed_zero.failure_reasons

    wall_only = settle_campaign(
        measured,
        total=ComputeCost.from_wall_only(0.1, source="ledger"),
    )
    assert wall_only.compliant is False
    assert _codes(wall_only) == [ACTUAL_DEVICE_GPU_HOURS_UNMEASURED]

    admission = settle_campaign(
        _manifest(),
        total=ComputeCost.measured(device_gpu_hours=99.0, wall_gpu_hours=0.1, source="ledger"),
    )
    assert admission.compliant is True, admission.failure_reasons


def test_the_wall_ceiling_settles_whether_or_not_device_time_was_measured() -> None:
    """The wall envelope is never demoted: it is a measurement by construction."""
    for device_time_measured in (False, True):
        manifest = _manifest(budget=_budget(device_time_measured=device_time_measured))
        verdict = settle_campaign(
            manifest,
            total=ComputeCost.measured(
                device_gpu_hours=0.0, wall_gpu_hours=1.60, source="ledger"
            ),
        )
        assert verdict.compliant is False, device_time_measured
        assert ACTUAL_WALL_GPU_HOURS_EXCEEDED in _codes(verdict)


def test_the_enforcement_record_matches_which_ceiling_was_actually_settled() -> None:
    """``_ceiling_enforcement`` is the record's claim about its own audit.

    Two declarations, two answers: a wall-only campaign must record its device
    ceiling as an admission constraint, and a campaign that declares device
    time measured must record it as settled -- but only when the total really
    is a measurement, or the record would claim an audit settlement refused.
    """
    wall_only = _manifest()
    assert _ceiling_enforcement(
        wall_only, ComputeCost.from_wall_only(0.1, source="ledger")
    ) == {
        "device": "admission:projected_plan",
        "wall": "settlement:measured",
        "project": "settlement:measured_wall",
    }

    measured = _manifest(budget=_budget(device_time_measured=True))
    settled = ComputeCost.measured(device_gpu_hours=0.1, wall_gpu_hours=0.1, source="ledger")
    assert _ceiling_enforcement(measured, settled)["device"] == "settlement:measured"
    # A declaration whose total is unmeasured settles nothing: the record must
    # not name a settlement the numbers cannot support.
    unmeasured = ComputeCost.from_wall_only(0.1, source="ledger")
    assert _ceiling_enforcement(measured, unmeasured)["device"] != "settlement:measured"


def test_the_campaign_projection_refuses_each_ceiling_on_its_own() -> None:
    """Admission is not settlement: a plan makes no measurement claim, and each
    declared campaign ceiling bounds the plan independently."""
    manifest = _manifest()
    assert settle_campaign_projection(
        manifest, projected=ComputeCost(0.0, 1.50, source="plan")
    ).compliant is True

    device_over = settle_campaign_projection(
        manifest, projected=ComputeCost(0.61, 0.10, source="plan")
    )
    assert _codes(device_over) == [PROJECTED_DEVICE_GPU_HOURS_EXCEEDED]

    wall_over = settle_campaign_projection(
        manifest, projected=ComputeCost(0.10, 1.51, source="plan")
    )
    assert _codes(wall_over) == [PROJECTED_WALL_GPU_HOURS_EXCEEDED]

    both = settle_campaign_projection(
        manifest, projected=ComputeCost(9.0, 9.0, source="plan")
    )
    assert _codes(both) == [
        PROJECTED_DEVICE_GPU_HOURS_EXCEEDED,
        PROJECTED_WALL_GPU_HOURS_EXCEEDED,
    ]


def test_the_ledger_total_is_the_sum_and_one_unmeasured_entry_demotes_it() -> None:
    """The total is only a measurement when every incremental part is one.

    A failed attempt's spend is still spend, so it is charged; and because its
    device figure was never separated, the campaign's device total is an
    estimate from that moment on -- which is exactly what makes the declared
    device ceiling unsettleable rather than satisfiable by a zero.
    """
    ledger = CycleCostLedger(cycle_id="audit")
    ledger.add(
        "recipe-a attempt",
        "training",
        ComputeCost.measured(device_gpu_hours=0.25, wall_gpu_hours=0.50, source="a"),
    )
    ledger.add(
        "recipe-b attempt",
        "failed_attempt",
        ComputeCost.from_wall_only(0.25, source="b"),
    )
    total = ledger.total()
    assert total.device_gpu_hours == pytest.approx(0.25)
    assert total.wall_gpu_hours == pytest.approx(0.75)
    assert total.device_measured is False


def test_a_reference_costs_nothing_and_does_not_demote_the_measurement() -> None:
    """A reused historical measurement is recorded, not paid for.

    The zero-cost reference is not incremental, so it contributes nothing to
    the total *and* cannot turn a measured total into an estimate. Summing it
    would inflate the campaign's spend; ANDing its measurement flag would
    demote a device ceiling the campaign really did measure.
    """
    ledger = CycleCostLedger(cycle_id="audit")
    measured = ComputeCost.measured(device_gpu_hours=0.40, wall_gpu_hours=0.90, source="train")
    ledger.add("recipe-a attempt", "training", measured)
    reference = ledger.add_reference("parent arm", "prepared-v10/parent-eval-report.json")
    assert reference.incremental is False

    total = ledger.total()
    assert total.device_gpu_hours == pytest.approx(0.40)
    assert total.wall_gpu_hours == pytest.approx(0.90)
    assert total.device_measured is True
    # The reference is still in the document, so an auditor sees the reuse.
    rendered = ledger.render()
    assert [entry["kind"] for entry in rendered["entries"]] == [
        "training",
        "baseline_reference",
    ]


def test_the_ledger_digest_is_recomputable_and_moves_with_any_edit() -> None:
    """The one identity an auditor can recompute from the bytes.

    ``ledger_digest`` is what the run pins in its record
    (``cost.accounting_digest``) and what the judge recomputes; a reader that
    re-derived the canonical form itself could drift from the writer and make
    every pinned artifact look moved. Any edit -- a total, an entry, a
    measurement flag -- changes the digest, which is the property the pin
    depends on.
    """
    ledger = CycleCostLedger(cycle_id="audit")
    ledger.add(
        "recipe-a attempt",
        "training",
        ComputeCost.measured(device_gpu_hours=0.25, wall_gpu_hours=0.50, source="a"),
    )
    document = ledger.render()
    assert document["digest_sha256"] == ledger_digest(document)

    twin = CycleCostLedger(cycle_id="audit")
    twin.add(
        "recipe-a attempt",
        "training",
        ComputeCost.measured(device_gpu_hours=0.25, wall_gpu_hours=0.50, source="a"),
    )
    assert twin.render()["digest_sha256"] == document["digest_sha256"]

    def _edit_total(edited: dict[str, Any]) -> None:
        edited["totals"]["incremental"]["wall_gpu_hours"] = 0.0

    def _edit_entry(edited: dict[str, Any]) -> None:
        edited["entries"][0]["cost"]["device_gpu_hours"] = 0.0

    def _edit_flag(edited: dict[str, Any]) -> None:
        edited["totals"]["incremental"]["device_measured"] = False

    for mutate in (_edit_total, _edit_entry, _edit_flag):
        edited = json.loads(json.dumps(document))
        mutate(edited)
        assert ledger_digest(edited) != document["digest_sha256"]


def test_the_settlement_vocabulary_is_read_from_the_shapes_production_writes() -> None:
    """Two writers, one vocabulary: the verdict and the refusal stamp.

    ``settlement_refusal`` must answer the same code for the same failure
    whichever half of the record carries it, or an over-budget attempt could be
    refused by one consumer and settled by another.
    """
    assert settlement_refusal(
        {
            "budget_settlement": {
                "budget_compliant": False,
                "budget_failure_reasons": [
                    f"{ACTUAL_WALL_GPU_HOURS_EXCEEDED}: actual wall 9.0 > ceiling 1.0"
                ],
            }
        }
    ) == ACTUAL_WALL_GPU_HOURS_EXCEEDED
    assert settlement_refusal(
        {
            "refused_by": "budget_settlement",
            "refusal_reason": f"{ACTUAL_DEVICE_GPU_HOURS_UNMEASURED}: no device figure",
        }
    ) == ACTUAL_DEVICE_GPU_HOURS_UNMEASURED
    # An unparseable reason still refuses; it just cannot name a unit.
    assert settlement_refusal(
        {"budget_settlement": {"budget_compliant": False, "budget_failure_reasons": [""]}}
    ) == "budget_settlement"


def test_a_compliant_or_absent_settlement_is_not_a_refusal() -> None:
    """Absence is not a refusal, and a compliant verdict is not evidence against."""
    assert settlement_refusal({}) is None
    assert settlement_refusal({"candidate_succeeded": True, "status": "SUCCEEDED"}) is None
    assert (
        settlement_refusal(
            {"budget_settlement": {"budget_compliant": True, "budget_failure_reasons": []}}
        )
        is None
    )


def test_one_predicate_owns_every_advancement_decision() -> None:
    """The search, the selection and the classifier must answer the same way.

    ``advanced`` decides who may earn a larger search budget, ``select_candidate``
    decides who may be measured for promotion, and ``classify_failure`` decides
    how the refusal is recorded. All three read the settlement fact the run
    wrote; if one of them missed it, an over-budget attempt -- which still
    carries ``candidate_succeeded=True`` because settlement runs after training
    -- could reach a larger budget or the promotion measurement.
    """
    refused = {
        "recipe_id": "recipe-a",
        "status": "SUCCEEDED",
        "candidate_succeeded": True,
        "artifact_ref": "attempt-01/adapter",
        "budget_settlement": {
            "budget_compliant": False,
            "budget_failure_reasons": [
                f"{ACTUAL_WALL_GPU_HOURS_EXCEEDED}: actual wall 9.0 > ceiling 1.0"
            ],
        },
    }
    settled = {
        **refused,
        "budget_settlement": {"budget_compliant": True, "budget_failure_reasons": []},
    }

    assert advanced([refused], survivor_count=1) == ()
    assert advanced([settled], survivor_count=1) == ("recipe-a",)
    assert select_candidate([refused]) is None
    assert select_candidate([settled]) is not None
    classification = classify_failure(refused)
    assert classification.failure_class == FailureClass.BUDGET_EXHAUSTED
    assert classification.evidence_state is None
    assert ACTUAL_WALL_GPU_HOURS_EXCEEDED in classification.reason


# ==========================================================================
# Layer 3 -- the run that spent the money, and the judge over its root
# ==========================================================================

#: A measured evaluation that does not fit the fixture's campaign envelope:
#: every attempt settled inside its own recipe ceiling, and the campaign total
#: still breaches. This is what makes the resource veto the deciding gate
#: rather than a second opinion on an attempt that never ran.
OVERRUN_EVALUATION = ComputeCost(
    device_gpu_hours=0.0,
    wall_gpu_hours=0.5,
    source="audit:overrunning evaluation",
    measurement_method="wall clock",
    device_measured=False,
)

ACCOUNTING = "cycle_compute_accounting.json"
RUN_RECORD = "campaign-run.json"

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
_spec = importlib.util.spec_from_file_location(
    "judge_gen2_settlement", GEN2 / "judge_gen2.py"
)
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

ROW = re.compile(r"^(T\d+|INFO)\s+(.+?)\s+(PASS|UNKNOWN|FAIL|INFO)\s+(.+)$")


def _judge(run_root: Path, manifest_path: Path) -> tuple[dict[str, tuple[str, str]], str, int]:
    """``{threshold: (status, detail)}``, the branch verdict, and the exit code."""
    original = judge_gen2.CAMPAIGN_MANIFEST
    judge_gen2.CAMPAIGN_MANIFEST = manifest_path
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = judge_gen2.judge(run_root)
    finally:
        judge_gen2.CAMPAIGN_MANIFEST = original
    rows: dict[str, tuple[str, str]] = {}
    final = ""
    for line in buffer.getvalue().splitlines():
        match = ROW.match(line.strip())
        if match and match.group(1).startswith("T"):
            rows[match.group(1)] = (match.group(3), match.group(4))
        if line.startswith("VERDICT: "):
            final = line.split(": ", 1)[1].strip()
    assert final, buffer.getvalue()
    return rows, final, exit_code


def _record(root: Path) -> dict[str, Any]:
    return json.loads((root / RUN_RECORD).read_text(encoding="utf-8"))


def _accounting(root: Path) -> dict[str, Any]:
    return json.loads((root / ACCOUNTING).read_text(encoding="utf-8"))


def _rerender_ledger(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Edit the accounting artifact the way a tamper would, digest included."""
    path = root / ACCOUNTING
    document = _accounting(root)
    mutate(document)
    document["digest_sha256"] = ledger_digest(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _run_over_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One campaign whose evaluation pushes it past its frozen wall ceiling.

    Training succeeds and the artifact survives; every attempt settles inside
    its own recipe envelope; the campaign total is what breaches. The run must
    record REJECTED with the resource veto, and the accounting artifact must be
    the settlement of the ledger it wrote.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner, cost=OVERRUN_EVALUATION)
    run = run_campaign(manifest)
    return manifest, run, Path(manifest.state_root)


def test_the_record_is_the_settlement_of_the_ledger_the_run_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one identity the judge's T23 rests on, asserted on a real run.

    ``ledger.write`` stamps the digest, the record pins it as
    ``cost.accounting_digest``, and the record's settlement must be exactly
    what production settles off the artifact's own totals. If any of the three
    came from a different computation, the pin would identify nothing.
    """
    manifest, run, root = _run_over_envelope(tmp_path, monkeypatch)

    assert run.verdict == "REJECTED"
    record = _record(root)
    document = _accounting(root)
    assert record["cost"]["accounting_digest"] == document["digest_sha256"]
    assert record["cost"]["accounting_digest"] == ledger_digest(document)

    total = ComputeCost.from_dict(document["totals"]["incremental"])
    recomputed = settle_campaign(manifest, total=total)
    assert recomputed.compliant is False
    assert record["settlement"]["budget_compliant"] == recomputed.compliant
    assert record["settlement"]["budget_failure_reasons"] == list(recomputed.failure_reasons)
    assert ACTUAL_WALL_GPU_HOURS_EXCEEDED in _codes(recomputed)

    # The veto is recorded as its own phase beside the rule's decision: the
    # rule rejected on the same envelope, and the record says which gate was
    # authoritative rather than leaving it to be inferred.
    phases = {phase["phase"]: phase for phase in record["phases"]}
    assert phases["resource_veto"]["verdict"] == "REJECTED"
    assert phases["promotion"]["verdict"] == "REJECTED"

    # The declaration keeps its device ceilings as admission constraints on
    # the projected plan; the wall envelope is the one settlement enforces.
    assert record["ceiling_enforcement"]["device"] == "admission:projected_plan"
    assert record["ceiling_enforcement"]["wall"] == "settlement:measured"


def test_the_judge_settles_the_same_artifact_the_run_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for T13: the judge's own settlement is production's answer.

    The run refused on its frozen envelope, and the judge pointed at that same
    root must refuse for the same reason -- the failure code in its detail, not
    just a red row. This is the agreement the tamper tests below attack.
    """
    _manifest, run, root = _run_over_envelope(tmp_path, monkeypatch)
    assert run.verdict == "REJECTED"

    rows, final, exit_code = _judge(root, tmp_path / "inputs" / "campaign.json")
    assert final != "PROMOTED", (final, rows)
    assert exit_code == 1
    assert rows["T13"][0] == "FAIL", rows["T13"]
    assert ACTUAL_WALL_GPU_HOURS_EXCEEDED in rows["T13"][1]
    # And T23 certifies the *agreement*: the artifact is the pinned one and it
    # settles exactly as the record says.
    assert rows["T23"][0] == "PASS", rows["T23"]


def test_the_judge_never_certifies_a_root_whose_accounting_artifact_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attack T23 exists for: edit the arithmetic, keep the refusal.

    A run REFUSED on its own frozen envelope, with its record of that refusal
    intact, must not become certifiable by editing the accounting artifact's
    totals. Before T23 the judge settled the edited bytes, found them
    compliant, and returned PROMOTED over a record that still said REJECTED --
    the settlement analogue of the declared-gate split-brain amendment 15
    closed. Reverting ``_settlement_artifact_gate`` (or dropping the digest
    clause) makes this test fail with ``VERDICT: PROMOTED``.
    """
    _manifest, run, root = _run_over_envelope(tmp_path, monkeypatch)
    assert run.verdict == "REJECTED"

    _rerender_ledger(
        root, lambda document: document["totals"]["incremental"].update({"wall_gpu_hours": 0.0})
    )
    # The record is untouched: it still pins the digest of the bytes it wrote,
    # and still records a non-compliant settlement.
    assert _record(root)["verdict"] == "REJECTED"
    assert _record(root)["settlement"]["budget_compliant"] is False

    rows, final, exit_code = _judge(root, tmp_path / "inputs" / "campaign.json")
    assert final != "PROMOTED", (final, rows)
    assert exit_code == 1
    assert rows["T23"][0] == "FAIL", rows["T23"]
    assert judge_gen2.ACCOUNTING_ARTIFACT_MOVED in rows["T23"][1]
    # And the edited totals read as compliant, which is why the pin is the
    # gate that matters: T13 alone would have passed them.
    assert rows["T13"][0] == "PASS", rows["T13"]


def test_the_judge_never_certifies_a_settlement_the_record_disagrees_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second half of T23: a consistently re-pinned artifact still disagrees.

    A tamperer who edits the artifact *and* re-pins its digest in the record
    has produced bytes whose digest matches the pin -- and left the record's
    own settlement verdict saying the campaign breached. The two halves of one
    record cannot both be trusted, so the judge refuses: the artifact settles
    compliant while the record says non-compliant, and no reading of that pair
    is a promotion.
    """
    _manifest, run, root = _run_over_envelope(tmp_path, monkeypatch)
    assert run.verdict == "REJECTED"

    _rerender_ledger(
        root, lambda document: document["totals"]["incremental"].update({"wall_gpu_hours": 0.0})
    )
    record_path = root / RUN_RECORD
    record = _record(root)
    record["cost"]["accounting_digest"] = _accounting(root)["digest_sha256"]
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    rows, final, exit_code = _judge(root, tmp_path / "inputs" / "campaign.json")
    assert final != "PROMOTED", (final, rows)
    assert exit_code == 1
    assert rows["T23"][0] == "FAIL", rows["T23"]
    assert judge_gen2.SETTLEMENT_DISAGREES_WITH_RECORD in rows["T23"][1]


def test_the_judge_still_certifies_the_settlement_of_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: T23 is a gate, not a way to refuse every run.

    A campaign inside its envelope records a compliant settlement of the
    ledger it pinned, and both T13 and T23 pass on the root it left behind --
    otherwise the new gate would be an unconditional refusal dressed as an
    audit.
    """
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    assert run.verdict == "PROMOTED"

    rows, _final, _exit = _judge(Path(manifest.state_root), tmp_path / "inputs" / "campaign.json")
    assert rows["T13"][0] == "PASS", rows["T13"]
    assert rows["T23"][0] == "PASS", rows["T23"]
    document = _accounting(Path(manifest.state_root))
    assert _record(Path(manifest.state_root))["cost"]["accounting_digest"] == ledger_digest(
        document
    )


def test_a_declared_device_measurement_the_ledger_did_not_take_refuses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declaring device time measured is a claim the ledger has to back.

    The executor reports wall only, so every attempt leaves device time
    unseparated and the campaign total is an estimate. A wall-only declaration
    keeps the device ceiling an admission constraint; declaring the
    measurement makes settlement refuse a device ceiling it cannot settle, and
    the run is rejected rather than certified against a placeholder zero.
    """
    manifest, runner, _document = _campaign(
        tmp_path,
        with_ancestor=True,
        budget={
            "device_gpu_hours_ceiling_per_recipe": 0.30,
            "wall_gpu_hours_ceiling_per_recipe": 0.75,
            "device_gpu_hours_ceiling_campaign": 0.60,
            "wall_gpu_hours_ceiling_campaign": 0.20,
            "device_time_measured": True,
        },
    )
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)

    assert run.cost["device_measured"] is False
    assert run.verdict == "REJECTED"
    assert run.settlement["budget_compliant"] is False
    assert ACTUAL_DEVICE_GPU_HOURS_UNMEASURED in " ".join(
        run.settlement["budget_failure_reasons"]
    )
    # The record must not name a device settlement the ledger cannot support.
    assert run.ceiling_enforcement["device"] != "settlement:measured"


def test_a_measured_device_overrun_refuses_the_campaign_after_every_leg_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign device ceiling has teeth when the declaration says measured.

    Each attempt measures 0.25 device against its own 0.30 recipe ceiling, the
    evaluation measures 0.20, and the campaign total of 0.70 breaches the
    declared 0.60. Nothing else failed: training succeeded, the artifact
    survived, the evaluation was produced. Only the campaign-level device
    envelope is violated, and that is a refusal.
    """
    measured_evaluation = ComputeCost.measured(
        device_gpu_hours=0.20,
        wall_gpu_hours=0.05,
        source="audit:measured evaluation",
        measurement_method="device-seconds",
    )
    manifest, runner, _document = _campaign(
        tmp_path,
        with_ancestor=True,
        budget={
            "device_gpu_hours_ceiling_per_recipe": 0.30,
            "wall_gpu_hours_ceiling_per_recipe": 0.75,
            "device_gpu_hours_ceiling_campaign": 0.60,
            "wall_gpu_hours_ceiling_campaign": 1.50,
            "device_time_measured": True,
        },
        runner=_RecordingRunner(
            gpu_hours=ATTEMPT_WALL_GPU_HOURS, device_gpu_hours=0.25
        ),
    )
    _patch_seams(monkeypatch, runner, cost=measured_evaluation)

    run = run_campaign(manifest)

    assert run.cost["device_measured"] is True
    assert run.cost["device_gpu_hours"] == pytest.approx(0.70)
    assert run.verdict == "REJECTED"
    assert _codes_settlement(run) == [ACTUAL_DEVICE_GPU_HOURS_EXCEEDED]
    assert run.ceiling_enforcement["device"] == "settlement:measured"
