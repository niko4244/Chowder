"""The Autonomous Growth workspace, driven through real widgets.

The screen is a client of ``AutonomousGrowthService``, so these tests drive it
with a *recording* service: every assertion is about what the screen did with
the service's answers, and about the rule that no widget decides anything the
service did not.

The properties that matter, and that a plausible-looking UI gets wrong:

* Start is enabled by the service's readiness verdict and by nothing else -- and
  a click that arrives anyway must not spend compute;
* a refusal keeps its individual checks and its machine reason code, rather than
  collapsing into one red state;
* Stop is a request the service records, not a cancellation the UI pretends to
  perform on a running trainer;
* history reports the generation the lineage actually stands on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from chowder.growth.service import (
    HistoryRow,
    LineageView,
    PlanningView,
    PreparationView,
    ReadinessView,
    StatusView,
)
from chowder.tui_growth import AutonomousGrowthScreen


def _lineage() -> LineageView:
    return LineageView(
        generation="gen2",
        base_model_path="F:/models/base",
        base_model_digest="a" * 64,
        adapter_path="F:/runs/gen2/adapter",
        adapter_digest="b" * 64,
        run_root="F:/runs/gen2",
        measured_arm_path="F:/runs/gen2/candidate_evaluation.json",
        trusted_ancestor="gen0",
        trusted_ancestor_arm="F:/runs/gen0/baseline-eval-report.json",
        declared_protected=("math500@2024-04",),
        protected_skills=("math.algebra", "math.competition"),
        profile_generation="gen2",
        profile_measured_skills=("instruction.formatting",),
        policy_digest="c" * 64,
    )


def _status(**overrides: Any) -> StatusView:
    document = {
        "state_root": "F:/runs/growth-state",
        "generations_recorded": 0,
        "current_parent": "gen2",
        "spent_wall_gpu_hours": 0.0,
        "remaining_wall_gpu_hours": 3.0,
        "maximum_total_wall_gpu_hours": 3.0,
        "last_decision": {},
        "operator_stop": {},
    }
    document.update(overrides)
    return StatusView(**document)


def _plan(decision: dict[str, Any] | None = None) -> PlanningView:
    if decision is not None:
        return PlanningView(parent_version="gen2", decision=decision)
    return PlanningView(
        parent_version="gen2",
        proposal={
            "target_skill": "instruction.formatting",
            "target_benchmarks": ["generation-diagnostics@gen2-response-surface-v1"],
            "priority": 0.3723,
            "confidence": 0.725,
            "expected_trainability": 1.0,
            "suggested_training_type": "targeted_repair",
            "expected_cost_gpu_hours": 0.4,
            "weakness_evidence": ["generation-diagnostics@gen2-response-surface-v1=0.5625"],
            "regression_risks": ["math.competition"],
            "why_not_other_targets": {
                "math.competition": "protected evidence: a gate this cannot regress through",
                "coding.generation": "insufficient evidence (unmeasured, not zero)",
            },
            "treatment_reason": "a low, well-measured skill with no prior attempt",
        },
    )


def _prepared(ready: bool, *, refused: bool = False) -> PreparationView:
    checks = (
        ReadinessView("schema", "ok", "schema matches FIELD_ENFORCEMENT"),
        ReadinessView("base_identity", "ok", "base verified"),
        ReadinessView(
            "ancestor_arm",
            "refused" if refused else ("refused" if not ready else "ok"),
            "the trusted-ancestor arm is absent" if refused else "arm parses",
            "READINESS_ANCESTOR_ARM" if refused else "",
        ),
        ReadinessView("campaign_projection", "skipped" if refused else "ok", "not evaluated: ancestor_arm did not pass"),
    )
    if refused:
        return PreparationView(checks=checks)
    return PreparationView(checks=checks)


class _RecordingService:
    """A service that records what the screen asked it for."""

    def __init__(
        self,
        *,
        ready: bool = True,
        refused: bool = False,
        plan_decision: dict[str, Any] | None = None,
        history: tuple[HistoryRow, ...] = (),
        status: StatusView | None = None,
        raise_on_open: Exception | None = None,
    ) -> None:
        self.ready = ready
        self.refused = refused
        self.plan_decision = plan_decision
        self._history = history
        self._status = status or _status()
        self.raise_on_open = raise_on_open
        self.calls: list[str] = []
        self.stopped: list[str] = []
        self.start_kwargs: dict[str, Any] = {}

    # the factory shape the screen calls
    def __call__(self, **kwargs: Any) -> "_RecordingService":
        if self.raise_on_open is not None:
            raise self.raise_on_open
        self.calls.append("open")
        self.opened_with = kwargs
        return self

    def inspect(self) -> LineageView:
        self.calls.append("inspect")
        return _lineage()

    def status(self) -> StatusView:
        self.calls.append("status")
        return self._status

    def plan_next(self) -> PlanningView:
        self.calls.append("plan_next")
        return _plan(self.plan_decision)

    def prepare_next(self) -> PreparationView:
        self.calls.append("prepare_next")
        return _prepared(self.ready, refused=self.refused)

    def history(self) -> tuple[HistoryRow, ...]:
        self.calls.append("history")
        return self._history

    def request_stop(self, *, reason: str = "") -> None:
        self.calls.append("request_stop")
        self.stopped.append(reason)

    def start(self, *, max_generations: int | None = None, resume: bool = False) -> Any:
        self.calls.append("start")
        self.start_kwargs = {"max_generations": max_generations, "resume": resume}
        from chowder.growth.growth_loop import LoopDecision, LoopRunReport

        return LoopRunReport(
            decision=LoopDecision("STOP_SUCCESS", "the generation limit was reached"),
            generations=(),
            budget=self._status.to_dict(),
            parent_version="gen2",
        )


class _Harness(App[None]):
    def __init__(self, service: _RecordingService) -> None:
        super().__init__()
        self._service = service

    def on_mount(self) -> None:
        self.push_screen(AutonomousGrowthScreen(service_factory=self._service))


async def _press(app: App[None], pilot: Any, widget_id: str) -> None:
    """Press a real widget and settle the message pump.

    The workspace is taller than any terminal a test opens, so ``pilot.click``
    cannot address a control below the fold. Pressing the actual ``Button``
    posts the same ``Pressed`` message a click does and goes through the same
    handler, which is what these tests are about -- the layout is the screen's
    business, not the decision's.
    """
    app.screen.query_one(f"#{widget_id}", Button).press()
    # ``pause`` drains the queue it can see when it starts counting, but a press
    # *bubbles* from the button to the screen, so the handler's message can land
    # in the screen's queue after the counters reached zero. Settle twice: the
    # second pause's counters are queued behind whatever the first left behind.
    await pilot.pause()
    await pilot.pause()


def _text(app: App[None], widget_id: str) -> str:
    """The rendered text of a Static panel, whatever this Textual exposes it as."""
    widget = app.screen.query_one(f"#{widget_id}", Static)
    for attribute in ("renderable", "_content"):
        value = getattr(widget, attribute, None)
        if value is not None:
            return str(getattr(value, "plain", value))
    return str(widget.render())


@pytest.mark.asyncio
async def test_the_workspace_renders_and_shows_what_it_resolved() -> None:
    service = _RecordingService()
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        resolved = _text(app, "growth_resolved")
        assert "policy" in resolved
        assert "parent declaration" in resolved
        assert "state root" in resolved
        # Nothing was opened just by rendering: resolving is not starting.
        assert "open" not in service.calls

        await _press(app, pilot, "growth_inspect")
        assert "inspect" in service.calls
        lineage = _text(app, "growth_lineage")
        assert "gen2" in lineage
        assert "gen0" in lineage  # the trusted ancestor
        assert "math.competition" in lineage  # protected skills, named


@pytest.mark.asyncio
async def test_plan_uses_the_service_and_shows_the_refusal_reason_code() -> None:
    service = _RecordingService(
        plan_decision={
            "action": "REQUIRES_HUMAN_REVIEW",
            "reason": "the target needs architecture_research",
            "reason_codes": ["TREATMENT_REQUIRES_REVIEW"],
            "terminal": True,
        }
    )
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_plan")

        assert "plan_next" in service.calls
        panel = _text(app, "growth_planning")
        assert "REQUIRES_HUMAN_REVIEW" in panel
        assert "TREATMENT_REQUIRES_REVIEW" in panel


@pytest.mark.asyncio
async def test_plan_shows_evidence_priority_confidence_and_why_not() -> None:
    service = _RecordingService()
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_plan")

        panel = _text(app, "growth_planning")
        assert "instruction.formatting" in panel
        assert "0.3723" in panel
        assert "0.725" in panel
        assert "targeted_repair" in panel
        assert "protected evidence" in panel
        assert "unmeasured" in panel


@pytest.mark.asyncio
async def test_readiness_keeps_every_check_distinct_when_it_refuses() -> None:
    service = _RecordingService(ready=False, refused=True)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_prepare")

        panel = _text(app, "growth_readiness")
        assert "✓ base_identity: ok" in panel
        assert "✗ ancestor_arm: refused" in panel
        assert "READINESS_ANCESTOR_ARM" in panel
        assert "– campaign_projection: skipped" in panel
        assert "READINESS_CAMPAIGN_PROJECTION" not in panel.split("skipped")[1]


@pytest.mark.asyncio
async def test_start_is_disabled_until_the_service_reports_ready() -> None:
    refused = _RecordingService(ready=False, refused=True)
    app = _Harness(refused)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        start = app.screen.query_one("#growth_start", Button)
        assert start.disabled, "start must be disabled before readiness is known"

        await _press(app, pilot, "growth_prepare")
        assert app.screen.query_one("#growth_start", Button).disabled


@pytest.mark.asyncio
async def test_start_is_enabled_only_on_ready_and_then_runs_the_service() -> None:
    service = _RecordingService(ready=True)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_prepare")
        start = app.screen.query_one("#growth_start", Button)
        assert not start.disabled

        await _press(app, pilot, "growth_start")
        for _ in range(20):
            await pilot.pause()
            if "start" in service.calls:
                break
        assert "start" in service.calls
        assert service.start_kwargs["max_generations"] == 1
        assert service.start_kwargs["resume"] is False
        assert "Stopped: STOP_SUCCESS" in _text(app, "growth_status")


@pytest.mark.asyncio
async def test_a_programmatic_start_on_an_unready_campaign_spends_nothing() -> None:
    """Defence in depth: a disabled button is not the only thing stopping it."""
    service = _RecordingService(ready=False, refused=True)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        app.screen._start()
        await pilot.pause()

        assert "start" not in service.calls
        assert "refused" in _text(app, "growth_status").lower()


@pytest.mark.asyncio
async def test_stop_asks_the_service_for_a_durable_request() -> None:
    service = _RecordingService(ready=True)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_stop")

        assert "request_stop" in service.calls
        assert service.stopped, "the reason is recorded, not just displayed"
        assert "stop after" in _text(app, "growth_status").lower()


@pytest.mark.asyncio
async def test_resume_goes_through_the_service_with_resume_set() -> None:
    service = _RecordingService(ready=True, status=_status(generations_recorded=1))
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_resume")
        for _ in range(20):
            await pilot.pause()
            if "start" in service.calls:
                break
        assert service.start_kwargs.get("resume") is True
        assert "request_stop" not in service.calls


@pytest.mark.asyncio
async def test_history_reports_the_effective_generation_not_the_candidate() -> None:
    rows = (
        HistoryRow(
            index=1,
            cycle_id="gen2-a1-instruction-formatting",
            generation="gen2",
            effective_generation="gen2",
            target_skill="instruction.formatting",
            treatment="targeted_repair",
            candidate_result="PROMOTED",
            promotion_result="promoted",
            wall_gpu_hours=0.42,
            measured_effect=0.18,
            regressions=(),
        ),
        HistoryRow(
            index=2,
            cycle_id="gen3-a1-instruction-formatting",
            generation="gen3",
            effective_generation="gen2",
            target_skill="instruction.formatting",
            treatment="sft",
            candidate_result="REJECTED",
            promotion_result="not_promoted",
            wall_gpu_hours=0.31,
            measured_effect=-0.01,
            regressions=("math500@2024-04",),
        ),
    )
    service = _RecordingService(history=rows)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_history")
        panel = _text(app, "growth_live")

        assert "lineage now: gen2  (promoted)" in panel
        # The rejection does NOT report gen3 as the current model.
        assert "lineage now: gen2  (not promoted)" in panel
        assert "gen3  (promoted)" not in panel
        assert "regressions: math500@2024-04" in panel


@pytest.mark.asyncio
async def test_a_service_that_cannot_open_refuses_visibly_and_disables_start() -> None:
    from chowder.growth.service import GrowthServiceRefusal

    service = _RecordingService(
        raise_on_open=GrowthServiceRefusal(
            "SERVICE_PROFILE_ABSENT: no measured parent capability profile"
        )
    )
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_inspect")

        status = _text(app, "growth_status")
        assert "SERVICE_PROFILE_ABSENT" in status or "refused" in status.lower()
        assert app.screen.query_one("#growth_start", Button).disabled
        log = app.screen.query_one("#growth_log")
        assert log is not None


@pytest.mark.asyncio
async def test_the_budget_display_reflects_the_service_spend() -> None:
    service = _RecordingService(
        status=_status(spent_wall_gpu_hours=0.42, remaining_wall_gpu_hours=2.58)
    )
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        await _press(app, pilot, "growth_stop")
        panel = _text(app, "growth_live")

        assert "GPU-hours spent        : 0.42" in panel
        assert "GPU-hours remaining    : 2.58" in panel


@pytest.mark.asyncio
async def test_the_max_generations_input_is_honoured() -> None:
    service = _RecordingService(ready=True)
    app = _Harness(service)
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        app.screen.query_one("#growth_max_generations", Input).value = "2"
        await _press(app, pilot, "growth_prepare")
        await _press(app, pilot, "growth_start")
        for _ in range(20):
            await pilot.pause()
            if "start" in service.calls:
                break
        assert service.start_kwargs["max_generations"] == 2


# Reaching the workspace from the real ``ChowderTUI`` is asserted in
# ``tests/test_tui.py``, where the app's own entry point is tested -- this file
# drives the screen itself.


def test_a_running_campaign_is_never_cancelled_mid_kernel() -> None:
    """Stop is a request, and the screen says so rather than implying a kill."""
    screen = AutonomousGrowthScreen(service_factory=_RecordingService())
    source = Path(__file__).resolve().parents[1] / "src" / "chowder" / "tui_growth.py"
    text = source.read_text(encoding="utf-8")
    assert "CancellationToken" not in text, (
        "the workspace must not claim a cancellation the campaign stack cannot "
        "perform; stop is honoured at a generation boundary"
    )
    assert "request_stop" in text


@pytest.mark.parametrize(
    "widget_id",
    [
        "growth_inspect",
        "growth_plan",
        "growth_prepare",
        "growth_start",
        "growth_stop",
        "growth_resume",
        "growth_history",
    ],
)
@pytest.mark.asyncio
async def test_every_documented_control_exists(widget_id: str) -> None:
    """The mission's control list, as buttons a user can actually press."""
    app = _Harness(_RecordingService())
    async with app.run_test(size=(140, 90)) as pilot:
        await pilot.pause()
        assert app.screen.query_one(f"#{widget_id}", Button) is not None
