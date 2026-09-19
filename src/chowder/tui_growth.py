"""The Autonomous Growth workspace: the growth loop, driven from the interface.

Everything the previous control plane could do from ``chowder growth loop ...``
is available here, and -- more importantly -- nothing here is *implemented* here.
Every button calls :class:`chowder.growth.service.AutonomousGrowthService`, which
is the same object the CLI commands are clients of, so the two cannot preview
different work or reach different verdicts.

Three rules shape the layout:

* **what is resolved is displayed.** Paths may be auto-discovered where the
  evidence makes the answer unique, but the panel always shows what was chosen,
  because a silently guessed parent run is a lineage pointing at another model;
* **a refusal is not one red state.** Every readiness check keeps its own
  ``ok`` / ``refused`` / ``skipped`` badge and its machine reason code, so a
  reader can tell "this input is missing" from "this gate said no";
* **no widget owns a decision.** Start is enabled by the service's readiness
  verdict and by nothing else, so a UI can never launch a campaign production
  would refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

#: The service factory the workspace calls. Injectable so a test drives a
#: recording service without a policy file, a GPU or a campaign.
ServiceFactory = Callable[..., Any]


def _default_service_factory(**kwargs: Any) -> Any:  # noqa: ANN401 - a service
    from .growth.service import AutonomousGrowthService

    return AutonomousGrowthService.open_from_paths(**kwargs)


_CSS = """
AutonomousGrowthScreen { layout: vertical; }
#growth_body { padding: 1 2; }
#growth_body Static.panel {
    padding: 1; border: round $accent; margin-bottom: 1; width: 100%;
}
#growth_body Static.section { margin-top: 1; text-style: bold; }
#growth_controls { height: auto; margin: 1 0; }
#growth_controls Button { margin-right: 1; }
#growth_log { height: 12; border: round $primary; }
#growth_status { padding: 0 1; height: 1; }
"""


class AutonomousGrowthScreen(Screen[None]):
    """Inspect, plan, prepare, start, stop and review autonomous growth."""

    CSS = _CSS
    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(
        self,
        *,
        service_factory: ServiceFactory | None = None,
        defaults: dict[str, str] | None = None,
        maximum_generations: int = 1,
    ) -> None:
        super().__init__()
        self._service_factory = service_factory or _default_service_factory
        self._defaults = dict(defaults or {})
        self._maximum_generations = int(maximum_generations)
        self._service: Any = None
        self._last_readiness_ready = False
        self._generations_run = 0
        self._worker = None

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="growth_body"):
            yield Static(
                "Advance a generation at a time. Every button here calls the same "
                "production service the `chowder growth loop` commands use, so what "
                "this screen previews is what a run executes.",
            )

            yield Static("Loop inputs", classes="section")
            yield Label("Loop policy (the immutable envelope)")
            yield Input(self._defaults.get("policy_path", ""), id="growth_policy")
            yield Label("Parent campaign declaration")
            yield Input(self._defaults.get("parent_declaration_path", ""), id="growth_parent")
            yield Label("Parent evidence (run root or arm report)")
            yield Input(self._defaults.get("parent_evidence_path", ""), id="growth_evidence")
            yield Label("Parent capability profile (optional if evidence is given)")
            yield Input(self._defaults.get("parent_profile_path", ""), id="growth_profile")
            yield Label("Growth-state root")
            yield Input(self._defaults.get("state_root", ""), id="growth_state")
            yield Label("Maximum generations for this session")
            yield Input(str(self._maximum_generations), id="growth_max_generations")

            yield Static(
                "Resolved inputs", id="growth_resolved", classes="panel", markup=False
            )

            yield Static("Current lineage", classes="section")
            yield Static(
                "Not inspected yet", id="growth_lineage", classes="panel", markup=False
            )

            yield Static("Planning", classes="section")
            yield Static(
                "Not planned yet", id="growth_planning", classes="panel", markup=False
            )

            yield Static("Readiness", classes="section")
            # These panels are plain text on purpose. Machine reason codes are
            # written in square brackets, which Textual would otherwise read as a
            # markup tag and silently drop -- a screen that hides the refusal
            # code is a screen an operator cannot act on.
            yield Static(
                "Not checked yet", id="growth_readiness", classes="panel", markup=False
            )

            yield Static("Frozen campaign", classes="section")
            yield Static(
                "Nothing frozen yet", id="growth_frozen", classes="panel", markup=False
            )

            yield Static("Live status", classes="section")
            yield Static("Idle", id="growth_live", classes="panel", markup=False)

            with Horizontal(id="growth_controls"):
                yield Button("Inspect", id="growth_inspect")
                yield Button("Plan Next", id="growth_plan")
                yield Button("Prepare + Check Readiness", id="growth_prepare")
                yield Button(
                    "Start Autonomous Growth", id="growth_start", variant="success", disabled=True
                )
                yield Button("Stop After Current Campaign", id="growth_stop", variant="warning")
                yield Button("Resume", id="growth_resume")
                yield Button("View Growth History", id="growth_history")
                yield Button("Back", id="growth_back")
            yield Static("Idle", id="growth_status")
            yield RichLog(id="growth_log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._report_resolved()

    # -- inputs ------------------------------------------------------------

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _report_resolved(self) -> None:
        """Show exactly which paths a session would open, before it opens them."""
        lines = [
            f"policy            : {self._value('growth_policy') or '(not set)'}",
            f"parent declaration: {self._value('growth_parent') or '(not set)'}",
            f"parent evidence   : {self._value('growth_evidence') or '(not set)'}",
            f"parent profile    : {self._value('growth_profile') or '(not set)'}",
            f"state root        : {self._value('growth_state') or '(default: beside the parent run root)'}",
            f"max generations   : {self._value('growth_max_generations') or '1'}",
        ]
        self.query_one("#growth_resolved", Static).update("\n".join(lines))

    def _open(self) -> Any:  # noqa: ANN401 - a service
        if self._service is None:
            self._service = self._service_factory(
                policy_path=self._value("growth_policy"),
                parent_declaration_path=self._value("growth_parent"),
                state_root=self._value("growth_state") or None,
                parent_profile_path=self._value("growth_profile"),
                parent_evidence_path=self._value("growth_evidence"),
            )
        return self._service

    def _set_status(self, text: str) -> None:
        self.query_one("#growth_status", Static).update(text)

    def _log(self, text: str) -> None:
        self.query_one("#growth_log", RichLog).write(text)

    def _fail(self, action: str, error: Exception) -> None:
        self._set_status(f"{action} refused: {error}")
        self._log(f"[red]{action} refused:[/] {type(error).__name__}: {error}")
        self.query_one("#growth_start", Button).disabled = True
        self._last_readiness_ready = False

    # -- rendering ---------------------------------------------------------

    def _render_lineage(self, payload: dict[str, Any]) -> None:
        digest = str(payload.get("adapter_digest") or "")
        measured = ", ".join(payload.get("profile_measured_skills") or ()) or (
            "none (the parent has no measurement under the declared protocol)"
        )
        lines = [
            f"parent generation  : {payload.get('generation', '')}",
            f"base identity      : {Path(str(payload.get('base_model_path', ''))).name}"
            f"  {str(payload.get('base_model_digest', ''))[:12]}",
            f"adapter identity   : {payload.get('adapter_path', '') or '(the base itself)'}"
            f"  {digest[:12]}",
            f"parent evidence    : {payload.get('run_root', '')}",
            f"measured arm       : {payload.get('measured_arm_path', '')}",
            f"trusted ancestor   : {payload.get('trusted_ancestor', '')}"
            f"  {payload.get('trusted_ancestor_arm', '')}",
            f"protected skills   : {', '.join(payload.get('protected_skills') or ()) or 'none'}",
            f"profile generation : {payload.get('profile_generation', '')}",
            f"profile measured   : {measured}",
            f"policy digest      : {str(payload.get('policy_digest', ''))[:12]}",
        ]
        self.query_one("#growth_lineage", Static).update("\n".join(lines))

    def _render_planning(self, payload: dict[str, Any]) -> None:
        decision = payload.get("decision")
        if decision:
            codes = ", ".join(decision.get("reason_codes") or ()) or "none"
            self.query_one("#growth_planning", Static).update(
                f"no campaign will be composed\n"
                f"decision     : {decision.get('action', '')}\n"
                f"reason       : {decision.get('reason', '')}\n"
                f"reason codes : {codes}"
            )
            return
        proposal = payload.get("proposal") or {}
        why_not = proposal.get("why_not_other_targets") or {}
        lines = [
            f"target            : {proposal.get('target_skill', '')}",
            f"benchmarks        : {', '.join(proposal.get('target_benchmarks') or ())}",
            f"priority          : {proposal.get('priority')}",
            f"confidence        : {proposal.get('confidence')}",
            f"trainability      : {proposal.get('expected_trainability')}",
            f"treatment         : {proposal.get('suggested_training_type', '')}",
            f"expected cost     : {proposal.get('expected_cost_gpu_hours')} wall GPU-hours",
            f"supporting        : {'; '.join(proposal.get('weakness_evidence') or ())}",
            f"why this target   : {proposal.get('treatment_reason', '')}",
            f"regression risks  : {', '.join(proposal.get('regression_risks') or ()) or 'none named'}",
            "why not others    :",
        ]
        lines.extend(
            f"  - {skill}: {reason}" for skill, reason in sorted(why_not.items())
        )
        self.query_one("#growth_planning", Static).update("\n".join(lines))

    def _render_readiness(self, payload: dict[str, Any]) -> None:
        if payload.get("decision"):
            self._render_planning({"decision": payload["decision"]})
            self.query_one("#growth_readiness", Static).update(
                "not checked: the attempt could not be prepared"
            )
            return
        badges = {"ok": "✓", "refused": "✗", "skipped": "–"}
        lines: list[str] = []
        for check in payload.get("checks") or ():
            status = str(check.get("status", ""))
            mark = badges.get(status, "?")
            line = f"{mark} {check.get('name', '')}: {status}"
            if check.get("reason_code"):
                line += f"  [{check['reason_code']}]"
            if check.get("detail"):
                line += f"\n    {check['detail']}"
            lines.append(line)
        if payload.get("ready"):
            lines.append("")
            lines.append("READY — Start Autonomous Growth is enabled")
        elif payload.get("refused"):
            lines.append("")
            lines.append("REFUSED — start is disabled until every check passes")
        self.query_one("#growth_readiness", Static).update("\n".join(lines) or "no checks")

    def _render_frozen(self, payload: dict[str, Any]) -> None:
        attempt = payload.get("attempt")
        if not attempt:
            self.query_one("#growth_frozen", Static).update("Nothing frozen yet")
            return
        lines = [
            f"cycle id            : {attempt.get('cycle_id', '')}",
            f"candidate generation: {attempt.get('candidate_version', '')}",
            f"frozen digest       : {attempt.get('frozen_digest', '')[:16]}",
            f"recipe count        : {len(attempt.get('recipe_ids') or ())}",
            f"recipes             : {', '.join(attempt.get('recipe_ids') or ())}",
            f"target benchmarks   : {', '.join((attempt.get('target') or {}).get('target_benchmarks') or ())}",
            f"directory           : {attempt.get('directory', '')}",
        ]
        self.query_one("#growth_frozen", Static).update("\n".join(lines))

    def _render_live(self, payload: dict[str, Any]) -> None:
        lines = [
            f"generations run        : {self._generations_run}",
            f"current parent         : {payload.get('current_parent', '')}",
            f"generations recorded   : {payload.get('generations_recorded')}",
            f"GPU-hours spent        : {payload.get('spent_wall_gpu_hours')}",
            f"GPU-hours remaining    : {payload.get('remaining_wall_gpu_hours')}",
            f"session envelope       : {payload.get('maximum_total_wall_gpu_hours')}",
        ]
        last = payload.get("last_decision") or {}
        if last:
            lines.append(f"last decision          : {last.get('action', '')}")
            lines.append(f"  {last.get('reason', '')}")
        stopped = payload.get("operator_stop") or {}
        if stopped:
            lines.append(f"stop requested         : {stopped.get('reason', '')}")
        self.query_one("#growth_live", Static).update("\n".join(lines))

    def _render_history(self, rows: tuple[Any, ...]) -> None:
        if not rows:
            self.query_one("#growth_live", Static).update(
                "no generations recorded for this session"
            )
            return
        lines = []
        for row in rows:
            lines.append(
                f"{row.index}. {row.cycle_id}  target={row.target_skill} "
                f"treatment={row.treatment} verdict={row.candidate_result} "
                f"cost={row.wall_gpu_hours} effect={row.measured_effect}"
            )
            lines.append(
                f"   lineage now: {row.effective_generation}"
                + ("  (promoted)" if row.promoted else "  (not promoted)")
                + (f"  regressions: {', '.join(row.regressions)}" if row.regressions else "")
            )
        self.query_one("#growth_live", Static).update("\n".join(lines))

    # -- actions -----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handler = {
            "growth_inspect": self._inspect,
            "growth_plan": self._plan,
            "growth_prepare": self._prepare,
            "growth_start": self._start,
            "growth_stop": self._stop,
            "growth_resume": self._resume,
            "growth_history": self._history,
            "growth_back": lambda: self.app.pop_screen(),
        }.get(event.button.id or "")
        if handler is not None:
            handler()

    def _inspect(self) -> None:
        try:
            self._report_resolved()
            service = self._open()
            self._render_lineage(service.inspect().to_dict())
            self._render_live(service.status().to_dict())
            self._set_status("Lineage resolved")
        except Exception as error:  # noqa: BLE001 - a refusal is shown, not raised
            self._fail("Inspect", error)

    def _plan(self) -> None:
        try:
            service = self._open()
            view = service.plan_next().to_dict()
            self._render_planning(view)
            self._set_status(
                "Planning refused; see the reason code"
                if view.get("decision")
                else "Planned a target"
            )
        except Exception as error:  # noqa: BLE001
            self._fail("Plan", error)

    def _prepare(self) -> None:
        """Plan, prepare and freeze, then gate on the production readiness report."""
        try:
            service = self._open()
            view = service.prepare_next()
            payload = view.to_dict()
            self._render_planning({"decision": payload["decision"]} if payload.get("decision") else view.to_dict())
            self._render_readiness(payload)
            self._render_frozen(payload)
            self._render_live(service.status().to_dict())
            self._last_readiness_ready = bool(payload.get("ready"))
            # Only the service's verdict enables Start. A UI that decided this
            # itself could launch a campaign production would refuse.
            self.query_one("#growth_start", Button).disabled = not self._last_readiness_ready
            self._set_status(
                "READY" if self._last_readiness_ready else "Not ready; start stays disabled"
            )
            for code in view.reason_codes:
                self._log(f"[yellow]refusal code:[/] {code}")
        except Exception as error:  # noqa: BLE001
            self._fail("Prepare", error)

    def _start(self) -> None:
        if not self._last_readiness_ready:
            # Defence in depth: the button is disabled, but a programmatic click
            # must not be able to spend compute on an unready campaign.
            self._set_status("Start refused: readiness has not reported READY")
            return
        try:
            service = self._open()
            maximum = int(self._value("growth_max_generations") or 1)
        except Exception as error:  # noqa: BLE001
            self._fail("Start", error)
            return
        self._set_status("Running…")
        self._log(f"[bold]Starting autonomous growth[/] (max {maximum} generation(s))")
        self._worker = self._run_loop(service, maximum)

    @work(thread=True, exclusive=True)
    def _run_loop(self, service: Any, maximum: int) -> None:  # noqa: ANN401 - a service
        """Run the loop off the UI thread, reporting each generation as it lands."""
        try:
            report = service.start(max_generations=maximum)
        except Exception as error:  # noqa: BLE001
            self.app.call_from_thread(self._set_status, f"Run failed: {type(error).__name__}: {error}")
            self.app.call_from_thread(self._log, f"[red]Run failed:[/] {error}")
            return
        self.app.call_from_thread(self._show_report, report)

    def _show_report(self, report: Any) -> None:  # noqa: ANN401 - a LoopRunReport
        self._generations_run = len(report.generations)
        self._set_status(f"Stopped: {report.decision.action}")
        self._log(
            f"[green]{report.decision.action}[/] — {report.decision.reason}"
        )
        for code in report.decision.reason_codes:
            self._log(f"[yellow]reason code:[/] {code}")
        for record in report.generations:
            self._log(
                f"gen {record.index}: {record.cycle_id} target={record.target_skill} "
                f"verdict={record.verdict} {record.wall_gpu_hours} wall GPU-hours"
            )
        try:
            self._render_live(self._open().status().to_dict())
            self._render_lineage(self._open().inspect().to_dict())
        except Exception:  # noqa: BLE001 - the run's report is the evidence
            pass
        self.query_one("#growth_start", Button).disabled = True
        self._last_readiness_ready = False

    def _stop(self) -> None:
        try:
            service = self._open()
            service.request_stop(reason="operator stop from the Chowder interface")
            self._render_live(service.status().to_dict())
            self._set_status("Will stop after the current campaign")
            self._log("[yellow]Stop requested:[/] no next generation will start")
        except Exception as error:  # noqa: BLE001
            self._fail("Stop", error)

    def _resume(self) -> None:
        try:
            service = self._open()
            payload = service.status().to_dict()
            self._render_live(payload)
            self._log(
                f"[bold]Resuming[/] from {payload.get('state_root')} "
                f"({payload.get('generations_recorded')} generation(s) recorded)"
            )
            maximum = int(self._value("growth_max_generations") or 1)
        except Exception as error:  # noqa: BLE001
            self._fail("Resume", error)
            return
        self._set_status("Resuming…")
        self._worker = self._run_resume(service, maximum)

    @work(thread=True, exclusive=True)
    def _run_resume(self, service: Any, maximum: int) -> None:  # noqa: ANN401
        try:
            report = service.start(max_generations=maximum, resume=True)
        except Exception as error:  # noqa: BLE001
            self.app.call_from_thread(self._set_status, f"Resume failed: {error}")
            return
        self.app.call_from_thread(self._show_report, report)

    def _history(self) -> None:
        try:
            self._render_history(self._open().history())
            self._set_status("Growth history shown")
        except Exception as error:  # noqa: BLE001
            self._fail("History", error)


def open_growth_screen(app: Any, **kwargs: Any) -> None:  # noqa: ANN401 - a ChowderTUI
    app.push_screen(AutonomousGrowthScreen(**kwargs))
