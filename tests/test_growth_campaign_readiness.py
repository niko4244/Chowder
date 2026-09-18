"""The zero-compute readiness inspection: what a campaign must satisfy to start.

``chowder growth campaign readiness`` answers "may this declaration train?" from
the declaration and the files it names, without starting a process or loading a
model. Its whole point is that no prerequisite a run checks *before* compute is
discoverable only *after* training, so the load-bearing assertion here is the
agreement between the two: a READY report is exactly a run that reaches training,
and a REFUSED one is exactly a run that stops before it.

The fixture machinery is ``test_growth_campaign_runner``'s: the same declaration,
the same on-disk inputs and the same recording subprocess seam, so the only thing
faked is the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from chowder.cli import main as chowder_main
from chowder.growth.campaign_runner import (
    READINESS_ANCESTOR_ARM,
    READINESS_BASE_IDENTITY,
    READINESS_DECLARED_INPUT,
    READINESS_EVALUATOR,
    READINESS_EVALUATOR_COVERAGE,
    READINESS_RECIPE_SET,
    CampaignRunRefusal,
    check_campaign_readiness,
    run_campaign,
)

from test_growth_campaign_runner import (
    ATTEMPT_WALL_GPU_HOURS,
    _campaign,
    _patch_seams,
    _redeclare,
    _verbs,
)
from test_growth_training_binding import _RecordingRunner

RUN_INPUTS = (
    "project_template_path",
    "training_material_path",
    "data_registry_path",
    "hardware_budget_path",
    "parent_profile_path",
    "parent_eval_report_path",
    "evaluation_material_path",
    "contamination_manifest_path",
)


def _manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "inputs" / "campaign.json"


# --------------------------------------------------------------------------
# a fully declared campaign is ready, and nothing is started to find out
# --------------------------------------------------------------------------


def test_a_fully_declared_campaign_is_ready_without_starting_compute(tmp_path: Path):
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)

    report = check_campaign_readiness(manifest)

    assert report.ready, report.to_dict()
    assert report.status == "READY"
    assert report.reason_codes == ()
    assert all(check.status == "ok" for check in report.checks), report.to_dict()
    # Zero compute by construction: inspecting readiness starts no subprocess.
    assert runner.commands == []


def test_readiness_reports_every_prerequisite_it_checked(tmp_path: Path):
    """The report is the checklist, in the order a run needs it, machine-readable."""
    manifest, _runner, _document = _campaign(tmp_path)

    report = check_campaign_readiness(manifest)
    names = [check.check for check in report.checks]

    assert names[0] == "schema"
    for expected in (
        "declared_inputs",
        "base_identity",
        "contamination",
        "parent_arm",
        "ancestor_arm",
        "protection_policy",
        "plan",
        "recipe_set",
        "campaign_projection",
        "evaluator",
        "evaluator_coverage",
    ):
        assert expected in names, names
    assert all(check.reason_code == "" for check in report.checks if check.status == "ok")


# --------------------------------------------------------------------------
# each knowable prerequisite refuses, and names itself
# --------------------------------------------------------------------------


def test_readiness_names_every_undeclared_input_at_once(tmp_path: Path):
    """One refusal for the declaration, not one per attempt after training."""
    _, _runner, document = _campaign(tmp_path)
    blank = {field: "" for field in RUN_INPUTS}
    manifest = _redeclare(document, **blank)

    report = check_campaign_readiness(manifest)

    assert report.status == "REFUSED"
    assert READINESS_DECLARED_INPUT in report.reason_codes
    detail = next(c.detail for c in report.checks if c.check == "declared_inputs")
    for field in RUN_INPUTS:
        assert field in detail, (field, detail)
    # A dependent check does not re-raise the same missing input as a second
    # failure; it is skipped, which is still not a pass.
    dependent = {c.check: c.status for c in report.checks}
    assert dependent["project_template"] == "skipped"
    assert dependent["parent_arm"] == "skipped"
    assert report.ready is False


def test_readiness_refuses_bytes_that_do_not_match_the_declared_digest(tmp_path: Path):
    _, _runner, document = _campaign(tmp_path)
    manifest = _redeclare(document, base_model_digest="0" * 64)

    report = check_campaign_readiness(manifest)

    assert READINESS_BASE_IDENTITY in report.reason_codes
    # The adapter identity depends on the base having been verified, so it is
    # skipped rather than reported against a base that is not the declared one.
    assert next(c for c in report.checks if c.check == "parent_adapter_identity").status == (
        "skipped"
    )


def test_readiness_refuses_a_campaign_with_no_trusted_ancestor(tmp_path: Path):
    manifest, _runner, _document = _campaign(tmp_path, with_ancestor=False)

    report = check_campaign_readiness(manifest)

    assert READINESS_ANCESTOR_ARM in report.reason_codes
    detail = next(c.detail for c in report.checks if c.check == "ancestor_arm")
    assert "baseline_eval_report_path" in detail


def test_readiness_refuses_a_recipe_set_the_planner_did_not_propose(tmp_path: Path):
    manifest, _runner, _document = _campaign(tmp_path, planned=False)

    report = check_campaign_readiness(manifest)

    assert READINESS_RECIPE_SET in report.reason_codes
    # A plan that cannot be selected cannot be projected either.
    assert next(c for c in report.checks if c.check == "campaign_projection").status == "skipped"


def test_readiness_refuses_when_the_instrument_has_no_material(tmp_path: Path):
    """A declared material path that does not exist is not an available evaluator."""
    _, _runner, document = _campaign(tmp_path)
    manifest = _redeclare(document, evaluation_material_path=str(tmp_path / "absent.json"))

    report = check_campaign_readiness(manifest)

    assert READINESS_EVALUATOR in report.reason_codes
    assert next(c for c in report.checks if c.check == "evaluator_coverage").status == "skipped"


def test_readiness_refuses_an_instrument_that_cannot_cover_a_declared_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Coverage is the evaluator's own admission seam, not the existence of an object."""
    manifest, runner, _document = _campaign(tmp_path)
    # An evaluator whose admission seam refuses the protected set: a real object
    # that simply cannot measure what the campaign declares.
    monkeypatch.setattr(
        "chowder.growth.campaign_runner.default_evaluator_factory",
        lambda manifest, *, state_root=None: _BlindEvaluator(),
    )

    report = check_campaign_readiness(manifest)

    assert READINESS_EVALUATOR_COVERAGE in report.reason_codes
    detail = next(c.detail for c in report.checks if c.check == "evaluator_coverage")
    assert "math500@2024-04" in detail
    assert runner.commands == []


class _BlindEvaluator:
    """An evaluator that is present but cannot measure the protected slice."""

    def admit(self, *, benchmarks: Any, protocol: Any) -> tuple[str, str]:
        return ("CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE", f"cannot measure {list(benchmarks)}")


# --------------------------------------------------------------------------
# the load-bearing invariant: readiness and a run agree
# --------------------------------------------------------------------------


def test_readiness_and_a_run_agree_about_whether_training_may_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A READY report is exactly a run that reaches training; a REFUSED one, one
    that stops before it.

    This is the property the whole command exists for: no prerequisite a run
    checks before compute may be discoverable only after the compute was spent,
    and a READY inspection may never be followed by an immediate pre-compute
    refusal.
    """
    scenarios: dict[str, Any] = {}

    ready, _runner, ready_document = _campaign(tmp_path / "ready")
    scenarios["ready"] = ready

    _, _runner, document = _campaign(tmp_path / "bad-digest")
    scenarios["bad-digest"] = _redeclare(document, base_model_digest="0" * 64)

    no_ancestor, _runner, _document = _campaign(tmp_path / "no-ancestor", with_ancestor=False)
    scenarios["no-ancestor"] = no_ancestor

    bad_recipes, _runner, _document = _campaign(tmp_path / "bad-recipes", planned=False)
    scenarios["bad-recipes"] = bad_recipes

    _, _runner, document = _campaign(tmp_path / "no-material")
    scenarios["no-material"] = _redeclare(
        document, evaluation_material_path=str(tmp_path / "no-material" / "absent.json")
    )

    for name, manifest in scenarios.items():
        runner = _RecordingRunner(gpu_hours=ATTEMPT_WALL_GPU_HOURS)
        _patch_seams(monkeypatch, runner)
        report = check_campaign_readiness(manifest)
        try:
            run_campaign(manifest)
        except CampaignRunRefusal:
            pass
        started = "train" in _verbs(runner)
        assert report.ready == started, (
            name,
            report.status,
            [c.to_dict() for c in report.checks if c.status != "ok"],
            _verbs(runner),
        )


# --------------------------------------------------------------------------
# the CLI surface: zero compute, gateable on its exit code
# --------------------------------------------------------------------------


def test_the_readiness_command_reports_ready_without_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _manifest, runner, _document = _campaign(tmp_path)

    monkeypatch.setattr(
        sys, "argv", ["chowder", "growth", "campaign", "readiness", str(_manifest_path(tmp_path))]
    )
    assert chowder_main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "READY"
    assert payload["reason_codes"] == []
    assert {check["check"] for check in payload["checks"]}
    assert runner.commands == []


def test_the_readiness_command_exits_non_zero_when_a_prerequisite_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _manifest, _runner, document = _campaign(tmp_path)
    broken = {**document, "base_model_digest": "0" * 64}
    path = tmp_path / "broken-campaign.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["chowder", "growth", "campaign", "readiness", str(path)])
    assert chowder_main() == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "REFUSED"
    assert READINESS_BASE_IDENTITY in payload["reason_codes"]
