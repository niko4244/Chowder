"""The six fake-compute scenarios: the loop's terminal states, pinned.

These tests do not assert that the loop "works". They assert that, for a table
of measured outcomes, the session ends in exactly one declared terminal state --
because the unattended-loop decision that matters most is the one that stops it.

Two of the six exist to catch defects this pass actually found, so the reason
codes are load-bearing rather than decorative:

* scenario F asserts ``CONSECUTIVE_NON_PROMOTIONS`` *and* three distinct targets.
  It originally asserted ``GENERATION_LIMIT_REACHED``, which would have passed
  for a loop that retried one failing target until the limit stopped it -- the
  exact behaviour the scenario is named after preventing.
* scenario D and E assert that zero campaigns were launched, so an envelope that
  cannot hold a campaign, and a target that needs a human, cost nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.growth import simulator

SCENARIOS = {scenario.name: scenario for scenario in simulator.SCENARIOS}


def test_every_named_scenario_is_present_and_unique() -> None:
    names = [scenario.name for scenario in simulator.SCENARIOS]
    assert len(names) == len(set(names)) == 6
    assert [name[0] for name in names] == list("ABCDEF")


@pytest.mark.parametrize("scenario", simulator.SCENARIOS, ids=lambda s: s.name)
def test_a_scenario_reaches_its_declared_terminal_state(
    scenario: simulator.Scenario, tmp_path: Path
) -> None:
    """``run_scenario`` asserts the terminal state internally; this re-checks it.

    The re-check is not redundant: ``run_scenario`` compares against the
    scenario's own declared expectations, so asserting the same fields here from
    the report proves the report *carries* them rather than that a helper
    happened to agree with itself.
    """
    root = tmp_path / scenario.name
    root.mkdir(parents=True, exist_ok=True)
    report = simulator.run_scenario(scenario, root=root)

    assert report.decision.action == scenario.expected_action
    assert report.decision.terminal, "a simulation must end in a terminal decision"
    for code in scenario.expected_reason_codes:
        assert code in report.decision.reason_codes
    assert len(report.generations) == scenario.expected_campaigns_run
    assert report.parent_version == scenario.expected_parent_version

    promotions = sum(
        1 for record in report.generations if record.verdict.upper() == simulator.PROMOTED
    )
    assert promotions == scenario.expected_promotions


@pytest.mark.parametrize("scenario", simulator.SCENARIOS, ids=lambda s: s.name)
def test_a_scenario_accounts_for_every_campaign_it_ran(
    scenario: simulator.Scenario, tmp_path: Path
) -> None:
    """Session spend is the sum of the campaigns' measured costs, exactly.

    An unaccounted campaign is how a bounded loop becomes unbounded, so this is
    an equality rather than a bound.
    """
    root = tmp_path / scenario.name
    root.mkdir(parents=True, exist_ok=True)
    report = simulator.run_scenario(scenario, root=root)

    declared = sum(
        float(campaign.wall_gpu_hours)
        for campaign in scenario.campaigns[: len(report.generations)]
    )
    assert report.budget["spent_wall_gpu_hours"] == pytest.approx(declared)
    assert report.budget["remaining_wall_gpu_hours"] == pytest.approx(
        report.budget["maximum_total_wall_gpu_hours"] - declared
    )


def test_the_failing_intervention_scenario_walks_targets_rather_than_retrying(
    tmp_path: Path,
) -> None:
    """The property scenario F is named after, asserted directly.

    Three campaigns, three *different* targets. A loop that retried its weakest
    target three times would also run three campaigns and also end non-promoted,
    so counting campaigns alone proves nothing here.
    """
    scenario = SCENARIOS["F-a-failing-intervention-is-not-repeated"]
    root = tmp_path / "F"
    root.mkdir(parents=True, exist_ok=True)
    report = simulator.run_scenario(scenario, root=root)

    targets = [record.target_skill for record in report.generations]
    assert len(targets) == len(set(targets)) == 3
    assert report.decision.action == simulator.STOP_PLATEAU
    assert "CONSECUTIVE_NON_PROMOTIONS" in report.decision.reason_codes
    assert report.parent_version == "gen2", "no promotion, so the parent cannot move"


def test_the_simulation_is_deterministic(tmp_path: Path) -> None:
    """Same inputs, same session -- or the scenarios prove nothing reproducible.

    ``preregistration_digest`` is compared separately and is deliberately *not*
    part of this equality: it covers the frozen declaration, and a declaration
    names its own run root, so two simulations in different directories must
    have different digests. Each one is still deterministic within its own root.
    """
    scenario = SCENARIOS["A-two-promotions-then-a-plateau"]
    reports = []
    for run in ("first", "second"):
        root = tmp_path / f"A-{run}"
        root.mkdir(parents=True, exist_ok=True)
        reports.append(simulator.run_scenario(scenario, root=root).to_dict())

    assert reports[0]["decision"] == reports[1]["decision"]
    assert reports[0]["budget"] == reports[1]["budget"]
    assert reports[0]["parent_version"] == reports[1]["parent_version"]
    for first, second in zip(reports[0]["generations"], reports[1]["generations"]):
        for key in (
            "index",
            "cycle_id",
            "candidate_version",
            "target_skill",
            "treatment",
            "verdict",
            "wall_gpu_hours",
            "measured_target_effect",
        ):
            assert first[key] == second[key]
        assert len(first["preregistration_digest"]) == 64
        assert len(second["preregistration_digest"]) == 64


def test_run_all_covers_every_scenario(tmp_path: Path) -> None:
    reports = simulator.run_all(root=tmp_path / "all")
    assert set(reports) == {scenario.name for scenario in simulator.SCENARIOS}


def test_a_missing_frozen_declaration_refuses_rather_than_inventing_limits(
    tmp_path: Path,
) -> None:
    """No declaration, no simulation: the policy is read, never assumed."""
    scenario = SCENARIOS["A-two-promotions-then-a-plateau"]
    with pytest.raises(simulator.SimulationError, match="frozen parent declaration"):
        simulator.run_scenario(
            scenario,
            root=tmp_path / "A",
            manifest_path=tmp_path / "absent.json",
        )
