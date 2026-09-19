"""The loop's CLI: one entrypoint, and what it does when it cannot proceed.

These tests run the real command handlers with the real production seams -- no
executor is injected anywhere. What they therefore prove is the property that
matters most about an unattended controller: when it cannot legitimately proceed,
it *stops* and says why, having spent nothing.

They deliberately do not assert a promotion. Reaching one needs a real campaign,
and a test that could reach one would be able to launch real training.
"""

from __future__ import annotations

import json
from pathlib import Path

from chowder.evals.result import (
    MEASURED_THIS_GENERATION,
    SUPPORTED,
    BenchmarkRun,
    EvalReport,
)
from fixtures_growth_loop import GEN2_MANIFEST, parent_manifest, policy_from

MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
FORMAT = "generation-diagnostics@gen2-response-surface-v1"


def _main(*argv):
    from chowder.cli import build_parser

    args = build_parser().parse_args(list(argv))
    return args.func(args)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A parent declaration, a policy, and a parent run root with one arm."""
    document = json.loads(GEN2_MANIFEST.read_text(encoding="utf-8"))
    document["state_root"] = str(tmp_path / "parent-run")
    manifest_path = tmp_path / "parent.json"
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    parent = parent_manifest(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(policy_from(parent).to_dict(), indent=2), encoding="utf-8"
    )

    evidence = tmp_path / "parent-run"
    evidence.mkdir(parents=True, exist_ok=True)
    EvalReport(
        generation_version="gen2",
        runs=(
            BenchmarkRun(
                benchmark_qualified_id=FORMAT,
                adapter="chowder_custom",
                generation_version="gen2",
                score=0.24,
                n_samples=16,
                support=SUPPORTED,
                measurement_origin=MEASURED_THIS_GENERATION,
            ),
            BenchmarkRun(
                benchmark_qualified_id=MATH,
                adapter="chowder_custom",
                generation_version="gen2",
                score=0.61,
                n_samples=16,
                support=SUPPORTED,
                measurement_origin=MEASURED_THIS_GENERATION,
            ),
            BenchmarkRun(
                benchmark_qualified_id=MGSM,
                adapter="chowder_custom",
                generation_version="gen2",
                score=0.58,
                n_samples=16,
                support=SUPPORTED,
                measurement_origin=MEASURED_THIS_GENERATION,
            ),
        ),
    ).save(evidence / "candidate_evaluation.json")
    return manifest_path, policy_path, evidence


def test_status_prints_the_durable_state_the_loop_decides_from(
    tmp_path: Path, capsys
) -> None:
    from chowder.growth.target_selection import GrowthState

    state = GrowthState(root=tmp_path / "growth-state")
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen3-a1-instruction-formatting",
        generation="gen3",
        cost_gpu_hours=0.42,
        candidate_result="PROMOTED",
        promotion_result="promoted",
        measured_effect=0.2,
        promoted_identity=("adapters/gen3", "a" * 64),
    )
    state.set_stopping_state(
        {
            "action": "STOP_SUCCESS",
            "reason": "the generation limit was reached",
            "reason_codes": ["GENERATION_LIMIT_REACHED"],
            "terminal": True,
        }
    )

    exit_code = _main("growth", "loop", "status", str(state.root))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["generations_recorded"] == 1
    assert payload["spent_wall_gpu_hours"] == 0.42
    assert payload["stopping_state"]["action"] == "STOP_SUCCESS"
    assert payload["targets"][0]["target_skill"] == "instruction.formatting"


def test_run_without_a_measured_parent_profile_refuses_before_composing(
    tmp_path: Path, capsys
) -> None:
    """No profile, no target: the loop must not choose one from nothing."""
    manifest_path, policy_path, _ = _fixture(tmp_path)

    exit_code = _main(
        "growth",
        "loop",
        "run",
        str(policy_path),
        "--parent",
        str(manifest_path),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "REFUSED"
    assert "capability profile" in payload["refusal_reason"]
    assert not (tmp_path / "growth-state").exists(), "a refusal must not start a session"


def test_a_promoted_generation_becomes_the_parent_the_next_one_trains_from() -> None:
    """The plumbing the loop depends on: a run's promoted artifact is its identity."""
    from chowder.growth.growth_loop import campaign_outcome

    class _Run:
        verdict = "PROMOTED"
        cost = {"wall_gpu_hours": 0.42}
        selection = {"artifact_ref": "adapters/gen3", "artifact_sha256": "b" * 64}
        certification: dict = {}
        settlement: dict = {}
        promotion: dict = {}
        record_path = ""

    identity = campaign_outcome(_Run()).parent_identity
    assert identity == ("adapters/gen3", "b" * 64)


def test_run_reaches_a_terminal_refusal_without_launching_training(
    tmp_path: Path, capsys
) -> None:
    """The composed generation is real; the spend is zero because it refused.

    Preparation is asked for the campaign's inputs from the parent generation's
    evidence, and this fixture's layout holds none, so the loop must refuse
    *before* the executor -- the whole point of gating readiness ahead of spend.
    """
    manifest_path, policy_path, evidence = _fixture(tmp_path)

    exit_code = _main(
        "growth",
        "loop",
        "run",
        str(policy_path),
        "--parent",
        str(manifest_path),
        "--parent-evidence",
        str(evidence),
        "--state-root",
        str(tmp_path / "loop-state"),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1, "a refusal is not success"
    assert payload["decision"]["terminal"] is True
    assert payload["decision"]["action"] in {"STOP_UNCERTAIN", "REQUIRES_HUMAN_REVIEW"}
    assert payload["decision_ok"] is False
    assert payload["generations"] == [], "no campaign may be recorded as run"
    assert payload["budget"]["spent_wall_gpu_hours"] == 0.0
    assert payload["budget"]["remaining_wall_gpu_hours"] == 6.0

    # The next generation was composed from evidence and frozen before the
    # refusal -- that is what makes the refusal about *this* campaign.
    frozen = sorted(tmp_path.glob("gen3-a1-*"))
    assert len(frozen) == 1
    assert (frozen[0] / "campaign.json").is_file()
    assert (frozen[0] / "preregistration.json").is_file()


def test_resume_adopts_the_durable_verdict_without_running_anything(
    tmp_path: Path, capsys
) -> None:
    """A session that already ended resumes into its own decision."""
    from chowder.growth.target_selection import GrowthState

    manifest_path, policy_path, evidence = _fixture(tmp_path)
    state_root = tmp_path / "loop-state"
    state = GrowthState(root=state_root)
    state.set_stopping_state(
        {
            "action": "STOP_PLATEAU",
            "reason": "3 consecutive generations failed to promote",
            "reason_codes": ["CONSECUTIVE_NON_PROMOTIONS"],
            "terminal": True,
        }
    )

    exit_code = _main(
        "growth",
        "loop",
        "resume",
        str(policy_path),
        "--parent",
        str(manifest_path),
        "--parent-evidence",
        str(evidence),
        "--state-root",
        str(state_root),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0, "resuming a settled session is the session's own outcome"
    assert payload["decision"]["action"] == "STOP_PLATEAU"
    assert payload["decision"]["reason_codes"] == ["CONSECUTIVE_NON_PROMOTIONS"]
    assert payload["generations"] == []
    assert payload["budget"]["spent_wall_gpu_hours"] == 0.0


def test_plan_prints_the_next_target_without_composing_a_declaration(
    tmp_path: Path, capsys
) -> None:
    """Planning is a read: it must not freeze a declaration as a side effect."""
    manifest_path, policy_path, evidence = _fixture(tmp_path)

    exit_code = _main(
        "growth",
        "loop",
        "plan",
        str(policy_path),
        "--parent",
        str(manifest_path),
        "--parent-evidence",
        str(evidence),
        "--state-root",
        str(tmp_path / "plan-state"),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["proposal"]["target_skill"]
    assert payload["proposal"]["target_benchmarks"]
    assert not list(tmp_path.glob("gen3-a1-*")), "planning must not freeze anything"
