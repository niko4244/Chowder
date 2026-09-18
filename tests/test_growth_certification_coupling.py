"""One run root is the run's output *and* the frozen judge's input.

The certification boundary is only real if the pipeline produces the evidence
the judge reads. Before this module's subject existed, the judge consumed five
artifacts no path in ``src/`` wrote: a manifest-driven campaign reported
PROMOTED while ``docs/gen2/judge_gen2.py`` pointed at that same run root
returned INCONCLUSIVE, so a gen2 verdict could only be reached by assembling
the arms by hand.

These tests drive a campaign through the real CLI and then hand the resulting
directory to the frozen judge, and they pin the two failure directions that
matter: a run whose evidence is absent or tampered must not certify, and the
runner must not invent an arm the manifest never declared. Nothing here
loosens a gate: the judge is imported unmodified from ``docs/gen2`` and the
promotion thresholds are untouched.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import re
import sys
from pathlib import Path

import pytest

from chowder.cli import main as chowder_main
from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION
from chowder.growth import campaign_runner
from chowder.growth.campaign import CampaignManifest
from chowder.growth.campaign_runner import CERTIFICATION_EVIDENCE

import test_growth_campaign_runner as campaign_fixture
from test_growth_gen2_judge import (  # the judge's own fixture builders
    judge_gen2,
    _slice_run,
    _write_arm,
)

REPO = Path(__file__).resolve().parents[1]
JUDGE_SOURCE = REPO / "docs" / "gen2" / "judge_gen2.py"

MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
INSTRUMENT = judge_gen2.INSTRUMENT_ID

#: Written by the same runner, through the cost ledger, rather than by the
#: evidence writer. Listed here so the drift guard below covers the whole set
#: of files the judge reads from a run root.
ACCOUNTING = "cycle_compute_accounting.json"

#: The one ``.json`` literal in the judge that is not a run-root artifact.
FROZEN_MANIFEST = "gen2_campaign.json"


# --------------------------------------------------------------------------
# a judge-complete campaign: declared arms carry the frozen protocol evidence
# --------------------------------------------------------------------------


def _slice(qualified_id: str, version: str, origin: str, score: float):
    """The judge's frozen protocol metadata, on the registry's declared metric.

    The judge checks the 16-item protocol; the cycle binds rows against the
    registry, which declares ``accuracy`` as MATH-500's and MGSM's primary
    metric, and refuses a row whose metric is not the declared one.
    """
    return dataclasses.replace(
        _slice_run(qualified_id, version, origin, score), metric="accuracy"
    )


def _arms(inputs: Path) -> dict[str, str]:
    """The three arms the judge reads, as *declared inputs* to the campaign.

    Provenance is the evaluator's declaration and is copied verbatim by the
    runner: the candidate arm is candidate-measured, the parent and ancestor
    arms are parent-measured, and every arm carries the frozen 16-item
    protocol metadata the judge checks.
    """
    inputs.mkdir(parents=True, exist_ok=True)
    candidate = inputs / "candidate-report.json"
    _write_arm(
        candidate,
        "gen2",
        MEASURED_THIS_GENERATION,
        dup=0,
        echo=0,
        correct=True,
        slices=(
            _slice(MATH, "gen2", MEASURED_THIS_GENERATION, 0.25),
            _slice(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0625),
        ),
    )
    parent = inputs / "parent-report.json"
    _write_arm(
        parent,
        "gen1",
        MEASURED_PARENT,
        dup=11,
        echo=7,
        correct=True,
        slices=(
            _slice(MATH, "gen1", MEASURED_PARENT, 0.0),
            _slice(MGSM, "gen1", MEASURED_PARENT, 0.0),
        ),
    )
    ancestor = inputs / "baseline-report.json"
    _write_arm(
        ancestor,
        "gen0",
        MEASURED_PARENT,
        with_instrument=False,
        slices=(
            _slice(MATH, "gen0", MEASURED_PARENT, 0.0),
            _slice(MGSM, "gen0", MEASURED_PARENT, 0.0),
        ),
    )
    return {
        "candidate_eval_report_path": str(candidate),
        "parent_eval_report_path": str(parent),
        "baseline_eval_report_path": str(ancestor),
    }


def _contamination(inputs: Path) -> str:
    """CLEAN on the frozen evaluated set, with a declared training source.

    Its own filename, so the shared campaign fixture cannot overwrite it: the
    file the run binds and the file the judge reads must be the same evidence.
    """
    path = inputs / "campaign-contamination.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    qualified_id: {"status": "CLEAN", "reason": "coupling fixture"}
                    for qualified_id in (INSTRUMENT, MATH, MGSM)
                },
                "policy": {},
                "training_sources": {"src-1": {"status": "CLEAN", "reason": "fixture"}},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> tuple[Path, int, dict]:
    """Write the declaration, run it through the real CLI, return its outcome."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    declared: dict[str, object] = {
        **_arms(inputs),
        "contamination_manifest_path": _contamination(inputs),
        **overrides,
    }
    manifest, runner, _document = campaign_fixture._campaign(
        tmp_path,
        # The judge's required set is the frozen manifest's; the campaign's own
        # declared sets are the catalog-pinned ids the cycle can bind. Both are
        # satisfied by rows the candidate arm may legally carry.
        target_benchmarks=[MATH],
        protected_benchmarks=[MGSM],
        broad_benchmarks=[MGSM],
        **declared,
    )
    manifest_path = inputs / "campaign.json"
    assert CampaignManifest.from_file(manifest_path).cycle_id == manifest.cycle_id
    monkeypatch.setattr(campaign_runner, "default_runner", runner)
    monkeypatch.setattr(
        sys, "argv", ["chowder", "growth", "campaign", "run", str(manifest_path)]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = chowder_main()
    return Path(manifest.state_root), code, json.loads(buffer.getvalue() or "{}")


def _judge(root: Path) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = judge_gen2.judge(root)
    return code, buffer.getvalue()


# --------------------------------------------------------------------------
# the coupling: the judge certifies the root the run just wrote
# --------------------------------------------------------------------------


def test_one_run_root_is_the_frozen_judges_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, code, payload = _campaign(tmp_path, monkeypatch)
    assert code == 0
    assert payload["verdict"] == "PROMOTED"

    # The runner writes every artifact the judge consumes, at the judged paths.
    for name in (*CERTIFICATION_EVIDENCE.values(), ACCOUNTING):
        assert (root / name).is_file(), f"{name} was not written by the run"

    verdict, output = _judge(root)
    assert verdict == 0, f"the frozen judge refused the run's own evidence:\n{output}"
    assert "VERDICT: PROMOTED" in output


def test_the_runner_does_not_invent_an_arm_the_manifest_never_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No declared ancestor report means no ancestor arm — not a placeholder."""
    root, code, _payload = _campaign(tmp_path, monkeypatch, baseline_eval_report_path="")
    assert code == 0
    assert not (root / CERTIFICATION_EVIDENCE["ancestor"]).exists()

    verdict, output = _judge(root)
    assert verdict == 1
    assert "trusted ancestor" in output


def test_absent_evidence_cannot_certify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a judged artifact after the run refuses the certification."""
    root, _code, _payload = _campaign(tmp_path, monkeypatch)
    assert _judge(root)[0] == 0

    (root / CERTIFICATION_EVIDENCE["candidate"]).unlink()
    verdict, output = _judge(root)
    assert verdict == 1
    assert "candidate_evaluation.json" in output


def test_a_declared_report_that_does_not_exist_refuses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared-but-missing arm is a refusal, not an arm that never existed."""
    missing = tmp_path / "inputs" / "does-not-exist.json"
    root, code, payload = _campaign(
        tmp_path, monkeypatch, candidate_eval_report_path=str(missing)
    )
    assert code == 1
    assert payload["verdict"] == "REFUSED"
    assert "candidate_eval_report_path" in payload["refusal_reason"]
    assert not (root / CERTIFICATION_EVIDENCE["candidate"]).exists()


@pytest.mark.parametrize(
    "mutation",
    ("origin_relabelled", "protocol_drifted", "artifact_mutated"),
)
def test_tampered_evidence_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, _code, _payload = _campaign(tmp_path, monkeypatch)
    assert _judge(root)[0] == 0

    candidate_arm = root / CERTIFICATION_EVIDENCE["candidate"]
    if mutation == "origin_relabelled":
        document = json.loads(candidate_arm.read_text(encoding="utf-8"))
        for run in document["runs"]:
            if run["benchmark_qualified_id"] == MATH:
                run["measurement_origin"] = "CARRIED_REFERENCE"
        candidate_arm.write_text(json.dumps(document), encoding="utf-8")
    elif mutation == "protocol_drifted":
        document = json.loads(candidate_arm.read_text(encoding="utf-8"))
        for run in document["runs"]:
            if run["benchmark_qualified_id"] == MGSM:
                run["metadata"]["seed"] = 99
        candidate_arm.write_text(json.dumps(document), encoding="utf-8")
    else:
        chosen = json.loads(
            (root / CERTIFICATION_EVIDENCE["selection"]).read_text(encoding="utf-8")
        )
        artifact = Path(chosen["artifact_ref"])
        target = artifact / "tampered.bin" if artifact.is_dir() else artifact
        target.write_text("this changed after the digest was recorded", encoding="utf-8")

    verdict, output = _judge(root)
    assert verdict == 1, f"the judge certified tampered evidence ({mutation}):\n{output}"


def test_the_judged_artifact_names_match_the_frozen_judge() -> None:
    """The runner writes the judge's contract, and drift fails the build.

    The judge is frozen policy, so the runner has to match it — the reverse
    (editing the judge to match a convenient filename) is exactly the drift
    this guard exists to prevent.
    """
    literals = set(re.findall(r"\"([a-zA-Z0-9_]+\.json)\"", JUDGE_SOURCE.read_text(encoding="utf-8")))
    judged = literals - {FROZEN_MANIFEST}
    written = set(CERTIFICATION_EVIDENCE.values()) | {ACCOUNTING}
    assert judged == written, (
        "the frozen judge reads artifacts the runner does not write (or vice "
        f"versa): judge={sorted(judged)} runner={sorted(written)}"
    )


def test_the_write_is_recorded_as_its_own_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One owner of the write, and the run record says so."""
    root, code, _payload = _campaign(tmp_path, monkeypatch)
    assert code == 0
    record = json.loads((root / "campaign-run.json").read_text(encoding="utf-8"))
    phases = [str(phase["phase"]) for phase in record["phases"]]
    assert phases.count("certification_evidence") == 1
