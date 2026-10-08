"""The e2e judge-agreement measurement: a rejected run root, then the frozen judge.

The stop-gate audit's open product question, measured instead of assumed: a
campaign whose manifest declares a retention gate, whose candidate breaches
it. The run REJECTS the promotion with the gate's machine code; the judge then
reads the same run root -- the runner writes exactly the artifacts the judge
reads, so one run root is both the run's output and the judge's input.

This is the measurement that motivated prereg amendment 15, and it now pins the
*coupled* behaviour. Before the amendment the judge returned INCONCLUSIVE on
this root with every gate it owns PASSING -- a certification path over a
candidate the run had refused -- because it read neither the run's decision nor
the declared profile. After it, the same root returns REJECTED, and the refusal
is attributable to a named gate rather than to an unknown instrument arm:

* the run REJECTED with `RETENTION_FLOOR: candidate 0.5 is below the absolute
  floor 0.5625`, and that reason is in the record the judge reads;
* T21 refuses: a candidate the run refused on a declared gate cannot be
  certified here, because this judge audits no declared gate;
* T22 passes: the judge recomputes the same declared constraint through
  production's own `evaluate_retention` + `retention_values` and reaches the
  same code, so the two authorities agree rather than merely being related;
* the judge's own gates still PASS on the same evidence -- the protected slice,
  both regressions, the trusted-ancestor protection, the identity chain,
  settlement, recipes and the judged contamination evidence. The coupling is
  what changed, not the branch rules.

The attribution rule is unchanged: the judge is pointed at the same declaration
the run used (the fixture manifest), which is the control the judge's own test
file uses. Judged against the *real* declaration instead, the root additionally
fails bookkeeping the fixture cannot carry (the real base/adapter digests, the
real recipe ids, the real contamination pin) -- fixture-vs-deployment mismatch,
noted in ``docs/AUTONOMY_05_REPORT.md``, not a fact about the judge's gates.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from chowder.growth.campaign_runner import run_campaign

from test_growth_campaign_runner import (
    PROTECTED_ID,
    _campaign,
    _patch_seams,
)

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
_spec = importlib.util.spec_from_file_location("judge_gen2_agreement", GEN2 / "judge_gen2.py")
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

#: The declared gate: an absolute floor above the candidate's protected
#: level. An absolute floor is the one declared shape the predeclared rule
#: has no check for (the rule is relative to the parent; a floor is not), so
#: the run's rejection below is attributable to the declared profile alone --
#: the sharp form of "a promotion the run rejects".
RETENTION_PROFILE = {
    "profile_id": "gen2-protection",
    "constraints": [
        {
            "dimension": "math500",
            "kind": "absolute-floor",
            "value": 0.5625,
            "benchmark": PROTECTED_ID,
        },
    ],
}


def _judge_rows_and_verdict(run_root: Path, manifest_path: Path) -> tuple[list[tuple], str, str, int]:
    """Run the judge and return (threshold rows, verdict, detail, exit).

    The judge's campaign global points at the declaration the run used, then
    is restored -- the same fixture control ``test_growth_gen2_judge.py``
    applies when it points the global at a missing file.
    """
    original = judge_gen2.CAMPAIGN_MANIFEST
    judge_gen2.CAMPAIGN_MANIFEST = manifest_path
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = judge_gen2.judge(run_root)
    finally:
        judge_gen2.CAMPAIGN_MANIFEST = original
    text = buffer.getvalue()
    rows = []
    final = ""
    detail = ""
    for line in text.splitlines():
        # The judge's table is space-aligned with a variable-width check
        # column, so split on the verdict cell itself: threshold, check,
        # PASS/UNKNOWN/FAIL/INFO, detail.
        match = re.match(
            r"^(T\d+|INFO)\s+(.+?)\s+(PASS|UNKNOWN|FAIL|INFO)\s+(.+)$", line.strip()
        )
        if match and match.group(1).startswith("T"):
            rows.append(match.groups())
        if line.startswith("VERDICT: "):
            final = line.split(": ", 1)[1].strip()
        if line.startswith("DETAIL: "):
            detail = line.split(": ", 1)[1].strip()
    assert final, f"the judge printed no VERDICT line:\n{text}"
    return rows, final, detail, exit_code


def test_a_run_rejected_by_the_declared_gate_is_refused_by_the_judge_too(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """The measured fact, and the coupling that now enforces it.

    The fixture candidate holds 0.5 on the protected slice -- exactly the
    parent's level and the trusted ancestor's, so every gate the judge owns
    passes on the same evidence -- while the declared floor 0.5625 is the only
    gate the run's promotion sees breached. One run root, two authorities, one
    verdict.
    """
    manifest, runner, _document = _campaign(tmp_path, retention_profile=RETENTION_PROFILE)
    _patch_seams(monkeypatch, runner)

    # Fact 1 -- the run rejects, and the rejection names the declared gate.
    run = run_campaign(manifest)
    assert run.verdict == "REJECTED"
    promotion = run.promotion["decision"]
    assert promotion["verdict"] == "REJECTED"
    retention_reasons = [
        str(reason)
        for reason in promotion["reasons"]
        if str(reason).startswith("RETENTION_FLOOR")
    ]
    assert retention_reasons, promotion["reasons"]
    assert "absolute floor 0.5625" in retention_reasons[0]

    run_root = Path(manifest.state_root)
    # The record's *name* is production's (``CampaignRun`` writes
    # ``campaign-run.json``); it is spelled out here rather than read from the
    # judge, so that a judge without the coupling fails on behaviour instead of
    # on a missing attribute.
    record = json.loads((run_root / "campaign-run.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "REJECTED"
    assert any(
        "RETENTION_FLOOR" in str(reason)
        for reason in record["promotion"]["decision"]["reasons"]
    ), record["promotion"]["decision"]["reasons"]

    # Fact 2 -- the judge reads the same run root, against the same declaration.
    rows, final, detail, exit_code = _judge_rows_and_verdict(
        run_root, tmp_path / "inputs" / "campaign.json"
    )
    print(f"MEASURED: run REJECTED via the declared gate; judge verdict {final} ({detail})")
    for row in rows:
        print(f"MEASURED row: {row}")

    # The judge's own branch rules still PASS on the very evidence the run
    # rejected -- what changed is that a refusal by the other authority is now
    # a refusal here too.
    passing_checks = (row[1] for row in rows if row[2] == "PASS")
    joined = " | ".join(passing_checks)
    assert "candidate math500@2024-04 measured + protocol-exact" in joined
    assert "math500@2024-04 candidate-vs-parent regression <= 0.0625" in joined
    assert "trusted-ancestor protection (vs gen0)" in joined
    assert "immediate-parent (gen1) protected regression" in joined
    assert "cost settles within the declared ceilings" in joined
    assert "all recipes accounted" in joined
    assert "judged contamination evidence is the pinned artifact" in joined

    # Fact 3, asserted first and in the sharpest form: the verdict itself. On a
    # judge without the coupling this is where the test fails, with the
    # measured gap restated -- INCONCLUSIVE over a root the run refused.
    assert final == "REJECTED", (
        f"the judge's verdict on a run-rejected root changed: {final} ({detail})"
    )
    assert exit_code == 1

    # The gate the run enforced is now in the judge's own record, named.
    coupling = {row[0]: row for row in rows if row[0] in {"T21", "T22"}}
    assert set(coupling) == {"T21", "T22"}, sorted(coupling)
    assert coupling["T21"][2] == "FAIL"
    assert "DECLARED_GATE_REJECTED_RUN" in coupling["T21"][3]
    assert "RETENTION_FLOOR" in coupling["T21"][3]
    # The recomputation agrees: production's own evaluator, on the arms the
    # judge already audited, reaches the same code the run recorded.
    assert coupling["T22"][2] == "PASS"
    assert "== recomputed" in coupling["T22"][3]
    assert "RETENTION_FLOOR" in coupling["T22"][3]

    unknown_rows = [row for row in rows if row[2] == "UNKNOWN"]
    # Only this fixture's synthetic candidate arm leaves the instrument gates
    # undecided; on a real Gen-2 arm they pass, and T21 is what refuses. T24
    # (amendment 17) is undecided for the same reason as T4/T5 -- the arm carries
    # no instrument run, so there are no per-prompt rows to compare -- and it
    # names that reason itself rather than joining a roster.
    assert unknown_rows and all(
        row[0]
        in {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T12", "T24"}
        for row in unknown_rows
    ), unknown_rows
