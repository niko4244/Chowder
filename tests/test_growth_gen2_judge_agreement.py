"""The e2e judge-agreement measurement: a rejected run root, then the frozen judge.

The stop-gate audit's open product question, measured instead of assumed: a
campaign whose manifest declares a retention gate, whose candidate breaches
it. The run REJECTS the promotion with the gate's machine code; the frozen
``docs/gen2/judge_gen2.py`` then reads the same run root -- the runner writes
exactly the artifacts the judge reads, so one run root is both the run's
output and the judge's input.

Attribution rule for this measurement: the judge is pointed at the same
declaration the run used (the fixture manifest), which is the control the
judge's own test file uses. Judged against the *real* frozen manifest
instead, the root additionally fails bookkeeping the fixture cannot carry
(the real base/adapter digests, the real recipe ids, the real contamination
pin) -- that is fixture-vs-deployment mismatch, noted in
``docs/AUTONOMY_05_REPORT.md``, not a fact about the judge's gates.

The measured fact, refined by running it: the judge returns INCONCLUSIVE on
the run-rejected root, but every protection, identity, settlement and recipe
gate it owns PASSES on the very evidence the run rejected -- the declared
gate the run enforced does not appear anywhere in the judge's record. The
INCONCLUSIVE comes only from the instrument gates being UNKNOWN on a
synthetic candidate arm.

No judge changes: this test records the measured fact and pins the
structural reason behind it.
"""

from __future__ import annotations

import importlib.util
import io
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
    """Run the frozen judge and return (threshold rows, verdict, detail, exit).

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


def test_a_run_rejected_by_the_declared_gate_through_the_frozen_judge(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """The measured fact, recorded.

    The fixture candidate holds 0.5 on the protected slice -- exactly the
    parent's level and the trusted ancestor's, so every gate the frozen
    judge owns passes on the same evidence -- while the declared floor
    0.5625 is the only gate the run's promotion sees breached.
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

    # Fact 2 -- the frozen judge reads the same run root, against the same
    # declaration. Recorded as measured, then pinned so a judge change
    # re-measures loudly.
    rows, final, detail, exit_code = _judge_rows_and_verdict(
        Path(manifest.state_root), tmp_path / "inputs" / "campaign.json"
    )
    print(f"MEASURED: run REJECTED via the declared gate; judge verdict {final} ({detail})")
    for row in rows:
        print(f"MEASURED row: {row}")

    # The judge's own protection gates PASS on the very evidence the run
    # rejected: the protected slice, the candidate-vs-parent regression, the
    # trusted-ancestor protection, the identity chain, settlement, recipes.
    passing_checks = (row[1] for row in rows if row[2] == "PASS")
    joined = " | ".join(passing_checks)
    assert "candidate math500@2024-04 measured + protocol-exact" in joined
    assert "math500@2024-04 candidate-vs-parent regression <= 0.0625" in joined
    assert "trusted-ancestor protection (vs gen0)" in joined
    assert "immediate-parent (gen1) protected regression" in joined
    assert "actual cost settled within the declared ceilings" in joined
    assert "all recipes accounted" in joined
    assert "judged contamination evidence is the pinned artifact" in joined

    # The gate the run enforced is absent from the judge's record: no row
    # carries the retention vocabulary or any absolute-floor verdict.
    assert not any("RETENTION_" in str(cell) for row in rows for cell in row)
    assert not any("absolute floor" in str(row[3]) or "0.5625" in str(row[3]) for row in rows)

    # The structural reason, pinned statically: the frozen judge never reads
    # the run's decision or the declared profile.
    judge_source = (GEN2 / "judge_gen2.py").read_text(encoding="utf-8")
    assert "retention_profile" not in judge_source
    assert "RETENTION_" not in judge_source

    # Fact 3 -- the recorded verdict: INCONCLUSIVE solely because the
    # instrument gates (T1-T10, plus the fixture's sourceless contamination
    # T12 row) are UNKNOWN on the synthetic candidate -- every protection,
    # identity, settlement and recipe gate the judge owns PASSES on the very
    # evidence the run rejected. With a real Gen-2 candidate arm carrying the
    # instrument metadata, certification would be reachable on this root.
    assert final == "INCONCLUSIVE", (
        f"the frozen judge's verdict on a run-rejected root changed: {final} ({detail})"
    )
    assert exit_code == 1
    unknown_rows = [row for row in rows if row[2] == "UNKNOWN"]
    assert unknown_rows and all(
        row[0] in {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T12"}
        for row in unknown_rows
    ), unknown_rows
