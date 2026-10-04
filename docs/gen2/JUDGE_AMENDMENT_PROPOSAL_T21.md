# Proposal: amendment 15 to the frozen Gen-2 judge (gate T21) — for review, not implemented

**Status: proposal only. `docs/gen2/judge_gen2.py` is unchanged in this
branch.** This document states the measured defect, the minimal amendment that
closes it, the exact insertion points, and what the amendment deliberately does
not do. Nothing here has been applied; the diff sketch below is the review
artifact.

## The defect, as measured

`tests/test_growth_gen2_judge_agreement.py` measures it on a real
`run_campaign` root: the run **REJECTS** with
`RETENTION_FLOOR: candidate 0.5 is below the absolute floor 0.5625`, and the
frozen judge, pointed at that same root and declaration, returns
**INCONCLUSIVE with every gate it owns PASSING** (T11 protected slice, T11
candidate-vs-parent, T16 trusted-ancestor, T17 immediate-parent, T19 identity
chain, T20 policy agreement, T12/T18 contamination, T13 settlement, T14
recipes, T15 artifact digest). The declared gate appears nowhere in its record.

Root cause, statically pinned: the judge reads no run decision and no declared
profile. `grep` finds zero occurrences of `retention_profile` or `RETENTION_`
in its 1,357 lines, and `judge()` (line 434) opens exactly three arm reports
plus `chosen_candidate.json`, the accounting and the pinned contamination
manifest. Everything the run decided is on disk beside the arms it reads:
`<run_root>/campaign-run.json` carries `verdict` and
`promotion.decision.{verdict,reasons,checks,...}` (`CampaignRun.to_dict`,
`PromotionAssembly.to_dict`).

So the judge's blindness is not that it computes the wrong thing. It is that
it never hears the run's answer, and its INCONCLUSIVE comes only from UNKNOWN
instrument metadata — which a real Gen-2 arm would supply.

## What "minimal" means here

The judge owns the frozen *policy* and delegates every mechanism to production
(this is stated in its own module docstring). The amendment therefore does not
re-derive retention arithmetic, does not recompute the promotion rule, and does
not add a verdict class. It adds one gate that reads what the run recorded and
fails closed when that record is absent or disagrees with the declaration —
`branch_verdict` already maps FAIL to REJECTED and UNKNOWN to INCONCLUSIVE, and
the exit code already treats anything but all-PASS as a refusal, so **no change
to the verdict algebra, the thresholds, or the exit code is required**.

## The amendment

### 1. Four named reasons (next to the existing `CONTAMINATION_*` block, ~line 120)

```python
#: Why the run's own decision could not be audited. Absence is UNKNOWN,
#: never an assumed pass -- the same rule every other gate here follows.
RUN_RECORD_ABSENT = "RUN_RECORD_ABSENT"
RUN_DECISION_ABSENT = "RUN_DECISION_ABSENT"
#: A run the branch already refused cannot be certified as anything else.
DECLARED_GATE_REJECTED_RUN = "DECLARED_GATE_REJECTED_RUN"
#: A gate that is not in the declaration rejected the run: the declaration is
#: not the policy the run enforced.
UNDECLARED_GATE_IN_RUN = "UNDECLARED_GATE_IN_RUN"
```

### 2. One gate function (beside `_protection_agreement_gate`, line 1158)

The reason vocabulary is re-exported from production rather than restated -- the
same move the judge already makes for the measurement codes at line ~390
(`MEASUREMENT_DIGEST_MISMATCH = _certification.…`, etc.). `RetentionViolation.code`
is a *property* on a frozen dataclass, so the codes are read off one violation
per reachable shape:

```python
from chowder.growth.retention import (  # noqa: E402
    RetentionConstraint,
    RetentionViolation,
)

#: Production's own declared-gate vocabulary, read off its owner rather than
#: restated here: every ``reason`` the promotion gate appends begins with one of
#: these three codes. ``RetentionViolation.code`` answers NaN -> UNMEASURED and
#: otherwise picks by constraint kind, so the three shapes are exhaustive.
RETENTION_CODES = frozenset(
    RetentionViolation(
        dimension="code-probe",
        constraint=RetentionConstraint(
            dimension="code-probe", kind=kind, value=0.0, benchmark="probe@2026-01"
        ),
        measured=measured,
        detail="",
    ).code
    for kind, measured in (
        ("max-regression", 0.0),      # RETENTION_REGRESSION
        ("absolute-floor", 0.0),      # RETENTION_FLOOR
        ("max-regression", float("nan")),  # RETENTION_UNMEASURED
    )
)
RETENTION_REASON_PREFIX = "RETENTION_"
```

The plain prefix test alone is sufficient for the gate; `RETENTION_CODES` exists
so a new code shape added to `retention.py` fails the amendment's test instead of
slipping past a prefix match. That test is part of the amendment, not optional.

```python
def _declared_gate_agreement(
    verdict: Verdict, campaign: CampaignManifest | None, run_root: Path
) -> None:
    """The run's recorded decision must agree with the declared profile.

    The run enforces ``retention_profile`` on both promotion paths
    (``GrowthCycle._apply_promotion_gates``); this judge never reads that
    verdict. So a candidate the run REJECTED on a declared gate can reach a
    judge table in which every audited gate passes -- a certification of
    something the branch already refused. This gate closes that by reading the
    run's own answer, and it fails closed when that answer is absent.

    It compares *codes*, not prose: a reason is classified by its production
    code prefix, so no human-readable string is parsed here. The declared
    profile's own shape (dimensions, benchmarks, tolerances) is reported as
    detail, not re-derived.
    """
    requirement = "the run's decision agrees with the declared retention profile"
    record = load_json(run_root / "campaign-run.json")  # quals_harness
    if record is None:
        verdict.add("T21", requirement, UNKNOWN,
                    f"{RUN_RECORD_ABSENT}: {run_root}/campaign-run.json is absent or "
                    "unreadable, so the run's own decision cannot be audited")
        return
    if campaign is not None and str(record.get("cycle_id", "")) != str(campaign.cycle_id):
        verdict.add("T21", requirement, UNKNOWN,
                    f"{RUN_DECISION_ABSENT}: the record in this run root belongs to "
                    f"cycle {record.get('cycle_id')!r}, not {campaign.cycle_id!r}")
        return
    decision = (record.get("promotion") or {}).get("decision") or {}
    if not decision:
        verdict.add("T21", requirement, UNKNOWN,
                    f"{RUN_DECISION_ABSENT}: the run record carries no promotion "
                    "decision (a run that refused before adjudication records none)")
        return
    run_verdict = str(record.get("verdict", ""))
    reasons = [str(reason) for reason in decision.get("reasons", ())]
    gate_reasons = [r for r in reasons if r.startswith(RETENTION_REASON_PREFIX)]
    declared = campaign.retention_profile if campaign is not None else None

    if gate_reasons and declared is None:
        verdict.add("T21", requirement, FAIL,
                    f"{UNDECLARED_GATE_IN_RUN}: the run recorded declared-gate breaches "
                    f"{gate_reasons!r} while the declaration names no retention profile")
        return
    if run_verdict == "PROMOTED" and gate_reasons:
        verdict.add("T21", requirement, FAIL,
                    f"{UNDECLARED_GATE_IN_RUN}: the run recorded PROMOTED together with "
                    f"declared-gate breaches {gate_reasons!r}")
        return
    if run_verdict in {"REJECTED", "TAINTED"} and gate_reasons:
        verdict.add("T21", requirement, FAIL,
                    f"{DECLARED_GATE_REJECTED_RUN}: the run refused this candidate on the "
                    f"declared gate(s) {gate_reasons!r}; this judge audits no declared gate, "
                    "so its own table cannot overturn that refusal")
        return
    if run_verdict in {"REJECTED", "TAINTED"} and declared is not None:
        # Rejected on the predeclared rule alone: the run's own reasons belong
        # in the table, even though T11/T16/T17 are the gates that audit them.
        verdict.add("T21", requirement, PASS,
                    f"the run refused this candidate without a declared-gate breach: "
                    f"{'; '.join(reasons) or 'no reasons recorded'}")
        return
    verdict.add("T21", requirement, PASS,
                f"run verdict {run_verdict or 'unrecorded'}; declared-gate reasons: "
                f"{gate_reasons or 'none'}; declared profile: "
                f"{declared.profile_id if declared is not None else 'none'}")
```

**Rejected option, for the record:** matching the breached *dimension* by
parsing the reason prose (`"… on 'protected-math' …"`) was considered and
rejected. It makes the judge depend on `RetentionViolation.detail`'s wording,
which is exactly the kind of coupling that let this gap go unnoticed. The code
prefix plus the declared profile's presence is enough to catch the measured
failure and cannot be broken by rewording a message.


def _declared_gate_agreement(
    verdict: Verdict, campaign: CampaignManifest | None, run_root: Path
) -> None:
    """The run's recorded decision must agree with the declared profile.

    The run enforces ``retention_profile`` on both promotion paths
    (``GrowthCycle._apply_promotion_gates``); this judge never reads that
    verdict. So a candidate the run REJECTED on a declared gate can reach a
    judge table in which every audited gate passes -- which is a certification
    of something the branch already refused. This gate closes that by reading
    the run's own answer and failing closed when it is missing.
    """
    requirement = "the run's decision agrees with the declared retention profile"
    record = load_json(run_root / "campaign-run.json")   # quals_harness
    if record is None:
        verdict.add("T21", requirement, UNKNOWN,
                    f"{RUN_RECORD_ABSENT}: {run_root}/campaign-run.json is absent "
                    "or unreadable, so the run's own decision cannot be audited")
        return
    decision = (record.get("promotion") or {}).get("decision") or {}
    verdict_record = str(record.get("verdict", ""))
    reasons = [str(reason) for reason in decision.get("reasons", ())]
    retention_reasons = [r for r in reasons if r.startswith(RETENTION_REASON_PREFIX)]

    if campaign is None or campaign.retention_profile is None:
        if retention_reasons:
            verdict.add("T21", requirement, FAIL,
                        f"{UNDECLARED_GATE_IN_RUN}: the run recorded "
                        f"{retention_reasons!r} but the declaration names no "
                        "retention profile")
            return
        verdict.add("T21", requirement, PASS,
                    "the declaration names no retention profile and the run "
                    "recorded no declared-gate breach")
        return

    declared = {
        f"{constraint.kind}:{constraint.dimension}"
        for constraint in campaign.retention_profile.constraints
    }
    breached = {
        reason.split(":", 1)[1].split(" on ", 1)[0].strip(" '")
        for reason in retention_reasons
    }
    undeclared = {
        dimension
        for dimension in breached
        if not any(dimension in entry for entry in declared)
    }

    if verdict_record == "PROMOTED" and retention_reasons:
        verdict.add("T21", requirement, FAIL,
                    f"{UNDECLARED_GATE_IN_RUN}: the run recorded PROMOTED while "
                    f"also recording {retention_reasons!r}")
        return
    if verdict_record in {"REJECTED", "TAINTED"} and retention_reasons:
        verdict.add("T21", requirement, FAIL,
                    f"{DECLARED_GATE_REJECTED_RUN}: the run refused this candidate "
                    f"on the declared gate(s) {retention_reasons!r}; this judge "
                    "audits no declared gate, so its own table cannot overturn it")
        return
    if undeclared:
        verdict.add("T21", requirement, FAIL,
                    f"{UNDECLARED_GATE_IN_RUN}: the run recorded breaches for "
                    f"{sorted(undeclared)!r}, which the declaration does not name")
        return
    verdict.add("T21", requirement, PASS,
                f"run verdict {verdict_record or 'REFUSED'}; declared-gate reasons "
                f"{retention_reasons or 'none'}")
```

### 3. One wiring line (in `judge()`, after `_protection_agreement_gate`, line 465)

```python
    _declared_gate_agreement(verdict, campaign, run_root)
```

### 4. Two documentation lines

- The module docstring's arm list gains `<run_root>/campaign-run.json` as a
  read-only input, with the reason it is read (the run's own decision is
  evidence, not a second implementation of the promotion rule).
- The `INFO` row naming the frozen policy gains `+ GEN2_PREREG_AMENDMENT15`
  (see below), so a judge table still names the policy it was frozen with.

## Governance: what amending the frozen judge requires

1. **A named prereg amendment**, in the same series as amendments 1-14:
   `docs/quals/GEN2_PREREG_AMENDMENT15_2026-10-04.md` (amendment 14, the
   declared-inputs declaration, lands on 2026-10-04 and is independent of this
   one). The judge is frozen
   *with* the prereg; it is amended by amending the prereg, in writing, before
   the change.
2. **The no-visible-candidate-results rule.** The freeze says thresholds may
   not change after candidate results are visible. Two things make this
   amendment admissible, and the reviewer should confirm both:
   - it changes **no threshold** — it adds a gate that reads an artifact and
     defers to the run's verdict, so no numeric policy moves;
   - no Gen-2 candidate evaluation exists: the ten
     `runs/2026-09-17-gen2-response-surface/attempts/*` hold training evidence
     only, and there is no `candidate_evaluation.json` or `chosen_candidate.json`
     anywhere in the Gen-2 state root. The run refused at the
     `candidate_evaluation` phase. (Amendment 14 of 2026-10-04 made the
     declaration fully runnable; it declared inputs and recipe ids, and
     measured nothing.)
3. **A test that fails without it.** The amendment's own proof is the inverse
   of `tests/test_growth_gen2_judge_agreement.py`: with T21 present, that test's
   fixture root must yield REJECTED (not INCONCLUSIVE), and its
   all-passing-a-run-rejected-root claim must be what the amendment retires.

## What the amendment deliberately does NOT do

- It does not recompute retention arithmetic. The declared constraint's
  threshold stays owned by `RetentionProfile`/`evaluate_retention`; the judge
  reads the outcome, exactly as T13/T14/T15 already defer settlement, digests
  and accounting to production.
- It does not trust the run's verdict blindly in the other direction. A run
  recording PROMOTED *and* a declared-gate breach is FAIL (an internally
  inconsistent record), and a breach on a dimension the declaration does not
  name is FAIL (the declaration is not the policy that ran).
- It does not turn a missing record into a pass: absent `campaign-run.json` is
  UNKNOWN, which the judge already treats as refusing to certify.
- It does not change the T1-T20 gates, the table, `branch_verdict`, or the exit
  code. It is one gate, four reason names, one wiring line.
- It does not close the wider gap (the judge's instrument metadata T1-T10 come
  from a historical Gen-1 driver, not from the run). That is a separate,
  larger amendment and should not be smuggled in here.

## Cost and risk

- Diff size: ~60 lines added, 0 lines of existing logic modified.
- Behaviour change on a real Gen-2 root: a run that PROMOTED keeps certifying
  (T21 PASS); a run REJECTED on a declared gate can no longer certify as
  INCONCLUSIVE-with-everything-passing; a run REJECTED on the predeclared rule
  alone still reports PASS with the run's own reasons visible in the table.
- Failure mode if wrong: the judge over-refuses (says REJECTED for a run whose
  `campaign-run.json` is from a different cycle). The sketch above already
  guards it: a `cycle_id` mismatch is UNKNOWN (`RUN_DECISION_ABSENT`), not FAIL.