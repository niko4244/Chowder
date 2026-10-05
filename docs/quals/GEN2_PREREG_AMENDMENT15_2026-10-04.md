# GEN2 prereg amendment 15 — 2026-10-04: the judge is coupled to the run's decision, and a corrected threshold sign

Additive to the 2026-09-18 series and to amendment 14 of the same date.
Admissible under the freeze on the same grounds amendment 14 stated: **no Gen-2
candidate evaluation exists** (`runs/2026-09-17-gen2-response-surface/attempts/*`
hold training evidence only; the state root has no `candidate_evaluation.json` or
`chosen_candidate.json`, and the run refused at the `candidate_evaluation`
phase), **and no threshold moves** — the number in section B is the number the
prereg already froze, with its sign corrected.

## A. The defect, measured

`tests/test_growth_gen2_judge_agreement.py` runs one real `run_campaign` root
twice over: the run **REJECTS** with
`RETENTION_FLOOR: candidate 0.5 is below the absolute floor 0.5625`, and the
frozen judge, pointed at the same root and the same declaration, returned
**INCONCLUSIVE with every gate it owns PASSING** (T11 protected slice, T11
candidate-vs-parent, T16, T17, T19 identity chain, T20 policy agreement,
T12/T18 contamination, T13 settlement, T14 recipes, T15 artifact digest). The
declared gate appeared nowhere in its record. The judge contained no occurrence
of `retention_profile` or `RETENTION_`.

That is the split-brain condition this system exists to prevent: two
authorities over one candidate, one enforcing a gate the other cannot see. With
a real instrument arm (T1–T10 decided), certification was reachable on a root
the run had refused.

## B. The correction: a declared threshold's sign

Writing the coupling surfaced a defect in **amendment 14's own declaration**, and
it is the reason this amendment exists rather than only the coupling.

A `max-regression` constraint's `value` is the **minimum acceptable
candidate-vs-parent delta**, so a *permitted* dip is declared **negative** —
pinned by production's own test,
`tests/test_growth_eval_tiers_retention_paired.py::test_small_declared_dip_within_budget_passes`
(`("max-regression", -0.02)` permits a −0.015 dip). Amendment 14 wrote
`value: 0.0625` and described it as "one 16th of a 16-item mini-slice", i.e. as
the frozen `protection.slice_regression_max = 0.0625` tolerance.

Under production's semantics that declaration requires the candidate to
**improve** by one sixteenth on *both* protected benchmarks, rather than
permitting a one-sixteenth regression. It is a materially stricter gate than the
one the prereg froze, and it contradicts the frozen judge, whose T11/T16/T17
apply `slice_regression_max` as a *permitted regression*. With Gen-1 measuring
0.0 on both protected slices, the declared gate as written would have rejected
a candidate the frozen branch-protection rule accepts.

**Both constraints are now `-0.0625`** — exactly `-(protection.slice_regression_max)`
— and the invariant is pinned against the shipped document by
`tests/test_growth_campaign_runner.py::test_the_declared_retention_profile_states_the_frozen_tolerance_with_the_right_sign`,
which asserts the value, asserts the *meaning* (a dip of exactly the tolerance
passes; a dip one hundredth past it yields `RETENTION_REGRESSION`), and fails
when the sign is flipped.

## C. The coupling: T21 and T22

One gate function, `_declared_gate_agreement`, reading one more artifact from
the directory the judge already reads: `campaign-run.json`, the run's own record
(`CampaignRun.to_dict`). It emits two rows.

**T21 — the recorded decision.** A candidate the run refused on a declared gate
is `FAIL`: `DECLARED_GATE_REJECTED_RUN`. This judge audits no declared gate, so
its table cannot overturn a refusal the branch has already recorded. Also
`FAIL`: a record that promotes a candidate it simultaneously recorded breaching
(`UNDECLARED_GATE_IN_RUN`), and a breach on a constraint the declaration does not
name (`UNDECLARED_GATE_IN_RUN`). `UNKNOWN` — refusing to certify, never assuming
pass — for a record that is absent (`RUN_RECORD_ABSENT`), belongs to another
cycle (`RUN_RECORD_WRONG_CYCLE`), or carries no decision because the run refused
before adjudicating (`RUN_DECISION_ABSENT`). A run refused on the predeclared
rule alone passes T21 with its own reasons printed, because T11/T16/T17 audit
that rule against the arms and a reader should see which refusal is certified.

**T22 — the recomputation.** The declared profile is evaluated *again* inside the
judge, on the arms it already audited, through production's own evaluator and
production's own provenance filter: `evaluate_retention` plus `retention_values`
(`chowder.growth.cycle`, renamed from the private `_retention_values` in this
amendment so the judge and the promotion path call one public function). The
recomputed violation *codes* must equal the codes the run recorded, or the gate
is `FAIL` with `RETENTION_RECOMPUTATION_DISAGREES` — two authorities computing
different answers about the same candidate, where neither may be assumed right.

Nothing is reimplemented. The declaration owns the constraint, production owns
the comparison and the rows that count, and the judge owns only the *agreement*
between the two answers — which is the only thing it was ever entitled to own.
`RETENTION_CODES` is read off `RetentionViolation.code` rather than restated, and
`test_growth_gen2_judge.py::test_the_judges_gate_vocabulary_is_productions_own`
asserts it equals the codes a real evaluation emits, so a new shape cannot pass
through a prefix match.

`branch_verdict`, the exit code, T1–T20 and every threshold are unchanged. A run
root without the record can no longer certify (UNKNOWN → INCONCLUSIVE), which is
the intended fail-closed cost: certification now requires a run that says what
it decided.

## D. Proof

- `tests/test_growth_gen2_judge_agreement.py` now asserts the *coupled*
  behaviour: REJECTED on the run-rejected root, with T21 `FAIL` naming
  `RETENTION_FLOOR` and T22 `PASS` recording that production's evaluator
  recomputes the same code. Reverting only `docs/gen2/judge_gen2.py` makes it
  fail with the measured gap restated —
  `the judge's verdict on a run-rejected root changed: INCONCLUSIVE`.
- `test_a_run_refused_on_a_declared_gate_cannot_be_certified` judges a root whose
  instrument gates T1–T10 are **all decided** (the case the original gap
  predicted would certify) and asserts T21 is the *only* failing row: the
  refusal cannot be attributed to thin evidence.
- `test_a_declared_gate_the_declaration_does_not_name_refuses`,
  `test_a_promoted_record_carrying_a_declared_gate_breach_refuses`,
  `test_the_recomputation_and_the_record_must_agree`,
  `test_a_run_root_without_the_runs_record_cannot_certify`,
  `test_a_record_from_another_cycle_is_not_this_runs_decision`,
  `test_a_run_root_whose_run_refused_before_adjudication_stays_inconclusive`,
  `test_a_clean_run_couples_too` (coupling is not a blanket refusal).
- The judge's own test fixtures now write the run record a run writes, with the
  decision computed through production's evaluator, so a fixture root can never
  carry a record a real run would not.

## E. What this amendment does not do

- It does not move a threshold (section B corrects a sign, not a magnitude).
- It does not add the target instrument's diagnostic metadata to a run output;
  the judge's T1–T10 still read a historical Gen-1 driver's shape.
- It does not start the Gen-2 run, and it does not change any campaign budget,
  protocol, recipe or protection constant.
- It does not make the judge a second promotion rule: it defers to the run's
  recorded decision and to production's evaluator in every branch.