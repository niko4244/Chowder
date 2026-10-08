# Gate-label audit — 2026-10-08: every judge row name against the quantity its code computes

**Status: audit record. Corrections are proposed, not applied.** The only row
this audit changes is the one amendment 17 already changed (T5); every other
finding is a proposal, because renaming a frozen gate row after the run's
thresholds were frozen is the repo owner's call, exactly as amending the judge
is (`JUDGE_AMENDMENT_PROPOSAL_T23.md`, `JUDGE_AMENDMENT_PROPOSAL_T24.md`).

## Why this audit exists

Amendment 17 found one row whose label named a quantity its code did not
compute: T5 said "answer correctness" and counted presence
(`docs/quals/GEN2_PREREG_AMENDMENT17_2026-10-08.md`). One instance is a defect;
the question this audit answers is whether it was the only one. The audit is
mechanical: enumerate every gating row `judge_gen2.py` can emit, read the code
that decides it, and compare the *name* an operator reads against the *quantity*
the code measures.

## Method

- The row set is taken from the judge itself, by running it on the clean
  run-root fixture and reading the table — not from documentation, so a row that
  only exists in code cannot be missed. All 24 thresholds plus the two `INFO`
  rows appear there and are covered below. (The two `INFO` rows gate nothing and
  are excluded from the findings.)
- Each row's code was then read in `docs/gen2/judge_gen2.py` and its quantity
  recorded in the operator's own words.
- A finding is raised when the name and the quantity differ in a way a reader
  could act on. Severity is stated per finding: **disclosure-only** (the label
  can mislead a reader but cannot change a pass/fail) or **gate-affecting** (the
  label describes a decision the code does not make).

## The rows

| row | what the row name says | what the code measures | audit |
| --- | --- | --- | --- |
| T1 | candidate instrument provenance | a duplicate-free instrument run carrying `MEASURED_THIS_GENERATION`, plus its `measurement_origin` | name is narrower than the quantity (F6) |
| T2 | answer-duplication target | share of completions whose post-think answer also appears in the reasoning, vs the parent, by the frozen paired-or-absolute rule | name matches; detail polarity is inverted (F2) |
| T3 | template-echo target | share of continuations opening with the prompt tail + `assistant`, vs the parent, same rule | name matches; detail polarity is inverted (F2) |
| T4 | constrained-prompt compliance == 8/8 | on the 8 declared prompts: answer surface non-empty, ≤ 40 chars, ≤ 1 newline | name matches the prereg's "format compliance" |
| T5 | ~~answer correctness~~ → **answer presence** | expected string present anywhere on the answer surface | **corrected by amendment 17** (F1) |
| T6 | EOS termination >= 0.9 | the run's recorded `eos_termination_rate` | matches |
| T7 | cap-hit < 0.1 | `max_token_cap_rate` | matches |
| T8 | obvious loops <= 0 | `obvious_loop_count` | matches |
| T9 | distinct-trigram >= 0.9 | `distinct_trigram_ratio_mean` — a **mean over prompts** | name drops "mean" (F5) |
| T10 | unclosed think <= 0.25 | `unclosed_think_rate` | matches |
| T11 | candidate {slice} measured + protocol-exact / {slice} candidate-vs-parent regression <= 0.0625 / no undeclared protected row substitutes | slice presence, protocol exactness, digest, and the score delta against the parent arm | matches |
| T12 | contamination CLEAN on the frozen evaluated set / training-source contamination CLEAN | the pinned manifest's statuses for the required benchmarks and for every training source | matches |
| T13 | **actual** cost settled within the declared ceilings | production's `settle_campaign` over `ComputeCost.from_dict(totals.incremental)`; the device dimension is settled from a number the run may have *not measured* | "actual" is too broad for the device dimension (F4) |
| T14 | all recipes accounted | the accounting artifact's recipe ids against the declared set, exactly (missing and extra both fail) | matches |
| T15 | candidate artifact identity (digest recomputed) | recorded digest vs recomputed digest of the selected artifact | matches |
| T16 | trusted-ancestor protection (vs gen0) | per-slice deltas against the ancestor arm | matches |
| T17 | immediate-parent (gen1) protected regression | per-slice deltas against the parent arm | matches |
| T18 | judged contamination evidence is the pinned artifact | the pin exists, lives in the run root, and is byte-identical to the copy judged | matches |
| T19 | {arm} report is labelled for the generation it claims / {arm} evidence names the declared artifact | the report's generation label and the digest of the bytes each arm measured | matches |
| T20 | the campaign declares the policy this judge enforces | the declared protection tuple vs the judge's frozen one | matches |
| T21 | the run's promotion decision **agrees with the declared profile** | only `RETENTION_`-prefixed recorded breaches against the run's verdict; a refusal on any other declared rule (the wall envelope) passes, by design | name is broader than the quantity (F3) |
| T22 | the judge recomputes the declared profile as the run did | recorded breach codes vs the codes recomputed through production's `evaluate_retention` | matches |
| T23 | the run's recorded settlement is the settlement of the artifact it pinned | pinned digest vs recomputed, and the recorded settlement vs the artifact's | matches |
| T24 | answer-readout disagreements <= 0 | items presence accepts that every declared production readout refuses | matches (new; amendment 17) |

## Findings

### F1 — T5 said "answer correctness" and counted presence

**Fixed by amendment 17** (`docs/quals/GEN2_PREREG_AMENDMENT17_2026-10-08.md`).
Recorded here because it is the audit's originating case and the standard the
others are held to: the name said correctness, the code counted the expected
string's presence anywhere on the answer surface, and production's own readout
scored the same completions 0.0. The correction renames the row to the prereg's
own quantity (`answer presence >= 15/16`), names the readout and its disclosure
row in the detail, and moves no threshold.

### F2 — the target rows' detail is inverted for lower-is-better rates (disclosure-only)

T2 and T3 are rates where **lower is better**, and their details print
production's `compare(...).verdict` verbatim
(`f"paired={comparison.verdict} (delta {comparison.delta:+.4f}, ...)"`).
`compare` is a higher-is-better comparison, and `_target_gate` maps it for the
decision (`comparison.verdict == "regressed"` means *improved* here), but the
detail string is not mapped, so the word a reader sees is the opposite of the
row's own direction. Measured on the judge's own fixtures:

```
T2  answer-duplication target  PASS  paired improvement — paired=regressed (delta -0.6875, min_effect 0.25); ...
T2  answer-duplication target  FAIL  paired=improved (delta +0.6875, min_effect 0.25); candidate rate 0.688 vs parent 0.000; ...
```

The second row is a candidate that made duplication *worse* — 0.688 against a
0.000 parent — and its detail says `paired=improved`. A reader auditing a
refusal is told the opposite of what happened.

**Proposed correction (needs a named prereg amendment).** Have `_target_gate`
name the direction in the row's own terms rather than production's:
`paired={'better' if paired_improved else 'not better'}` — or keep the raw
verdict but label its polarity, `paired=regressed (lower rate is better, so this
is the improvement the rule wants)`. Nothing about the decision changes; this is
a detail string in two rows.

### F3 — T21's name says "the declared profile", the code reads only retention codes (disclosure-only)

T21's requirement string is `"the run's promotion decision agrees with the
declared profile"`, and what the code collects is
`reasons starting with RETENTION_REASON_PREFIX`. A run refused on its own
declared **wall envelope** — a rule the campaign does declare, which is exactly
what the fixture run refused on — is a PASS, and its detail says so:

```
T21  the run's promotion decision agrees with the declared profile  PASS  the run refused this candidate without a declared-gate breach: actual wall GPU-h 0.5120 exceeds preregistered wall ceiling 0.2000
```

Amendment 16 states this PASS branch is deliberate (the judge audits the
predeclared rule itself under T11/T16/T17 and may disagree with it), so the
decision is not in question; the *name* is. A reader can take "agrees with the
declared profile" as agreement about the whole declaration, which the row never
checked.

**Proposed correction.** Name the profile the row actually reads:
`"the run's decision agrees on the declared retention profile"`, keeping T22's
agreement row and the detail that lists the run's own reasons. The wall envelope
keeps its own gate (T13) and its own pin (T23).

### F4 — T13's "actual cost" is one word too broad while `device_measured=false` (disclosure-only)

T13's name is `"actual cost settled within the declared ceilings"`; the detail
it prints is honest — `device 0.4000 (unmeasured) / wall 1.1000 against the
campaign envelope` — and amendment 1 declares that with
`device_time_measured=false` the device ceilings are **admission** (projected
plan) constraints while wall is the post-run settlement. So for the device
dimension the settled number is not an actual measurement, and the row name says
it is.

**Proposed correction.** Either widen the name to what is settled
(`"cost settled within the declared ceilings (wall measured; device admission
while unmeasured)"`) or shorten it to `"cost settles within the declared
ceilings"` and leave the dimension-by-dimension truth in the detail, where it
already is.

### F5 — T9's name drops the "mean" (disclosure-only)

T9 reads `distinct_trigram_ratio_mean` — a mean over the instrument's prompts —
and the name says `distinct-trigram >= 0.9`, which reads as a per-prompt
property. The value in the detail (`measured 0.97; parent 0.973`) is the mean.

**Proposed correction.** `"distinct-trigram ratio mean >= 0.9"`, which is also
the name the constant's own key uses.

### F6 — T1's name covers provenance, the code also enforces row identity (disclosure-only)

T1 reports three different facts under "candidate instrument provenance":
duplicate prompt identities in the candidate arm
(`candidate arm duplicates rows for [...]`), the absence of a single instrument
run carrying `MEASURED_THIS_GENERATION`, and — when all that holds — the
`measurement_origin`. Only the last is provenance in the narrow sense; a reader
seeing "provenance PASS" cannot tell that identity was also checked.

**Proposed correction.** `"candidate instrument provenance and row identity"`
(the row's own detail already names which check it is reporting).

## What the audit did not find

- **No label can certify what the code refuses.** Every finding above is
  disclosure-only: the pass/fail of all 24 rows is decided by the quantity the
  code computes, and no finding changes a decision. F1 was the exception in kind,
  not in outcome (T5's threshold never moved and its decision never changed).
- **No threshold moved.** This audit changed no constant, no comparison and no
  gate, and it is the reason amendment 17's change is limited to T5's name, its
  detail, and the new T24 row.
- **No missing row.** The table above is every gating row the judge can emit, and
  every one of them decides on the clean-run fixture (none is unreachable).

## Governance

Each proposal above is a rename or a detail string, so each needs the same
treatment as any frozen-judge change: a named prereg amendment, and the
no-visible-candidate-results check that governs amendments 15-17 (`evidence/`
holds zero `candidate_evaluation.json` and zero `chosen_candidate.json`, and the
declared Gen-2 state root is not present in this checkout). Nothing here is
applied, and nothing here is urgent in the way F1 was: none of these names can
certify a candidate the code refuses.
