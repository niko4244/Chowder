# The autonomous growth control plane

This document describes the layer that turns Chowder's single-generation
growth engine (Model N → N+1) into a loop that can advance generation after
generation. It states plainly what is built and what is not: an overstated
control plane is worse than a missing one, because a loop built on it can only
be trusted as far as its own documentation.

The trusted lower layer is unchanged: training binding, evaluation binding,
certification, settlement and the frozen judge. The control plane orchestrates
campaigns; it never replaces or loosens them.

## Status

| Component | State |
|---|---|
| Production parent-measurement path (`measure-parent`) | **delivered** (#191) |
| Offloaded evaluation path stability (both arms measurable) | **delivered** (#191) |
| Persistent growth state | **delivered** (`target_selection.GrowthState`) |
| Benchmark → skill attributed profiling | **delivered** (`build_skill_profile`) |
| Intervention classifier | **delivered** (`classify_intervention`) |
| NextTargetSelector + TargetProposal | **delivered** (`NextTargetSelector`) |
| Persistent FailureBank restore (`FailureBank.from_records`) | **delivered** |
| Automatic next-campaign builder + preregistration freeze | **delivered** (`NextCampaignBuilder`) |
| Loop/session budget, plateau rules, stop/review policy | **delivered** (`GrowthLoop`, `detect_plateau`, `LoopDecision`) |
| `GrowthLoop` outer controller, resume/recovery | **delivered** |
| Deterministic fake-compute simulator (six scenarios) | **delivered** (`simulator`) |
| CLI entrypoint (`growth loop status\|plan\|run\|resume`) | **delivered** |
| One production service used by CLI, TUI, loop and tests | **delivered** (`AutonomousGrowthService`) |
| Autonomous Growth workspace in the Chowder interface | **delivered** (`ChowderTUI` → Autonomous Growth) |
| One authoritative benchmark-attributed profile | **delivered** (`SkillProfile`; `CapabilityProfile` is a named derived view) |
| Explicit parent-evidence lineage object | **delivered** (`ParentEvidenceRef` in durable state, verified against disk) |
| Exact production recipe ids frozen before preregistration | **delivered** (`FinalCampaignManifest`) |
| Protected skills derived from the policy's protected benchmarks | **delivered** (`protected_skills_for_policy`) |
| One shared plan/run decision path | **delivered** (`GrowthLoop.plan_next`, used by CLI and TUI) |
| Task-specific training-data providers + corpus quality gate | **not built** |
| Bounded production candidate search (successive halving) | **not built** |
| Gen-0 trusted-ancestor arm | **measured** — `math500@2024-04` 0.0, `mgsm@2022-11` 0.0 (16 rows each) |
| Gen-1 parent arm under the *Gen-2* instrument | **measured** — target `generation-diagnostics@gen2-response-surface-v1` 0.5625, `math500@2024-04` 0.0, `mgsm@2022-11` 0.0 (16 rows each, `MEASURED_PARENT`, bound to the gen1 adapter digest) |
| Gen-2 readiness gate | **READY** (all pre-compute prerequisites pass) |
| Real Gen-2 candidate training run | **not run** |

Readiness being READY is a statement about prerequisites, not about outcome --
but it now also means every comparison a Gen-2 verdict is made of has a measured
arm on both sides, because both arms have since been measured (the target row for
the parent is 0.5625, so the target gate is decidable rather than vacuous). What
READY still does *not* say is how the protected and broad gates will behave: both
arms sit at 0.0 there (see below), so those gates compare two measured zeros.

Both arm measurements were made under the frozen 16-item protocol with the dense
weights off the card: the trusted-ancestor arm is 2 suites in 2818 s, the parent
arm is 3 suites in 3706 s. Both fit the declared 7200 s worker timeout, and the
parent arm only does so with the adapter placement fix recorded in PR #194 --
without it the same three suites could not finish inside the timeout at all.

## The pieces that exist

### Persistent state (`chowder.growth.target_selection.GrowthState`)

Append-only JSONL memory under one root, read back before the next generation
is planned:

```text
growth-state/
  failure-bank.jsonl            failures, with recurrence and repair state
  intervention-history.jsonl    what was tried, what it cost, what it did
  target-history.jsonl          every TargetProposal ever made
  capability-history.jsonl      every generation's measured profile
  stopping-state.json           the loop's own durable state
```

Nothing is overwritten. `failure_bank()` rebuilds a live `FailureBank` from
disk, so generation N's failures are present while generation N+1 is planned —
the behaviour the per-campaign `FailureBank()` never had.

### Benchmark-attributed profiling (`build_skill_profile`)

A skill's estimate is computed **only** from the benchmarks the registry
declares for that skill, weighted by each measurement's sample support and
provenance. Consequences:

* a skill nobody measured is `estimate=None`, `confidence=0`, `uncertainty=1`
  — **unknown, not zero**;
* two skills measured by different benchmarks get different estimates (the old
  flat mean gave one number to every skill);
* carried/grey rows contribute nothing: a reference is not a measurement;
* the aggregation method is recorded on every estimate.

### Intervention classification (`classify_intervention`)

Before a campaign exists, the weakness is classified:

```text
targeted_repair | sft | continued_pretrain | preference
data_acquisition | evaluation_needed | architecture_research
untrainable_with_current_path
```

Order matters and encodes the falsification the mission asks for: no evidence →
*measure*, don't train; a known structural limit → research; a calibration
defect → preference; missing external knowledge → data acquisition; a target
already tried to its ceiling → stop. `evaluation_needed`,
`architecture_research` and `untrainable_with_current_path` require human
review and never start a campaign on their own.

### Target selection (`NextTargetSelector`)

The selector's inputs are the parent's measured profile, durable memory and
policy. It **cannot** see a campaign's candidate results: `propose()` takes no
runs, and a test pins its signature so an undeclared campaign's scores can never
become a selection signal. Protected skills are excluded from candidacy
entirely — they are gates, never objectives.

The score is an explicit product of named factors, damped by named penalties
(see `TargetScoreFactors`), not "the lowest benchmark":

```text
weakness × confidence × importance × (1 + recurrence) × (1 + frontier gap)
        × trainability × (0.5 + novelty) × efficiency
  damped by (1 − 0.5·regression_risk)(1 − 0.5·repeat_penalty)(1 − 0.5·uncertainty)
```

Every `TargetProposal` records its `weakness_evidence`, its `factors`, and a
`why_not_other_targets` map that names unmeasured candidates as
`insufficient evidence (unmeasured, not zero)` before it names the scored
runners-up.

### The loop (`chowder.growth.growth_loop.GrowthLoop`)

One iteration: resolve the parent, load durable state, select a target, classify
the intervention, compose and freeze the campaign, prepare, check readiness, run
it, learn from its outcome, then decide to continue or stop. What the loop owns
and nothing else may own:

* **A finite stopping condition.** Every iteration reaches one of a closed set
  of terminal decisions; the envelope is re-checked before each generation is
  allowed to spend anything.
* **Budgets from measurement.** The remaining envelope is decremented by what the
  campaign's accounting artifact says was spent. A campaign that ran without
  reporting a measured cost ends the session with a durable decision -- it is
  not charged an estimate, and it is not treated as free.
* **No self-authorisation.** The policy supplies every ceiling, the protected
  set, the trusted ancestor, the declared execution throughput and the allowed
  treatments, and the loop can only read them. A target the policy does not allow
  becomes human review, not a campaign.
* **Promotion that cannot advance on nothing.** A promotion the loop cannot name
  an adapter for, or that reported no measurement of the model it promoted,
  stops the session instead of advancing the parent pointer.

`run(resume=True)` reads the durable record before it spends anything: a session
the record says already ended returns its stored decision and launches nothing,
and a recorded promotion restores both the declaration it ran under and the
adapter it promoted, so the next generation trains from the artifact the run
produced rather than from the one the process happened to be holding.

The simulator runs the *real* loop against a deterministic table of measured
outcomes, because a controller that decides whether to spend must be exercised
on rejections, plateaus and exhausted envelopes -- none of which a GPU will
produce on request. Six scenarios are pinned: two promotions then a plateau, a
rejection then a promotion, a protected regression that does not advance the
parent, an envelope too small to launch anything, a structural target that goes
to review unspent, and a repeatedly failing intervention that walks targets
instead of retrying one.

```text
chowder growth loop status <state-root>
chowder growth loop plan   <policy.json> --parent <manifest> [--profile <p.json>|--parent-evidence <run-root>]
chowder growth loop run    <policy.json> --parent <manifest> [--profile|--parent-evidence] [--max-generations N]
chowder growth loop resume <policy.json> --parent <manifest> [--state-root <dir>]
```

The CLI takes no injectable train/evaluate seam, so a run that cannot
legitimately proceed refuses having spent nothing. Its exit code is 0 only for
the terminal decisions that mean the loop stopped *correctly* (finished
improving, exhausted its envelope, stopped repeating itself); refusals exit 1.

## How the interface, the CLI and the loop stay one system

`chowder.growth.service.AutonomousGrowthService` is the only place that composes
production objects. `inspect()`, `plan_next()`, `prepare_next()`, `status()`,
`history()`, `request_stop()` and `start()` are thin orchestrations of
`NextCampaignBuilder`, `campaign_prepare`, `check_campaign_readiness`,
`GrowthLoop` and `GrowthState` -- no decision is made in the service, the CLI or
the interface.

```
AutonomousGrowthService            (composition only)
        Δ                                      Δ
   chowder growth loop ...            AutonomousGrowthScreen (ChowderTUI)
        Δ                                      Δ
        └──────────── GrowthLoop.plan_next / _one_generation ────────┘
```

Three rules are enforced rather than documented:

* **`plan` and `run` ask the same question.** `GrowthLoop.plan_next()` runs the
  same profile, target, treatment, protected-exclusion and envelope gates the run
  applies, and returns a `TargetProposal` or a terminal `LoopDecision` without
  writing a frozen campaign or spending compute. The CLI `plan` and the interface
  both go through it; neither calls `selector.propose()` directly.
* **the profile must belong to the parent.** A profile measured on another
  generation is `PROFILE_GENERATION_MISMATCH` before any composition, on `run`,
  on `plan` and on `resume` -- resuming after a promotion means the *promoted*
  run's profile, not the one the session started with.
* **protected evidence is not optimizable.** The protected skill set the selector
  excludes is derived from the policy's `protected_benchmarks` through the
  benchmark registry (`protected_skills_for_policy`), so a skill known only
  through a protected benchmark can never be proposed as a target. A skill with
  both targetable and protected evidence keeps its targetable evidence and is
  reported in `regression_risks`.

## The pieces that do not exist yet

Still manual or missing for the autonomous case:

* task-specific training-data providers (the corpus is still the protocol-repair
  template), and the corpus quality gate that refuses a poor self-generated set;
* bounded production candidate search (successive halving) -- an existing library
  implementation that is not yet wired, not something to reimplement;
* a measured Gen-2 outcome: readiness is READY and no real Gen-2 candidate
  training has been run, so nothing here is evidence about Gen-2's verdict.

Until the first two exist, Chowder can compose, freeze, plan and monitor the next
campaign without a human, but it cannot yet *choose better data* for it. It must
not claim otherwise.

## What the measured arms imply

Both arms are measured now, and on the protected and broad slices they read the
same number:

```
                          math500@2024-04   mgsm@2022-11   generation-diagnostics@gen2-response-surface-v1
Gen-0 base (ancestor)          0.000000       0.000000        (not measured under this protocol)
Gen-1 parent (gen1 adapter)    0.000000       0.000000        0.562500
```

The untouched dense base emits no EOS, so every generation runs the full
512-token cap and no answer is extracted; the gen1 parent, measured for the
first time under *this* instrument, terminates (16/16 EOS-terminated, mean 31
tokens) and scores 0.5625 on the target -- the protocol repair, visible on the
instrument that is supposed to see it. On math500 and mgsm it also reads 0.0.

Two consequences, stated plainly because a reader should not have to infer them:

* the **target** gate is meaningful and non-vacuous: 0.5625 is a real bar that a
  Gen-2 candidate has to clear under its own measurement;
* the **protected/broad** regression gates are decided between two measured
  zeros, so `candidate - parent >= 0` cannot fail. The gate is not broken -- it
  compares like-for-like measured arms, which is the property that matters --
  but it is weak here, and "no measured regression on math500" should be read as
  "neither model answers these items under this protocol" rather than as
  evidence of retained capability.

## Invariants the control plane must never break

1. Candidate results cannot alter frozen thresholds.
2. Protected evaluation content never enters training material.
3. Parent evidence cannot masquerade as candidate evidence.
4. Carried evidence cannot masquerade as a fresh measurement.
5. Every promoted candidate is bound to exact artifact bytes.
6. All compute after campaign start is durably accounted.
7. Promotion cannot occur unless production certification passes.
8. Production `PROMOTED` and the frozen judge may not disagree for the same
   immutable run root.
9. New generations cannot overwrite historical evidence.
10. Missing evidence remains UNKNOWN/UNMEASURED, never zero.
11. A target cannot be selected from protected-set performance of competing
    candidates.
12. Loop policy cannot enlarge its own global budget.
13. The control plane cannot modify its safety/integrity gates in response to
    candidate results.
14. Architecture/model-family changes require explicit review.
15. There must always be a finite stopping condition.

The delivered pieces honour 1–15 by construction. Invariant 15 now has a named
owner (`LoopDecision` plus the loop's generation, non-promotion, plateau and
envelope limits) and a test that pins each of its terminal states, including the
one that proves a session can never be unbounded.
