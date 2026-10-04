# First 0.5 campaign: design (not run)

Status: **design, preregistered-style, NOT launched** (Phase 14). No training
runs from this document. It exists so that launching is a decision about
*this text*, not about whatever the operator feels on the day — the
preregistration rule that makes a post-hoc threshold change detectable.

Every cost number below is measured, not estimated: it comes from the
Kaggle acceptance runs recorded in `docs/KAGGLE_PROVIDER_ACCEPTANCE.md`
(scientist lane) — screening-lane A/B at 300 steps measured **12.781 s wall
on 2×T4 = 0.0071 device-GPU-h**, and the replay-decay falsification run
(Run 4) measured **14.4 s = 0.008026 device-GPU-h** for a full 3-seed A/B.
Local steady-state reference: 28.9 s/step = 0.0080 GPU-h/step.

## 1. Objective

Discover, under the 0.5 rules, whether any *qualified* intervention family
measurably improves the current model's target capability without breaching
the retention profile — and produce evidence either way. A campaign whose
every candidate falsifies is a success of the architecture, not of the
model: it buys the next generation a smaller, sharper search space.

## 2. Cost basis and budget (all measured)

| Quantity | Value | Source |
| --- | --- | --- |
| screening A/B (300 steps, 3 seeds, 2×T4) | 0.008026 device-GPU-h | Run 4, settled |
| per-step at screening scale | ≈ 2.7e-5 device-GPU-h | Run 4 ÷ 300 steps |
| operator weekly Kaggle quota | 30 device-GPU-h | Kaggle `quota` |

**Declaration** (rounds budgets in steps; continuation pricing per Phase 2):

- rounds = 3, initial_max_steps = 300, step_multiplier = 2.0
  (round budgets 300 → 600 → 1200), survival_fraction = 0.5, min_survivors = 1.
- Round deltas: 300 / 300 / 600 steps → 0.008 / 0.008 / 0.016 device-GPU-h
  per attempt.
- 4 candidates, worst case (everyone survives every round):
  4×0.008 + 2×0.008 + 1×0.016 = **0.064 device-GPU-h** search spend.
- Survivor evaluation (tier 2, 3-seed paired, per surviving lineage):
  ≈ 0.024 device-GPU-h.
- Promotion evaluation (tier 3: winner + parent on the frozen suites):
  ≈ 0.05 device-GPU-h.
- **Campaign total ≈ 0.14 device-GPU-h ≈ 0.5 % of the weekly quota.**
- Declared ceilings (the numbers the runner enforces, per
  `CandidateSearchDeclaration`): device 0.30 GPU-h, wall 0.20 GPU-h for the
  search; campaign ceilings 1.0 / 1.0 — five-fold margin over the
  projection, one-thirtieth of the quota.

## 3. Candidate set (4, from the families registry)

Maturity policy: `experimental_interventions: [sft families]`,
`research_campaign: []` — the RESEARCH families (conditional-ffn,
hybrid-lm, low-rank-vocab, ptq, retrieval, speculative) are **out** this
campaign: they are declared with fold-artifact basis but no in-scope
evidence, and the maturity gate refuses them without a research policy.

1. `sft.replay-balanced-a` — curriculum replay rebalance at replay_rate 0.10
   (PRODUCTION family, unchanged parameters — the control-adjacent arm).
2. `sft.replay-balanced-b` — replay_rate 0.05, decay OFF (the parameter
   *narrowing* of Run 4's falsified decay hypothesis — evidence-driven by
   the failure taxonomy: the mechanism was refuted *with decay at 0.25*,
   the family was not).
3. `sft.curriculum-order-hard-first` — curriculum item ordering (PRODUCTION
   family, new parameter axis).
4. `sft.adapter-continuation` — LoRA rank continuation schedule (PRODUCTION
   family, new parameter axis).

**Budget-ladder verdict, computed not asserted:** the evidence store holds
history for exactly one family (training.sft-curriculum, Run 4 FAILED).
Three candidates' families have no in-scope record, so `ladder_stage`
returns **DETERMINISTIC** with the reason "no evidence record" — the
campaign runs the declared order. The ladder escalates only after this
campaign's own evidence lands.

## 4. Evaluation tiers (the wall)

| Benchmark | Tier | Read by |
| --- | --- | --- |
| dev-loss, training stability, artifact validity | search-evidence | screen |
| capability probes (3-seed paired) | survivor-evidence | survivor round |
| protected retained-capabilities suite | promotion-evidence (reserved name) | promotion only |
| gsm8k-heldout | promotion-evidence (reserved name) | promotion only |
| contamination screen | promotion-evidence (reserved name) | gate |

`assert_search_isolation` refuses the campaign at plan time if any tier-3
name appears in a search surface. Unclassified names default to promotion
evidence — fail closed.

## 5. Retention profile (preregistered; freezes at launch)

| Dimension | Kind | Value | Benchmark (tier 3) |
| --- | --- | --- | --- |
| reasoning accuracy | max-regression | 0.00 | protected retained suite |
| tool validity | absolute-floor | 0.85 | protected retained suite |
| termination health | absolute-floor | 0.95 | protected retained suite |
| contamination signal | absolute-floor | 0.00 | contamination screen |

The mandate's arithmetic — target 0.60→0.78 with reasoning 0.71→0.51 — is
a refusal, not a promotion with a footnote. An unmeasured dimension is a
violation, not compliance. (These values are the operator's to freeze; the
*kinds* and the fail-closed semantics are the architecture's and do not
move.)

## 6. Decision rules (preregistered)

- Advancement: training-side screen only (`candidate_succeeded` +
  artifact), declaration order — the ladder is DETERMINISTIC this campaign.
- Continuation: every round after the first resumes the survivor's own
  checkpoint (Phase 2 semantics); a reported `not-a-resume` ends the
  lineage and keeps the spend.
- Winner: `cycle.select_candidate` over the final round, then retention
  profile evaluation against the parent, then paired-delta significance
  (bootstrap CI excludes 0) on the survivor evidence.
- Failure: every non-promoted attempt is classified by
  `growth.failure_taxonomy`; MECHANISM_FALSIFIED writes FAILED evidence and
  closes the exact parameter repeat; EVAL_NOISE replicates with more
  evidence and writes INCONCLUSIVE; infra/budget classes write nothing.
- No threshold may change after the first attempt's evidence exists. A
  change to this document after launch invalidates the campaign.

## 7. What would falsify this design

- Measured attempt costs exceeding 2× the projected per-round spend before
  any result — the cost model (§2) is wrong and the campaign pauses.
- A checkpoint that cannot be resumed from in round 1 for a *technical*
  (not lineage) reason — the Phase 2 continuation contract is broken.
- A promotion evaluation that finds the retention measurements incomplete —
  the tier separation failed and the gate must refuse.

## 8. Explicitly out of scope

Running it. This document is the design; the execution requires the
Kaggle backend bar (`docs/KAGGLE_BACKEND_REQUIREMENTS.md`) to clear R1–R8,
the retention/eval-isolation gates wired into the campaign runner, and an
operator go.
