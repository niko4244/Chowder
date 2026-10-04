# 0.5 architecture preparation — implementation report

Branch `feature/autonomous-05-architecture` (base: origin/feat/candidate-search
@ 2aef4e6). Five implementation commits; every claim below is backed by an
executable test in this branch. Test commands at the end.

## Commit 1 — b997c5c: checkpoint-resuming progressive halving (Phase 2)

| File | Change | Invariant enforced |
| --- | --- | --- |
| `src/chowder/growth/candidate_search.py` | `round_recipe(..., resume_from=)`; `plan_search` prices rounds ≥ 1 at the step DELTA; `run_search` rewritten: survivors carry checkpoint-per-candidate, missing checkpoint ⇒ `lineage_stops`, executor-reported `not-a-resume` ⇒ lineage disqualified with spend kept; `SearchProgress` resumes an interrupted search only onto a validated rounds-prefix; `SearchRun.candidate_cumulative` | a survivor earns a larger budget only by continuing its own lineage; a restart can never win progressive allocation; an interrupted search resumes as the same search |
| `src/chowder/growth/recipe_planner.py` | `to_config_patch` emits `router_healing.resume_from` / `backend.resume_from_checkpoint` | the continuation reaches the backend in the namespace its reader actually reads |
| `src/chowder/successive_halving.py` | exported public `latest_checkpoint_dir` | one checkpoint-resolution code path for both stacks |
| `tests/test_growth_candidate_search.py` (+16 tests), `tests/test_growth_training_binding.py`, `tests/test_growth_campaign_runner.py` | E2E: round-1 project config carries `resume_from == round-0 checkpoint`; `_RecordingRunner` writes a real `trainer/checkpoint-4/optimizer.pt` | continuation is observable in the composed project, not just in memory |

## Commit 2 — 4b797d1: search-axis contract (Phase 3)

| File | Change | Invariant |
| --- | --- | --- |
| `src/chowder/growth/recipe_planner.py` | `CONSUMED_RECIPE_FIELDS` corrected against real readers (peft: lr/scheduler/warmup/max_steps/seq_len/lora_rank/lora_alpha/target_modules/batch_size/gradient_accumulation/resume_from_checkpoint; router: + batch_size/scheduler/warmup_steps); `UNSAFE_SEARCH_FIELDS` (max_steps halving-owned, resume, curriculum ids, dataset manifest, recipe id, projections, notes); `PLANNED_UNMAPPED_FIELDS` named; `classify_recipe_fields()`; `assert_search_axes_consumed` refuses unsafe too; mapper emits batch_size/gradient_accumulation/target_modules | a declared search axis is either consumed by a real backend reader or refused — never silently inert |
| `tests/test_growth_search_axis_contract.py` (10 tests) | classification completeness; mapper-keys ⊆ reader-keys via **source regex census**; consumed ⊆ mapper-emitted (with the peft name map); **live** `TransformersPeftRunSpec.from_resolved_config` parse (skipif no transformers); changed-field→changed-spec; notes-only→same-spec | the contract is checked against the readers' actual source, not against a copy of it |

## Commit 3 — cc497c0: families, evidence, hypotheses (Phases 4/5/6/13)

| File | Change | Invariant |
| --- | --- | --- |
| `src/chowder/growth/interventions.py` | `Maturity` (PRODUCTION/QUALIFIED_EXPERIMENTAL/RESEARCH/REJECTED; REJECTED requires basis), `InterventionFamily` (target failure class, typed parameter ranges, valid architectures, evidence required, compute class, risks, eval dimensions), 9-family registry, `assert_maturity_permits` (policy keys experimental_interventions / research_campaign / reopen:{family: hypothesis}), `register_family` (test-extensible), `families_for_campaign`, `family_from_id` fail-closed | a family trains only under a maturity its evidence has earned; reopening a REJECTED family requires naming the hypothesis that reopens it |
| `src/chowder/growth/evidence.py` | `EvidenceState` (8 states), `EvidenceRecord` (every scope field required: model_family, checkpoint_identity, architecture, eval_suite, parameters, software_runtime, hardware_class, sample_size, evidence_quality, measured_effect, date, source), `EvidenceStore` append-only JSONL **with sha256 hash chain** (`audit()`), duplicate record_id refused, `records_for_family`, `prior_for_family` → multiplier (incompatible→0/excluded, FAILED→0.35^min(n,3), NOT_REPRODUCED→0.5, PROMISING→1.25, PRODUCTION_QUALIFIED→1.5, no history→1.0) | evidence is scoped, tamper-evident, and produces priors that starve falsified directions without ever auto-promoting |
| `src/chowder/growth/hypotheses.py` | `Observation` (weakness vs threshold, direction-aware), `Hypothesis` (suspected cause, predicted improvement AND risks, required measurements, falsification criterion, prior), `generate_hypotheses` (weakness-only, maturity gate, `_addresses` metric-class matching, excluded prior skipped), `hypothesis_candidate_brief` (refuses family mismatch) | a candidate begins as a falsifiable experiment, not a recipe with a wish attached |
| `tests/test_growth_interventions_evidence_hypotheses.py` (19 tests) | tamper-evidence, duplicate refusal, prior arithmetic, maturity refusals, register/cleanup | |

## Commit 4 — 798cdb7 + 54705f7: the evaluation wall, retention, paired deltas, budget ladder (Phases 7/8/9/10)

| File | Change | Invariant |
| --- | --- | --- |
| `src/chowder/growth/eval_isolation.py` | 3 evidence tiers; unclassified defaults to PROMOTION (fail closed); reserved protected-suite names refuse downward classification **at declaration time and read time**; `assert_search_isolation` refuses any tier-3 name in a search-readable surface or a selection policy | the search cannot optimize against the promotion gate, structurally, before compute |
| `src/chowder/growth/retention.py` | `RetentionConstraint` (max-regression / absolute-floor; must name a tier-3 benchmark), `RetentionProfile` (non-empty, unique dimensions), `evaluate_retention` — unmeasured candidate dimension ⇒ violation; missing parent ⇒ violation for max-regression | a capability trade (0.60→0.78 with 0.71→0.51) is a refusal; a gate that cannot be measured was not passed |
| `src/chowder/growth/paired_deltas.py` | `pair_outcomes` refuses differing task SETS (not intersection) and per-task protocol-key mismatches (decoding/seed/evaluator/evaluator_version/hardware_class/protocol_sha256); wins/losses/ties ±1e-12; `bootstrap_delta_interval` deterministic (seed 20261003) percentile CI | a delta across a protocol change is not causal evidence; an effect must be separable from noise |
| `src/chowder/growth/budget_ladder.py` | `ladder_stage` (policy is the cap, evidence is the climb, reasons recorded); `entry_order` (prior-desc, tie by id; prior 0 refuses — exclusion is the maturity gate's); `ucb1_order` (explore-first, deterministic); `survivor_orderer`; `allocate` reorders a plan **keeping per-round sets, totals and schedule verbatim** | adaptivity reorders inside the declared envelope; ceilings and budgets are never the ladder's to change; the adaptive reward is training-side only |
| `src/chowder/growth/candidate_search.py` | optional `order_survivors` hook in `run_search` (identity by default; refuses a hook that drops/adds candidates) | the seam cannot become a side door |
| `tests/test_growth_eval_tiers_retention_paired.py` (24), `tests/test_growth_budget_ladder.py` (24 incl. run_search integration) | |

## Commit 5 — 49a73a2 + 07bc00a: failure classification + docs + mutation checks (Phases 11/12/14/15)

| File | Change | Invariant |
| --- | --- | --- |
| `src/chowder/growth/attempt_failure.py` | 7 failure classes, explicit precedence (settlement/non-resume → no record; failed training → infra; incomplete measurement → inconclusive; paired CI < 0 → clean falsification; unreadable → operator); class → evidence-state mapping (infra/budget record NOTHING — recording one would be fabrication; parameter misses record neutral, not failed); `record_refuses_disproven_repeat` refuses an exact failed repeat in scope | every failure teaches the next generation the right amount — no more (fabricated FAILED rows), no less (unclassified retries) |
| `src/chowder/growth/failure_taxonomy.py` | **restored verbatim from origin/main** — the attempt classifier had taken this filename (owned since #166) and broke `test_growth_decisions` collection via `failure_bank`'s import | a module's name belongs to its lineage; collisions are found by import, not by review |
| `docs/KAGGLE_BACKEND_REQUIREMENTS.md` | R1–R8 production bar for PR #200 (source binding + SHA verify, artifact manifests + hashes, settlement through compute_cost, resume vocabulary, classified failures, pinned env + recorded provenance, one-attempt-per-kernel, quota as ceiling), each with status-in-#200 | remote spend is held to the same scientific standard as local spend |
| `docs/CAMPAIGN_DESIGN_05.md` | first 0.5 campaign, NOT run: 4 candidates, 3 rounds (300→600→1200 steps), ~0.14 device-GPU-h total from **measured** costs (Run 4: 0.008026 device-GPU-h per 3-seed A/B), preregistered tiers/retention/decision rules, ladder verdict computed (DETERMINISTIC — 3 of 4 families have no history) | launching is a decision about the document, so a post-hoc threshold change is detectable |
| `tests/test_growth_mutation_checks.py` (6 tests) | the six mandated sabotages, each run against its real enforcement point and caught | the guards cannot rot silently |

## The one next action, wired: settlement refusals stop advancing, promotion consults the gates

The audit's single most important open change, implemented on top of the
commit stack above:

- **One owner of the settlement-refusal vocabulary.**
  `compute_cost.settlement_refusal(evidence)` reads exactly the two shapes
  production writes (the `budget_settlement` verdict and the `refused_by`
  stamp) and returns the
  machine-readable identifier, or `None`. `attempt_failure.classify_failure`
  derives its settlement branch from the same predicate, so the classifier
  and the runner can never disagree about what was refused — the
  classifier's old `settle_refusal`-only read could never fire on a real
  production record.
- **Settlement-refused attempts stop advancing.** `run_search` ends the
  lineage of an attempt that settled over budget (the spend stays in the
  accounting; the stop lands in `lineage_stops`), and both advance surfaces —
  `candidate_search.advanced` and `cycle.select_candidate` — refuse such
  rows. `candidate_succeeded` is set *before* settlement runs, so "it
  trained" was never "it may win".
- **The promotion path consults the preregistered gates.**
  `GrowthCycle._apply_promotion_gates` runs in both `decide_promotion` and
  `decide_promotion_from_runs`: a declared `RetentionProfile` is evaluated
  fail-closed (unmeasured constraint = violation; verdict downgraded to
  REJECTED with `RETENTION_REGRESSION` / `RETENTION_FLOOR` /
  `RETENTION_UNMEASURED` reasons), and a declared `EvalTierPolicy` refuses a
  constraint measured on search-readable evidence outright — that is
  campaign wiring, not a measured outcome.
- **The wiring caught the predicted second instance of the same disease.**
  With the gate in place, the campaign fixtures' "clean" runner (a fixed
  0.05 wall against a ~0.006 projection) refused at settlement and the
  dry-run matrix's clean-promotion scenario stopped promoting: it had only
  ever promoted because selection ignored settlement refusals — exactly the
  #204 hole. The fixtures now report a settleable cost (`ATTEMPT_WALL_GPU_HOURS
  = 0.006`) with the deliberate overruns made explicit (`OVERRUN_WALL_GPU_HOURS`),
  and the three scenarios that relied on the hole assert the honest refusal
  path. The REFUSED-at-selection record also now carries the `stopping`
  phase, which it previously dropped.

New tests: `tests/test_growth_runner_gates.py` (20 — the predicate's
vocabulary, the advance/selection/search refusals, the classifier agreement,
and the retention/tier gates on both promotion paths).

## The loader seam, closed: the gates bind from the manifest

The promotion gates above were reachable from programmatically constructed
cycles only; the manifest → CycleConfig loader now closes that gap:

- **Two new optional manifest sections.** `retention_profile` (profile id +
  a non-empty list of constraints, each exactly
  `{dimension, kind, value, benchmark}`) parses into the domain's
  `RetentionProfile`; `eval_tier_policy` (`{classification:
  {benchmark@version: tier}}`) parses into `EvalTierPolicy`. Both live in
  `campaign.py` beside the other declared sections and are passed through
  `_build_cycle` into `CycleConfig` — a declared gate reaches the promotion
  path as the same object a programmatic construction would pass.
- **Fail-closed at load.** Unknown section fields, an empty profile, a
  constraint with a missing/unknown-kind/non-finite-value field, an
  unpinned benchmark, a constraint naming a benchmark the declared
  measurement sets never cover (an unmeasurable gate is a guaranteed
  rejection, not a constraint), an unknown tier name, a reserved-name
  demotion, and a tier policy that demotes a declared constraint into the
  search's view (the pair is checked at load, not first at promotion) all
  refuse as `CampaignManifestError` from `from_mapping`. A manifest
  declaring neither loads with both unset: every pre-existing manifest runs
  unchanged.
- **Schema/table discipline holds.** Both fields are named in
  `FIELD_ENFORCEMENT`, so `assert_every_field_enforced` still proves no
  declared field is decorative.

New tests: `tests/test_growth_manifest_promotion_gates.py` (8) — parse-valid
binding through the real fixture and builder with an observed gate downgrade
(the same regression that promotes without the section is REJECTED with it),
twelve malformed-declaration refusals, an unknown-top-level-key control, and
the unchanged-default path.

### Ownership consolidation (design review, behavior-preserving)

- The `RETENTION_REGRESSION` / `RETENTION_FLOOR` / `RETENTION_UNMEASURED`
  machine codes moved from `cycle._retention_reason` into `retention.py` as
  `RetentionViolation.code` (with `reason` = code + detail); the cycle
  consumes `violation.reason`, so a reason-code change lands in one place.
  `test_growth_runner_gates.py` pins the reason to the domain's own `code`.
- The phantom `settle_refusal` arm of `settlement_refusal` was deleted: it
  was a vocabulary for a field nothing in `src` writes. The predicate reads
  exactly the two shapes production writes -- the `budget_settlement`
  verdict and the `refused_by` stamp (a later commit, 6e67427, owned the
  correction of this section's earlier "three shapes" claim, which rested
  on a phantom `settlement_failed` marker). The tests that exercised the
  phantom shape were re-pointed to real ones (the classifier test now feeds
  the binding's actual refusal record), keeping coverage, not deleting it.
- The constraint `kind` validation has one owner —
  `RetentionConstraint.__post_init__` — and the manifest loader wraps the
  domain error with source context (`CampaignManifestError` at load, never a
  raw domain exception), pinned by an ownership test asserting both the
  domain message and the loader's `<source>` context.

### The parent side of a declared constraint demands earned provenance

The retention gate's two sides now agree on what counts as evidence. The
parent side of `_retention_values` trusted any row whose
`measurement_origin` was not `UNMEASURED`, so a `CARRIED_REFERENCE` row —
a quotation from history — could silently anchor a declared constraint
with a number nothing measured. The contract is decided by the codebase's
own vocabulary (`evals/result.py`: carried rows "never satisfy a
regression or improvement gate"; `MODEL_GROWTH_SYSTEM.md`: "a carried
parent row reads inconclusive, never not-regressed") and by the provenance
owner (`metric_binding` refuses carried rows on the parent role).

- `BenchmarkResult.parent_measured` is the wall, owned beside
  `gate_eligible` in `promotion.py`: a baseline is earned evidence
  (`MEASURED_PARENT`, or `MEASURED_THIS_GENERATION` from the parent's own
  cycle) or it is not a baseline. `cycle._retention_values` applies it, so
  an unearned non-`UNMEASURED` row reads as unmeasured and the gate
  refuses with `RETENTION_UNMEASURED` — promotion refused, never a silent
  compare against a borrowed number. On the production binder path this is
  defense in depth, pinned as a no-op: the binder already refuses carried
  parent rows, and the parent arm only writes earned `MEASURED_PARENT` rows.
- The single owner of violation shapes surfaced a latent mislabel the new
  pin test caught: the parent-side missing-measurement violation carried
  the candidate's score as `measured`, so `RetentionViolation.code` named
  the shape `RETENTION_REGRESSION` when nothing had been measured.
  `evaluate_retention` now records the parent's absence as NaN (the shape
  `code` already defines as `RETENTION_UNMEASURED`) and keeps the
  candidate's value in the detail.

Tests: `tests/test_growth_runner_gates.py` — a measured candidate against
a carried baseline refuses `RETENTION_UNMEASURED` through `decide_promotion`,
and the binder-path no-op is pinned directly.

## Verification

```
cd F:/chowder-worktrees/autonomy05
python -m pytest tests/test_growth_candidate_search.py \
  tests/test_growth_search_axis_contract.py \
  tests/test_growth_interventions_evidence_hypotheses.py \
  tests/test_growth_eval_tiers_retention_paired.py \
  tests/test_growth_budget_ladder.py \
  tests/test_growth_attempt_failure.py \
  tests/test_growth_mutation_checks.py \
  tests/test_growth_campaign_runner.py tests/test_growth_campaign_readiness.py \
  tests/test_growth_training_binding.py tests/test_growth_campaign_prepare.py \
  tests/test_growth_campaign.py tests/test_growth_decisions.py \
  tests/test_growth_next_campaign.py tests/test_growth_budget_settlement.py \
  tests/test_growth_candidate_selection.py tests/test_growth_metric_binding.py \
  tests/test_growth_target_selection.py \
  tests/test_growth_runner_gates.py \
  tests/test_growth_manifest_promotion_gates.py -q
python -m ruff check src/chowder/growth/
```

At commit time: all of the above green (261-growth-regression + 24 + 24 + 16
+ 6 + 20 + 8 additions; the campaign-runner, certification-coupling,
evaluation-binding and dry-run-matrix suites re-pointed to the honest
settlement semantics), ruff clean; full-suite run recorded in the PR body.
