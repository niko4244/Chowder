# Model Growth System

The Autonomous Model Growth System sits above Chowder's training/evaluation
infrastructure. It turns the platform from *something that can safely run
training experiments* into *something that can systematically measure a model,
decide what it should learn next, source verified material for it, train
competing candidates, verify genuine improvement, preserve prior capability,
and iterate toward stronger generations*.

Everything here is evidence-driven: no decision is "autonomous" because an
LLM was asked what to do next. Every phase answers **"why did Chowder do
this?"** from measurements, explicit policies, uncertainty, budgets, and
provenance.

## The Model N → Model N+1 lifecycle

```
MODEL N
  → comprehensive evaluation        (evals/ adapters, tiers)
  → capability profile              (capability.py: skills + raw scores)
  → failure analysis                (failure_bank.py, failure_taxonomy.py)
  → skill-gap map                   (curriculum.py prioritize)
  → curriculum plan                 (curriculum.py plan, with decision traces)
  → data discovery / verification   (discovery.py, data_registry.py, contamination.py)
  → recipe search                   (recipe_planner.py, synthetic.py)
  → multiple candidates             (cycle.py train_candidates)
  → targeted + protected + frontier eval   (eval_tiers.py)
  → promotion / rejection           (promotion.py: predeclared, multi-objective)
  → MODEL N+1                       (lineage.py GenerationLedger)
  → repeat
```

`growth/cycle.py` orchestrates the phases and records machine-readable
provenance for each one. It never overrides hard constraints (contamination,
budgets, protected regressions) and never invents evidence.

## Module map

| Module | Responsibility |
| --- | --- |
| `capability.py` | Closed skill list (57 skills, 16 categories); profiles keep raw benchmark scores visible beside normalized skill estimates |
| `benchmark_registry.py` + `catalog.py` | 40 version-pinned benchmarks across the mandated categories, with status/lifecycle/split/adapter rules |
| `contamination.py` | The non-negotiable firewall: protected/development/training sets, MinHash + canary + substring + exact/normalized detection |
| `data_registry.py` | GOLD/SILVER/BRONZE/QUARANTINE trust classes with verification floors; licensing is required for registration |
| `discovery.py` | DISCOVER → INSPECT → REGISTER; candidates enter QUARANTINE, never trainable by default |
| `failure_bank.py`, `failure_taxonomy.py` | Durable, classified failure evidence feeding curriculum priority |
| `difficulty.py` | Evidence-based difficulty calibration into bands |
| `curriculum.py` | Weakness → prioritized plan with weighted decision traces and mixture (TARGET/PRESERVE/GENERAL/REPLAY/STRETCH) |
| `synthetic.py` | Generator → independent critic → objective verifier → dedup → contamination check pipeline |
| `recipe_planner.py` | Bounded candidate recipes within measured hardware envelopes |
| `eval_tiers.py` | Tier 0 smoke → Tier 4 frontier, bound by decision stakes and budget |
| `statistics.py` | Welch's t-test, Wilson CIs, Cohen's d, pass@k; a 1-question swing is not proof |
| `promotion.py` | The single predeclared promotion rule (see below) |
| `lineage.py` | Append-only generation ledger + regression-probe memory (anti-forgetting) |
| `frontier_reference.py` | Five reference levels, protocol-honest comparability, frozen snapshots |
| `research_kb.py` | Recorded research results screened for applicability to our 9B/local scale |
| `cycle.py` | The lifecycle orchestrator with per-phase provenance |

Companion package `chowder/evals/` normalizes Inspect AI, lm-eval-harness,
benchmark-native agent suites, and Chowder-native diagnostics into one
`EvalReport` schema with honest non-measurements.

## Promotion (never one number)

A candidate is **PROMOTED** only when *all* of the predeclared checks pass —
and since the integrity pass (2026-09-17), every check consumes only
evidence that can prove its provenance and its cost:

1. target improvement, statistically significant and ≥ the declared minimum,
   measured on the candidate (`measurement_origin=MEASURED_THIS_GENERATION`);
2. no protected regression beyond tolerance, from candidate-measured rows
   only — a carried parent row reads inconclusive, never "not-regressed";
3. no material broad-battery deterioration, from candidate-measured rows
   only;
4. no calibration/hallucination regression;
5. reliability (survives repeated samples where practical);
6. evidence integrity — no contamination (`TAINTED` otherwise);
7. resource envelope respected **by settlement as well as admission**:
   projected cost admits the run, actual measured cost (all recipes,
   evaluations, failed attempts — via the cycle compute accounting artifact)
   must stay under the frozen device/wall ceilings after execution or the
   cycle refuses. A ceiling is only settled against the unit it was declared
   in: wall and project ceilings are settled from the trainer's measurements,
   and a device ceiling is settled only when the run actually separated
   device time (`ComputeCost.device_measured`) — an unmeasured device figure
   fails closed with `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` rather than reading
   as "device free", so a device ceiling that a run cannot measure stays an
   admission constraint on the projected plan and the evidence says so
   (`ceiling_enforcement`). See
   `docs/growth/DEVICE_CEILING_FAIL_CLOSED_ADDENDUM_2026-09-18.md`.

A real target repair with incomplete full-promotion evidence is
**INCONCLUSIVE** with `target_repair_validated=true` — deliberately not a
new verdict class: PROMOTED must mean "complete evidence, all gates passed".
Historical verdicts are corrected only by append-only adjudication revisions
(`GenerationLedger.append_adjudication_revision`); original records never
change.

Frontier scores never decide promotion. A candidate below GPT-class can be a
legitimate generation if it beats its parent without regressions.

## Anti-forgetting

Convincingly repaired failure classes become protected regression probes
(`lineage.RegressionMemory`). Every future generation must keep passing them;
growth is cumulative, not one capability traded for another.

## The five frontier levels

Every benchmark can be viewed against: Generation-0 floor → parent →
comparable peer (7–9B class) → open-weight frontier → absolute frontier.
Comparisons are only emitted when benchmark version, tool setting, and
reasoning setting align; otherwise the report says
`NOT DIRECTLY COMPARABLE`. Snapshots are frozen at generation time
(`SnapshotStore`, never rewritten) so "Chowder improved" can be separated
from "the frontier moved faster."

## CLI

```text
chowder eval catalog          # the pinned benchmark catalog
chowder eval scoreboard FILE  # render an eval report honestly
chowder data audit            # registered sources + trainability
chowder data register ...     # the one mutating path; enters QUARANTINE
chowder data contamination    # firewall checks against protected material
chowder growth profile FILE   # capability profile
chowder growth curriculum FILE [--protected IDs]   # plan with decision traces
chowder growth status FILE    # cycle outcome summary
chowder growth campaign validate FILE   # is this a declaration this code can execute?
chowder growth campaign plan FILE       # the curriculum items and recipe ids a run will honor
chowder growth campaign run FILE        # execute the declared campaign end to end
chowder growth campaign settle FILE     # re-settle a campaign from its accounting artifact
```

## Gen-2 certification boundary

The frozen gen2 judge (`docs/gen2/judge_gen2.py`, frozen with
`docs/quals/GEN2_PREREG_2026-09-17.md` and
`GEN2_PREREG_AMENDMENT1_2026-09-18.md`) owns the frozen policy — benchmark
set, thresholds, branch rules, verdict composition — and no second
implementation of anything production already decides. It hashes artifacts
with `training_binding.directory_digest`, settles resources with
`campaign.settle_campaign`, reads provenance from `evals.result`, interprets
contamination with `metric_binding.MetricBinder`, compares with
`statistics.compare`, and parses arms with `evals.result.EvalReport`. Three
provenance-bound arms are judged: the gen2 candidate
(`MEASURED_THIS_GENERATION`), the gen1 parent and the trusted ancestor gen0
(`MEASURED_PARENT`), each measured on the same frozen 16-prompt instrument
and the same 16-item protected mini-slices. Promotion requires the target
rule *and* no protected regression against the parent *and* the ancestor, so
a candidate cannot promote by matching an unresolved parent. Missing
evidence is INCONCLUSIVE; contamination is TAINTED; a hard failure is
REJECTED. The pass that hardened this boundary before any Gen-2 compute,
with its adversarial re-attacks and its verification counts, is
`docs/growth/GEN2_CERTIFICATION_HARDENING_REPORT_2026-09-18.md`.

The run root *is* the judge's input. `campaign_runner` materialises the judged
evidence set from the run's own measurements: the three provenance-bound arms
(`candidate_eval_report_path`, `parent_eval_report_path`, and the new
`baseline_eval_report_path` naming the gen0 arm branch protection is judged
against), the winner's identity in `chosen_candidate.json`, and the
contamination manifest the firewall bound. `chowder growth campaign run`
followed by `python docs/gen2/judge_gen2.py <state_root>` therefore reads one
directory. Provenance is copied verbatim -- a row's `measurement_origin` is the
evaluator's declaration, never the runner's election -- an input the manifest
does not declare produces no file rather than a placeholder, and
`tests/test_growth_certification_coupling.py` proves both directions: the
frozen judge certifies the root a CLI run just wrote, and absent or tampered
evidence refuses.

## Campaign manifests: preregistration as configuration

A campaign manifest (`campaign.CampaignManifest`) is the preregistration a
run executes: parent identity and digest, promotion sets, recipe set, per
recipe and campaign ceilings in named units, stopping rules, promotion
policy version, and every input path the run reads. `campaign_runner`
turns it into a cycle by *composing* the pieces that already own each
decision — `GrowthCycle` for sequencing/curriculum/selection/promotion,
`SubprocessTrainingFn` for execution and projected-cost admission,
`MetricBinder` for declared metric semantics, `settle_cost`/
`settle_campaign` for post-run settlement — rather than adding a second
engine. No phase invents an input: a path the manifest did not declare is
a named refusal, and the campaign's resource gate is authoritative over
the lineage record, so a run that overran its own frozen envelope cannot
record a promoted generation. Every field the schema accepts is listed in
`campaign_runner.FIELD_ENFORCEMENT` with the behavior it drives, and
`assert_every_field_enforced` fails if the schema and that table diverge:
a declared key nobody acts on is refused rather than silently ignored.

## Status

Generation 0 is frozen: the dense Qwen3.8-9B abliterated parent
(content digest `59e767aa…7555f`) was evaluated evaluation-only at
`docs/gen0/GEN0_EVAL_RESULT_2026-09-17.md` (freeze digest `5c8b18ab…`,
immutable snapshot `gen0-frontier`, ledger record `gen0`). The cycle's
TrainingFn is bound to the real trainer and its promotion inputs to
declared metric semantics (#167), with the binding qualified in
`docs/GROWTH_TRAINFN_BINDING_QUALIFICATION_2026-09-16.md` and
`docs/GROWTH_PROMOTION_BINDING_QUALIFICATION_2026-09-16.md`. The first
real Model N → N+1 cycle has now **executed and closed: gen1 was recorded
PROMOTED, and after the integrity audit its effective verdict is
INCONCLUSIVE with target_repair_validated=true**
(`docs/growth/GEN1_READJUDICATION_ADDENDUM_2026-09-17.md`; original result
`docs/quals/GEN1_RESULT_2026-09-17.md`; prereg
`docs/quals/GEN1_PREREG_2026-09-17.md` under amendments 1–3 — both recipes
trained 30/30 steps through the production binding, and the frozen rule
adjudicated the candidate from independent evaluation). The proven,
repeatable path for every future cycle: prereg → measured preflight →
production train → independent evaluation → mechanical promotion → ledger.

Known limitations are listed in `docs/DATA_POLICY.md` and
`docs/BENCHMARK_CONTAMINATION_POLICY.md`.
