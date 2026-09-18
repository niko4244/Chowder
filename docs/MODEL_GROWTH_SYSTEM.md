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
   cycle refuses.

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
```

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
