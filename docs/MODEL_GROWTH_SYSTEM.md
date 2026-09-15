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

A candidate is **PROMOTED** only when *all* of the predeclared checks pass:

1. target improvement, statistically significant and ≥ the declared minimum;
2. no protected regression beyond tolerance;
3. no material broad-battery deterioration;
4. no calibration/hallucination regression;
5. reliability (survives repeated samples where practical);
6. evidence integrity — no contamination (`TAINTED` otherwise);
7. resource envelope respected.

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

Infrastructure complete and tested (62 dedicated tests; full suite 2056
passed / 77 skipped). The first real campaign awaits a frozen Generation-0
evaluation of the 9B parent — see `docs/FIRST_MODEL_GROWTH_CAMPAIGN.md`.

Known limitations are listed in `docs/DATA_POLICY.md` and
`docs/BENCHMARK_CONTAMINATION_POLICY.md`.
