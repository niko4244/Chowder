# Evaluation Matrix

The growth system measures a model the way modern laboratories do: many
benchmarks, many categories, version-pinned, with honest statuses. We do not
build one giant homemade benchmark; we integrate established public
evaluations and record everything needed to reproduce — or to refuse — a
comparison.

## Registry structure

`benchmark_registry.py` enforces the rules; `catalog.py` holds the seed data
(40 entries). Every entry pins:

- identity: `benchmark_id` + `version` (`latest` is refused at construction);
- capability mapping: `category`, `subcategory`, and skills from the closed
  `capability.ALL_SKILLS` list (57 skills, 16 categories);
- provenance: dataset source, implementation source, publisher, license,
  release date;
- evaluation posture: scorer type, primary metric, random/human baselines,
  contamination risk;
- availability: one of `RUNNABLE_PUBLIC`, `PUBLIC_SCORE_ONLY`, `PRIVATE`,
  `INTERNAL_REFERENCE_ONLY`, `UNSUPPORTED_BY_CURRENT_MODALITY`;
- lifecycle: `ACTIVE_FRONTIER`, `ACTIVE_DIAGNOSTIC`, `LEGACY`, `SATURATED`,
  `RETIRED`;
- split policy: `protected` (default) / `dev` / `none`;
- adapter: every `RUNNABLE_PUBLIC` entry must name the harness that runs it.

## Categories covered

reasoning · math · coding · agentic · tools · research · knowledge ·
instruction · context · multilingual · professional · science · health ·
self_improvement · multimodal · safety.

Multimodal entries (e.g. OSWorld) are present so a text-only model shows
`NOT_APPLICABLE_MODALITY` — never a zero — and cannot sit in tiers 0–2.

## Score-only and private benchmarks

Benchmarks the frontier labs publish but we cannot run (private test sets,
unreleased environments) are registered as `PUBLIC_SCORE_ONLY` or `PRIVATE`.
The dashboard can display their frontier reference numbers without pretending
Chowder ran them; `may_score_against()` distinguishes display from execution.

## Training use is not the default

`training_use_permitted` is false by default. A runnable benchmark with a
protected split may only permit training use when it explicitly names a
non-protected `training_split` (validated at construction). The contamination
firewall independently enforces the protected set; see
`docs/BENCHMARK_CONTAMINATION_POLICY.md`.

## Versioning discipline

Scores are always recorded as `benchmark_id@version`. A new benchmark version
is a new row, never a silent replacement, so historical Chowder generations
remain fairly comparable. Saturated benchmarks are marked `SATURATED` and
retired from optimization (`optimizable()` returns false) while their scores
remain visible in historical reports.

## Rotation summary

| Lifecycle | Meaning |
| --- | --- |
| ACTIVE_FRONTIER | Separates strong models; legitimate optimization target |
| ACTIVE_DIAGNOSTIC | Still informative; lighter weight |
| LEGACY | Kept for historical comparison only |
| SATURATED | No longer separates strong models; do not optimize against it |
| RETIRED | Removed from active batteries; scores preserved |
