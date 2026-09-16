# Growth promotion binding — qualification record (2026-09-16)

Branch `feat/growth-trainfn-binding` (on top of the TrainingFn binding, `b8f8347`,
which is on top of `main` = `1a4d35f`). Worktree
`F:\chowder-worktrees\growth-trainfn`. Files: `git status` at the time of
writing lists exactly `benchmark_registry.py`, `catalog.py`, `cycle.py`,
`metric_binding.py` (new), `tests/test_growth_metric_binding.py` (new),
`tests/test_growth_benchmark_registry.py` — nothing else.

## What closed

`chowder.growth.metric_binding.MetricBinder` maps a run's **measured** metrics
onto the `BenchmarkResult`s `evaluate_promotion` consumes, using **declared**
directions and 0..1 scales from the benchmark registry. The conversion is
never computed at bind time: `catalog.METRIC_SEMANTICS` declares each metric's
polarity and scale once, with a written rationale, and the binder only applies
that declaration — refusing everything it cannot apply by name.

The cycle gained `GrowthCycle.decide_promotion_from_runs(binder, ...)`: it
binds candidate runs under the candidate's version pin and parent runs under
the parent's, then hands the cycle-owned promotion sets, declared tolerances,
and device-hour ceiling to `binder.promotion_input`, which returns a
`PromotionAssembly` (input + verdict + both binding reports).

```python
assembly = cycle.decide_promotion_from_runs(
    binder, candidate_runs=candidate_runs, parent_runs=parent_runs,
    device_gpu_hours=measured_device_hours,
)
assembly.decision.verdict   # PROMOTED / REJECTED / INCONCLUSIVE / ...
```

A full Model N → N+1 attempt is now adjudicable end to end:
`cycle.train_candidates(...)` (real `chowder train` subprocesses, qualified in
`GROWTH_TRAINFN_BINDING_QUALIFICATION_2026-09-16.md`) →
`decide_promotion_from_runs` → the predeclared promotion rule.

## The declared scales

`METRIC_SEMANTICS` declares six metrics as 0..1 proportions with `identity`
normalization (`accuracy`, `pass@1`, `resolved_rate`, `success_rate`,
`strict_accuracy`, `win_rate_vs_human`) and one metric —
`speedup_at_correctness` — as **explicitly unscaleable**: it is a ratio with no
zero floor and no one ceiling, and anchoring it would require pinning a
reference implementation and a timing protocol, which the catalog refuses to
invent. `semantics_for()` refuses an undeclared metric rather than defaulting
its polarity. Every catalog entry carries a direction and a normalization
decision; a row may not redeclare metric semantics locally.

## The refusals (each one mutation-proved)

| # | Refusal | Caught by |
|---|---|---|
| 1 | Run measured on a different generation than the pin | `test_a_run_from_the_wrong_generation_is_refused` |
| 2 | Benchmark absent from the registry | `test_a_benchmark_absent_from_the_registry_is_refused` |
| 3 | Unsupported run (honest non-measurement) — never bound as zero | `test_an_unsupported_run_is_refused_and_never_bound_as_zero` |
| 4 | Supported run with no score — absent evidence is not zero | `test_a_supported_run_with_no_score_is_refused_as_absent_not_zero` |
| 5 | Run's metric name disagrees with the entry's primary metric | `test_a_metric_name_that_disagrees_with_the_registry_is_refused` |
| 6 | Entry declares no 0..1 scale (`speedup_at_correctness`) | `test_an_entry_without_a_declared_scale_is_refused` |
| 7 | Anchors read from the run's own metadata — ignored, registry only | `test_anchors_are_read_from_the_registry_never_from_the_run` |
| 8 | Raw value bound without applying the declared scale | `test_a_measured_run_binds_onto_the_declared_scale` |
| 9 | Per-sample scores left in raw units | `test_per_sample_scores_are_normalized_onto_the_same_declared_scale` |
| 10 | Contamination read from the run instead of the firewall manifest | `test_contamination_status_comes_from_the_manifest_not_the_run` |
| 11 | Manifest `benchmarks` section ignored by `from_manifest` | `test_a_binder_can_be_built_from_a_firewall_manifest` |
| 12 | One benchmark bound twice in a generation | `test_binding_the_same_benchmark_twice_refuses_the_second` |
| 13 | Promotion set names an unregistered benchmark → `PromotionBindingError` | `test_a_promotion_set_naming_an_unregistered_benchmark_is_refused` |
| 14 | Cycle's declared tolerance dropped before promotion | `test_the_cycle_passes_its_declared_tolerances_into_promotion` |
| 15 | Cycle's device-hour ceiling dropped before promotion | `test_the_cycle_passes_its_device_ceiling_into_promotion` |
| 16 | Parent runs bound under the candidate's version pin | `test_the_cycle_adjudicates_a_measured_attempt_end_to_end` |
| 17 | Catalog row redeclares metric semantics | `test_a_catalog_row_cannot_redeclare_its_metric_semantics` |
| 18 | Undeclared metric silently defaulted | `test_a_metric_with_no_declared_semantics_is_refused_not_defaulted` |
| 19 | Unscaled metric quietly declares a scale | `test_every_catalog_entry_declares_a_direction_and_a_normalization_decision` |

Probe: `Chowder-Protected/runs/2026-09-16-growth-promotion-binding/mutation_probe.py`,
**19/19 caught**, tree restored byte-identical afterwards (verified: no `MUTANT`
markers, focused suite green post-probe).

Two things are deliberately **not** refusals: an aggregate that disagrees with
the mean of its per-sample scores (recorded as `sample_mean` on the
`BoundMeasurement` — visible evidence, since adapters legitimately aggregate
differently), and `UNKNOWN` contamination (recorded on the result, where
promotion treats it as inconclusive, not clean).

## Gates

| Gate | Result |
|---|---|
| Focused suite (`test_growth_metric_binding.py` + `test_growth_benchmark_registry.py`) | **56 passed** |
| Mutation probe | **19/19 caught** |
| Full suite | **2124 passed, 77 skipped** (one environmental failure, see below) |
| `ruff check` on all touched files | clean |
| `python -m build` (sdist + wheel) | `chowder_ai-0.3.0` built |

### The environmental failure, root-caused

The first full run failed 8 tests across four modules with
`sqlite3.OperationalError: database or disk is full`, and a rerun failed one
disk-sensitive test (`test_dependency_preflight.py::...memory_preflight...`)
because its refusal message showed the disk preflight firing (`0.96 GB free`)
before the memory preflight could. Root cause: the system `C:` drive was at
100% (923 MB free) from ~29 GB of stale `dm-guided-*`/`bis-branch-*`/`entry-chain-*`
experiment directories in `%LOCALAPPDATA%\Temp` belonging to other live
sessions' tooling. Only directories untouched for >6 hours were deleted
(~14.7 GB freed; actively-touched directories left alone). Both failures
reproduce green afterwards; no code change was involved and none was made.

## Scope notes

- `CycleConfig` has no `reliability_benchmarks` field; `promotion_input`
  supports one and defaults it empty. A cycle that needs a reliability set
  adds the field — the binder side is ready.
- One automatic `baseline` row per registry (from the TrainingFn record)
  still applies: score multiple candidates through per-candidate registries.
- The end-to-end adjudication demonstrated in tests uses measured runs
  binding onto declared identity scales; a real generation's first campaign
  still begins with the Generation-0 evaluation freeze, per the campaign doc.
