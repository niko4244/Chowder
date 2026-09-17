# Growth `TrainingFn` → production `chowder train`: binding qualification plan (2026-09-16)

**Status: plan, frozen as written. The acceptance criteria below were not
changed after the fact.** Implementation and the S0-S7 ladder results:
[`GROWTH_TRAINFN_BINDING_QUALIFICATION_2026-09-16.md`](GROWTH_TRAINFN_BINDING_QUALIFICATION_2026-09-16.md).
No Generation-1 campaign has started.

## Why this is a separate task

PR #166 merged the Autonomous Model Growth System onto `main`. Its cycle is
honest about its own seam:

```python
# src/chowder/growth/cycle.py
TrainingFn = Callable[[TrainingRecipe, Sequence[CurriculumItem]], Mapping[str, Any]]
```

> "the caller supplies a `TrainingFn` that executes one recipe (typically through
> `chowder train` / `run_project`) and returns the artifact/evaluation evidence.
> The orchestrator is trainer-agnostic by design and never invokes a trainer
> itself."

That seam has only ever been executed against **tiny local fakes**. A fake
trainer cannot demonstrate project validation, budget enforcement, registry
lifecycle, contamination binding, worker source identity, or independent
evaluation. Until the seam is bound to the real trainer **and qualified**, the
growth system is infrastructure, not authority.

Corollary, stated as a rule: **the merged framework is not permission to start
autonomous real training.** No Generation-1 campaign may launch before this
qualification passes.

## Existing seam honesty (must not be "fixed" by assumption)

`docs/ROADMAP.md` records that wiring `run_successive_halving()` /
`prioritize_candidates()` into `project_runner.py` is open and **"must not be
inferred from the library implementation or its integration tests"**. There is no
`search` project config, and `search.variants` does not exist. The binding below
must therefore **not** invent a search-controller config. Pattern for now:

- the growth recipe planner emits a valid project config patch;
- a supplied `TrainingFn` executes it;
- promotion follows the predeclared growth rule.

Wiring successive halving is its own tested production-integration change.

## Required properties (all nine are acceptance criteria)

1. **No alternate trainer.** The binding invokes the production entry points
   (`chowder project-validate` then `chowder train`), or the in-process
   equivalents those commands use. No reimplementation, no side-channel trainer.
2. **Existing project validation.** The recipe's config patch is validated by the
   real validator before any compute; refusal surfaces as a terminal experiment.
3. **Existing registries.** Runs and results land in `RunRegistry` through the
   normal lifecycle — no parallel bookkeeping. The registry audit for
   result-carrying non-terminal rows must stay clean.
4. **Existing budget enforcement.** GPU-hour ceilings, decomposed sub-budgets and
   the measured preflight govern the run; the growth envelope does not bypass them.
5. **Existing contamination/data binding.** Training material is drawn only from
   sources the data registry admits and the contamination firewall clears. A
   refusal is a refusal, not a warning.
6. **Worker source identity preserved.** The run records `chowder_source_identity()`
   and it is verified on the worker side, exactly as the router path does.
7. **Independent evaluation preserved.** Candidate scoring happens in an
   independent process from training, against a frozen, pinned evaluation corpus
   and setting.
8. **Hard promotion gate preserved.** `evaluate_promotion` remains the only
   promotion authority: predeclared multi-objective rule; frontier context
   recorded, never deciding regressions or contamination.
9. **Failed/refused experiments remain terminal durable evidence.** A failed run
   is never deleted, never retried in place, and never leaves a non-terminal row
   carrying a result.

## Proof sequence (tiny and real — not a campaign)

Each stage is a bounded, real-subprocess proof with its own tests. Do not skip
ahead; a later stage's PASS does not substitute for an earlier stage's.

| Stage | Proof | Must be real |
|---|---|---|
| S0 | Unit contract: a fake `TrainingFn` still drives the cycle end to end | already green (existing growth tests) |
| S1 | **tiny real model + tiny real corpus**, minimal bounded steps | real weights, real tokenizer, real forward/backward |
| S2 | the same run goes through the **real subprocess** worker path | real process boundary, real worker result artifact |
| S3 | the run is bound to the **real registry** | `runs.db` rows reach terminal status; no stranded results |
| S4 | an **actual adapter/payload artifact** is produced and reloaded | artifact digest recorded; reload verified key-for-key |
| S5 | **independent evaluation** scores the candidate in a separate process | distinct process, pinned eval corpus |
| S6 | **refusal proofs**: contaminated source, over-budget recipe, invalid config | each refuses *before* compute and leaves terminal evidence |

Only after S1–S6 pass may a real Model N → N+1 campaign be considered — and even
then, at the tiniest scale first.

## Preconditions before starting implementation

- `main` is green (all six protected checks) at the commit the branch is cut from.
- The Rung-4 question is closed or explicitly parked, so the GPU and the router
  evidence are not contested by two workstreams at once.
- No concurrent large CUDA job; no local-model workload (Ollama / `llama-server`)
  resident while a qualification run executes.
- The branch is cut from `origin/main` and the prerequisite list above is copied
  into its preregistration as *acceptance criteria*, not narrative.

## Deliberately out of scope

- Starting a Generation-1 campaign.
- Implementing the router-rung instrumentation gap (separate issue; see
  `../Chowder-Protected/runs/2026-09-16-router-healing-rung4/`).
- Restoring a retired hard active-parameter gate. The standing objective is the
  lowest active parameter count actually achieved that retains meaningful
  capability under the frozen accounting convention.
- Wiring successive halving into `project_runner.py`.
