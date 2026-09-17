# Growth `TrainingFn` → production `chowder train`: binding qualification (2026-09-16)

Judged against the frozen plan
[`GROWTH_TRAINFN_BINDING_PLAN_2026-09-16.md`](GROWTH_TRAINFN_BINDING_PLAN_2026-09-16.md),
whose nine acceptance properties and S0–S6 proof ladder were written before this
implementation. Nothing in that plan was relaxed; the criteria are restated here as
acceptance criteria with the evidence that meets them.

**Verdict: the seam is bound and qualified at the ladder's tiny scale.** A
`GrowthCycle` handed this binding trains through real `chowder project-validate` and
`chowder train` processes, and every gate that governs those runs governs this one.

## What was closed

`GrowthCycle` takes a `TrainingFn` and never invokes a trainer itself. That seam had
only ever been executed against local fakes, which cannot demonstrate project
validation, budget enforcement, registry lifecycle, contamination binding, worker
source identity or independent evaluation. `chowder.growth.training_binding` is the
binding:

```python
from chowder.growth.training_binding import GrowthEnvelope, SubprocessTrainingFn
from chowder.growth.cycle import CycleConfig, GrowthCycle

binding = SubprocessTrainingFn(
    run_root=run_root,
    project_template=project_payload,   # a real project, naming {corpus}
    envelope=GrowthEnvelope(
        device_gpu_hours_ceiling=0.05,
        wall_gpu_hours_ceiling=0.20,
        project_gpu_hour_budget=0.05,
    ),
    registry=admitted_data_registry,
    firewall=contamination_firewall,
    sources={item.item_id: source_id for item in items},
    material={item.item_id: lines for item in items},
)
cycle = GrowthCycle(..., train_fn=binding)
results = cycle.train_candidates(items, recipes)     # real training
```

Each call reserves a never-reused `<run_root>/attempt-NN`, and returns — and writes to
`<attempt>/training-evidence.json` — the run's own evidence: process return codes and
captured output, the composed project file, the materialized corpus and its hash, the
registry rows and the stranded-result audit, the evaluation outcomes, the worker
artifacts, the artifact directory digest, the verified source identity, and either a
named refusal or a named failure reason.

## The nine properties

| # | Property | How the binding satisfies it | Test |
|---|---|---|---|
| 1 | No alternate trainer | runs `python -m chowder.cli project-validate` then `train`; no training code in the module | `test_s1_s2_s3_...`, `test_s7_...` |
| 2 | Existing project validation | real validator approves the composed project before compute; a refusal starts no trainer | `test_a_validation_refusal_never_starts_training` |
| 3 | Existing registries | rows written by the run itself into an ordinary `RunRegistry`; the binding only reads and audits | `test_s1_s2_s3_...`, `test_s7_...` |
| 4 | Existing budget enforcement | project ceiling is authoritative; an envelope-looser **or** stricter-than-project mismatch refuses rather than silently rewriting either | `test_the_binding_refuses_a_recipe_over_the_growth_envelope`, `test_the_binding_refuses_a_project_budget_looser_than_the_envelope` |
| 5 | Existing contamination/data binding | source must be registry-admitted and firewall-CLEAN; a refusal is a refusal | `test_the_binding_refuses_a_source_the_data_registry_has_not_admitted`, `test_the_binding_refuses_protected_material_offered_as_training_text`, `test_s6_...` |
| 6 | Worker source identity | child environment built by `worker_env`; every recorded identity matched parent-side; missing identity is a failure | `test_the_real_runner_child_imports_this_checkout_not_a_competing_one`, `test_evidence_records_the_chowder_source_identity_...`, `test_a_missing_source_identity_is_a_failure_not_a_pass` |
| 7 | Independent evaluation | scoring legs are separate processes, recorded by artifact kind and registry outcome | `test_s1_s2_s3_...` (worker kinds + outcome count) |
| 8 | Hard promotion gate | the binding never calls `evaluate_promotion`; it reports `succeeded`, metrics and the row status factually | `test_s7_...` |
| 9 | Terminal durable evidence | attempts never reused, refusals/failures recorded, stranded sibling rows surfaced | `test_attempt_directories_are_never_reused`, `test_a_refused_recipe_leaves_durable_terminal_evidence`, `test_the_binding_reports_rather_than_launders_a_stranded_registry_result` |

## Ladder results

| Stage | Proof | Result |
|---|---|---|
| S0 | a fake `TrainingFn` drives the cycle end to end | **PASS** — but see the correction below: it had **no coverage at all** before this change |
| S1 | tiny real model + tiny real corpus, bounded steps | **PASS** — real Qwen3 MoE (E=4, k=2, hidden 16, 2 layers), real tokenizer, 4 steps |
| S2 | the same run through the real subprocess boundary | **PASS** — both commands as real child processes, both exit 0 |
| S3 | bound to the real registry | **PASS** — `baseline=passed`, `growth-ladder-a01=rejected`, audit empty |
| S4 | a real artifact produced and reloaded | **PASS** — payload directory digested; `load_router_payload` reloads it key-for-key from the recorded reference |
| S5 | independent evaluation in a separate process | **PASS** — 2 evaluation outcomes and 3 worker artifacts: baseline eval, candidate eval, training |
| S6 | refusal proofs before compute | **PASS** — invalid config (real validator), over-budget recipe (envelope), protected material (firewall); none started a trainer |
| S7 | **the growth cycle itself drives real training** | **PASS** — one cycle, two recipes, two attempts, two registries, both SUCCEEDED |

**Correction to the plan's S0 row.** The plan recorded S0 as "already green (existing
growth tests)". It was not: `GrowthCycle` had no test and no caller anywhere in the
repository, so the orchestrator the binding plugs into was itself unverified. S0 is now
a real test, including the promotion decision and the lineage record.

## The real run, verbatim

```
$ python -m chowder.cli project-validate .../attempt-01/project.json      # exit 0
$ python -m chowder.cli train            .../attempt-01/project.json      # exit 0

[prepare] Loaded project 'growth-binding'
[baseline] Automatic baseline established: dead_experts=1.0000, experts_per_token=2.0000, holdout_loss=4.1752
[train] Starting growth-ladder-a01 with 0.1 reserved GPU-hours
[evaluate] Evaluation complete: dead_experts=0.0000, experts_per_token=2.0000, holdout_loss=4.1754
[rejected] Candidate completed but did not pass the promotion gate
{
  "artifact_ref": ".../runs/growth-ladder-a01-96937472f5fd/output/payload",
  "error": null,
  "experiment_id": "growth-ladder-a01",
  "gpu_hours": 0.0,
  "metrics": {"dead_experts": 0.0, "experts_per_token": 2.0, "holdout_loss": 4.175413370132446},
  "project": "growth-binding",
  "promoted_experiment_id": null,
  "succeeded": true
}
```

Registry: `baseline=passed`, `growth-ladder-a01=rejected`, stranded-result audit empty.
Source identity `d8b08b96…` verified against all three recorded
`chowder-identity.json` files (training run, baseline eval, candidate eval).

Three honest readings of that transcript:

- **`succeeded: true` with `rejected` is not a contradiction.** `chowder train` reports
  whether the candidate *ran*; the registry row carries the gate verdict. Four training
  steps on a two-layer toy move a holdout loss from 4.1752 to 4.1754, so the gate
  correctly rejected it. The binding reports both facts and decides neither.
- **The gate's verdict is genuinely production's.** `rejected` comes from
  `evaluate_promotion` inside the run, not from this binding.
- **`gpu_hours: 0.0` is real, not missing.** A CPU run of a toy model costs ~0 GPU-hours;
  the ladder proves the wiring, not a cost model. The binding records `None` when a
  number is absent and `0.0` only when the trainer reported zero.

## Two production constraints the ladder found

Neither is visible to a fake trainer, and both were found by running the real path.

**1. The child must inherit the parent's checkout.** The repository's
`test_every_chowder_worker_launch_passes_worker_env` guard failed on the first version of
this binding: `default_runner` passed `env=dict(environment)`, so the child resolved
`import chowder` through whatever its environment pointed at — the exact failure
`chowder/worker_env.py` documents, where a worktree run trained and evaluated with the
main checkout's code and recorded results against the wrong code. `default_runner` now
builds the child environment with `worker_env`, and the binding refuses a caller-supplied
`PYTHONPATH` outright. `test_worker_env.py` now passes.

**2. A registry can hold exactly one automatic baseline.** Production only supports
`baseline.mode` `"auto"` or `"fixed"`, and `"auto"` records an experiment literally named
`baseline`; the registry refuses a duplicate id. A shared, cycle-level registry therefore
cannot score two candidates — the second attempt dies inside
`_run_automatic_baseline` with `duplicate persisted experiment id: baseline`. The binding
defaults to **one registry per attempt** (each attempt self-contained, individually
audited, with its own automatic baseline measured on the same frozen holdout by the same
scorer), accepts an explicit shared registry when the project declares
`baseline.mode='fixed'`, and **refuses before compute** when a shared registry already
holds a baseline — naming the constraint and both supported resolutions rather than
letting a child traceback explain it after a model load. Rewriting the caller's baseline
mode, or making `record_experiment` tolerant of duplicates, were both rejected: the first
is silent drift in a qualified config, the second weakens a production invariant for a
caller's convenience.

## Mutation evidence

Every guard was mutated one at a time and each mutation had to be caught by its own test
— **13/13 caught**:

growth-envelope cost gate removed · data-registry trainable check removed · contamination
verdict ignored · corpus placeholder check removed · project budget comparison removed ·
attempt directories reused · attempt-id collision check removed · declared shared registry
ignored · shared-registry baseline conflict check removed · missing source identity
treated as verified · stranded registry audit ignored · unparsable summary replaced by a
synthetic one · child environment not built from `worker_env`.

The stranded-audit test was rewritten after its first mutation went uncaught: stranding the
attempt's *own* row was already caught by the terminal-status check, so the test now
strands a **sibling** experiment's row and leaves this attempt's lifecycle clean, which only
a registry-wide audit can see.

## Deliberate non-changes

- **No `search` section.** `run_project` has no search config and the successive-halving
  controller has no production caller, so the binding refuses a template containing one
  rather than pretending to drive a controller that is not wired.
- **No promotion decision.** The binding reports; `evaluate_promotion` decides.
- **No silent config rewriting.** A template looser than the envelope, or with no declared
  budget, or naming no corpus, is refused with the reason. Guessing which knob takes the
  corpus would train on whatever the template already pointed at.
- **No retry in place.** A new attempt is a new directory with a new experiment id.

## What this licenses, and what it does not

**Licenses:** a real, tiny Model N → N+1 attempt on this machine where the promotion phase
is fed genuine benchmark measurements, because the training half of the loop now produces
real, audited, identity-verified evidence.

**Does not license:** a Generation-1 campaign on the 9B. Before that:

1. **The promotion phase still consumes `BenchmarkResult`s that no run produces.** S7
   deliberately stops at training; mapping a run's measured metrics onto promotion inputs
   requires declared directions and normalizations from the benchmark registry, and
   inventing them from `holdout_loss` here would be arithmetic on an undeclared scale.
2. **The cycle-level registry question above needs an answer** for a campaign that scores
   more than one candidate: either per-candidate registries plus an aggregation story, or
   a tournament project that establishes the baseline once.
3. **The campaign's own first-run checklist** in
   [`FIRST_MODEL_GROWTH_CAMPAIGN.md`](FIRST_MODEL_GROWTH_CAMPAIGN.md) is unchanged, and the
   router backend's qualified-device boundary still applies to whatever model the first
   real campaign uses.

## Reproduction

```
PYTHONPATH=src python -m pytest tests/test_growth_training_binding.py -q        # 22 passed
PYTHONPATH=src python -m pytest tests/test_growth_training_binding.py \
    -q -k "test_s1 or test_s4 or test_s6 or test_s7"                            # 4 real-subprocess
PYTHONPATH=src python -m pytest -q                                              # 2081 passed, 77 skipped
ruff check src tests                                                            # All checks passed
```

The S1–S7 tests are gated on torch/transformers and run in the real-ML CI job, so a silent
skip shows up in the skip count. The ladder was executed on CPU with no GPU job and no
resident local-model workload; the machine's resident inference server was left untouched.
