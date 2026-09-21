# Generation-1 cycle driver — design (2026-09-17)

Preregistration: `docs/quals/GEN1_PREREG_2026-09-17.md` (committed and
pushed, `b04e6e3`, before any of this was implemented). This file records
the driver's exact shape; the driver itself is the executable next step.

## Path

```
GrowthCycle(config, ..., train_fn=SubprocessTrainingFn(...))
  -> plan_curriculum(profile)      # CapabilityProfile from the Gen-0 freeze
  -> train_candidates(recipes)     # REAL chowder project-validate + train subprocesses
  -> decide_promotion_from_runs(parent_runs, candidate_runs)  # MetricBinder + evaluate_promotion
  -> finalize(...)                 # GenerationLedger record
```

## Seams resolved from the merged code

1. **Corpus format**: `SubprocessTrainingFn._write_corpus` materializes
   curriculum material as plain-text lines (`corpus.txt`); the
   transformers-peft backend loads datasets with
   `load_dataset("json", data_files=...)`. Bridge: the curriculum material
   is authored AS JSONL rows `{"text": "<chat-formatted training text>"}`,
   so the corpus file is exactly the dataset the backend ingests, with its
   sha256 recorded in the evidence.
2. **Instrument**: `generation-diagnostics@gen1-eval-protocol-v1` is
   registered in a `BenchmarkRegistry` (via the existing registry
   constructor from `BenchmarkEntry` rows) with
   `primary_metric="eos_termination_rate"`, an identity 0..1
   normalization, and `skills=()` — behavioral instruments must not leak
   into capability-skill estimates. `METRIC_SEMANTICS` gains the metric
   name with `higher_is_better` identity semantics; per-row redeclaration
   stays refused (existing tests).
3. **Independence**: parent and candidate diagnostic runs are separate
   measurement passes, each recorded as its own evidence JSON with
   generation version, content digest of the evaluated weights, and the
   frozen sampling contract (greedy, seed 1234, batch 32, 128 tokens).
4. **Promotion**: `MetricBinder.bind` refuses unregistered benchmarks,
   wrong-generation runs, unsupported/None rows, metric disagreements, and
   undeclared scales; `evaluate_promotion` applies the frozen thresholds
   (target >= 0.90, protected regression <= 0.02).

## Budget contract

Training 0.25 device GPU-h/recipe; evaluation 1.00 GPU-h aggregate with
sub-budgets (load 0.02, target 0.12, protected 0.55). Any exceedance
refuses/demotes with recorded arithmetic. Wall envelope x1.5 of projected.

## What success means

A mechanical PROMOTED or REJECTED verdict from the frozen rule, durably
recorded in the GenerationLedger with parent/candidate identities,
corpus digest, training evidence, and independent evaluations — after
which the repeatability check (a second cycle constructed purely from the
durable records) proves the loop is closed.
