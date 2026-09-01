# Repair candidate construction

After Chowder harvests failures, creates a `RepairPlan`, sources clean repair examples, and passes the contamination audit, it can construct reproducible child experiments.

## Verified repair dataset

`VerifiedRepairDataset` binds three pieces of evidence:

- the repair dataset file SHA-256;
- the contamination audit and holdout-index SHA-256 values;
- the canonical repair-example index SHA-256.

Before a candidate is created, Chowder re-hashes the dataset file and recomputes every prompt/pair fingerprint from the written rows. A clean audit from one dataset therefore cannot be reused with a different dataset, even if the caller updates only the file SHA.

## Protocol preservation

A repair candidate may patch only:

```text
backend.dataset
backend.training.*
backend.lora.*
```

It does not patch:

```text
backend.base_model
backend.revision
backend.precision
backend.quantization
evaluation.*
```

Those values remain inherited through the experiment graph. This keeps repair experiments comparable to their parent under Chowder's protocol-match gate.

## Deterministic identity

Candidate IDs are derived from:

- parent experiment ID;
- repair-plan ID;
- repair dataset SHA-256;
- repair-example index SHA-256;
- holdout-index SHA-256 values;
- variant name;
- training patch;
- LoRA patch;
- expected benchmark deltas.

Running the same repair intervention twice produces the same candidate identity. Changing the holdout audit, dataset, or training intervention produces a different branch.

## Candidate populations

`build_repair_population()` creates several competing repair branches from one verified dataset. For example, variants can test different learning rates or LoRA ranks while keeping the evaluation protocol fixed.

```text
RepairPlan + verified repair data
               |
        +------+------+------+
        |             |      |
      lr-low        lr-high  rank-low
        |             |      |
        +------+------+------+
               |
        train/evaluate
               |
          tournament
               |
          promote/reject
```

The resulting `Experiment` objects can be passed to `EvolutionEngine.propose()`, which applies the existing compute-budget and parallel-candidate limits before execution.
