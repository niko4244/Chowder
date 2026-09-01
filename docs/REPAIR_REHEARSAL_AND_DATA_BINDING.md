# Repair rehearsal and immutable training-data bindings

Targeted repair training can improve one failure cluster while degrading unrelated capabilities. Chowder therefore treats rehearsal and input-byte identity as part of the repair experiment itself.

## Default autonomous-repair behavior

A single-hop autonomous repair now uses two training sources:

1. the contamination-checked independent repair dataset,
2. a deterministic rehearsal sample from the exact dataset that trained the rejected parent.

The replay source is derived from the rejected candidate's resolved backend config and the `dataset_sha256` recorded in its `TrainingArtifact`. The repair-source provider never receives the replay dataset, its contents, or its path.

```text
rejected candidate
      |
      +--> recorded parent dataset SHA ---------+
      |                                         |
      v                                         v
independent repair provider              verified replay source
      |                                         |
      v                                         |
contamination-safe repair data                  |
      |                                         |
      +-------------------+---------------------+
                          v
                deterministic mixed training
                          |
                          v
                independent reevaluation
```

`replay_ratio=1.0` is the autonomous default. `None` explicitly disables rehearsal.

## Deterministic replay selection

For `P` repair rows, `R` available replay rows, and requested ratio `q`, the worker selects:

```text
min(R, ceil(P * q))
```

replay rows using the experiment seed, concatenates them with the repair rows, then deterministically shuffles the mixed training rows using the same seed.

The worker records:

- primary dataset SHA-256,
- replay dataset SHA-256,
- primary row count,
- available replay row count,
- selected replay row count,
- requested replay ratio,
- selection seed,
- SHA-256 of the final mixed training text sequence.

## Immutable input boundary

Repair candidate identity includes:

- repair dataset SHA-256,
- repair contamination-index SHA-256,
- holdout fingerprint-index identities,
- independent-source manifest SHA-256,
- replay dataset SHA-256,
- replay ratio,
- variant/training/LoRA patches.

The Transformers controller checks all declared input hashes immediately before process launch. The isolated worker independently checks them again before loading any model or dataset and again after training. A mutation at any point invalidates the run.

For ordinary non-repair training where the caller did not predeclare `backend.dataset_sha256`, the executor computes the current dataset SHA-256 immediately before launch and writes that binding into the immutable worker run spec.

## GPU-hour accounting

Replay is not free. Repair variants historically estimated repair-only compute, so Chowder conservatively reserves:

```text
repair_only_estimate * (1 + replay_ratio)
```

when replay is enabled. The worker may use fewer replay rows when the parent dataset is smaller, but the controller does not knowingly under-reserve compute simply because rehearsal is enabled.

Actual training/evaluation GPU-hours continue to settle through the normal generation accounting path.

## Safety boundary

Replay is inherited only from data that already trained the rejected candidate. It is not considered an independent repair source and cannot satisfy the independent-source requirement by itself. New repair examples must still pass source provenance and holdout-contamination checks before candidate creation.
