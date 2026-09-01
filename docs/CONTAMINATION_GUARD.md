# Holdout contamination guard

Chowder must not improve a benchmark by accidentally training on the benchmark.

The failure-repair pipeline therefore has two independent boundaries:

1. **source-role eligibility** — gate/holdout and development failures are not directly training-eligible;
2. **content overlap auditing** — even an independently labeled repair source is rejected when its prompt overlaps the promotion holdout.

## Hash-only holdout index

The auditable `transformers-text` evaluator emits a fingerprint index for every evaluation row. It stores only:

- normalized prompt SHA-256;
- normalized prompt + expected-answer SHA-256.

Raw prompt/answer text is not written into this index. The parent evaluator verifies the index file remains inside the evaluation directory and that its SHA-256 matches the worker result before accepting the evidence.

Normalization collapses whitespace, strips leading/trailing space, and case-folds text. This means variants such as:

```text
What is 2 + 2?
WHAT   is 2 + 2?
```

produce the same prompt fingerprint.

## Repair admission

`write_verified_repair_dataset()` and `write_verified_failure_repair_dataset()` require one or more verified holdout fingerprint files. They reject the repair set when either:

- a normalized repair prompt matches a holdout prompt; or
- the normalized prompt/expected pair exactly matches a holdout pair.

Prompt overlap alone is sufficient to reject the row, even if the candidate repair answer differs.

The successful audit records only overlap hashes and holdout-index digests, preserving provenance without reproducing benchmark content.

## Scope

This first implementation covers Chowder's custom `transformers-text` evaluation suites because their row-level datasets are explicitly available and auditable. Standard `lm-evaluation-harness` suites still rely on task-level protocol fingerprints; adding a task-dataset contamination index for harness-managed datasets is a later extension.
