# Local Corpus Repair Provider

Chowder's first concrete `RepairSourceProvider` is intentionally deterministic and non-generative.

## Why start here

Before allowing an LLM or remote agent to synthesize repair examples, Chowder needs a baseline provider whose selection process is fully auditable. `LocalCorpusRepairProvider` receives only the sanitized `RepairRequest` and selects examples using declared metadata:

- evaluation suite
- controlled repair strategy
- optional failure class
- deterministic priority

It never receives the failed holdout prompt, expected answer, prediction, row index, failure hash, or free-form diagnosis text.

## Corpus format

One JSON object per line:

```json
{"example_id":"math-001","suite":"reasoning","strategy":"near_neighbor_reasoning","prompt":"What is 3+3?","expected":"6","priority":10}
```

Optional fields:

- `suite`: suite name or `*` (default `*`)
- `strategy` / `strategies`: one or more `RepairStrategy` values or `*`
- `failure_kind` / `failure_kinds`: optional failure-class filter
- `priority`: numeric deterministic selection priority

Each selected corpus file is read once, SHA-256 hashed from those exact bytes, and represented as a `RepairSource`. This makes local corpus provenance verifiable rather than a caller-supplied hash assertion.

## Orchestration

`prepare_and_propose_repair_population()` performs:

```text
FailureCluster + RepairPlan
        |
        v
sanitized RepairRequest
        |
        v
RepairSourceProvider.propose
        |
        v
provider identity validation
        |
        v
source provenance + holdout contamination audit
        |
        v
VerifiedRepairDataset
        |
        v
deterministic RepairVariant population
        |
        v
EvolutionEngine.propose
```

The orchestrator stops before training. `EvolutionEngine.propose` remains the only path that reserves repair-candidate GPU-hour budget and parallel slots.

## Failure behavior

Provider/source/contamination/candidate-construction failures occur before engine proposal. Partial repair files are removed so a rejected proposal cannot leave a poisoned deterministic retry directory.

## Current limitation

Selection is capability-directed, not semantic. The provider cannot search for examples similar to a hidden failed prompt because it deliberately never receives that prompt. A future research/source provider can use public task taxonomies, independently generated capability descriptors, or trusted external retrieval while preserving the same request boundary and contamination gate.
