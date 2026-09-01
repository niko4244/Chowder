# Failure harvesting and targeted repair

Chowder treats failed evaluation rows as diagnostic evidence, not automatically as training data.

## Flow

```text
evaluate
  ↓
failed item evidence
  ↓
FailureRecord
  ↓
deterministic clustering
  ↓
RepairPlan
  ↓
independent repair-source data
  ↓
train repair candidate
  ↓
same-protocol evaluation
  ↓
promotion gate
```

Each `FailureRecord` preserves the experiment/evaluation IDs, evaluator and suite, protocol SHA-256, artifact SHA-256, prompt/expected/prediction, score, source role, and a deterministic failure ID.

## Contamination boundary

Source roles are explicit:

- `gate_holdout`: promotion evidence. Diagnostic use only.
- `development`: diagnostic/development evidence. Not directly training-eligible.
- `repair_source`: independently designated repair material that may be converted to training rows.

`write_direct_repair_dataset()` refuses any input containing holdout or development failures. A gate failure may inform *what kind* of new example should be sourced or generated, but its prompt/answer pair may not be copied into the repair set.

This prevents Chowder from improving its measured score by memorizing its own holdout benchmark.

## Generation-cycle integration

`ExperimentCycleRunner` accepts an optional `failure_harvester`. When present it runs after independent evaluation and before adjudication:

1. validate failure IDs and experiment/evaluation lineage;
2. verify failure protocol SHA against the evaluation protocol;
3. cluster failures by evaluator, suite, protocol, source role, and failure kind;
4. create deterministic repair plans;
5. persist failure records and plans when a `RunRegistry` is configured.

Diagnostic processing is non-blocking. If harvesting fails, Chowder records `diagnostic_error` but still allows valid evaluation evidence to reach the promotion gate.

## Current limitation

The first clustering pass is intentionally deterministic and heuristic. It recognizes broad classes such as empty prediction, refusal/unknown, overlong mismatch, and generic answer mismatch. The next layer can add semantic clustering and independent counterexample generation without weakening the provenance or contamination boundaries defined here.
