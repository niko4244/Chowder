# Evaluation protocol fingerprints

A benchmark score is only comparable to another score when the evaluation protocol is materially the same. Chowder therefore supports strict protocol matching at the promotion gate.

Enable it on a goal:

```python
Goal(
    metrics=(...),
    gpu_hour_budget=20,
    require_protocol_match=True,
)
```

When strict matching is enabled, both the baseline and candidate must carry an `evaluation_protocol_sha256`, and the two fingerprints must match exactly. A higher score under a different protocol is rejected rather than treated as model improvement.

## What belongs in a protocol

A protocol fingerprint intentionally excludes candidate-specific state such as adapter paths, output directories, run IDs, wall time, and the candidate artifact hash.

It includes evidence that can change the meaning of a score:

- evaluator backend and package versions;
- base model identity and pinned revision;
- benchmark/task definitions;
- evaluation dataset content hashes when Chowder owns the dataset;
- task-config hash for `lm-evaluation-harness`;
- exact metric mapping;
- prompt/expected fields and scoring rules;
- few-shot count and chat-template policy;
- generation limits;
- precision and quantization;
- resolved execution device;
- random seed.

The protocol is stored both as a canonical evidence object and as its SHA-256 fingerprint.

## Custom text evaluator

For `TransformersTextEvaluator`, each suite contributes its dataset SHA-256 plus scoring/generation configuration. Changing a single evaluation row changes the protocol fingerprint.

## lm-evaluation-harness

For `LmEvalEvaluator`, Chowder hashes the resolved `configs` returned by the harness. This protects against a task name remaining the same while its task definition changes. The `lm-eval` package version, exact task list, metric map, few-shot/chat settings, precision, quantization, and runtime device are also included.

## Baseline migration

Existing manually created baselines may not have protocol evidence. Keep `require_protocol_match=False` while establishing a fresh evaluated baseline, then enable strict matching for autonomous generations.

For long-running autonomous optimization, strict protocol matching should be treated as the default operating mode once that baseline exists.
