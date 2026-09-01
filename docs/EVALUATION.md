# Independent evaluation and promotion

Chowder deliberately separates **training output** from **promotion evidence**.

A training backend returns a `TrainingArtifact`. An evaluator returns an `EvaluationOutcome`. Neither object can be sent directly to the promotion gate. `ExperimentCycleRunner` combines the training and evaluation compute cost into an `ExperimentResult`, validates the benchmark metrics, and only then asks the `EvolutionEngine` to rank/promote the candidate.

```text
Experiment
  │
  ├─ TrainingExecutor ──> TrainingArtifact
  │                         │
  │                         ├─ immutable artifact SHA checked
  │                         ▼
  ├─ EvaluationExecutor -> EvaluationOutcome
  │                         │
  │                         ▼
  └─ ExperimentCycleRunner -> ExperimentResult
                              │
                              ▼
                    regression gate/tournament
                              │
                       promote or reject
```

## Why the split matters

Training loss, throughput, or step count are useful telemetry but are not independent evidence that a model improved. The type boundary prevents a training backend from promoting its own output.

The cycle runner also owns lifecycle compute accounting. `EvaluationOutcome.gpu_hours` is **evaluation-only** compute; `TrainingArtifact.gpu_hours` is training compute. Their sum becomes `ExperimentResult.gpu_hours`, which is the value settled against the generation's compute reservation.

Failed runs are charged conservatively. If actual compute is unavailable after a crash, Chowder charges the reserved estimate instead of treating the failure as free compute.

## Transformers text evaluator

`TransformersTextEvaluator` is the first independent evaluator. It runs in a subprocess and:

- verifies the trained adapter directory still matches its recorded SHA-256 before evaluation;
- pins the base model to the commit resolved during training when that provenance is available;
- disables `trust_remote_code`;
- supports deterministic exact-match and normalized-exact-match JSONL suites;
- records evaluation dataset hashes, spec hash, result hash, package versions, logs, seed, and runtime device;
- reports zero GPU-hours when the evaluator actually ran on CPU;
- rejects malformed, missing, non-finite, or mismatched metric evidence before gate entry.

Example resolved configuration:

```json
{
  "evaluation": {
    "type": "transformers-text",
    "device": "auto",
    "precision": "inherit",
    "quantization": "inherit",
    "suites": [
      {
        "name": "instruction_exactness",
        "dataset": "data/eval/instruction_exactness.jsonl",
        "prompt_field": "prompt",
        "expected_field": "expected",
        "scoring": "normalized_exact_match",
        "max_new_tokens": 64
      }
    ]
  }
}
```

Each JSONL row must contain the configured prompt and expected-answer fields.

## Standard benchmark harnesses

The custom evaluator is intentionally small and auditable. It is not intended to replace standard benchmark suites. The next evaluator adapter targets EleutherAI's `lm-evaluation-harness`, which supports local Hugging Face models plus PEFT adapters and returns task/metric result dictionaries suitable for mapping into Chowder goals.
