# Standard benchmarks with lm-evaluation-harness

Chowder's `LmEvalEvaluator` runs EleutherAI `lm-evaluation-harness` in an isolated worker and returns an `EvaluationOutcome` for the normal Chowder generation cycle.

Install the optional evaluation stack:

```bash
pip install -e '.[eval]'
```

The adapter uses the Hugging Face model backend with the trained PEFT adapter passed separately from the base model. Chowder keeps `trust_remote_code` disabled and verifies the adapter directory SHA-256 before the worker starts.

## Configuration

```json
{
  "evaluation": {
    "type": "lm-eval",
    "device": "auto",
    "batch_size": "auto",
    "num_fewshot": 0,
    "tasks": ["hellaswag", "arc_easy"],
    "metric_map": {
      "reasoning_hellaswag": "hellaswag:acc_norm,none",
      "knowledge_arc_easy": "arc_easy:acc_norm,none"
    },
    "precision": "inherit",
    "quantization": "inherit",
    "apply_chat_template": false,
    "fewshot_as_multiturn": true,
    "runtime": {
      "timeout_seconds": 7200
    }
  }
}
```

## Why `metric_map` is mandatory

`lm-evaluation-harness` tasks commonly report several fields: raw accuracy, normalized accuracy, stderr, exact match, and task-specific metrics. Chowder will not guess which field controls model promotion.

Each entry maps a Chowder metric name to an exact `task:metric` key. If that task or metric is absent, the evaluation fails instead of silently substituting another score.

This also lets Chowder goals stay stable when the external benchmark's native field names are awkward:

```text
reasoning_hellaswag -> hellaswag:acc_norm,none
```

## Evidence

Each run records:

- trained adapter SHA-256;
- evaluation spec SHA-256;
- exact task list and metric map;
- SHA-256 of the complete raw lm-eval result payload;
- package versions;
- execution device and GPU count;
- stdout/stderr logs;
- random seed and wall time.

The worker's GPU time is evaluation-only. `ExperimentCycleRunner` adds it to training GPU time before the candidate is adjudicated.

## Current scope

- Hugging Face causal models with PEFT adapters;
- optional inherited 4-bit loading;
- standard `lm-eval` tasks and groups;
- one explicit global `num_fewshot`, batch-size policy, and device per evaluation run.

Future work will add task-set version pinning, benchmark contamination metadata, and richer per-sample failure harvesting for Chowder's automatic curriculum builder.
