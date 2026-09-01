# Training backends

Chowder keeps scientific decision-making separate from framework execution. A training backend produces a `TrainingArtifact`; only a separate evaluator may produce an `ExperimentResult` that can enter a promotion gate.

## Transformers + PEFT

Install the optional training stack:

```bash
pip install -e '.[train]'
```

For 4-bit QLoRA, also install the quantization extra:

```bash
pip install -e '.[train,qlora]'
```

The first backend supports causal-language-model SFT with LoRA or CUDA QLoRA. Heavy ML imports occur only inside an isolated worker subprocess, so the Chowder controller remains lightweight and a framework crash does not run inside the decision engine.

### Resolved configuration

```json
{
  "seed": 17,
  "backend": {
    "type": "transformers-peft",
    "base_model": "Qwen/Qwen3-8B",
    "revision": "optional-hub-revision",
    "dataset": "data/train.jsonl",
    "text_field": "text",
    "max_length": 512,
    "precision": "auto",
    "quantization": "4bit",
    "training": {
      "learning_rate": 0.0002,
      "epochs": 1,
      "batch_size": 1,
      "gradient_accumulation_steps": 4,
      "logging_steps": 10,
      "gradient_checkpointing": true
    },
    "lora": {
      "r": 16,
      "alpha": 32,
      "dropout": 0.05,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "use_rslora": false
    },
    "runtime": {
      "timeout_seconds": 7200
    }
  }
}
```

The experiment graph may store only child patches. `ExperimentGraph.resolve_config()` resolves root-to-child patches before the backend sees the configuration.

### Evidence captured for every successful run

The returned `TrainingArtifact` records:

- exact execution-spec SHA-256;
- path-independent recipe SHA-256;
- training-dataset content SHA-256;
- resolved Chowder config SHA-256;
- worker-result SHA-256;
- worker stdout/stderr paths;
- random seed;
- installed framework versions;
- requested model/revision;
- model commit resolved by Transformers when available;
- training telemetry such as loss, step count, runtime and peak allocated VRAM.

Training telemetry is intentionally not benchmark evidence and cannot be passed directly through Chowder's promotion gate.

### Safety and isolation

`trust_remote_code` is hard-disabled for the autonomous backend. Model repositories that require arbitrary Python execution are therefore unsupported by this backend until Chowder has a separately designed sandbox/approval boundary.

The backend runs training in a subprocess and supports timeout termination. Run logs and manifests remain in `.chowder/runs/<run-id>/` for diagnosis.

### Current limits

- This initial worker handles local JSON/JSONL text datasets and causal-LM SFT.
- 4-bit QLoRA currently requires CUDA.
- Multi-GPU/FSDP/DeepSpeed are not yet exposed by this backend.
- The worker trains an artifact only; an independent evaluator is the next layer required for autonomous promotion.
- Hardware calibration evidence is not yet automatically converted into backend batch/placement decisions.
