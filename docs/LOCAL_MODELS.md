# Local-first model sources

Chowder should reuse model weights that already exist on the machine. A training or evaluation run must not require a second copy of a model merely because the original source was Hugging Face.

## Policy

`backend.base_model` is a model source, not necessarily a Hub repository id.

Resolution order:

1. If the configured value names an existing local model directory, use that directory directly.
2. If a relative local path is explicitly resolved by a caller against the project/work directory, use that resolved directory directly.
3. Otherwise treat the value as a Hugging Face model id and use the normal cache/Hub path.

A local model directory bypasses Hugging Face cache probing. It must never be submitted to `try_to_load_from_cache()` as though the filesystem path were a Hub repo id.

## Disk-space rule

For a local source Chowder must not stage, duplicate, or redownload the base weights. Disk preflight should budget only for artifacts the requested experiment can actually create, such as:

- LoRA/PEFT adapters;
- optimizer/checkpoint state when checkpointing is enabled;
- evaluation evidence and manifests;
- explicitly requested distilled/pruned model outputs.

Model conversion is a separate explicit operation. A run should never silently create a second full-precision copy as a side effect of selecting a backend.

## Example

```json
{
  "backend": {
    "type": "transformers-peft",
    "base_model": "D:/AI/models/Qwen3.6-35B-A3B-abliterated",
    "offline": true,
    "dataset": "data/chowder_train.jsonl"
  }
}
```

`offline: true` is useful when the directory is self-contained because it turns an accidental missing local asset into an immediate error instead of a network fetch. It is not required for a valid local directory to be recognized as local.

## Supported local shape

The current Transformers/PEFT and Unsloth training paths expect a Hugging Face-style model directory that their respective `from_pretrained()` loaders can open: model config, tokenizer assets, and compatible weight shards.

A standalone GGUF is not the same thing as a trainable Transformers checkpoint and is not automatically converted. GGUF support should be added only through an explicit backend/conversion path with its own disk estimate and provenance.

## Provenance

The configured local path should remain visible in run evidence as the requested base model. Future hardening should add an optional content identity for large local bases without requiring a full re-hash on every run (for example, a manifest of shard names/sizes plus selected config/tokenizer hashes, with an explicit full-hash mode for publication-grade runs).
