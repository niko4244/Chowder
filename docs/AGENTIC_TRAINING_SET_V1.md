# Agentic SFT sets v1 — the Fable arm and its control (2026-10-18)

Two files under `C:\llm-work\train-sets\agentic-v1\`, built by
`chowder_batch/build_agentic_training_set.py`:

- **`train-fable.jsonl`** — the three owner-approved Fable agentic sources
  (glint local, lordx64, swarm), cross-source deduped by
  `FableAgenticProvider`, plus the shared Nemotron block.
- **`train-control.jsonl`** — the same Nemotron block and **no** Fable row.

The arms differ by the Fable rows and nothing else, so any measured delta is
the Fable exposure, not a sampling artefact. The control exists because Fable
output is AGPL-3.0 (decision N1) and the Anthropic usage question (N2) is still
open; `docs/REAP_MOE_LINEAGE.md` requires a Fable-free arm until N2 is recorded.

## Rebuild

```
python chowder_batch/build_agentic_training_set.py --out C:\llm-work\train-sets\agentic-v1
```

Defaults: `--fable-rows 1500 --nemotron-rows 1500 --seed 20261018`, template
validation on. The four benchmark test splits are read from the **offline**
Hugging Face caches (`HF_DATASETS_OFFLINE=1`); the tokenizer defaults to
`G:\llm-cold\bf16\Qwen3.6-35B-A3B` and, if absent, template validation warns and
is skipped rather than failing. The manifest is written beside the two sets:
per-source path/sha256/licence, rows in/out, every drop reason, the seed, the
protected-set sizes, and per-output sha256 + row count.

## Row shape and backend config

Rows are `{"messages": [...], "tools": "<json string>" or ""}`. Tool-call
`function.arguments` are JSON **strings**, not objects: `datasets` loads JSONL
through Arrow, and Arrow unifies nested dicts across rows, adding other rows'
keys as nulls. The worker parses the strings back through
`_validate_chat_messages`; a tool call rendered with a string argument raises
inside the chat template, so this shape is load-bearing, not cosmetic.

```yaml
dataset_format: chat
messages_field: messages
tools_field: tools
```

`transformers_worker` reads only `messages` and (when `tools_field` is set)
`tools`; the tools schema reaches the model through the template's
tools argument, exactly as the worker's chat path renders it.

## What never gets trained on

Protected evaluation items (GSM8K, MATH-500, MGSM, HumanEval test splits and the
`tool-use@reap-moe-v1` items) and every Nemotron row the evaluation split
reserves (`is_reserved_for_evaluation`) are dropped **and counted**, never
silently. Drop reasons per source: `reserved_uuid`, `duplicate`, `contaminated`,
`validation_error`, `no_assistant_turn`, `template_error`, `empty_label_span`,
`over_cap`.
