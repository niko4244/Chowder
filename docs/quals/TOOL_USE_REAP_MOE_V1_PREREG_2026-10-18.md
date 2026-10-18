# Pre-registration — `tool-use@reap-moe-v1` (2026-10-18)

**What this registers.** The target instrument for the REAP MoE pruning
question: does an expert-pruned, LoRA-tuned Qwen3.6 MoE still choose the
right tool, with the right arguments, on an agentic tool-call benchmark?
The instrument is frozen here, before any candidate is measured, so the
number a campaign reports cannot be redefined after seeing it.

**Items.** 240 items, committed at
`src/chowder/growth/instruments/tool_use_reap_moe_v1.jsonl`, one per line:

| Field | Meaning |
|---|---|
| `uuid` | the source row's uuid (provenance; not measured) |
| `tools` | the tool schemas offered in the context |
| `messages` | the conversation *before* the assistant turn being predicted |
| `expected` | `{"name": ..., "arguments": {...}}` — the call to make |

Each item is one conversation **cut at the first assistant turn that makes
exactly one parseable tool call**. Turns with zero or several calls do not
qualify; a call whose `arguments` are not a JSON object does not qualify. A
row that never qualifies contributes no item.

**Source.** `nvidia__Nemotron-Agentic-v1` `interactive_agent.jsonl`
(19,028 rows, CC BY 4.0), pinned by path and sha256 in the manifest.

**Held-out rule.** A source row is dropped when

```python
int(hashlib.sha256(row["uuid"].encode()).hexdigest()[:8], 16) % 10 == 0
```

— the public `is_reserved_for_evaluation(uuid)` predicate, so the boundary
is reproducible and independent of dataset order. On this source it drops
1,896 of 19,028 rows.

**Sampling.** From the qualifying, non-held-out, non-leaking candidates
(14,907 here), the instrument takes 240 by **round-robin over the sorted
tool names**, capped at **24 items per tool** (10%). No single tool may
dominate the score. Chosen items are finally sorted by
`(name, uuid, turn_index)`. Seed `20261018` is recorded; the sample is
deterministic without randomness, so the seed is provenance, not a shuffle.

**Leak check.** Any candidate with a message text of ≥ 64 characters
appearing verbatim in the REAP calibration corpus (`reap/calib-v2`) is
dropped — an item that shapes the pruned weights cannot also measure them.
The count dropped is recorded in the manifest (0 on this build).

**Scoring (`tool_call_exact`).** Score the next turn as JSON:
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (JSON form) or
`<function=name><parameter=k>v</parameter></function>` (Qwen3.5/3.6 XML
form, `chat_template.jinja:122`). The item scores **1.0 iff**, in order:

1. exactly one block parses as a tool call, else `no_call` / `multiple_calls`;
2. its `name` equals the expected `name`, else `wrong_name`;
3. its `arguments` satisfy the expected tool's `parameters` schema, else
   `schema_invalid`;
4. its `arguments` equal the expected `arguments` (order-insensitive,
   `5` == `5.0`, `"5"` != `5`), else `args_mismatch`.

Only `ok` scores 1.0; every other reason scores 0.0. Parsing is
tagged-blocks-only: bare JSON with no `<tool_call>` wrapper is `no_call`.
`explain_tool_call(completion, item)` returns the reason.

**Determinism.** The builder is deterministic (sorted traversal, fixed
seed) and the manifest carries no timestamp, so two builds on the same
inputs are byte-identical; the committed items are verified against the
manifest's `items_sha256` before any read, and a mismatch refuses.

**Falsification.** This instrument's number is meaningful only as a
*difference* against the same base's un-pruned score, measured with this
instrument's bytes. If the manifest's `items_sha256` differs from
`a5890a618415…`, the run is a different instrument and its number is not
comparable.
