# MoE Gen-1 training mix v1 — the Fable arm and its control (2026-10-19)

Two files under `C:\llm-work\train-sets\moe-gen1-v1\`, built by
`chowder_batch/build_moe_gen1_mix.py`, implementing the approved mix in
`docs/PLAN-MOE-FINETUNE-BEAT-BASELINES-2026-10-19.md` §3:

- **`train-fable.jsonl`** — available approved math/code sources, the
  Nemotron tool-use block, and the Fable tool-use block, subject to per-source
  caps, screening, and documented shortfalls.
- **`train-control.jsonl`** — the same rows with **no** Fable row: a strict
  subsequence of the arm file, identical row order, identical Nemotron block.

The arms differ by the Fable rows and nothing else, so a measured delta is the
Fable exposure and not a sampling artefact. The control exists because Fable
output is AGPL-3.0 (decision N1) and the Anthropic usage question (N2) is still
open; `docs/REAP_MOE_LINEAGE.md` requires a Fable-free arm until N2 is recorded.

## Sources and caps

| Source | Cap | Licence |
|---|---|---|
| OpenR1-Math (verified solutions, local sample) | 400 | Apache-2.0 |
| GSM8K **train** split (offline HF cache) | 200 | MIT |
| Ornith curriculum (`curriculum-all/ornith_sft.jsonl`) | 39 | not declared |
| OpenThoughts-3 code reasoning (`ot3_sft.jsonl`) | 150 | Apache-2.0 |
| Nemotron-Agentic `interactive_agent` (from `agentic-v1/train-control.jsonl`) | 120 | CC BY 4.0 |
| Fable sets (rows in `agentic-v1/train-fable.jsonl` not in its control) | 120 | **AGPL-3.0** (N2 open) |
| Self-OSS-Instruct-SC2 execution-filtered Python (`selfoss_code`, optional) | 250 | ODC-BY (verify source terms before release) |

The earlier ~1,030-row Fable arm / ~910-row control estimates are historical,
not guarantees for current builds. The optional Self-OSS code source and stricter
length screening change available row counts. A source with fewer clean rows
than its cap contributes all of them and the shortfall is recorded in the
manifest — never padded, never swapped for another source.

## Row shape

`{"messages": [...], "tools": "<json string>" or ""}` — the worker's chat row
shape (`docs/AGENTIC_TRAINING_SET_V1.md`). Math and code rows are one user turn
and one assistant turn; GSM8K keeps its worked steps and closing `#### n` line
verbatim. Tool-call `function.arguments` are JSON **strings** (Arrow unifies
nested dicts and would add every other row's argument keys as nulls).

## What never gets trained on

Every candidate row — including the Ornith and agentic rows, re-checked — runs
through the agentic-v1 contamination firewall (exact, normalized, shingle
overlap, MinHash) **and** the plan's 13-gram rule against the GSM8K test,
MATH-500, MGSM, HumanEval test splits and the committed
`tool-use@reap-moe-v1` items. Any hit is dropped and counted; each source's
manifest entry lists rows in, examined, kept, shortfall, every drop reason and
which detectors fired. The tool *schema* is excluded from the fingerprint (the
shared API surface every agentic row offers), and so is the `system` turn on
both sides of the 13-gram check: Nemotron's agent policy is one paragraph
repeated in every row *and* in the `tool-use@reap-moe-v1` items built from it,
so folding it in flags the whole family. The user/assistant content — what an
item actually asks and answers — stays checked.

Selection is a seeded permutation of each source's deduplicated pool, then the
first `cap` clean rows; a uniform sample without replacement where the pool is
large enough, and every row examined where it is not.

## Sequence-length handling (corrected 2026-10-22)

The merged builder now defaults to `--max-length 2048`, matching the
Gen-1 training recipe. When tokenizer/template validation is available, it
measures each complete rendered example and **drops** an overlength row
(`drops.over_length`) rather than truncating it and losing a worked
answer or assistant label span. The source pool is scanned for another
eligible row until the cap is reached or the pool is exhausted; shortfalls
remain explicit. The earlier 4,096-screened 1,028-row arm and the report
that 136 examples would be truncated describe a **superseded build**, not
the currently committed default.

Long Nemotron and OpenThoughts-3 reasoning rows may produce significant
shortfalls at 2,048. The optional, execution-filtered `selfoss_code`
source provides shorter code examples without silently substituting
for those sources. If the tokenizer cannot be loaded, template/length
validation is skipped with a warning; such a build is **not qualified**
to claim the 2,048-token screen was enforced. Retain the generated
manifest, actual row counts and source hashes for every arm.

## Rebuild

```
python chowder_batch/build_moe_gen1_mix.py --out C:\llm-work\train-sets\moe-gen1-v1
```

Defaults: `--seed 20261019`, template validation on against
`G:\llm-cold\bf16\Qwen3.6-35B-A3B` (CPU only; warns and skips if absent). The
protected splits are read from the **offline** Hugging Face caches
(`HF_DATASETS_OFFLINE=1`). `manifest.json` beside the sets records the seed,
the plan file, the tokenizer path, protected-set sizes, per-source
path/sha256/licence/cap/kept/drops and per-output row count + sha256.
