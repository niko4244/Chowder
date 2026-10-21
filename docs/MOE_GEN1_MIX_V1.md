# MoE Gen-1 training mix v1 — the Fable arm and its control (2026-10-19)

Two files under `C:\llm-work\train-sets\moe-gen1-v1\`, built by
`chowder_batch/build_moe_gen1_mix.py`, implementing the approved mix in
`docs/PLAN-MOE-FINETUNE-BEAT-BASELINES-2026-10-19.md` §3:

- **`train-fable.jsonl`** — all six sources (the four math/code sources, the
  Nemotron tool-use block, and the Fable tool-use block).
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

About 1,030 rows in the arm and 910 in the control. A source with fewer clean
rows than its cap contributes all of them and the shortfall is recorded in the
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

## Sequence length: built at 4,096, plan §4 trains at 2,048

The mix is screened at `--max-length 4096`, the length the agentic-v1 block was
validated at. The plan §4 recipe uses 2,048, and the two numbers matter:
screened at 2,048, Nemotron's long agentic rows collapse — 757 of 873 unique
rows render with no assistant tokens left (`empty_label_span`) and the block
yields nothing at cap 120. Screening at 4,096 keeps the block whole; the trainer
then truncates. On the 1,028-row arm, 136 rows exceed 2,048 tokens and will be
truncated by §4's recipe, of which some lose their answer span. Either raise §4
to 4,096 or accept the truncated tail — this mix does not silently choose.

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
