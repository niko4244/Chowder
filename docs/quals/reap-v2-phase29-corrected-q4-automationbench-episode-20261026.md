# Phase 29 — corrected Q4 AutomationBench episode: 6/6 tasks completed, 0 passed

**Date:** 2026-10-26, 15:55–16:27 CDT
**Host:** Windows, RTX 5060 Ti 16,311 MiB
**Status:** scientific selection remains **BLOCKED**, `chosen: null`, `release_approved: false`

## Why the GPU had to be cleared, and what was cleared

By 15:55 the Q4 experiment itself was no longer running. Two unrelated resident
model servers held 10.7 GiB of the card:

| PID | Port | Role |
|---|---|---|
| 36372 | 8766 | preset loader (`models-local.ini`, `--models-max 1`) |
| 12624 | 8767 | `spark-x2.5-4b`, `-ngl 999 --ctx-size 131072` |

Their exact command lines are preserved at
`F:\reap-v2-automationbench-private-20261026\stopped-resident-servers-20261026.json`
before anything was stopped. Both were stopped on explicit user instruction, and
they were **not** restarted, because restarting them would undo that
instruction. To bring them back:

```
"F:\llama.cpp\bin\llama-server.exe" --host 127.0.0.1 --port 8766 \
  --models-preset "F:\llama.cpp\models-local.ini" --models-max 1
"F:\llama.cpp\bin\llama-server.exe" --host 127.0.0.1 --port 8767 \
  -m F:\Ollama\models\blobs\sha256-5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2 \
  --alias spark-x2.5-4b -ngl 999 --ctx-size 131072 --jinja -fa on \
  --top-p 0.95 --temp 0.7 --min-p 0.0
```

## Trial 2 exposed a second, different failure

The first corrected attempt (`-c 65536 --max-steps 45`, dir `...-six-v1`) ran
three tasks cleanly, then died at task 4 with `n_tokens = 65535, truncated = 1`
and 29+ invalid-JSON 500s.

The prompt at that point was only **11,180** tokens. The conversation had not
grown into the wall — **generation ran ~54,355 tokens in a single turn** and was
cut at the ceiling. That is a degenerate generation, not context exhaustion, so
enlarging the context cannot fix it. Phase 28's arithmetic was right about the
prompt-growth path and silent about this one.

## The configuration that actually worked

Per-turn generation had to be bounded as well as the context:

| Knob | Value |
|---|---|
| context | `-c 65536` (1 slot) |
| per-turn cap | `--n-predict 4096` |
| evaluator steps | `--max-steps 44` |
| arithmetic | 1,067 + 1,382.9 × 43 + 4,096 = **64,629** ≤ 65,536 |
| GPU layers | 41/41, 0 experts on CPU |
| model | `G:\reap128-v2-q4_k_m-20261025.gguf`, sha256 `2d64b786…63bf7`, 11,563,915,264 bytes |
| evaluator | public AutomationBench 1.0.6, commit `4a8e1061…`, clean checkout |

4,096 is ~4.7× the largest legitimate generation observed in these tasks, so the
cap bounds a runaway without truncating a real tool call. Nothing in the
successful run hit the cap.

## Result

All six pinned public tasks completed. No task aborted, no evaluator error, no
empty response.

| Task | Passed | Score | Model calls | Tool calls |
|---|---:|---:|---:|---:|
| sales.apply_project_label | no | 0.000 | 8 | 13 |
| sales.advance_opportunity_stage | no | 0.000 | 13 | 33 |
| sales.create_contact_for_account | no | 0.000 | 24 | 30 |
| sales.mark_vip_emails_read | no | 0.000 | 32 | 130 |
| marketing.event_followup_personalization | no | **0.556** | 16 | 49 |
| hr.visa_expiration_monitoring | no | 0.000 | 32 | 71 |

- **Pass rate 0/6 (0%).** Mean score **0.093**.
- 125 model calls, 326 tool calls, 657 s wall clock.
- 51,232 output tokens over 654 s of model time → **~78 tok/s aggregate**.
- 2,289,441 input tokens, of which 2,343,532 were cache-hits.

`marketing.event_followup_personalization` reached 0.556, so the model was doing
real work rather than failing to call tools. The rest produced no credit.

## The harness failure is genuinely gone

An independent audit re-derived every aggregate from the item rows and read the
native server log directly:

| Check | 32k trial (phase 28) | this episode |
|---|---:|---:|
| Requests served | 24 | **125** |
| Max prompt | 32,639 | 56,873 |
| Truncated generations | 1 | **0** |
| Invalid-JSON 500s | 66+ | **0** |
| Scorable episodes | **0** | **6** |

`truncation_eliminated: true`. This is the first local Q4 AutomationBench run in
this project that produced completed, scored episodes instead of a retry storm.

## What this does and does not establish

It establishes the runnable configuration: 64k context plus a 4k per-turn cap
completes the pinned six-task slice on a 16 GiB card.

It does **not** produce a leaderboard number. This is 6 of 600 public tasks, one
run, one seed, described as `NONCOMPARABLE_Q4_CTX64K_1X65536_STEPS44_CAP4096` —
a different context geometry and step budget from the frozen four-slot 16,384
cohort. `official_private_leaderboard_comparable: false`. No Gen-0 gate moves.

## Reproduce

```
python -m chowder_batch.reap_q4_automationbench_ctx64k_20261026 \
  --execute --out F:/reap-v2-automationbench-private-20261026/<new-dir>
```

Read-only preflight is the default and refuses on memory, GPU, port, disk or
episode-arithmetic grounds. Receipts land in `experimental-manifest.private.json`,
`native.private.log`, `resources.private.jsonl`,
`public-automationbench.private.json` and `independent-audit.private.json`.
