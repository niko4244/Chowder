# Phase 28 — local Q4 admission succeeded; the blocker is context, not memory

**Date:** 2026-10-26, 15:35–15:44 CDT
**Host:** Windows, RTX 5060 Ti 16,311 MiB, 63.90 GiB physical RAM
**Status:** scientific selection remains **BLOCKED**, `chosen: null`, `release_approved: false`

## The premise this phase overturns

Every previous phase reported local Q4 launch as blocked on memory: 10.57 GiB
free against a 29.31 GiB requirement. That was true when it was measured, and it
is **no longer true**. A local Q4 trial admitted and started at 15:34:52 today:

| Gate | Floor | Measured at admission (15:35:22) | Result |
|---|---:|---:|---|
| RAM available | 29.31 GiB | **33.46 GiB** | pass |
| Windows commit | 34.31 GiB | **40.69 GiB** | pass |
| GPU free | 12,288 MiB | **14,693 MiB** | pass |
| Disk free | 8 GiB | 35.15 GiB | pass |

The token ceiling, the 512 MiB extra-KV allowance and the frozen first-start
gate all passed. So **no suspension was needed, and none was performed**: I did
not suspend, stop or reconfigure any process. In particular the `llama-server`
holding 13.04 GiB (PID 88668) and its parent harness (PID 18828) belong to the
`chowder-reap128v2-lab-testing-20261026` worktree and were left untouched while
live.

An earlier free-memory reading of 10.57 GiB was real at the time; the host has
since freed ~10 GiB. Quoting either figure without its timestamp is how the
"Q4 is memory-blocked" belief survived several phases after it stopped holding.

## What actually blocked the run

The trial did not fail on memory. It failed on **context geometry**, and it
failed deterministically. From its own native server log:

| Observation | Value |
|---|---|
| Requests served before failure | 24 |
| Prompt at first request | 1,067 tokens |
| Prompt at 24th request | 32,639 tokens |
| Mean growth per request | **1,372.7 tokens** |
| Served context (`-c 32768 -np 1`) | 32,768 |
| Terminal release line | `n_tokens = 32767, truncated = 1` |
| HTTP 500 `Failed to parse tool call arguments as JSON` | **66 and counting** |

The causal chain, each step read from the log rather than inferred:

1. AutomationBench runs 50 model steps. Each step appends the assistant turn
   plus a tool result, so the conversation grows ~1,373 tokens per request.
2. At request 24 the prompt was 32,639 of 32,768 tokens. Only 129 tokens
   remained for generation.
3. The model began a tool call and was cut at the ceiling: `truncated = 1`.
4. A truncated tool call is invalid JSON, so llama-server answered HTTP 500 at
   a fixed parse offset (`column 89`, missing closing quote).
5. The harness retried **the same conversation**. Its prompt was unchanged and
   still at the ceiling, so every retry overflowed at the same byte. Backoff
   stretched the interval (0.5 s → 2 min) without changing the outcome.

Retries cannot recover: the request that overflows only grows. The run is
arithmetically dead and will spend its full 3,600 s task watchdog (until
~16:34) producing zero scorable episodes. The official evaluator still shows
`0/6`.

## Arithmetic that should have refused the launch

```
required = first_prompt + growth x (max_steps - 1) + generation_reserve
         = 1,067 + 1,372.7 x 49 + 2,048 = 70,378 tokens
shortfall against 32,768      = 37,610 tokens
affordable steps at 32,768    = 22 of 50
```

The frozen original geometry is worse: at the original `-c 16384` only **10 of
the 50 steps** fit. AutomationBench at 50 steps is structurally incompatible
with the frozen 16,384-token context at this task's tool-result verbosity — an
independent reason the original four-slot cohort could not complete these tasks.

## Implemented guard

[`chowder_batch/reap_context_budget_20261026.py`](../chowder_batch/reap_context_budget_20261026.py)
is a pure, stdlib-only admission check that parses a native server log and
refuses a launch whose episode cannot finish. Against the real log:

```
status            REFUSED           (exit 2)
first_prompt      1067
tokens/request    1372.7            observed_native_log
required_context  70378
shortfall         37610
affordable_steps  22
reasons           EPISODE_CANNOT_REACH_MAX_STEPS_IN_CONTEXT
                  TRUNCATION_GUARANTEED_BEFORE_MAX_STEPS
                  OBSERVED_TRUNCATED_TOOL_CALL_REJECTED_AS_INVALID_JSON
                  RETRY_CANNOT_RECOVER_UNCHANGED_OVERFLOW
```

17 tests in [`tests/test_reap_context_budget_20261026.py`](../tests/test_reap_context_budget_20261026.py)
cover the real trajectory mean, reclaim events that must not count as growth, a
frozen overflow that must be reported rather than raised, and the refusal/admit
boundary at 32k / 64k / 72k.

## Executable next step

GPU model buffer is 10,619.88 MiB with 13,031 MiB in use at 32k context, leaving
~3,021 MiB free; KV grows ~20 KiB/token.

| Launch | Context needed | Fits? | Extra KV vs 32k |
|---|---:|---|---:|
| `-c 16384`, 50 steps (frozen) | 70,378 | no — 10 steps only | — |
| `-c 32768`, 50 steps (this trial) | 70,378 | no — 22 steps only | — |
| **`-c 65536`, 45 steps** | 63,514 | **yes** | +320 MiB |
| `-c 73728`, 50 steps | 70,378 | yes | +800 MiB |

Recommended: run the corrected trial at `-c 65536` with `--max-steps 45`, which
fits with ~2.7 GiB GPU headroom, and pass the native log through
`reap_context_budget_20261026.py` before launching so an infeasible episode is
refused in milliseconds instead of after an hour.

This cannot start until the dead trial releases the GPU at its 3,600 s watchdog
(~16:34). Its receipts under
`F:\reap-v2-automationbench-private-20261026\q4-expctx-32768-six-v2\` document
the negative result and are preserved.

## What this does not change

No scientific result is affected. No Q4 benchmark score is produced or
implied here: this phase measures why one attempt produced none. The frozen
selector is untouched.
