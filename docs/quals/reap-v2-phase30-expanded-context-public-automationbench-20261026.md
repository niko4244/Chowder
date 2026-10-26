# Phase 30: Real Q4 expanded-context AutomationBench public-task testing

**Date:** 2026-10-26. **Scope:** original REAP128_v2 Q4_K_M model weights;
isolated **experimental** serving profile only. Gen-0 qualification and
Amendment 4 original `--fit on -np 4 -c 16384 --jinja` remain unchanged,
`status:BLOCKED`, `chosen:null`, `release_approved:false`. Draft PR #282
remains unmerged. No private model bytes, completions, API worlds, or original
scientific receipts are committed.

## Protocol separation, independent resources and native admission

The exploratory model was the original GGUF
`G:\reap128-v2-q4_k_m-20261025.gguf`, SHA-256
`2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.
**Only** native concurrency/context was varied to:

`--fit on -np 1 -c 32768 --jinja`

The native startup log independently verified **1 sequence, 32,768 actual
context tokens, 41/41 CUDA layers, zero CPU expert offload**. This cannot
be mixed into the original four-way CUDA/throughput scientific gate, or
treated as a directly comparable published vendor leaderboard protocol.

Public AutomationBench **1.0.6**, official commit
`4a8e1061254004d9dac807054eed33fad7d1ff14`, 600-task source digest
`1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5`
and official Python3.13 API WorldState simulated-tool scorer were retained.
No model-generated code was executed in Windows or without a sandbox.

The separate experimental runner reused the original source-derived start
admission, required **512 MiB additional free VRAM** for enlarged CUDA KV,
and checked every measured runtime physical RAM/Windows commit/disk floor.
An initial preflight failed normally on short RAM and commit; an admitted
launch later measured **33.3156 GiB RAM, 39.6313 GiB commit, and 14,622 MiB
free VRAM** after reversible reclamation. It did not bypass a refusal.

Spark `llama-server` PID 72452 had zero active slots and was stopped only
after private process identity/command receipt. The `~/.hermes/supervisor.ps1`
PID 24280 was **temporarily suspended in memory**, preserving its identity,
to prevent it automatically reloading Ollama during the experiment. Ollama
master PID 74760 was unloaded, as were five verified Chrome renderers
(PIDs 34372, 55960, 59636, 54752, 78692). Chrome's master remained alive.
Previous browser tabs may require manual reload and unsaved in-tab state
cannot be restored by resuming a renderer. These actions had the user's
explicit authorization. Ledger and raw commands stay private:

`F:\reap-v2-automationbench-private-20261026\expanded-32768-reclamation-20261026-v1`

## Actual source-bound task outcomes

First multi-task startup required a native log verifier correction:
`llama.cpp` calls its real observed property `n_ctx_seq`, not
`n_ctx_per_seq`. The corrected parser and adversarial tests now prove native
one-slot context was actually 32768, not inferred solely from CLI flags.

One source-pinned six-task run then encountered repeated native HTTP 500
errors due to **malformed model-generated JSON tool-call arguments**;
the official evaluator retried exponentially instead of producing a
completed task. That private attempt was stopped with only its owned
process tree after reproducing the failure. **No score from it**. The
source-grounded fail-fast watchdog now detects at least five repeated
native JSON parsing failures and terminates that owned experimental task,
without repairing or fabricating model tool calls.

Separate **official single-task** cohorts then successfully completed:

| Source-pinned public task | Official completed score | Assertions | Model calls | API tool calls | Generated tokens |
|:--|--:|--:|--:|--:|--:|
| `sales.apply_project_label` | **0.0 (fail)** | 0/7 | 11 | 30 | 5,478 |
| `sales.mark_vip_emails_read` | **0.0 (fail)** | 0/6 | 29 | 52 | 8,061 |
| **Two completed tasks** | **0/2 passed** | **0/13** | **40** | **82** | **13,539** |

These are real measured **negative** scores, not seven formerly
context-aborted attempts falsely graded 0/7. Both source-bound exports
verified `aborted_tasks=[]`, source-known exact task ID, evaluator
`benchmark_version=1.0.6`, `toolset=api`, native original GGUF SHA,
actual single-slot 32768 native geometry and 41/41 CUDA offload.

Source receipts (raw private responses NOT on GitHub):

- `F:\reap-v2-automationbench-private-20261026\q4-expctx-32768-apply-label-v1`
  official JSON SHA-256:
  `8c986f812a24038b0ddec69c98c718587c1b85185dc56186cd82f28ff38665bf`.
  27 live resource observations; minimum available physical RAM **21.2758
  GiB**, available commit **26.3375 GiB**, free VRAM **3,031 MiB**.
- `F:\reap-v2-automationbench-private-20261026\q4-expctx-32768-vip-email-v1`
  official JSON SHA-256:
  `262634deaf24772521e051a0d82a7947d201ef5112f5b5cec556c3930909b6ba`.
  41 live resource observations; minimum RAM **20.2159 GiB**, available
  commit **24.5692 GiB**, free VRAM **3,014 MiB**.

**Zero** sampled original runtime-floor violations in either complete run.
These are *two chosen public benchmark tasks*, so 0/2 is not a statistically
valid estimate of the full **600-task** AutomationBench score, nor of
the harder/private AutomationBench leaderboard.

An attempt to start a third isolated task (`sales.create_contact_for_account`)
lost its command session **before a result directory or owned native server
was created**. It is not counted as an attempt, score or failure. No duplicate
model run was launched without verifying that the process had exited.

## Restoration and remaining work

The owned experimental native server and evaluator were confirmed shut
down after the two completed cohorts. Spark was restarted with its original
captured command; new PID **12624** initially bound original port **8767**,
but **did not remain resident** at later follow-up checks. The exact Hermes
supervisor PID **24280** was resumed with `NtResumeProcess`, not replaced;
it restarted Ollama (new PID **91052** on port 11434, `HTTP 200`). At the
latest 20:57 UTC observation, Spark port 8767 and Hermes router port 8766
were **not listening**. Their full restoration is **NOT VERIFIED**; do not
represent the earlier transient Spark listener as lasting recovery.

Importantly, a **different concurrently active** Q4 experimental runner,
`reap_q4_automationbench_ctx64k_20261026`, parent PID **76808**, owned
native model PID **73808** on port **8892** with context **65536** was
independently observed. It had ~15.4 GiB private commit residency and
used scarce RAM/VRAM. That scientific work is a **protected independent
job**. We did not touch its processes, files, resources or scores; forcing
Spark/router model loads before its safe completion could compromise it.
Thus their restoration must be reconciled with that job's owner/availability,
and remains an explicit unresolved operational blocker.

An individual browser renderer cannot be restored exactly; reload
affected Chrome tabs if needed. The original Chrome browser master remained
running; original process commands are saved privately for later repair.

The original 11-frontier-category requests retain **no official REAP Q4
score** other than the very limited public AutomationBench two-task
experimental observation. The HLE Kaggle job was separately running at
last checked status; no final HLE score was reviewed here. Remaining
benchmark harnesses and sandbox/CI gates, prospective noninferiority,
Q8/Q4 numerical parity, REAP160 and dense first-200 comparison are not
certified by these exploratory samples. Gen-0 remains **BLOCKED**, chosen
null; never merge PR #282 or initiate Gen-1 from these data.
