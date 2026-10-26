# REAP128_v2 Q4: live Windows capacity reclamation and actual inference

**Date:** 2026-10-26. **Branch:** `experiment/reap128v2-test-unblocked-20261026`.

## Decision and scope

The user explicitly authorized suspending nonessential workloads to obtain
original Q4 inference. We performed reversible targeted workload reclamation,
**not** a machine reboot or unqualified admission override. The original
BF16/Q8_0/Q4_K_M scientific gates and Amendment 4 thresholds are unchanged;
Gen-0 is `{ "status": "BLOCKED", "chosen": null, "release_approved": false }`.
No original GGUF, frozen canonical prompts/scorer, scientific receipt, remote
Kaggle HLE/random128 job, or draft PR #282 was changed or cancelled.

## Minimum safe Q4 launch resources: independently reconciled

The original aborted four-way Windows speed run measured 14.43 GiB available
RAM loss and 18.43 GiB available commit loss. Its frozen source-grounded
startup admission additionally includes 5.8846 GiB possible prompt-cache
growth, 2 GiB safety headroom, and mandatory runtime floors of 7 GiB
available RAM and 8 GiB available commit. Existing frozen admission:

| Metric | Frozen minimum available to start |
|:--|--:|
| Physical RAM | 29.3146 GiB |
| Windows commit | 34.3146 GiB |
| Free GPU VRAM | 12,288 MiB |
| Destination free disk | 8 GiB |

The **already completed and independently source-audited** native Windows
Q4 first-200 run provides a useful second observation: initial available
RAM 31.1077 GiB fell to 10.0643 GiB (21.0434 GiB decrease); available
Windows commit 38.1057 GiB fell to 14.7239 GiB (23.3818 GiB decrease).
The simple empirical observed-drop-plus-original-floor-plus-2-GiB margin
is therefore **30.0434 GiB RAM / 33.3818 GiB commit**. The higher of this
independent RAM cross-check and frozen gate is ~30.04 GiB RAM; the higher
commit gate remains **34.3146 GiB**. These numbers are conservative
observed-workload launch budgets, not a new inference of the absolute
theoretical model minimum or permission to change frozen admission.

Source-bound original CUDA geometry remains 41/41 GPU-offloaded layers,
four simultaneous 4,096-token sequences (`-np 4 -c 16384`), 320 MiB CUDA
KV cache, and zero CPU expert offload. Original GGUF SHA-256:
`2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.

## Reclamation and measured admission

We first stopped the 4-slot idle Spark `llama-server.exe` PID **49124**
only after zero active `/slots` requests and zero established clients,
capturing the exact original command, executable and identity privately.
Observed available resources after that targeted unload were 19.3084 GiB
RAM, 21.8523 GiB commit and 14,037 MiB free VRAM. We unloaded the currently
resident Ollama embedding model, later preventing supervisor-triggered
Ollama reloading by temporarily suspending only the identified
`~/.hermes/supervisor.ps1` loop PID **24280** and stopping its owned
Ollama child. The original scheduled `OllamaWatchdog` task was already
**Disabled**; we did not alter its setting.

Browser Chrome had **99 processes** with about 26.37 GiB private commit and
20.73 GiB working set. Three identity-verified batches terminated **25
browser renderer processes**, deliberately leaving Chrome's main process,
the remote-control application and all system processes alive. Terminated
tabs may need a normal reload when revisited; closing a renderer is not the
same as suspending and resuming its exact in-memory state. These actions
were performed under the user's explicit aggressive-reclamation permission.

Initial admission passed after the first two renderer batches at
29.9712 GiB available RAM, 35.3426 GiB commit and 14,498 MiB free VRAM.
An intervening test attempt was correctly **REFUSED**: Hermes had
automatically restarted Ollama, reducing VRAM and commit before launch.
After identifying/suspending the real supervisor and two further renderer
unloads, the final source-bound admission passed at **30.9065 GiB RAM,
36.8292 GiB commit, 14,612 MiB free VRAM**, with the same frozen values.
We did not kill Windows Terminal, Codex, WSL, Hermes/Brainz endpoints or
remote active scientific jobs because measured admission was attainable
without taking those additional destructive steps.

All restoration source commands and process IDs are retained **off GitHub**:

- `F:\reap-v2-automationbench-private-20261026\spark-reversible-unload-20261026-1\original-private-restore.json`
- `F:\reap-v2-automationbench-private-20261026\ollama-reversible-unload-20261026-1\restore.private.json`
- `F:\reap-v2-automationbench-private-20261026\ollama-master-restoration.private.json`
- `F:\reap-v2-automationbench-private-20261026\hermes-supervisor-paused-20261026.private.json`
- Three private Chrome renderer batch receipts in the same root.

## Genuine Q4 official public AutomationBench 1.0.6 inference

The exact upstream public benchmark Git commit is
`4a8e1061254004d9dac807054eed33fad7d1ff14`, with all 600 tasks and
frozen prompt source digest
`1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5`.
We launched the **original** GGUF with the **unmodified** registered native
`--fit on -np 4 -c 16384 --jinja`, verified native 41/41 CUDA placement
and zero CPU expert offload, and used the **official** Python3.13 evaluator,
WorldState API toolset, scorer and task IDs. The model performed genuine
inference; all raw task contents remain private.

| Experiment | Genuine Q4 model calls | Official API tool calls | Q4 output tokens | Completed scored tasks | Context-aborted unscored tasks |
|:--|--:|--:|--:|--:|--:|
| First public task | 1 | 5 | 430 | 0 | 1 |
| Prospectively named short-prompt six-task slice | 15 | 30 | 3,046 | 0 | 6 |
| **Total** | **16** | **35** | **3,476** | **0** | **7** |

The first task hit a 4,776-token next-turn request, exceeding native
`n_ctx_seq=4096`. All six source-pinned short-prompt tasks also aborted
with follow-up prompt requirements of **4,174–5,658 tokens**, despite
five to seven real tool calls on most tasks. The official raw evaluator
export reported zero passes, but because **all seven trajectories aborted
before completion**, **0/7 is NOT a reportable completed-task accuracy**.
The new independent CPU-only triage also returned
`ALL_OFFICIAL_ROLLOUTS_ABORTED_UNSCORED`, with `completed_tasks=0`.
Never use either partial six-task score or the vendor's noncomparable
private leaderboard result as an official 600-task comparable metric.

Source raw result hashes: first task
`cea46fcd01a3587ed1fdab2b596cfb22ac359946aaee6570c4dc286af8608ec8`,
six-task slice
`19c904212041d568fde82ea3a113d3f9a68f53e9a595c821e6c295a6e7d3a7c7`.
The six-task native experiment recorded 20 resource samples, with minimum
17.5738 GiB available RAM, 21.0687 GiB available commit, and 3,022 MiB
free GPU VRAM; **zero original runtime-floor violations**.

The observed barrier is **per-sequence context**, not physical memory.
Expanding `-c` or reducing `-np` would alter the frozen original serving
geometry; it may be studied in a new explicitly marked exploratory profile,
but such scores **cannot be substituted for unchanged registered-arm
Amendment 4 results**. The source or scorer was never patched to truncate
tool outputs merely to manufacture a pass.

## Additional completed registered Q4 Windows GSM8K run

A new separate first-16 frozen-source GSM8K experiment completed without
runtime aborts. Its rows and summary were rechecked independently:

| Property | Actual original-Q4 measurement |
|:--|--:|
| Canonical original items | **16 distinct / 16 requested** |
| Correct | **15 / 16 = 93.75%** |
| Naturally terminated | **16 / 16 = 100%** |
| Generated completion tokens | **6,550** |
| Native serving | `--fit on -np 4 -c 16384 --jinja` |
| Original source population SHA | `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89` |
| Single native serving session | `aeb9f9022ad745d9b27c2edd081119ec` |
| Raw row SHA | `ffa55aab131ea27cf961c2d0da67448573783d89783f55b1039f0bd6ab3e0856` |

These are **genuinely generated and scored** results on the custom model,
but a short descriptive repeat, not any of the eleven requested frontier
benchmarks, a matched dense comparison or a fresh full-population gate.
The prior immutable full original Linux Q4 research populations remain
1,229/1,319 GSM8K and 151/164 originally scored HumanEval; HumanEval
independent execution sandbox attestation is still pending.

## Actual restoration and health

After the Q4 native runners shut down and released port 8891, Spark was
restarted using the saved original command as PID **72452**, with port
8767 responding **HTTP 200** and all four slots idle. The original Hermes
supervisor PID **24280** was explicitly `NtResumeProcess`-resumed rather
than replaced; it automatically restored Ollama. The live Ollama server
was PID **74760** on port 11434, **HTTP 200**. We issued a fixed benign
embedding health probe, which reloaded original
`nomic-embed-text:latest` successfully. Hermes/router port 8766 also
returned **HTTP 200**. Baseline watchdog scheduling was left unchanged.

Browser master survived all renderer shutdowns; **individual crashed
Chrome tabs may need user reload**. We cannot claim their previous
unsaved in-memory JS/tab state was perfectly restored. No original model
weight, source dataset, original gate receipt, or protected external
scientific job was changed.

## Reproduction and release disposition

From the authorized experimental worktree, perform read-only admission
before any new run:

```powershell
python -m chowder_batch.reap_q4_automationbench --out F:\reap-v2-automationbench-private-20261026\fresh-run --count 6
```

An actual six-task execution adds `--execute` **only** after the measured
fresh gate succeeds and the supervisor is safely isolated. The frozen
registered Windows GSM8K smoke requires the independent admission and a
new exclusive path:

```powershell
python -m chowder_batch.reap_v2_lab --execute --limit 16 --out F:\reap-v2-automationbench-private-20261026\fresh-gsm8k16
```

All private prompts, completions, benchmark JSONL and native logs remain
on the user's F: private research volume; only code, tests, sourced
aggregate summaries, and source SHA identities may enter the experimental
Git branch. The existing original gate evidence and historical REAP128-v1
Gate1 FAIL are not overridden. **Gen-0 remains BLOCKED, chosen null, and
release_approved false.** No draft PR merge, deployment, or Gen-1 training.
