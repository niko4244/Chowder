# REAP128_v2 phase 28 — Q4 execution gate and original item-level witness

**Date:** 2026-10-26. **Scope:** experimental branch only; original scientific
selection stays `{ "status":"BLOCKED", "chosen":null, "release_approved":false }`.
Draft PR #282 is not merged. Existing original immutable models, sessions,
score files and protected user processes were not edited.

## Actual inference eligibility, not simulated admissions

Two *real* Kaggle GPU workloads were observed as `KernelWorkerStatus.RUNNING`
through the authenticated Kaggle CLI on this phase:
`nikmarco/reap128v2-random128-full-postload` (protected original control)
and `nikmarco/reap128v2-q4-hle-textonly-20261026` (independent Q4 HLE research).
No task output of the latter had been published to the API during inspection,
and it is **not** an official HLE without-tools result until exact task source,
judge, scope and completed output are independently audited. No competing GPU
job was submitted, replaced, cancelled or modified.

The Q4 native Windows first-start admission, using unchanged original
source-derived floors, returned `REFUSED` on this phase. A representative
sample, 2026-10-26 19:12:14 UTC:

| Metric | Observed available | Pinned minimum | Result |
|:--|--:|--:|:--|
| Physical RAM GiB | 14.31515884 | 29.31459863 | REFUSED |
| Windows commit GiB | 11.50782394 | 34.31459863 | REFUSED |
| VRAM MiB | 538 | 12,288 | REFUSED |
| Destination free disk GiB | 35.15152740 | 8 | PASS |

The GPU resource is actively used by separate user model services; WDDM
sampling fluctuated widely even within this phase. The prior independently
measured *optimistic upper bound* on reclaiming Spark plus one Ollama child
still left physical RAM and commit below the frozen minima. The only way to
reach the remaining deficit by terminating Windows Terminal, Codex,
Hermes/Brainz, WSL or other active developer processes would violate the
protection contract. Therefore **no model service, registered runtime,
watchdog, interactive agent, Kaggle job or other protected process was
stopped**. No restoration/restart was necessary. Ports 11434, 8766, and
8767 remained listening; no REAP Q4 private listener was launched.

This is a fresh resource refusal, **not** a claim that the model failed a
quality benchmark. No release thresholds were reduced to produce a PASS.

## New source-bound original item-level evidence (already completed inference)

The original Linux sessions were independently reread from the exact private
row files; no fresh expensive generation or unsafe HumanEval program execution
occurred. This phase created an **exclusive, private** item-level proof:

`F:\reap-v2-automationbench-private-20261026\original-q8-q4-recount-wave-20261026-v1`

Its CSV contains one matched row for each of the 1,483 original tasks, with
original `correct`, `terminated`, and generated token count bits for Q8 and
Q4 side by side, but no completion text, prompts, code or credentials.
The item-level CSV SHA-256 is
`ab2144ff1a7239751066efd52fc37cbc409b6c98d8002b8090f667832619c752`.

| Original unchanged population | Q8_0 | Q4_K_M |
|:--|--:|--:|
| GSM8K correct | 1220/1319 | 1229/1319 |
| GSM8K terminated | 1280/1319 | 1281/1319 |
| GSM8K completion tokens | 467,560 | 456,671 |
| GSM8K generated tokens/wall-second | 81.6421 | 98.4385 |
| HumanEval original scored correct | 152/164 | 151/164 |
| HumanEval terminated | 162/164 | 162/164 |
| HumanEval completion tokens | 40,281 | 40,803 |
| HumanEval generated tokens/wall-second | 74.7642 | 89.2712 |

The GSM8K paired original reconstruction produced exactly 1198 both-correct,
68 both-wrong, 22 Q8-only correct and 31 Q4-only correct, without selecting a
new statistical CI method. The original HumanEval **independent sandbox v2**
and original Windows native gate remain *NOT PROVEN*. These historical Linux
percentages are **not** HLE, AutomationBench, or same-host dense/Q4 speed.

## Substantive Windows public AutomationBench runner changes

The `chowder_batch/reap_q4_automationbench.py` runner now writes
`resource-samples.private.jsonl` after an admitted run starts: every sampled
native-load/evaluation measurement is append-only, fsynced, time/phase/PID
tagged and records measured physical RAM, Windows commit, disk, GPU free and
any original-floor failures. Invalid or nonfinite samples fail closed; none
may be silently promoted into a result. This recording is entirely in the
exclusive private output folder, never committed or uploaded to GitHub.

For the installed official `automation-bench==1.0.6`, the frozen public
600-task source/ID/prompt digest passed earlier. The unchanged official
CLI in this phase also passed a fresh isolated Python3.13 `-I -m ... --help`
check. The runner now uses isolated Python `-I` for source inspection and
official evaluation to reduce unintended user-package/shadow imports,
rejects checkout-local `.env` and sitecustomize shadows even if Git ignores
them, and passes local dummy API authentication only to its loopback model.
The official WorldState API tool adapter and strict binary scorer are still
owned by the upstream evaluator; no bespoke replacement metric is used.

Focused RED/GREEN+original item-suite tests: **44 PASS**; source Ruff clean,
plus previous official chart source validation **11 PASS**. This does not
constitute official inference or an admitted native CUDA session.

## Reproducible command and restoration

From the experimental worktree, a read-only admission:

```powershell
python -m chowder_batch.reap_q4_automationbench `
  --out F:\reap-v2-automationbench-private-20261026\new-q4-task `
  --count 1
```

Only on a host that has *freshly* passed original Q4 RAM/commit/VRAM/startup
admission, `--execute` starts an owned model server on port 8891 with immutable
`--fit on -np 4 -c 16384 --jinja` and invokes the source-pinned official
1.0.6 evaluator. Samples must be reviewed before accepting any score. A
600-task public score requires 600 distinct source-scoped tasks and a separate
full-row scorer audit. Official **private leaderboard equivalence must not be
claimed** from public task output.

If independently verified idle model services become genuinely reclaimable,
record the exact original PID, hash/executable, command line, creation time,
ports, listeners, `/slots` and clients, supervisor/watcher and config SHA;
restore from the captured command under the same ownership and check all
previously healthy endpoints, slots and configuration. Never disable an
active watchdog, stop a service required by Hermes/Brainz, or reclaim memory
from unrelated developer sessions merely to manufacture a benchmark PASS.

The original Gen-0 Gate0, Gate1, Gate2, BF16/CUDA identity, dense-first200
matched speed, REAP160/random128 control, Q8/Q4 numerical parity and
prospective noninferiority method retain their original separate statuses.
No newly obtained item-level data is used to change Amendment 4 statistical
methods, thresholds or historical REAP128-v1 Gate1 FAIL.
