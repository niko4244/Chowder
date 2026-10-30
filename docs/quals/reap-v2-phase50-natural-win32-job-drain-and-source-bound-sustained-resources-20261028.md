# Phase50 — genuine Windows Job lifetime safety and full1319 resource envelope

**OPEN, scientific BLOCKED, chosen:null, release_approved:false.**
Target: branch `feature/reap128v2-selection-20261025`, open **draft**
[niko4244/Chowder#282](https://github.com/niko4244/Chowder/pull/282).
This document supplements, does not rewrite or replace, original prereg,
frozen source files, Phase48/49 certificates, scorer or prior completed rows.
**Engineering validation of this source wave:** 588 passed, 1 skipped in
38.67s on the materially affected frozen-selector, original source-lock,
HumanEval certificate/intake, Windows Job supervisor/lineage, resource
diagnostics and scorer suites. Ruff and Git whitespace checks passed.
The committed-source portable archive test intentionally requires the
subsequent source commit before it can be verified; there is no new original
GPU model-run qualification claimed by these CPU and benign-Job tests.

## New material engineering and genuine Windows evidence

* **Independent receipt-join false positive closed.** The existing
  `reap_v2_supervisor_integration_guard_20261027.assess_bound_receipts`
  previously admitted a caller-constructed clean terminal although
  `fully_witnessed_native_fit`, full private-Job lineage, naturally extinct
  pre-cleanup children and complete cumulative Job process accounting were
  all false. The evaluator and its source SHA remained unchanged. The
  source-bound join now separately refuses nonzero exit, nonempty breaches,
  absent complete root/server/evaluator PID+birth roles, missing original
  successful-fit SHA and active native source observation, invalid native
  cumulative process totals or non-natural pre-cleanup extinction. The
  unchanged `JobObjectBasicAccountingInformation` lifetime counter catches
  missed short-lived helpers rather than silently excluding them.
  Adversarial RED/GREEN test cases exercise each critical contradiction.
  Runtime receipt consistency is **not** scientific approval.
* **Legitimate Windows console helper survives Python root exit.** Two
  actual benign private Win32 Job CPU probes found a System32
  `conhost.exe` child of original owner Python whose lifecycle persisted
  ~150ms **after the root exited 0**. Immediate Job kill + postkill
  extinction would create a real false refusal or false runtime proof.
  New `reap_v2_phase50_postexit_job_drain_20261028.py`, integrated into
  the actual owner supervisor, gives an already-verified original active
  root+native+evaluator Job **at most 40 × 50ms** read-only observations
  after exit. It allows only source-bound OS System32 console children,
  never post-root native server/evaluator; requires two stable empty Job
  censuses and `ActiveProcesses=0`, plus exact `TotalProcesses` equal
  to independently observed PID+birth identities. Natural exit proof occurs
  **before** private-Job cleanup. Timeout, foreign image, running evaluator,
  changed Job limit, identity race or missed birth REFUSES; cleanup only
  ever targets explicitly owned private Job handles. No fixed native
  serving flags, model tokens, or scores changed.
* **Kernel-derived helper identity.** The Job lineage guard obtains
  System32 from Win32 `GetSystemDirectoryW`, rather than a potentially
  user-overridden inherited `SystemRoot` or `PATH`. Original console,
  `nvidia-smi.exe` and frozen `powershell.exe` helpers are matched to
  their kernel-derived system executable paths and exact source-known
  arguments, parent PID, Job membership and birth; arbitrary same-named
  binaries REFUSE. A separate genuine bounded CPU probe confirmed
  `conhost.exe` actual native image/argv/parent and Job BasicAccounting.
* **Unseen short-lived helper remains an explicit blocker.** A further
  actual benign private Windows Job (Python launching a short-lived
  Python child) showed the cumulative `TotalProcesses=3` while a
  ~3-second live-PID sample saw only root+conhost (2). `NEW_PROCESS`
  completion-port event receipt plus immutable creation-time command
  attestation is still needed for exhaustive production provenance. Merely
  dropping the total-birth check would create a false PASS, so Phase50
  correctly preserves **UNKNOWN/REFUSED** if any natural helper exits
  between polls without full source-bound identity evidence.
* **More useful non-intrusive sampled memory telemetry** in the actual
  private-Job supervisor: bounded original PID+birth role working-set,
  peak-so-far, private-commit and page-fault observations now persist up to
  128 *raw, time-aligned* samples with simultaneous host RAM/commit/WDDM
  counters. Capture is attempted at the first active role witness, every
  32 samples, near RAM/commit floors, and immediately before a safety
  abort—not just after successful samples. Unavailable process counters
  stay UNKNOWN; no memory safety floor can be bypassed, no foreign process
  is inspected for termination. This does **not** measure hard pagefile I/O,
  model-exclusive CUDA peak or a full CPU lifetime peak on the current host.

## Actual historical native Q4 memory and KV root cause

Pinned source set: Phase49 resource diagnostics; original 12-row abort
SHA256 `836e4e886f11819075d75a461eec028f0453b9fd3d00e2c9553e0d3177dd81ed`,
original full1319 822-row watchdog
`72f9ab7fbbc345b4cfb4e19990a1a44f2ca1364156a71331f653ce66a7141641`,
original complete terminal
`fabd6e8381ab68f73473b8a57bd80a94c2d2cf675968b7527fa5a37795bb0697`.
Unchanged original source verifier reopens the complete immutable files.

The abort stopped at physical **6.98 GiB free** (registered 7 GiB
runtime reserve), while minimum commit headroom **10.41 GiB**
remained above its 8 GiB reserve. Host-wide free RAM fell 14.43 GiB,
commit 18.43 GiB. During startup, an 11.15 GiB commit-headroom drop
preceded a sharp 8.82 GiB RAM drop; original GPU free dropped from
13,993 MiB to 3,149 MiB as the CUDA model loaded. This is consistent
with CPU commit/residency and mmap/cache pressure during native model
loading, but overlapping protected applications preclude attributing
all host memory change to the model or asserting observed swap I/O.

An actual complete original four-slot Q4 startup uses **ONE** CUDA model
buffer **10,619.88 MiB**, **320 MiB** CUDA KV for 4 × 4096 contexts,
**251.25 MiB** recurrent and **104.05 MiB** CUDA compute buffers:
**11,295.18 MiB recorded buffer sum**, not four model replicas and not
an isolated measured VRAM lifetime peak. CPU-mapped source buffer
**397.85 MiB**, native 41/41 CUDA layers, zero MoE expert CPU spill,
source `common_fit_params` success. Frozen flags remain
`--fit on -np 4 -c 16384 --jinja`.
Experimental **one-slot/73,728 context** uses 1,440 MiB KV and
12,294.74 MiB reported CUDA buffers: **999.56 MiB greater** than
original four-slot buffer total, not an interchangeable frozen session.

| Pinned original Q4 historical host sequence | Actually observed |
|---|---:|
| First full1319 sample physical free | 39.087059 GiB |
| Minimum through 822 original full1319 samples | 11.278446 GiB |
| Maximum drop from first sample | 27.808613 GiB |
| First full1319 commit headroom | 64.811462 GiB |
| Minimum commit headroom | 31.294880 GiB |
| Maximum drop from first sample | 33.516582 GiB |
| Derived full1319 RAM prestart envelope (7 GiB reserve + 2 GiB margin + original host drop) | **36.808613 GiB** |
| Derived full1319 commit prestart envelope (8 GiB reserve + 2 GiB margin + original host drop) | **43.516582 GiB** |

These are **host-wide source-bound estimates**, not isolated model
allocations or permission to run. They supersede the earlier aborted
12-row-derived smaller 29.3146/34.3146 GiB *as sustained-run sizing*,
not as a change to any original registered startup/abort contract.
The already-registered **48 GiB genuinely available physical RAM,
48 GiB Windows commit headroom, 48 GiB-class CUDA GPU with ≥40 GiB
free VRAM** dedicated-host provisioning is stricter and remains unchanged.

The separate original one-slot 24.135s diagnostic sampled native
working-set maximum **22,162,624,512 bytes**, private commit
**23,464,726,528 bytes**, and 408,782 process page faults delta.
Those are **not** full1319 lifetime maxima or hard swap I/O.
Current shared RTX 5060 Ti 16 GiB has only ~61–121 MiB free during
the read-only Phase50 observations, and protected Ollama, Freebuff,
Hermes and Podman may not be disturbed. **No new original model was
launched** or unowned process terminated.

## Existing original scientific results and complete gate ledger

Total stored parameters **18,543,997,568**, ~**3,444,503,168** active
parameters per token with 128 stored/8 active experts per MoE layer.
Original 26 BF16 shards **37,981,266,120 bytes**; BF16 GGUF
**37,122,439,488 bytes**; Q8_0 **19,748,435,968 bytes**;
Q4_K_M **11,563,915,264 bytes**. Original dense Q4
**21,166,757,984**, original dense Q8 **36,903,139,424**,
random128 Q8 **19,748,435,872 bytes**.

Original authentic Q4 Windows GSM8K **1,224/1,319 (92.797574%)**,
1,281 natural terminations; Q4 first200 **189/200** and **40.7703
native predicted tok/s**, missing matched dense Q4 original same-host
first200; original Q4 and Q8 HumanEval research **151/164** versus
**152/164**, NOT the independently executed original Q8/dense Gate2.
Separate Linux Q4/Q8/random GSM8K **1229/1319**, **1220/1319**,
**964/1319** on incompatible platform/sessions. Historical
REAP128-v1 D0 and Gate0 **PASS**, Gate1 **FAIL** one-sided paired Wald
lower **−0.03780741197234255**, strict original >−0.03.
REAP160 native speed is **UNKNOWN**; original isolated BF16 layer
parity is not 40-layer whole-model parity.

| Required frozen decision | Status |
|---|---|
| REAP128_v2 BF16 full D0, whole-model CPU/CUDA parity, EOS/trigram Gate0 | UNKNOWN |
| Q8 vs original dense full1319 GSM8K Gate1, sample-SD one-sided 95% Wald LB >−0.03 | UNKNOWN |
| Q8 vs original dense genuinely isolated full164 HumanEval Gate2, LB >−0.10 | UNKNOWN |
| Original random128 Q8 source-paired full1319 control | UNKNOWN |
| Q4 candidate/dense first200 matched native same-host speed ≥1.3× | UNKNOWN |
| REAP160 full native/accuracy/speed gates | UNKNOWN |
| Amendment4 optional paired −0.02 McNemar-compatible 95% CI vs §4 Z1.645 sample-SD Wald | BLOCKED — prospective independent statistical adjudication |
| Original Gen-0 deterministic selector | **BLOCKED / chosen:null / release_approved:false** |

Originally registered eleven frontiers: HLE, TerminalBench4, DeepSWE,
ProgramBench, NL2Repo, AutomationBench, Agents' Last Exam, ApexBench
**INCOMPLETE / UNKNOWN official score**; Chartography, BabyVision,
ZeroBench **NOT NATIVE ELIGIBLE** for original text-only source.
Previous partial/heuristic research results in Phase49 remain separated
and cannot be pooled as official scores or security approvals.

## Executable next stage — **only on independently authorized capable host**

Use the source-bound Phase50 code-only portable bundle and CPython/ABI
verification. Recreate registered exact originals:
`G:\reap128-v2-q4_k_m-20261025.gguf`,
`I:\reap128-v2-q8_0-20261025.gguf`,
`G:\llm-cold\gguf\dense-q4_k_m.gguf`,
`G:\llm-cold\gguf\dense-q8_0.gguf`,
`G:\llm-cold\gguf\random128-q8_0.gguf`, and
`F:\llama.cpp\bin\llama-server.exe`. Preserve immutable original
model sidecar absolute paths, model/quantizer/source SHA256, original
HF offline ordered canonical 1319/164 sets and original frozen evaluator.
Run read-only `python -m chowder_batch.reap_v2_phase49_resource_diagnostics_20261028`
and `python -m chowder_batch.reap_v2_dedicated_runner_admission_20261027
--arm q4 --model G:\reap128-v2-q4_k_m-20261025.gguf
--server F:\llama.cpp\bin\llama-server.exe --port 8790
--out evidence/new-exclusive-preflight.json` on that host.
Only independently approved fresh RAM/commit/GPU/native model+KV fit,
no foreign GPU/owner/ports, complete original source lock and
authorized fresh private Win32 Job can initiate a new original arm.
No copied original rows from separate sessions, no altered `--fit on
-np 4 -c 16384 --jinja`, no changed prompt/decode/score/threshold.
Run candidate/dense Q8 full1319, random control and paired Q4 first200
on matching approved native hardware **serialized**, then independently
isolated and externally reviewed original Q8/dense full164 HumanEval,
original BF16 complete parity and REAP160; eligible registered
frontiers only through original harnesses. **Do not self-approve**
the source-controlled empty external execution/security approval maps.

Until a native Job event-complete source/command witness and independent
reviewer are available, a missing short-lived helper's identity still
correctly blocks a clean terminal; the bounded natural conhost drain is
NOT a way around that safeguard. No available approved host satisfies
the dedicated floor on this shared workstation, and no authentic
additional complete original gate evidence was generated this wave.
The original frozen selector must remain **BLOCKED** until its full
preregistered scientific inputs and prospective statistical decision exist.
