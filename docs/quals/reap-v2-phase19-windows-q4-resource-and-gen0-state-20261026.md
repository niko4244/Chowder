# REAP128_v2 phase 19 — independently source-bound original Windows Q4 resources

**2026-10-26; PUBLIC-SAFE measurement report; PR #282 stays OPEN/DRAFT/UNMERGED.**
This is a substantive read-only verification of *already completed* original
Windows Q4 candidate evidence, accompanied by new fail-closed watchdog
hardening. It is **not** a fresh model run, source replacement, dense speed
certification, actual instantaneous GPU allocator peak, or Gen-0 approval.

## Original evidence identities and replay

The original Windows Q4 first-200 private run and its native source-bound
certificate are immutable. Session:
`e3f9c268b2bb4d958cc55648c0c4c8e0`. Exact Q4 GGUF SHA:
`2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.
Native flags `--fit on -np 4 -c 16384 --jinja`; all 41/41 model layers
GPU-offloaded, **zero routed experts** offloaded to CPU. Exactly 200 of the
original 1,319 GSM8K items recorded in a single exclusive session,
187/200 correct (93.5%), 192/200 naturally terminated, 73,224 generated
tokens / 398 seconds rounded native scoring wall time = **approximately
183.98 generated tokens/s**. This numerator/denominator is a scoring wall
measurement and not decoded-token kernel-only speed.

The new CPU-only verifier
`chowder_batch/reap_q4_windows_sourcebound_resource_20261026.py`
rehashes the **exact original** first-200 JSONL, serving receipt, 148
resource samples, original native server log and terminal watchdog receipt;
rejects any altered bytes, malformed duplicate/nonfinite source JSON,
source population/ID changes, stale session, wrong model/flags/offload,
missing speed proof, impossible native placement/KV/cache and violated or
gapped RAM/commit/disk/VRAM admission data. It emits an exclusive private
**aggregate-only** receipt, not prompts or generated responses. The newly
produced private audit SHA-256 is
`09fab65b63c1a08265472cca59e2e5645bd8fa00b73137de9aff20315971a7a6`.
It keeps `gen0_status=BLOCKED`, `chosen=null`, `release_approved=false`.

## Real resource, allocator and four-way geometry observations

| Source-bound observation | Original Windows Q4 candidate |
|---|---:|
| 148 watchdog samples, longest interval | 3.086361 s |
| Initial physical available RAM | 31.108 GiB |
| Minimum sampled physical available | **10.064 GiB** |
| Largest sampled decline from start in physical availability | **21.043 GiB** |
| Initial Windows available commit | 38.106 GiB |
| Minimum sampled available commit | **14.724 GiB** |
| Largest sampled decline from start in available commit | **23.382 GiB** |
| Initial free RTX 5060 Ti VRAM | 14,346 MiB |
| Lowest sampled free VRAM | **2,859 MiB** |
| Largest sampled decline from start in free VRAM | **11,487 MiB** |
| Native CUDA model buffer | **10,619.88 MiB** |
| Native CUDA KV buffer | **320.00 MiB** |
| Native prompt-cache maximum observed | **8,150.615 MiB** |
| Original native prompt-cache limit | 8,192 MiB |
| Native serving context | 16,384 total, 4 parallel × 4,096 tokens |
| Native model layer placement | 41/41 on CUDA, 0 CPU experts |
| Minimum sampled disk-free capacity | 35.782 GiB |

**Interpretation:** sample minima and differences are *host-wide available*
memory/commit/VRAM measurements, not instantaneous per-process allocation
peaks or attribution of each byte to model buffers. Model + KV recorded
buffer sizes must not be added mechanically to WDDM counters or a host commit
delta as though they are disjoint. The 8-GiB prompt cache was genuinely
near its configured cap (203 native cache observations; up to 41 cached
prompts). This helps explain why memory headroom changed during sustained
four-slot serving, but does not prove how much was WDDM pageable GPU,
anonymous CPU allocation, OS file cache or copy-on-write. Original CUDA
allocator peaks, instantaneous native RAM/VRAM peaks, dynamic CUDA
shared-library dependency identity and forward/logit parity remain **UNKNOWN**.

**Never repeat this candidate benchmark just to obtain the same score.**
The previously successful native session satisfies the original candidate-only
first-200 source/sealing requirement. The full uninterrupted original Windows
Q4 **1,319-item** suite is still absent; do not splice this first-200 with
Linux or another serving session to claim full Windows quality.

## Concrete runtime/source safety improvements

Two pre-existing benchmark-supervisor fail-open conditions were verified by
adversarial RED/GREEN:

1. `NaN`, `Infinity`, invalid types and missing sample fields could bypass
   `available < floor` checks, or make strict JSON serialization fail
   before durable terminal evidence. The new watchdog explicitly refuses
   all nonfinite/missing/invalid RAM, commit, VRAM and disk telemetry,
   serializes a finite error-only abort sample, and retains the exact
   original 7-GiB RAM, 8-GiB commit and 8-GiB disk runtime floors.
2. Wrapper exit 0 + 200 rows + a speed dictionary did not independently
   demonstrate the owned native server died: the child could remain resident
   or another listener could occupy the original port. The revised
   **new future-run** watchdog checks native PID disappearance and absence
   of *any* port 8790 listener, without killing unrelated processes. It
   refuses `COMPLETE` if either remains or a process census fails.
   It preserves the successful historical receipt without retroactive edits.

New synthetic adversarial tests also reject changed source IDs, timing
gaps/duplicates, wrong flags, cache/KV/placement inconsistency, corrupted
bytes, malformed scores and false speed certificates. No models, generated
code, GPU allocations, protected services or original evaluation/scoring
source were touched by these tests.

## Scientifically complete evidence vs mandatory missing controls

- **Weights/geometry:** 40 layers, 128 routed experts/layer, top-8 routed;
  total stored weight elements **18,543,997,568**, non-routed
  **2,437,870,208**, all routed **16,106,127,360**, selected routed
  **1,006,632,960** per token, approx. **3,444,503,168** active weight
  elements/token (**18.575%** of stored). This is *weight counting*, not a
  measured FLOP count. There are 320 routed expert selections over 40
  layers/token; non-routed operations and expert arithmetic are not
  represented by the raw active-weight count.
- **Model sizes (verified filesystem byte counts, prior full hashes):** BF16
  **37,122,439,488** bytes; Q8_0 **19,748,435,968**; Q4_K_M
  **11,563,915,264**. Dense Q4 **21,166,757,984**; independent
  random128 Q8 **19,748,435,872** bytes. Reuse historical frozen
  full-file hashes, do not needlessly rehash 100+ GiB of protected GGUFs.
- **Original continuous Linux Q4 research full cohort:** GSM8K
  **1,229/1,319** correct, HumanEval **151/164** original scorer
  correct; 98.4385 GSM8K and 89.2712 HumanEval generated tokens/s.
  Historical independently re-executed Q4 164 program outcomes matched,
  but full network/namespace isolation was not independently established.
- **Original continuous Linux Q8 research full cohort:** GSM8K
  **1,220/1,319**, HumanEval **152/164 original scorer correct**,
  81.6421 / 74.7642 generated tokens/s, respectively; original full
  1,319 source-bound paired GSM8K results had 1,198 both correct, 68 both
  wrong, 22 Q8-only correct, 31 Q4-only correct:
  **Q8−Q4 = −0.6823 percentage points**. Not an original Windows
  paired noninferiority PASS or a separate isolated Q8 HumanEval v2 replay.
- **Original independent random128 full experiment:** still
  `KernelWorkerStatus.RUNNING` at this reporting checkpoint. Its
  partial/checkpoint or historical negative-control bytes are **not** a
  fresh independent full quality/control PASS.
- **Dense first-200:** original attempt safely aborted with only
  29.444 GiB unused cgroup before model hashing/load and 9.992 GiB at
  abort; never became a complete 200-item source-bound serving session.
  The new unlaunched dense runner requires at least **33.452 GiB
  genuinely unused cgroup before the first 21.17-GB read**, with more
  (preferably >=48 GiB), qualified host/GPU/free-disk and dynamic CUDA
  library identity. A 4-GiB Podman VM or currently occupied 16-GiB 5060
  Ti is not an eligible substitute. No candidate/dense **>=1.3x** claim.
- **Q8 independent HumanEval:** pinned source/canonical original 164 and
  strict audit/preflight pass without executing generated code. Rootless
  768-MiB Podman exact-profile harmless census and one fixed-benign
  parent-plus-child timeout probe were executed. The latter confirmed
  a disposable container was absent after its native timeout, but did
  **not** independently demonstrate host child PID/cgroup extinction,
  exact original Python 3.13 execution, or safe handling of all malicious
  programs. The new generated-code replay is therefore intentionally
  **disabled** and independent Q8 correctness remains pending.
- **Statistical governance:** original preregistered §4 paired Wald
  method and optional Amendment 4 nonzero-margin McNemar-style CI wording
  conflict. No post-outcome method shopping/automatic optional CI
  supersession or fabricated statistical authorization. Original
  REAP128-v1 Gate1 **FAIL**, canonical REAP160 speed **UNKNOWN**.

| Original qualification gate | Verified status |
|---|---|
| Q8/Q4 original GGUF integrity/topology | Certified construction identities; forward parity UNKNOWN |
| Continuous original Linux Q4 and Q8 full cohorts | COMPLETE as research, not original Windows Gate1/Gate2 PASS |
| Windows Q4 same-session first-200 candidate speed | COMPLETE candidate-only |
| Original full Windows Q4 GSM8K/HumanEval | UNKNOWN |
| Mandatory matched dense first-200 and sustained >=1.3x | UNKNOWN |
| Fresh complete random128 negative-control pairing | UNKNOWN, original full kernel running |
| Original REAP160 canonical throughput | UNKNOWN |
| Independently qualified all-164 Q8 HumanEval isolated replay | UNKNOWN, generated execution disabled |
| BF16 CUDA/logit and Windows/Linux numerical parity | UNKNOWN |
| Original full Gate0/Gate1/Gate2 and paired noninferiority | UNKNOWN/incomplete |
| Historical REAP128-v1 Gate1 | **FAIL** (preserved) |
| Deterministic Amendment 4 Gen-0 selector | **BLOCKED**; chosen null; release_approved false |

**Next executable engineering action:** independently verify the already
running original random128 kernel's terminal state. If and only if COMPLETE,
recover its native source/rows/telemetry/hashes into **new exclusive private
storage** and independently re-audit its 1,319+164 identities and paired
control; do not relaunch. In parallel, qualify a *separate* dedicated
large-memory dense host (real >=33.452 GiB unused cgroup, preferably >=48 GiB,
appropriate CUDA memory/libraries) before starting the unchanged original
dense first-200 workload. Qualify an actual isolated and reproducible
Python-3.13 HumanEval v2 execution runtime and trusted per-task attestation
before lifting the source-enforced generated-code execution block.

No protected services or GGUFs were changed, no original model workload was
rerun, and PR #282 must remain OPEN/DRAFT/UNMERGED. Do not merge, select,
release, deploy or start Gen-1 until every mandatory original gate is
genuinely evidenced.

## Phase 20 follow-up — full original Q4/Q8 independent source rechecks

Without loading models or executing generated Python, the unchanged strict
original-source external auditor was rerun against the original complete
Q4 and Q8 machine receipts, all 1,319 original GSM8K and 164 original
HumanEval rows per arm, native 41/41 GPU startup/session logs, frozen source
population, scorer and decoding witnesses. Both original continuous serving
sessions passed; both retained
`human_eval_independent_execution=PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION`
because the historical optional six-field replay PASS has been correctly
rejected as insufficient isolated v2 provenance.

The new exclusive private **Q4 full source audit** receipt SHA-256 is
`582e57557c40abad170592db1718f5aea3e61f8d3dae2db2e71a73c99ee19ea6`.
The new exclusive **Q8 full strict source audit** SHA-256 is
`107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683`,
the same trusted source-witness SHA as the previously certified exact
phase5 Q8 output. The complete 1,319-item GSM8K paired descriptive
consumer independently reprocessed the new audits and original exact
rows: both right **1,198**, both wrong **68**, Q8-only **22**, Q4-only **31**;
Q8−Q4 **−0.0068233510**. Its exclusive private SHA is
`9ee3943012ce2632082b452e4ab855c5aab315a7bdd92798ed04d1dd078e88a0`.
It explicitly leaves `paired_confidence_interval=null`,
`amendment4_eligibility=false` and
`statistical_method=NONE_DESCRIPTIVE_ONLY_UNRESOLVED_MCNEMAR_VS_WALD`.
No full HumanEval paired v2 PASS was fabricated or inferred.

A separate fixed harmless rootless OCI parent-plus-child timeout probe
(exact owned 768-MiB profile, no generated model source) actually
reproduced one disposable container timeout and observed it absent after
`--rm`; private receipt SHA
`068a6cae5ca45708dd305125d9190c18c486ac64b8a781b57813a8284355da0b`.
The proof expressly does **not** establish independent extinction of
all host-visible PIDs/cgroup, external threat-resistant sandbox
qualification or original Python 3.13 parity, and never unlocks the
source-disabled HumanEval replay. The original random128 full Kaggle
kernel continued to report RUNNING, not COMPLETE, after this recheck.
