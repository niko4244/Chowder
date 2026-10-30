# Phase47 — Original REAP128_v2 frozen runtime execution and complete source readiness

**Campaign:** OPEN, unmerged, no deployment or Gen-1. **Authorized branch:**
`feature/reap128v2-selection-20261025` / draft [Chowder PR #282](https://github.com/niko4244/Chowder/pull/282).
**Source investigation:** 2026-10-27 CDT. Read alongside the complete
[Phase46 frozen gate and eleven-frontier register](reap-v2-phase46-original-frozen-wave-and-real-resource-admission-20261027.md).
The prior Phase46 implementation stopped at a **capacity-only probe** even on
valid future hardware. Phase47 substantively implements an owner-bound,
continuously guarded, explicit execution path, an independently replayed
original frozen offline benchmark source lock, a separately source-admitted
original HumanEval164 capture/execution-witness protocol, and a reproducible
source-only dedicated-host bundle. **No GPU inference or paid remote
infrastructure was admitted on this workstation.**

## Completed actual work and what was proven

1. **Original frozen source actually replayed offline.**
   `chowder_batch/reap_v2_original_offline_source_lock_20261027.py` loaded
   the original production `reap_eval._load`, **without network access**, from
   locally available HF cache. Full canonical ordered prompts, answers and
   164 executable HumanEval tests matched the preregistered rendered SHA:
   - GSM8K test **1,319/1,319**, source SHA
     `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`;
     exact frozen first200-subset SHA
     `60ba9e741b73fe6cbbcbd33047fc74ffc944a4c6c4d59fa4633cf5d862a87f47`;
     first and last IDs `gsm8k-0` through `gsm8k-1318`.
   - HumanEval test **164/164**, source SHA
     `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420`;
     IDs `HumanEval/0` through `HumanEval/163`. This does **not** run generated
     model Python, score a candidate, or qualify Gate2.
   - The exact source verifier/scorer implementation `reap_eval.py` SHA
     `9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1`
     is locked. Original receipt SHA256:
     `7f990d22519c5a0e18e69c4414b2f1632b6640534532b52bcec4e97474e15e64`
     at `evidence/reap-v2-original-offline-source-reverified-20261027.json`.

2. **Actual executable original frozen GSM8K path guarded by private Win32
   owner.** The Phase46 original wave now routes `--execute
   --authorize-owned-run --receipt-dir` to additive
   `reap_v2_owner_bound_runtime_supervisor_20261027.py` rather than refusing
   every attempt. The supervisor permits model startup **only** after:
   source-locked wave/arm/command validation, original model *full* SHA256 and
   size, Windows physical RAM and real commit floor, GPU physical device class
   and available VRAM, disk, active foreign process/listening port census, an
   immediate second Windows/GPU observation, and a fresh **offline full
   GSM8K1319+HumanEval164 source-proof receipt**. No source claim from a
   self-authored prior JSON can replace fresh remeasurement. The subprocess
   is created **suspended**, assigned to a newly created private Job Object
   with kill-on-close and max-16 processes, PID/create-time verified, and
   *only then* resumed. Runtime samples enforce the fixed bounds and watch
   the original 41/41 GPU layer fit, four-slot context 16,384, up to 4GiB
   measured CUDA KV, zero REAP128 expert spill, listener and Job member
   ownership, foreign resident service arrival and pressure. On a breach,
   only the newly owned Job is terminated; Ollama, Hermes, LiveBench,
   Freebuff and unrelated Podman processes are never termination targets.
   Successful child exit is **runtime witnessed only**, not automatic
   throughput, benchmark PASS or selection.

3. **Real Windows owner Job checked with a benign child.** The worker
   performed a genuine `CREATE_SUSPENDED`/Job-assignment/resume/termination
   test on an owned benign Python sleep subprocess, proving owned-PID death
   without starting any model, GPU, Podman, extra service or app. The wrapper
   had an actual CLI/import duplicate-dataclass study-identity defect:
   original source-safe preflight first refused erroneously with
   `FROZEN_STUDY_DRIFT`. RED regression, exact field-by-field comparison and
   canonical module-object rebind fixed it. The same explicit native command
   now correctly reaches the Windows physical preflight and **REFUSES**
   `DEDICATED_HOST_NOT_INDEPENDENTLY_ADMITTED` for genuine machine limits;
   `model_executed:false`. Output/session ownership was not created.

4. **Original frozen HumanEval capture and approved-broker handoff,
   still execution-blocked.** New
   `reap_v2_original_humaneval_frozen_executor_20261027.py` implements
   source-hashed `frozen_request` for exactly 164 original prompts with
   system `You are a helpful assistant.`, temperature0, top_k1, seed7,
   max_tokens768 and thinking disabled; source- and original model-bound
   `capture_case`/`seal_capture`, candidate original-program SHA, outbound
   stdin-only payload, independent per-case sandbox and outside-host witness
   checks, and in-memory selector-compatible row/summary publication only
   after verified scoring. The original `reap_eval._humaneval_pass` uses
   local `python -I`, which is **not** by itself a sufficient protected host
   sandbox. Both original Q8/dense serving-session reviewer approvals,
   full164 Q8/dense capture source SHA, approved broker executable and sandbox
   capability SHA and 164 external process-extinction/10s program witnesses
   are not yet supplied; their source-owned approval constants remain EMPTY.
   Actual generated-program execution is therefore **REFUSED**, not marked
   fail or zero. Existing original Q4 151/164 and Q8 152/164 are separately
   authenticated research; no frozen dense comparison or Gate2 PASS follows.

5. **Code-only reproducible dedicated runner package.**
   `reap_v2_dedicated_portable_bundle_20261027.py` builds a deterministic
   archive of **only the exact committed frozen Python sources and prereg
   documents**. It requires `git show HEAD:<source>` bytes identical to
   working-tree bytes, pins original evaluator/serving/gate/selector/scorer
   SHA256 and freezes exact Python dependency versions. The ZIP contains
   **zero GGUF/BF16 model bytes, tokenizer outputs, datasets, generated
   answers, credential files, wheels or CUDA binaries**. `--verify` checks
   every archive entry, source digest and target Python/package ABI; it is
   a code-only prerequisite, **not a machine-ready runnable model image**.
   The builder deliberately refuses before its source has been committed.
   The original GGUFs and dataset-cache source identities must be separately
   transported/verified by authorized operators. See steps below.

## Native host and exact scientific state

Most recent protected host inspection observed Windows approximately 17.57GiB
physical RAM free and 33.23GiB commit headroom, and RTX 5060Ti
**16,311MiB physical total**, approximately **3,693MiB free**, with protected
Q4 original LiveBench llama-server PID55032 on 127.0.0.1:8890 and independently
owned Ollama/Freebuff/Hermes services. The numbers fluctuate and are not
per-model lifetime peaks. The preceding Phase46 bounded witness recorded
minimum 9.99GiB free RAM, 23.92GiB free commit, 3,726MiB free GPU and
max native llama-server sampled working set 19.19GiB; even its host-wide
12,325MiB GPU-used sample was **not** an isolated peak. The original frozen
Q4 four-slot host admission minima of **29.315GiB RAM free, 34.315GiB
commit free, 12,288MiB VRAM free** remain unchanged. The **dedicated**
provisioning targets are **48GiB actually unused physical and commit** and
**48GiB CUDA-class card with at least 40GiB truly free VRAM**, subject to
real remeasurement, model/KV fit and sustained four-slot observability.
No dedicated host was provisioned or model-run this round.

| Immutable model/science item | Verified evidence or required missing comparison |
|---|---|
| Parameters | 18,543,997,568 stored; approximately 3,444,503,168 active/token, 128 routed stored experts with 8 selected/token |
| Size/quantization | 26 BF16 shards 37,981,266,120 bytes; original intermediate BF16 GGUF 37,122,439,488 bytes; Q8_0 GGUF 19,748,435,968 bytes; Q4_K_M GGUF 11,563,915,264 bytes |
| Windows Q4 GSM8K | **1,224/1,319 = 92.7976%**, original source-bound research; original dense Q8 or Q8 candidate/dense same-platform still missing |
| Original Q4/Q8 HumanEval | **151/164** and **152/164** respectively, original source-bound research with independently witnessed generated-code work; original frozen Q8-vs-dense Gate2 still missing |
| Q4 first200 | **189/200** original correct; native mean **40.7702977203 predicted tokens/s**, 41/41 CUDA layers and 0 expert CPU offload in native four-slot Q4 research; new matching native dense Q4 first200 never executed |
| Distinct Linux Q4/Q8/control GSM8K | Q4 **1229/1319**, Q8 **1220/1319**, random128 Q8 **964/1319**; cannot be paired into same Windows official sessions |
| BF16 isolated numerical parity | 127,806/163,840 isolated scalars bit-equal, max abs difference 0.052734375; no whole-model logits/full calibrated CUDA parity PASS |
| REAP160 | Original historical **1244/1319** in separate source/protocol evidence; native full qualification/first200 zero-offload or permitted partial-offload speed still UNKNOWN |
| AutomationBench original 600 | Guard.v3 87/600 strict3, independent historical guard.v1 43/600 strict0 under DIFFERENT sessions/policies; both remain partial, not pooled |
| Official eleven frontier suites | No newly complete official unscored suite; HLE needs official judge, TerminalBench/agent harness and ProgramBench need authentic complete solutions, remaining visual suites not natively eligible. See Phase46 source register. |
| Native lifetime model-specific peak RAM/commit/VRAM, sustained tokens/s | **UNKNOWN**, no four-slot dedicated successful model load this round; earlier isolated host-wide samples are not these numbers |

Original immutable historical REAP128-v1 BF16 D0 PASS, Gate0 PASS and
**Gate1 FAIL** under strict lower-bound `-0.03780741197234255` versus
`-0.03`. The existing deterministic selector's Amendment4 fallback therefore
requires full REAP128_v2 and REAP160 genuine gates. The optional v2-v1
nonzero-margin one-sided 95% paired McNemar-compatible CI is **prospectively
unresolved** versus the original §4 sample-SD Wald formula; no data-selected
new CI method was substituted. No new original-model Q8-versus-dense pair,
random control or valid density-matched first200 speed was produced.

| Original decision | Latest authentic state |
|---|---|
| Historical v1 Gate1 | **FAIL** |
| v2 D0/full CUDA numerical safety | UNKNOWN frozen |
| v2 Gate0 EOS/degeneracy | UNKNOWN |
| v2 Gate1 full1319 Q8/dense paired Wald lower bound > −0.03 | UNKNOWN |
| v2 Gate2 full164 executable HumanEval Q8/dense paired Wald lower bound > −0.10 | UNKNOWN |
| Original random128 Q8 negative control lower bound > 0 | UNKNOWN |
| Candidate/dense same-native-host first200 Q4 throughput ≥1.3× | UNKNOWN |
| REAP160 original speed/fit | UNKNOWN |
| Amendment4 optional v2/v1 paired CI | BLOCKED pending independent prospective method adjudication |
| Gen0 deterministic selector | `{"status":"BLOCKED","chosen":null,"release_approved":false}` |

## Executable continuation without touching protected services

All commands below are **for a separately provisioned, authorized and idle
dedicated Windows runner** unless explicitly marked read-only. Set `PYTHONPATH`
to the extracted bundle root plus `src` and use exactly the sealed Python
and package versions, frozen llama.cpp source/build and original models.
No separate operating-system/global policy changes or infrastructure purchase
are implicitly authorized.

```powershell
# Read-only frozen source input proof (works with an original offline HF cache)
$env:HF_HUB_OFFLINE='1'; $env:HF_DATASETS_OFFLINE='1'
python -m chowder_batch.reap_v2_original_offline_source_lock_20261027 `
  --out C:\reap\receipts\frozen-source-original-1.json

# After these sources are committed and pushed, build/verify a CODE-ONLY ZIP
python -m chowder_batch.reap_v2_dedicated_portable_bundle_20261027 `
  --out C:\reap\bundles\original-reap-runner-1.zip
python -m chowder_batch.reap_v2_dedicated_portable_bundle_20261027 `
  --verify C:\reap\bundles\original-reap-runner-1.zip

# Read-only original physical resources, protected owners and exact model hash
python -m chowder_batch.reap_v2_dedicated_runner_admission_20261027 `
  --arm dense_q4 --model C:\reap\models\dense_v2_q4.gguf `
  --server C:\llama.cpp\bin\llama-server.exe --port 8790 `
  --out C:\reap\receipts\dense-q4-preflight-1.json

# On REAL admitted host ONLY, with separate output/session per registered arm
# and an existing empty exclusive receipt directory, launch owned Job first200
python -m chowder_batch.reap_v2_frozen_wave_20261027 `
  --execute --authorize-owned-run --arm dense_v2_q4 `
  --model C:\reap\models\dense_v2_q4.gguf `
  --server C:\llama.cpp\bin\llama-server.exe `
  --out C:\reap\eval-wave-1 --receipt-dir C:\reap\receipts\dense-q4-1 `
  --port 8790
```

The original supervised runner always rechecks admission, full model SHA,
offline source and loopback owners **again**; a previous positive JSON is not
authority. Its bounded stdout/JSON terminal verdict
`CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY` is a process-lifecycle observation,
**not original quality or speed gate PASS**. On a dedicated eligible host,
sequence original `reap128_v2_q4` first200 after dense Q4, then original
`dense_v2`, `reap128_v2`, `random128_v2` full GSM8K1319, without mixing
per-model sessions. For original HumanEval, separately finish and approve the
new source-pinned Windows response-capture and broker sandbox review, and
produce independent Q8 and dense full164 execution witnesses, then feed
source-pinned receipts into the existing frozen evidence bridge.

**Operational requirement:** Never reuse existing arm result directories,
guess offload, substitute a port-free check for device exclusivity, turn
model-not-executed into a score of zero, or treat a successful code bundle
as proof of capacity. Keep PR #282 DRAFT and do not merge, deploy or train
Gen-1 until the unchanged original deterministic selector has complete
authentic evidence and human approval.
