# Phase49 — original Q4 memory diagnosis, owned Win32 lineage and HumanEval164 trace intake

**Project state: OPEN. Gen-0 BLOCKED; chosen:null; release_approved:false.**
Feature `feature/reap128v2-selection-20261025`, [OPEN DRAFT PR #282](https://github.com/niko4244/Chowder/pull/282).
This is a source-controlled engineering and preserved-evidence qualification
wave, not a full BF16/CUDA benchmark, new four-slot native run, official
frontier aggregate, statistical method amendment, or model selection. Read
with Phase34, Phase37 and Phase45–48 evidence. No frozen `reap_eval.py`,
`reap_gates.py`, `reap_serve.py`, `reap_select.py`, quantization, tokenizer,
model hash, prompts, decoding or frozen thresholds were altered.

## Source-verified original resource investigation

CPU-only `python -m chowder_batch.reap_v2_phase49_resource_diagnostics_20261028`
authenticates historical immutable original Win32 source logs, an aborted
12-row run, the successful full1319 native four-slot session, an isolated
24-second one-slot diagnostic and the Phase48 dedicated-host refusal.
Its pinned report status is
`PASS_PRESERVED_RESOURCE_EVIDENCE_DIAGNOSTIC_ONLY` and does not start or
stop models, Podman, native servers or CUDA. Every historical original partial
item remains incomplete, not zero or pooled into official scores.

* Original aborted 12-row four-slot run (50 actual OS samples):
  **6.98 GiB** minimum physically available RAM was **below the frozen
  7 GiB runtime reserve**, so the direct stop condition was **physical RAM**.
  Commit minimum **10.41 GiB** remained **above** the 8 GiB commit reserve.
  During this run host-wide free physical memory declined **14.43 GiB** and
  commit headroom **18.43 GiB**. A host-wide difference is not solely the
  original Q4 model's working set. Its bounded 8,192 MiB prompt cache had
  **2,166.171 MiB** observed used, with approximately 5.8846 GiB remaining
  theoretical headroom. It is not additive disjoint to the observed
  system-memory delta or guaranteed future allocation.
* Conservative first200 prestart calculations *from this documented failure*
  (not a qualified runtime or scientific score) are **29.3146 GiB free
  physical RAM**, **34.3146 GiB free Windows commit headroom**, and
  **12,288 MiB native GPU free**. They preserve a 2 GiB safety reserve
  on top of earlier measured host decline and unfilled prompt-cache ceiling.
  Existing stricter dedicated-host provisioning remains **48 GiB truly free
  physical**, **48 GiB genuine commit headroom**, **48 GiB-class CUDA card,
  ≥40 GiB genuinely unallocated GPU VRAM**, ≥8 GiB free disk, verified
  exclusive owner/port, model and binary SHA admission.
* Original source-bound four-slot full1319 Windows Q4: 822 independent
  runtime samples, minimum **11.278446 GiB** physical available,
  **31.29488 GiB** commit headroom and **34.32457 GiB** disk free,
  all above original runtime floors. The original model's authoritative
  native startup shows **41/41 CUDA layers**, successful original
  `common_fit_params` fit, context 16,384, four × 4096 slots and
  **320 MiB CUDA KV**. This session is real historical Q4 research only.
* Distinct *experimental* one-slot guard.v3 run with **73,728-token
  context**: 12 Win32 samples over **24.135 seconds** showed native
  `llama-server` maximum *sampled* working set
  **22,162,624,512 bytes** (20.64 GiB) and maximum sampled private
  committed charge **23,464,726,528 bytes** (21.85 GiB);
  minimum 14,456,483,840 bytes host physical available,
  minimum 31,103,373,312 bytes host commit headroom, minimum
  239 MiB GPU free, 408,782 cumulative process page faults *delta*.
  Original one-slot reported CUDA buffers: model 10,619.88 MiB,
  KV 1,440 MiB, compute 172.05 MiB and recurrent 62.81 MiB.
  These **are not frozen four-slot model lifetime peaks**. Windows
  `pagefile`/private bytes represent private commit, **not demonstrated
  pagefile disk I/O**. Page fault count includes soft/hard faults and cannot
  establish actual hard paging. GPU free/used is host-wide WDDM, **not
  source-isolated process VRAM peak**.
* During Phase49 continuation, the *shared* RTX 5060 Ti was genuinely
  only **77–86 MiB free** of **16,311 MiB**, ~70–73% GPU utilization.
  Read-only Windows observation was ~25.33 GB physically available and
  ~19.57 GB commit headroom (samples fluctuate). Ollama's native
  `llama-server`, Freebuff and Podman remain protected and unmodified.
  These measurements categorically **refuse** unchanged native original
  four-slot relaunch on this shared host; no attempt was made.

## Integrated production code and adversarial safety outcomes

1. `reap_v2_live_job_lineage_guard_20261028.py` now feeds a fresh
   **actual private Job handle**, Win32 limit and PID membership, Windows
   process create times, parent chain, executable image path, exact root
   `reap_serve.py`/native `llama-server.exe`/frozen evaluator argv and
   loopback listener census into the source-controlled supervisor.
   Native Job policy must be **exactly kill-on-close plus max-process limit**;
   breakaway/unreviewed flags, unexpected helpers, missing/changed process
   images and PID reuse fail closed. Only bounded original read-only
   PowerShell ownership and NVIDIA memory helpers, plus an OS-owned
   Windows System32 `conhost.exe` console child with exact parent,
   image and constrained console-ID arguments, are allowed. This last
   child was discovered in an actual short-lived disposable Windows
   `CREATE_SUSPENDED`/private-Job CPU probe: Job lifetime process count
   rose to **2** (root Python and Windows `conhost.exe`), both with
   witnessed parent and exact image paths. A generic `conhost.exe`
   name, altered argv or non-System32 executable does **not** qualify.
2. `reap_v2_owner_bound_runtime_supervisor_20261027.py` requires the
   complete native root+server+evaluator **ACTIVE** witness before
   promoting original server fit. A source log with only 41/41 CUDA
   layers, but missing native successful-fit line, cannot produce success.
   During bounded genuine wrapper teardown after previously witnessed
   active fit, listener disappearance is not confused with *new*
   startup-grace expiration; foreign listener/resource changes still abort.
   Natural process lifetime is now checked **BEFORE** terminating the
   private Job: root exit zero with a still-running server/evaluator/helper
   **refuses**, rather than terminating them and pretending the entire
   experiment finished normally.
3. Source-bound Win32 `JOBOBJECT_BASIC_ACCOUNTING_INFORMATION` cumulative
   process totals are compared to independently observed PID+create-time
   lineage. If an extremely short-lived child was missed between polls,
   the terminal must **REFUSE** under
   `OWNED_JOB_UNOBSERVED_TRANSIENT_CHILDREN_OR_CENSUS_DRIFT`.
   This is conservative: without an event-complete Job completion-port
   recorder, a missed helper can prevent a valid wave from being
   certified; there is **no fabricated process-completeness PASS**.
   Full native model lifecycle acceptance still awaits an independently
   admitted dedicated host. A bounded three-try native PID census handles
   legitimate helper-exit snapshot races before raising a persistent
   Job-identity failure.
4. `reap_v2_phase49_owned_memory_witness_20261028.py` adds strictly
   read-only working set/private commit/peak-so-far page-fault counters for
   only **live, named, birth-verified private-Job members**. At most 128
   active-role samples (1/32 supervisor cycles) are summarized. A transient
   missing process records diagnostic UNKNOWN and does not change frozen
   original resource floors. Neither sampled per-process sums nor Windows
   reported peak-so-far imply a four-slot isolated lifetime VRAM/CPU peak.
5. Source-controlled `reap_v2_frozen_gate_evidence_bridge_20261027.py`
   now requires each original Q8 and dense HumanEval164 independent
   trace-certificate path manifest to be *separately approved by source-owned
   SHA pins*, reopens all original case files using
   `reap_v2_humaneval_certificate_verifier_20261027.py`, checks exact
   capture/model/session/host/source/termination/correctness against the
   original 164 scored rows and verifies the original per-file bytes against
   approved original-artifact digests. This prevents a forged in-memory
   `build_selector_rows` grade or source-looking trace-hex string from
   crossing the production selector boundary. It enforces **different
   original model sessions** for incompatible checkpoints and **same
   physical host + hardware profile** for Q8/dense/random and Q4-speed
   pairs. The old `reap_eval` JSONL writer and certificate publisher use
   **different serialization and timing precision**; source-pinned byte
   hashes are verified for *each* original stream, with item-level semantic
   join across them, rather than falsely claiming identical file bytes.
   All 164 trace sources are rechecked **after** selector evaluation too,
   rejecting concurrent replaced certificate inputs where detected.
   Independent native capture/sandbox/reviewer/traces **still are not
   approved**, and all production approval maps remain **EMPTY**.
   Positive test fixtures explicitly monkeypatch synthesized approvals;
   they are **not scientific or production approvals**.

## Original authentic scientific evidence, explicitly NOT a frozen new pair

Independent CPU-only source validation: **574 passed / 1 skipped** on the
integrated original selector, frozen sources, HumanEval, Job, resource and
admission test set before the final OS console-host discovery; after that
specific Win32 fix **94/94** owner/lineage/memory tests passed, followed by
**84/84** owner/lineage tests after the bounded PID-census retry regression.
Ruff and tracked `git diff --check` passed; the post-commit portable-source
identity test must run only after source is committed. Disposable native
private-Job benign Python probes verified actual
`JOBOBJECT_BASIC_ACCOUNTING_INFORMATION` counters, real executable images
and source-known OS `conhost.exe` origin; no original model, GPU or Podman
job was launched. These test counts are engineering confidence only.

| Original source-bound item | Recorded result | Qualification limitation |
|---|---|---|
| Stored model | **18,543,997,568 parameters**; ~**3,444,503,168 active/token**, 128 stored routed experts, 8 selected per MoE layer | Parameter inventory/source construction |
| Original BF16 26 shards | **37,981,266,120 bytes**; BF16 GGUF **37,122,439,488 bytes** | Whole-model BF16/CUDA logit parity absent |
| REAP Q8_0 candidate GGUF | **19,748,435,968 bytes**, registered SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6` | Full-file rehash blocked by host admission |
| REAP Q4_K_M candidate GGUF | **11,563,915,264 bytes**, registered SHA `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` | Construction sidecar (733 tensors) verified; not a quality gate |
| Original dense Q4 / dense Q8 / random128 Q8 GGUF | **21,166,757,984** / **36,903,139,424** / **19,748,435,872 bytes** | Full original same-host comparison incomplete |
| Windows REAP Q4 GSM8K | **1,224/1,319 (92.797574%)** original, 1,281 terminated, 459,264 tokens; real 41/41 and KV 320 MiB | No original Q8-versus-dense Gate1 |
| Windows Q4 native first200 | **189/200**, mean **40.7702977203 predicted tokens/s** | Native same-host original dense Q4 first200 missing; speed gate UNKNOWN |
| Original Q4 and Q8 HumanEval research | **151/164** vs **152/164**; both correct 147, Q8-only5, Q4-only4, both wrong8 | Original Q8-vs-dense 164 independently executed frozen Gate2 missing |
| Linux original Q4/Q8/random128 GSM8K | **1229/1319**, **1220/1319**, **964/1319** | Different serving/platform; not pooled into Windows official gate |
| BF16 isolated layer CUDA parity | **127,806 / 163,840** scalars bit exact, max |abs diff| **0.052734375** | Not complete 40-layer whole-model numerical parity |
| Historical REAP128-v1 | D0 PASS, Gate0 PASS, **Gate1 FAIL**: bound **−0.03780741197234255** vs strict >**−0.03** | Historical FAIL retained |
| REAP160 historical GSM8K | **1,244/1,319** separate research | Native REAP160 fit/throughput/gates UNKNOWN |
| Original AutomationBench600 | Partial guard.v3 87/600 (strict3); separate guard.v1 43/600 (strict0) | Separate incompatible partial cohorts, official UNKNOWN |
| Q4 separate LiveBench | **139/200** across four subsets | Not official aggregate, outside registered eleven |

Frozen ordered canonical GSM8K1319 SHA256
`913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`;
HumanEval164 source SHA256
`49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420`;
original evaluator Windows bytes SHA256
`9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1`.
All original quantization/source fingerprints and the initial Phase48
model sidecar receipts remain immutable. New read-only checks did not replace
registered weights or rerun expensive completed evaluations.

## Scientific gates / registered eleven frontiers

| Scientific prerequisite | Actual registered disposition |
|---|---|
| v2 BF16 whole-model D0, CPU/CUDA full parity, EOS/trigram Gate0 | **UNKNOWN** |
| Original candidate Q8-versus-dense Q8 GSM8K **full1319**, paired sample-SD Wald one-sided 95% LB > **−0.03** | **UNKNOWN** |
| Original Q8-versus-dense Q8 independent external-sandbox HumanEval **full164**, original sample-SD Wald LB > **−0.10** | **UNKNOWN** |
| Original random128 Q8 paired control | **UNKNOWN** |
| Matched native Q4 candidate/dense first200 ≥**1.3×** speed | **UNKNOWN** |
| REAP160 original full gates/native fit/speed | **UNKNOWN** |
| Amendment4 optional v2-v1 McNemar-compatible one-sided 95% paired CI at **−0.02** vs §4 **Z=1.645 sample-SD Wald** | **BLOCKED** until one independently approved method frozen **prospectively**, not selected from results |
| Source-approved frozen selector | **BLOCKED / chosen:null / release_approved:false** |

The eleven original suite statuses remain: **HLE** (Q4 128 original
answers, heuristic 3/128; Q8 99, heuristic 0/99, official judge absent)
**UNKNOWN**; **TerminalBench4.0** (66, partial Q4 tool trace, no official
grader) **UNKNOWN**; **DeepSWE1.1** (113) **UNKNOWN**;
**ProgramBench1.0.1** (200 source tasks and 248,089 eligible tests
authenticated; no complete original submitted programs) **UNKNOWN**;
**NL2Repo-Bench** (104) **UNKNOWN**; **AutomationBench1.0.6**
(600; incomplete separate original prefixes) **UNKNOWN**; **Agents'
Last Exam1.1** (122) **UNKNOWN**; **ApexBench1.1** (original v1.1
source/protocol incomplete) **UNKNOWN**; **Chartography** (100),
**BabyVision** (388), and **ZeroBench-main** (100 pass@5) are
**NOT NATIVE-ELIGIBLE** with this text-only model, with no unapproved
external OCR/VLM substituted.

## Exact safe next execution after independent host authorization

1. Restore all five registered original model paths and unchanged sidecars:
   dense Q4 `G:\llm-cold\gguf\dense-q4_k_m.gguf`,
   candidate Q4 `G:\reap128-v2-q4_k_m-20261025.gguf`,
   dense Q8 `G:\llm-cold\gguf\dense-q8_0.gguf`,
   candidate Q8 `I:\reap128-v2-q8_0-20261025.gguf`,
   random Q8 `G:\llm-cold\gguf\random128-q8_0.gguf`;
   original native binary
   `F:\llama.cpp\bin\llama-server.exe`. The original Q4/Q8 sidecars
   bind those absolute paths; NEVER silently transplant to C: or modify
   source-receipt content.
2. Independently authorize a physically **dedicated** Windows host meeting
   measured resource floors above; use original source-only runner bundle
   and exact supported CPython/dependency ABI, verify full registered model
   and native server file hashes, frozen offline HF1319/164 source and original
   sidecars, source-controlled code SHA and no foreign GPU/PID/port owners.
   Perform fresh physical/commit/VRAM admission for EACH owned arm; no
   preexisting partial row adoption. Native serving flags **exactly**
   `--fit on -np 4 -c 16384 --jinja`; do not alter context, number of slots,
   prompt/seed, decoding, evaluator, original D0 or scorer to force fit.
3. Only with authenticated admission, run independently sealed original full
   Q8/dense/random GSM8K1319, then candidate and original dense Q4 first200
   **separate** complete serving sessions on one approved host and identical
   frozen protocol; use first200 matched throughput with no mixed logs.
   Capture full original BF16 40-layer numerical and D0 safety and REAP160.
4. Obtain separately reviewed real outside-host generated-code capability,
   native capture owner, sandbox broker, complete independent per-case
   process/cgroup/trace approvals for **both** original Q8 and dense Q8
   before launching any original HumanEval generated programs. The
   source-owned approvals must be set by an **independent reviewer**, not by
   this implementation; full164 official rows must match original frozen
   semantics and certified process termination. Attempt other eligible
   original frontier suites only under their registered harnesses.
5. Freeze the disputed Amendment4 method BEFORE a held-out confirmation
   comparison through independent statistical review; never use an ad hoc
   favorable McNemar/Wald choice. Pass all original complete pinned evidence
   to unchanged gate/selector and inspect all gate status and strict margins.
   No deployment, merge, or Gen-1 until the original selector genuinely
   admits it. At this phase **`{"status":"BLOCKED","chosen":null,"release_approved":false}`**.

Known remaining limitations: original native model host unavailable,
independent HumanEval approvals absent, end-to-end full native Job lineage
acceptance untested on genuine dedicated model service, missed helper events
conservatively refuse without a completion-port witness, new exact native
process memory samples not yet collected on a qualifying four-slot run,
isolated GPU VRAM peak/hard paging unmeasured, and frozen gate data incomplete.
Passing CPU-only unit tests and source packaging does not change any of those.
