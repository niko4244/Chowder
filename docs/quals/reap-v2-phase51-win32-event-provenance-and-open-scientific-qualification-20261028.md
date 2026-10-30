# REAP128_v2 Phase51 — source-bound Win32 Job event provenance and OPEN scientific qualification

Date: 2026-10-28. Scope: authorized feature/reap128v2-selection-20261025 worktree only, open DRAFT Chowder PR #282. This document records the Phase51 engineering and scientific evidence boundary, not a model release or gate certificate. Baseline committed HEAD at start: ab9b26932b72442912755241145f7bac2067b618 (Phase50).

## Disposition

    {"status":"BLOCKED","chosen":null,"release_approved":false}

Phase51 adds an owned Win32 Job completion-port birth/exit source to reduce the unsampled short-lived helper gap. Native events alone are insufficient: each NEW_PROCESS needs captured, source-joined PID, original kernel process HANDLE, process creation timestamp, actual image, argv, parent, Job membership and full registered command semantics. The independent receipt verifier checks the lifetime count against the Job BasicAccounting.TotalProcesses. Event packets may be lost by Windows; missing event or exited-before-command-capture means REFUSED, not an inferred helper or scientific PASS. All registered original science remains OPEN pending independent full runs, genuine baselines and approvals.

## New production enforcement and adversarial checks

1. Attach an IO completion port to the empty private Job before CreateProcessW. Assign the directly created suspended root to that Job, verify its own kernel HANDLE and birth, set the private owner active, start the reader, then ResumeThread; never let the event reader's membership check see an unauthenticated or inactive owner. Failures terminate only the owned direct HANDLE and close the owned Job.
2. GetQueuedCompletionStatus messages use the exact owner completion key, limited queue size, NEW_PROCESS/EXIT_PROCESS/ABNORMAL_EXIT_PROCESS/ACTIVE_PROCESS_ZERO codes and PID values without dereferencing packet pointers. On NEW_PROCESS use OpenProcess and the original held HANDLE, native GetProcessTimes, QueryFullProcessImageNameW, psutil parent/argv/image and the same private Job. Preserve process handles until terminal cleanup. Native HANDLE and psutil birth must agree within 0.05 seconds. Unknown processes, spoofed source, duplicate births, malformed exits or zero packets and abnormal exits fail closed.
3. Retain the exact original registered root, native server and evaluator command arrays as independently verified by the frozen wave and unchanged reap_serve.py. Accept only narrowly verified OS System32 conhost or exact read-only nvidia-smi/PowerShell probes from that frozen source, never a general helper class. HumanEval code-generated subprocesses are NOT authorized by this GSM8K-only four-slot wave; those require separately approved execution isolation and provenance.
4. During post-root natural shutdown allow the original bounded 40 x 50ms Job census and delayed source-verified births; wait for a bounded two quiet IOCP polls before stopping the event reader; require a complete terminal EXIT pairing and ACTIVE_PROCESS_ZERO, source-vetted birth ledger matching native cumulative TotalProcesses, and zero live Job members. Complete the ledger before the independent postclose PID/birth extinction check. Any missing kernel packet, unexpected reader death, timeout, Job-close uncertainty, source/receipt mutation or listener mismatch REFUSES. No process from a foreign owner is killed.
5. Original HumanEval independent approval maps remain empty. This code does not grant a process sandbox, grader certificate, native model launch, Gen-0 or Gen-1 training authority.

Verified CPU-only Phase51 test results (after final changes): 137 focused native/lineage/receipt/drain tests PASS; broad registered production/selector/scorer/source-identity battery 609 PASS and 1 skipped; Ruff on all owned Phase51 changes PASS; no original model inference or GPU experiment. The benign native test creates a short Python subprocess in an independently owned disposable private Job and verifies an actual root event before owner teardown. A skipped existing Windows CRLF test is not an evaluation gate. The Phase50 prior publication was 588 passed, 1 skipped; these counts are distinct.

## Original model geometry and frozen artifact evidence

Registered model geometry is 18,543,997,568 stored parameters, approximately 3,444,503,168 active per token, 128 stored routed experts and 8 expert choices per MoE layer. Frozen original BF16 is 26 checkpoint shards totalling 37,981,266,120 bytes; BF16 intermediate GGUF is 37,122,439,488 bytes. The original Q8 is 19,748,435,968 bytes and original Q4 11,563,915,264 bytes. Exact immutable files listed below were checked for path, regular-file presence and expected length in Phase51; they were NOT rehashed multi-gigabyte end to end on this heavily occupied shared host.

| Original artifact | Absolute path | SHA256 provenance / state |
| --- | --- | --- |
| BF16 GGUF intermediate | F:/reap128-v2-bf16-intermediate-20261025.gguf | Registered full SHA in frozen source receipts; prefix 80341aa7; not rehashed Phase51 |
| Candidate Q8 | I:/reap128-v2-q8_0-20261025.gguf | 89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6 |
| Candidate Q4 | G:/reap128-v2-q4_k_m-20261025.gguf | 2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7 |
| Dense Q8 | G:/llm-cold/gguf/dense-q8_0.gguf | 2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2 |
| Dense Q4 | G:/llm-cold/gguf/dense-q4_k_m.gguf | 2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049 |
| Random128 Q8 | G:/llm-cold/gguf/random128-q8_0.gguf | 601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b |

Candidate Q4 .reap-chunked.json original sidecar SHA256 cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c, Q8 sidecar a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d. The small sidecars and immutable reap_serve.py (f62a0de3669acae33252a9e178eea134f7a31d98f5b8d2e6746ff8a603cab3dd), reap_eval.py (9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1) were rehashed and matched. Pruning, original calibration, weight files, tokenizer, dataset order, serving configuration, scoring and frozen selector have not been changed. Original serving flags remain --fit on -np 4 -c 16384 --jinja.

Native original Q4 prior CUDA startup reported 41/41 GPU layers, zero CPU expert spill, 10,619.88 MiB CUDA model buffer, four-slot KV 320 MiB, recurrent 251.25 MiB, compute 104.05 MiB; listed CUDA buffer sum 11,295.18 MiB is NOT measured isolated peak VRAM. A separate one-slot experimental resource sample reached 22,162,624,512 bytes process working set, 23,464,726,528 bytes private commit, and 408,782 cumulative process page faults, NOT a validated four-slot full-study peak and NOT hard pagefile disk I/O. Original Q4 native first200 prior isolated research throughput 40.7702977203 predicted tok/s; original dense Q4 same-host matching ratio remains UNKNOWN.

## Live resource and independent admission

An independent 2026-10-28 read-only snapshot observed 14,831,005,696 free physical bytes (13.81245 GiB), 14,739,320,832 free commit bytes (13.72706 GiB), and RTX 5060 Ti 16,311 MiB VRAM, 15,973 MiB used, just 78 MiB free, 88% GPU utilization. Protected Ollama, Freebuff, Podman and other user processes were active. GPU hardware cannot ever satisfy the pinned dedicated >=48 GiB total-class criterion; even terminating unrelated services would not fix the hardware admission. No foreign services were stopped, no model loaded, no GPU context acquired, no multi-GB hashes repeated. Host-wide RAM/commit pressure is not a measured exclusive model peak. A known aborted 12-row original four-slot experiment crossed the 7 GiB physical runtime floor (min 6.98 GiB) while commit min remained 10.41 GiB; do not relaunch its unsafe unchanged configuration.

For a separately authorized Windows machine, provision ideally >=128 GiB host RAM and >=96 GiB CUDA VRAM to include BF16/dense comparators, with hard preflight: >=48 GiB genuinely FREE physical memory; >=48 GiB FREE commit headroom; dedicated GPU >=48 GiB total and >=40 GiB genuinely FREE, <=1 GiB used and <=10% busy with fixed GPU UUID; >=8 GiB free destination; no foreign port, process, GPU or Job ownership. Original construction receipts and GGUFs must remain at the canonical G:/I:/F: absolute paths; original executable F:/llama.cpp/bin/llama-server.exe must pass its full native SHA verification on that target. No dedicated capable/authorized machine has been established. A read-only READY_FOR_CAPACITY_PROBE_ONLY verdict never authorizes an actual experiment.

Only after a separately authorized capable host is identified, create an exclusive experiment parent and a distinct existing empty receipt folder; refuse reuse. Example read-only original capacity preflight:

    python -m chowder_batch.reap_v2_dedicated_runner_admission_20261027 --arm dense_q4 --model G:/llm-cold/gguf/dense-q4_k_m.gguf --server F:/llama.cpp/bin/llama-server.exe --port 8790 --out F:/reap-dedicated-phase51/dense-q4-preflight.json

After fresh independent admission AND explicit separate owner-run authorization, use one registered frozen arm and unique output/receipt identity per run:

    python -m chowder_batch.reap_v2_frozen_wave_20261027 --execute --authorize-owned-run --arm dense_v2_q4 --model G:/llm-cold/gguf/dense-q4_k_m.gguf --server F:/llama.cpp/bin/llama-server.exe --port 8790 --out F:/reap-dedicated-phase51/dense-q4-wave --receipt-dir F:/reap-dedicated-phase51/dense-q4-receipts
    python -m chowder_batch.reap_v2_frozen_wave_20261027 --execute --authorize-owned-run --arm reap128_v2_q4 --model G:/reap128-v2-q4_k_m-20261025.gguf --server F:/llama.cpp/bin/llama-server.exe --port 8790 --out F:/reap-dedicated-phase51/candidate-q4-wave --receipt-dir F:/reap-dedicated-phase51/candidate-q4-receipts

Sequentially, after each original source/Job extinction receipt is independently approved, perform remaining full1319 registered source arms, with separate unique output/receipts for each: reap128_v2 at I:/reap128-v2-q8_0-20261025.gguf, dense_v2 at G:/llm-cold/gguf/dense-q8_0.gguf and random128_v2 at G:/llm-cold/gguf/random128-q8_0.gguf. The frozen wave uses limit 200 for each Q4 first200 and 0 = full1319 for Q8, max 1024 generated tokens, parallel 4. Distinct checkpoints require distinct resident native sessions; do not splice rows, partial prefixes, changed profiles or Windows/Linux. All commands above are proposed future instructions only, not executed or authorized here.

Full source GSM8K1319 fingerprint: 913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89. Original HumanEval164 source fingerprint: 49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420. HumanEval requires independent approved code execution broker/isolation, 164 actual traces per Q8 and dense Q8, and original sandbox provenance. Future HumanEval per-case Python child processes have a different parent topology than this GSM8K-specific private-Job source gate; separately registered and approved process-witness semantics are required, not a permissive change to the frozen GSM8K allowlist.

## Actual original scientific evidence and frozen gates

Historical source-bound research, not newly qualified registered paired gates: original Windows Q4 GSM8K 1,224/1,319 = 92.797574%, 1,281 natural terminations; Q4 native first200 189/200 and historical speed above. Original Q4 HumanEval research 151/164 and original Q8 research 152/164, genuinely reconciled as different original model sessions, NOT Q8 versus dense Q8 HumanEval Gate2. Original Linux GSM8K research Q4 1,229/1,319; Q8 1,220/1,319; random128 Q8 964/1,319, all separate Linux model/profile sessions. Historical REAP128-v1 Gate0 PASS and Gate1 FAIL are immutable; paired Wald lower -0.03780741197234255 is not > -0.03. REAP160 historical 1,244/1,319 is a separate research cohort; native speed UNKNOWN.

| Registered gate or control | Status | Required missing proof |
| --- | --- | --- |
| REAP128-v1 D0 / Gate0 | Historical PASS | Preserve historical identity only |
| REAP128-v1 Gate1 | Historical FAIL | Immutable recorded noninferiority failure |
| REAP128_v2 whole BF16 D0, CPU/CUDA parity, EOS/trigram Gate0 | UNKNOWN | Full original BF16 parity and Gate0 evidence |
| Q8 candidate versus dense Q8 paired complete GSM8K 1,319 Gate1 | UNKNOWN | Both original full same-protocol results; one-sided Wald lower > -0.03 |
| Q8 versus dense Q8 independently sandboxed original HumanEval164 Gate2 | UNKNOWN | Two complete 164 executed source-bound cohorts; lower > -0.10 |
| Random128 original control | UNKNOWN | Complete registered same-study native cohort |
| Original candidate vs dense Q4 matched first200 native speed | UNKNOWN | Same physically matched authorized host/protocol, separate original sessions, >=1.3x |
| REAP160 original full native speed/accuracy | UNKNOWN | Unexecuted registered native comparator |
| Optional v2-v1 paired -0.02 noninferiority | BLOCKED | Prospectively independent adjudication of conflicting Amendment4 paired CI methods before margin test |
| Registered deterministic Gen-0 selection | BLOCKED | All frozen gates and original independent approvals; chosen=null, release_approved=false |

Eleven registered frontier suites: HLE, TerminalBench4, DeepSWE, ProgramBench, NL2Repo, AutomationBench, Agents' Last Exam and ApexBench UNKNOWN/incomplete for original official release scoring. Chartography, BabyVision and ZeroBench NOT NATIVE ELIGIBLE on this original text-only checkpoint absent an originally authorized vision input/harness; no fabricated text-only equivalence. Current original AutomationBench guard.v3 87/600 remains partial research. Original LCB transport latest 1,055 population, 288 journal rows, 200 transport successes, 88 refused and 105 internally passed remains official score null: independent source/security approvals and complete execution absent.

## Remaining blockers, preservation and next execution

The private Windows completion port is a genuine source-attested improvement, not a guarantee of observing a 1ms process: legitimate PowerShell or nvidia-smi may exit before a NEW packet is dequeued, in which case the source correctly REFUSES. A future authenticated process-creation broker or equivalent registered, separately approved source witness may be necessary; do not inject DLLs, change original serve source, suppress errors or infer missing commands from native counters. Windows Job completion events also cannot themselves prove genuine original benchmark correctness. No claimed release approvals or genuine full Q8/dense gate runs were performed in Phase51.

Preserve all immutable frozen code/weights/sidecars/certificates. Collect new source-bound full original datasets and matched native receipts only on independently admitted real hardware, with original evaluation protocol, physically separate model resident sessions, independently approved HumanEval execution, and prospective Amendment4 CI method adjudication. Apply unchanged reap_gates.py and reap_select.py only to a complete independent certificate set. Until then Gen-0 remains BLOCKED / chosen:null / release_approved:false; no PR merge, release, deployment or Gen-1 training. Phase51 code-only portable ZIP (to be separately committed and independently SHA verified) contains NO GGUF, model weights, private dataset rows or executable grader outputs. Phase50 package and previous research/other contributors' untracked artifacts remain untouched.
