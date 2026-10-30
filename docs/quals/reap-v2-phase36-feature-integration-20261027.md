# REAP128_v2: original-source feature integration and bounded BF16/CUDA evidence

**2026-10-27 CDT. Decision: `BLOCKED`, `chosen: null`, `release_approved: false`.** This integrates independently verified prior work into the **authorized feature worktree** without altering original sources, scoring, serving flags, threshold semantics, models, or the open draft PR #282. It is a partial qualification record, **not eleven completed frontier suite scores or Gen-0 authorization**.

## Source/branch identities and preservation

- Authorized worktree `C:\Users\nikma\wt\chowder-reap128v2-selection-20261025`, `feature/reap128v2-selection-20261025`; predecessor `3ee98362826cb7ecb8043a1fd06ed514244479d8` is confirmed ancestor of the verified integration base `435e2e99e6b7e60e26bb14dab5065061cd30a59a`.
- Preserve contributor modifications to `chowder_batch/reap_q4_automationbench_ctx64k_20261026.py` and untracked `chowder_batch/reap_context_guard_proxy_20261026.py`, `tests/test_reap_context_guard_proxy_20261026.py`. Do not stage, reset, overwrite or commit them here.
- Original checkpoint source verification is independently pinned to `docs/quals/reap128-v2-full-source-equivalence-20261025.json`, SHA-256 `0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d` (validated live on the feature branch). The bounded layer runner rehashes the **complete current bytes of each selected original weight shard** against this earlier full-source proof.
- Original Windows Q4 GSM8K (not part of the registered eleven) is already complete. `docs/quals/reap-v2-phase34-original-windows-q4-1319-20261027.md` records independent frozen-source scoring: **1224/1319, 92.797574%; all 1319 rows scored; 1281 natural stops; 41/41 CUDA layers; clean shutdown**. No model rerun was performed for this integration.

## Actual model/resource identities

| Quantity | Original observed or source-bound value | Scope |
|---|---:|---|
| REAP128_v2 saved parameters | 18,543,997,568 | 40 layers, 128 routed experts per layer |
| Estimated active parameters per token | 3,444,503,168 (18.575%) | 8 routed experts/token; **not measured FLOPs** |
| Pruned BF16 safetensors on local disk | 37,981,266,120 bytes across 26 shards | Local original checkpoint; not GGUF |
| BF16 intermediate GGUF | 37,122,439,488 bytes | Original frozen intermediate, not full numerical-parity proof |
| Original Q8_0 GGUF | 19,748,435,968 bytes | SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6` |
| Original Q4_K_M GGUF | 11,563,915,264 bytes | SHA `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` |
| Original full Q4 Windows startup | 39.106 GiB physical free; 64.823 GiB commit free; 14,823 MiB GPU free | Earlier 1319 run pre-admission |
| Original full Q4 Windows runtime minima | 11.278 GiB physical free; 31.295 GiB commit free; 34.325 GiB F: free | No host physical/commit/disk floor breached |
| Minimum host-wide GPU-free during Q4 1319 | 179 MiB | **Confounded by another Ollama reload; not isolated model peak VRAM** |
| Local layer 3 retry at 03:11 CDT | 7,592 MiB physical GPU free vs 8,192 MiB probe start reserve | Refused without model-layer execution; unrelated Ollama retained |

The completed Q4 original 1319 therefore **did not end in a Windows host-memory exhaustion failure**. The distinct original dense-Q4 first200 Linux attempt was safely aborted after unused cgroup memory fell from 29.444 to 9.992 GiB during incomplete model loading; zero eligible dense speed rows. Do not conflate these two events or waive original floors. Isolated peak GPU/RAM for original dense, Q8 and BF16 full-model serving remains **UNKNOWN**.

**Later actual original-Q8 missing29 recovery (03:25 CDT):** Existing private Kaggle kernel `nikmarco/original-q8-hle-missing29-private-20261027` changed to COMPLETE at the platform level, and its authentic output was downloaded **once** into the new exclusive private folder `F:\reap-v2-q8-hle-missing29-output-20261027-feature-first-complete-recovery`. The source/selection/serving-locked private receipt SHA-256 is `2489554a5ade3cbfec74001d5d1fcf8b7be6028bf283b564fcdf0ecb5a27a99d`; native loading log SHA-256 `2d92b64bf690815514fbb12dbcef49c41a31a256d1a24381d42270b84e64b09f`. Actual status **`REFUSED_OR_ABORTED`**, error **`Q8 conservative 10 GiB cgroup headroom floor breached`**. Initial unused cgroup **28.806 GiB**, host available physical **29.770 GiB**, both GPUs free **14,912 MiB**. The native server had begun mmap model-tensor loading but had not completed it; zero new scored JSONL rows, zero 29-item serving session, zero eligible heuristic or official scores. Exact source-set 99+29 verification remains only an **ID census**; the strict native two-session audit rightly refused (exit 2). Do not confuse Kaggle COMPLETE with model-qualified COMPLETE.

The new `reap_q8_hle_failed_host_admission_20261027.py` is a strict read-only SHA-bound negative-evidence auditor and a **prospective** higher-capacity host census, without any new Kaggle submission. Measured consumption before incomplete load exceeded **18.806 GiB** of cgroup headroom. Next runner prestart requires at least **48 GiB truly unused cgroup**, **48 GiB host physical available**, 12,288 MiB free per two GPUs and 8 GiB disk; these are conservative next-run filters, **not a claim that 48 GiB guarantees complete load**. Linux-only native measurement reads real cgroup-v2 usage, MemAvailable, physical GPU free and filesystem headroom. Even satisfying all these prestart filters does not prove postload 14 GiB reserve, original 41/41 CUDA offload, actual source scoring or scientific PASS. Do not repeat the measured 28.806-GiB-cgroup host or lower its 10-GiB runtime floor. Focused admission tests **18/18 PASS**, Ruff PASS, original private failure check PASS.

## Real physical BF16 CPU/CUDA layer numerics

The standalone `reap_bf16_cuda_layer_parity_20261027.py` uses the actual pruned checkpoint, genuine PyTorch CPU BF16 and RTX 5060 Ti CUDA BF16, equal fixed non-model-graded input activations, the same Hugging Face decoder-layer class, a source-pinned earlier full-checkpoint verification and native GPU/host floor sampling. It executes no generated code, full model or changed quantized weights.

| Independently source-verified prior native receipt | Layer 0, linear attention | Layer 3, full attention |
|---|---:|---:|
| BF16 elements compared | 4096 | 4096 |
| Exactly equal | 3816 (93.16%) | 2390 (58.35%) |
| Maximum absolute error | 0.00390625 | 0.0078125 |
| Mean absolute error | 0.0000575576 | 0.0007555038 |
| RMSE | 0.0003232623 | 0.0014008750 |
| Original private receipt SHA-256 | `0a632ba0f87db8604159728e79bc5edb3a299cdc195e3fa6150d6599d053c22b` | `9228f76bb505f95680b6b49163644cec57a1eecbe7bd1d83a35aea6369a95f6e` |

These are **two isolated layer comparisons**, not 40-layer chained forward/logit parity, 704-row D0, cross-device deterministic generation equivalence, original GGUF BF16/Q8/Q4 numerical acceptance, or a Gen-0 PASS. No bound on end-to-end error was derived. A later feature-worktree attempt to repeat layer3 refused with insufficient physical VRAM; the earlier genuine receipt remains preserved and is not pooled as a new run.

The new `reap_bf16_cuda_all40_research_20261027.py` has an original-source, per-layer isolated-child, bounded resumption plan: it reuses earlier exact original layer0 and3 receipt SHA rather than rerunning, refuses unowned evidence roots/modified proof, and keeps model and Gen0 gates BLOCKED. Its initial invocation **without** `--execute` returned `ORIGINAL_ALL40_PLAN_NOT_EXECUTED`; the 40-layer experiment has **not** been claimed complete. Actual execution requires independently admitted GPU capacity and no disruption to protected Ollama/WSL/Hermes jobs.

## Eleven original registered frontier benchmarks — honest qualification status

| # | Registered original suite | Authentically observed original progress | Frozen official qualification |
|---:|---|---|---|
| 1 | HLE text/no-tools | Q4 original text-only 128/128 responses, **3 heuristic exact matches**, *not official judge*. Q8 first **99/128 original responses**, 0/99 heuristic. Exact 29 distinct missing source IDs; completed Kaggle continuation **cgroup-aborted during load, 0 new rows** | **UNKNOWN**, original HLE judge and valid new-session 29 source rows missing |
| 2 | Terminal-Bench 4.0 | No official full original Q4/Q8 task-container/Harbor run | **UNKNOWN** |
| 3 | DeepSWE v1.1 | No official full original **113** task execution | **UNKNOWN** |
| 4 | ProgramBench | No official full original **200** binary/docs task execution | **UNKNOWN** |
| 5 | NL2Repo-Bench | No official full original **104** generated repository/build-test execution | **UNKNOWN** |
| 6 | AutomationBench 1.0.6 | Official source **600** distinct domain/task identities, Q4 original isolated pilots **0/6** and **0/2** strict across nonpoolable profiles. Q8 original CPU-only Kaggle kernel COMPLETE but its source-locked output is still inaccessible (HTTP429) | **UNKNOWN**, neither model scored on full600 |
| 7 | Agents' Last Exam v1.1 | No official full original **122** agent task execution | **UNKNOWN** |
| 8 | ApexBench v1.1 | Exact official source/version/harness and full original task count not authenticated | **UNKNOWN** |
| 9 | Chartography with tools | Registered vision/chart images cannot be consumed by original unmodified text-only GGUF | **INELIGIBLE original modality; no score** |
| 10 | BabyVision with tools | Original vision input unsupported by frozen text-only GGUF | **INELIGIBLE original modality; no score** |
| 11 | ZeroBench-main with tools | Original image input/pass@5 unsupported by frozen text-only GGUF | **INELIGIBLE original modality; no score** |

Unsupported modalities, absent grader/source, interrupted jobs and partial pilots are **not zero-correct scores**. No OCR, captions, different VLM, alternate harness, shortened population, or invented official judge may replace the original registration.

## Other frozen original comparisons / gates

- Original Q4 Windows first200: **189/200** correct, mean original per-item `predicted_per_second` **40.7702977 tok/s**, native shutdown and source proof completed. Comparable original same-session/same-hardware **dense first200 missing**; original speed >=1.3x gate **UNKNOWN**. Linux research tokens/sec must not be used as its denominator.
- Original research Linux GSM8K1319: Q4 **1229/1319**; Q8 **1220/1319**; random128 Q8 **964/1319**. Original research HumanEval164 reported Q4 **151/164**, Q8 **152/164**, random128 **45/164**, but original generated code lacks approved **independent sandbox** execution witness; official HumanEval Gate2 **UNKNOWN**.
- Frozen D0 full BF16 forward NLL calibration / CUDA parity: **UNKNOWN for the required accepted source/configuration**; bounded layer probes do not qualify.
- Gate0 original EOS/trigram differences vs matching dense cohort: **UNKNOWN**. Gate1 original paired Wald 1319 lower bound vs dense **UNKNOWN**. Gate2 original paired Wald HumanEval164 **UNKNOWN**. Random128 original source-matched research negative control exists, but entire selector control admission **UNKNOWN**.
- Historical original REAP128-v1 Gate1 paired lower bound **-0.03780741197234255**, strictly fails **>-0.03**. Ordered fallback REAP128_v2 and REAP160 are not thereby automatically selected; REAP160 quality/speed remain **UNKNOWN**.
- Amendment 4 optional v2-v1 paired one-sided 95% McNemar-style interval at margin **-0.02** is **not uniquely defined** by frozen text; original §4 instead specifies paired sample-Wald. Prior outcomes were exposed. An independent qualified reviewer must freeze the exact nonzero-margin paired-risk interval and blind new nonoverlapping confirmation before analyzing outcome data. Neither prospective signoff nor eligible substitute original validation exists; optional route **BLOCKED**.
- Original frozen serving `--fit on -np 4 -c 16384 --jinja` unchanged. HLE text-only research used its separately original registered `-np 2` serving; **never** claim 99+29 is a single native session or applicable to 4-slot release gate.

## Implemented integrations, independent checks, and next execution

The feature branch has imported original source-bound full1319 report/validator, Q8 HLE missing29 second-session generator/auditor and adversarial tests, locked upstream Python3.13 AutomationBench 108-wheelhouse/CPU source600 checker, distinct Q8 six/guard2 pilot source and CPU-receipt fail-closed gate, and BF16 layer/all40 research probes. Imports are source-identical to independently tested experimental files, copied to **new paths only**. No *new* Kaggle job, code-executing HumanEval replay, GPU model serving, system-service suspension, scorer, source set, model weight or protected contributor edit occurred for these integrations. The separately launched preexisting missing29 job was retrieved and audited only after actual Kaggle completion, without restarts or extra rate-limited polls.

Verification, from the authorized feature root:

```powershell
python -m pytest -q tests/test_reap_bf16_cuda_layer_parity_20261027.py tests/test_reap_bf16_cuda_all40_research_20261027.py tests/test_reap_q8_hle_missing29_resume_20261027.py tests/test_reap_q8_automationbench_offline_wheelhouse_20261027.py tests/test_reap_q8_automationbench_cpu_receipt_gate_20261027.py tests/test_reap_q4_windows_full1319_20261027.py
python -m chowder_batch.reap_q8_automationbench_cpu_receipt_gate_20261027
python -m chowder_batch.reap_q8_hle_failed_host_admission_20261027
python -m chowder_batch.reap_bf16_cuda_all40_research_20261027 --model 'G:\llm-cold\bf16\Qwen3.6-REAP128-V2' --out 'F:\reap-v2-bf16-cuda-all40-research-20261027-feature'
```

Additional original feature-tree-wide `pytest -q tests -k reap --maxfail=1` run: **889 passed, 1 skipped, 4247 deselected, 1 failed** in 182.22 s. The sole failure `test_reap_kaggle_runner_snapshots_20261026.py::test_each_snapshot_is_exactly_hashed_source_without_private_model_bytes` is a strict snapshot-folder inventory mismatch: local extra `q4_hle_textonly.py` and `__pycache__` are present. They were **not created, modified or removed** by this scoped integration. The failed assertion is recorded; the full test suite is **not claimed PASS**. Focused new-path checks and Ruff pass independently.

Executable next gates: (1) move the exact **original HLE 29 missing IDs** to a genuinely larger independently eligible runner, validating its true cgroup/physical/GPU headroom using the new admission gate before any model load; preserve original source/model/prompt/decoding/score and *new native session* so the existing strict two-session auditor can accept actual rows; **do not resubmit the previous insufficient host**; (2) obtain the existing completed AutomationBench CPU source-bound JSON without repeated rate-limited output API, then verify with `--original-cpu-receipt`; (3) approve independently observed disposable Python3.13 HumanEval host witness after fixed benign canaries, before any generated code; (4) schedule remaining bounded layer probes **only when genuine 8-GiB physical free VRAM and Windows host floors hold**; (5) use an independently eligible dedicated host with measured 48+ GiB free memory/commit, enough usable VRAM and original native flags for original dense and matched throughput; (6) obtain actual original suite sources/tool agents and independently approved Amendment 4 prospective method, then run the frozen selector.

**PR #282 remains OPEN/DRAFT/UNMERGED. Do not merge, approve, deploy or start Gen-1.**
