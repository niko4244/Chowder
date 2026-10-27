# REAP128_v2 — phase 33 original-model qualification evidence (26 October 2026)

**Release:** `BLOCKED`, `chosen:null`, `release_approved:false`. No Gen-1 approval. Draft PR #282 remains unmerged. This is an **experimental** branch report and must not replace the frozen preregistered selection contract.

## Original-weight identity and compute accounting

* Original Q4_K_M GGUF `G:/reap128-v2-q4_k_m-20261025.gguf` — 11,563,915,264 bytes, SHA256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.
* Original Q8_0 GGUF — 19,748,435,968 bytes, SHA256 `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
* 18,543,997,568 total stored weight elements; 40 decoder layers, 128 routed experts per layer, top-8 expert selection. Approximate active weight elements/token = 3,444,503,168 (18.575% of stored elements), **not** a measured FLOP count.
* Frozen original native serving: `--fit on -np 4 -c 16384 --jinja`, 41/41 CUDA layers, zero CPU expert offload, four 4,096-token native slots. Context-expanded AutomationBench tests are explicitly exploratory alternate serving configurations.

## Complete original Linux research cohorts (NOT original Windows release-gate results)

| Model | Canonical GSM8K, 1,319 | Original HumanEval, 164 | Interpretation |
| --- | ---: | ---: | --- |
| Q4_K_M | 1,229/1,319 = 93.1766% | 151/164 = 92.0732% | Complete original-model source/native research; generated-code replay not independently sandbox certified |
| Q8_0 | 1,220/1,319 = 92.4943% | 152/164 = 92.6829% | Complete original-model source/native research; generated-code replay not independently sandbox certified |
| random128_v2 Q8 control | 964/1,319 = 73.0857% | 45/164 = 27.4390% | Independently audited original source/native control; generated-code execution certification pending |

Original research output throughput receipts: Q4 GSM8K 98.4385 tok/s and HumanEval 89.2712 tok/s; Q8 GSM8K 81.6421 tok/s and HumanEval 74.7642 tok/s. These are **Linux research rates**, not the preregistered same-hardware dense-Q4 first-200 speed ratio. Original paired GSM8K Q8 minus Q4: both right 1,198; both wrong 68; Q8-only 22; Q4-only 31; delta −0.6823 percentage points. **Descriptive only**, not the original Amendment 4 paired noninferiority CI.

Additional research-only paired GSM8K calculations from independently source-bound 1,319-ID contingency populations (one-sided 95% sample-Wald lower bound `mean(d)-1.645*sample_sd(d)/sqrt(1319)`, not Amendment 4's unresolved paired McNemar inversion):

| First arm minus second arm | First-only correct | Second-only correct | Difference, percentage points | Exploratory one-sided 95% lower bound, percentage points |
| --- | ---: | ---: | ---: | ---: |
| Q8 minus Q4 | 22 | 31 | −0.6823 | −1.5901 |
| Q4 minus random128 Q8 | 291 | 26 | +20.0910 | +18.0647 |
| Q8 minus random128 Q8 | 296 | 40 | +19.4086 | +17.2976 |

These are **not** source-reproducible independent Windows release Gate0/1/2 outcomes or the prospectively unresolved v2-v1 Amendment 4 CI. Never feed them to the original selector as completed release evidence.

## Exact registered eleven-benchmark scorecard

| # | Registered benchmark | Q4_K_M completed valid evidence | Q8_0 completed valid evidence | Exact outstanding qualification |
| --- | --- | --- | --- | --- |
| 1 | Humanity's Last Exam, no tools | 128/128 text-only Q4 research rows; **3/128 heuristic exact/normalized matches**, no official LLM judge | Matched Q8 private Kaggle **v4 running** at last verification, not scored | Same 128 IDs, strict source/model/native audit and official judge needed; do not conflate 128/600 exploratory sample with full HLE |
| 2 | Terminal-Bench 4.0 | No official completed Q4 suite | No official completed Q8 suite | Published terminal task containers and official grader, independently qualified execution host/harness |
| 3 | DeepSWE v1.1 | No official completed Q4 suite | No official completed Q8 suite | Original 113-task container/repository workload and official verifier/harness |
| 4 | ProgramBench | No official completed Q4 suite | No official completed Q8 suite | Original 200-task binary-to-program harness and hidden behavioral verification |
| 5 | NL2Repo-Bench | No official completed Q4 suite | No official completed Q8 suite | Original source/build/test harness with isolated code execution |
| 6 | AutomationBench v1.0.6 | **10 scored official PUBLIC attempts, 8 unique, zero strict binary passes** across noncomparable exploratory context profiles; 7 other original-context task aborts are UNSCORED | **No verified matched Q8 official public tasks** | 600 exact original official-source population, task hashes, original model/native profile, binary rubric/worldstate replay; protected private leaderboard not claimed |
| 7 | Agents' Last Exam | No official completed Q4 suite | No official completed Q8 suite | Original agent harness and 122-task v1.1 benchmark execution/scoring |
| 8 | ApexBench v1.1 | No verified score | No verified score | Confirm official dataset/harness, licensing and execution eligibility |
| 9 | Chartography, with tools | No verified score | No verified score | Original chart visual inputs, vision/tool pipeline and original judge; no invented image-token support |
| 10 | BabyVision, with tools | No verified score | No verified score | Original image inputs, visual tool protocol, scorer and reproducible original-model path |
| 11 | ZeroBench-main, with tools | No verified score | No verified score | Original visual image/tool prompts and scorer; no substituted multimodal model |

**Outside the eleven:** GSM8K and HumanEval are the complete research qualification baselines in the preceding table. A Q4 local Windows 16-question smoke had 15/16 but is not the frozen 1,319-item release gate. The 10 AutomationBench completed attempts cannot be pooled into 10 unique tasks or a 600-task accuracy percentage.

## New source-bound public-600 hardening

Authoritative public AutomationBench 1.0.6 source at Git SHA `4a8e1061254004d9dac807054eed33fad7d1ff14`, fixed six-domain original population and `(example_id,prompt)` ordered SHA256 `1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5`. Verified **600 unique task names, 100 per domain, but only 570 globally unique numeric example IDs**: numeric `example_id` legitimately repeats across domains. Use scoped domain+ID, never unique-global numeric ID or exported result ordinal.

Pinned upstream `automationbench/runner.py` computes `_task_contract_sha256`, but `automationbench/scripts/eval.py` drops `_task_contract_*` and `example_id` when projecting outputs to `export_results`; hence official JSON **does not carry its own cryptographic input-to-rollout binding**. Upstream `tasks[].id` is an export ordinal, not its original `example_id`; `score` is partial reward, while strict binary `task_completed_correctly` is 1 only if the partial reward equals 1. Existing triage that merely checked 600 names was insufficient.

New independent `reap_automationbench_population_20261026.verify_population` reconstructs original 600 names, scoped identities, prompt prefixes and assertions from the verified original source, rechecks exported scored/excluded assertion arithmetic, strict raw binary reporting, usage task ordinal/name/total tokens, and rejects unexecuted trailing tool turns, forged task swaps and partial exports. **It deliberately does not claim independent end-state worldstate execution or certify frozen Gen-0**. `reap_q4_automationbench.py` invokes this proof only for requested full 600, preserving prior six-task exploratory census behavior.

Validation: **310 passed** source/resource/runner/sandbox/statistics regression tests; Ruff PASS; Git diff whitespace PASS. Test fixtures simulate only adversarial data; **no fake benchmark/model scores** were produced. Experimental commits `f23d5d6` (Q8 native offload logging) and `ca79f95` (source-identity auditor, tests) pushed to `experiment/reap128v2-test-unblocked-20261026`; protected PR #282 remains open draft.

## Original Windows Q4 first-200 admission and execution

Fresh actual source-bound admission 26 October 2026: `ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED`. Minimum free physical RAM **29.314598632812498 GiB** before startup; commit **34.3145986328125 GiB**; VRAM **12,288 MiB**; F: disk **8 GiB**. These thresholds derive from the original SHA-pinned 12-row historical abort + original prompt cache, and preserve continuous 7-GiB physical/8-GiB commit/8-GiB disk runtime safety floors. No essential Codex/Hermes/Brainz/WSL/other-user jobs were stopped; idle Ollama had no loaded model at admission.

The new original frozen 200-item Windows supervised run **COMPLETED**, at private exclusive root `F:/reap-v2-q4-windows-200-20261026-phase33-original/original-first200`, with original admission `original-admission.json`. This is genuine original Q4 native inference, not substitute scores. `watchdog-terminal.json` reports exactly `COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY`, **200 original unique frozen IDs (`gsm8k-0` through `gsm8k-199`), 189/200 correct (94.5%), 192/200 naturally terminated (96%), zero reported trigram degeneration, 72,390 completion tokens**, 425 append-only resource observations, 1,340.391 seconds total including startup, and a verified shutdown. The original session ID is `c30968efd57d4498ab5f64ab3e6462c5`; native server PID 19480, port 8790, 41/41 CUDA offload and **zero CPU expert offload**. Frozen serving flags, Q4 GGUF SHA and the independent original source-gate speed verifier all match.

The sealed original `serve.json.speed_session` is identical to `watchdog-terminal.json.source_bound_speed_session`, schema `chowder.reap-q4-same-session-speed.v1`; source digest `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`; 200-item JSONL SHA256 **`d83ec4aac8387552d3da722b59dff24749a6a11d5068435cb4e1e29518ab5aec`**. The production source verifier `reap_gates._verify_new_q4_speed_session` returned **true** against actual private bytes. Original first200 mean per-item `predicted_per_second` **40.770297720291175 output tok/s** (the metric used by the original frozen speed gate); item aggregate tokens divided by summed per-item seconds is **17.94376677812838 tok/s**, **not** simultaneous-system generation throughput. Do not substitute one definition for the other or compare cross-host Linux Q8 rates as a frozen speed ratio.

Observed watchdog minima: free physical RAM **7.77728 GiB**, Windows free commit **30.35173 GiB**, free WDDM VRAM **43 MiB**, F: output space remained above 35 GiB. These are host-wide sampled minima, **not isolated instantaneous Q4 peak allocations**; another independently owned Ollama model loaded while the benchmark was active, and its resources cannot be attributed to this Q4 session. The run remained above the original 7-GiB RAM, 8-GiB commit and 8-GiB disk runtime floors, but RAM was within 0.78 GiB of the floor. Independent Ollama/WSL/ChatGPT jobs were preserved, not forcibly stopped.

This is a completed original **Windows Q4 first200 CANDIDATE** result. It is **not** a dense-vs-Q4 ≥1.3× speed comparison, 1,319-item Windows Gate0/Gate1 completion, HumanEval164 Gate2 completion, BF16 parity, Q8 matched Windows benchmark, or any release/Gen0 selection. The original dense Q4 matched 200-item reference is still missing; no valid speed ratio may be calculated. A matching dense model is 21.166 GiB and cannot be presumed safe on this single 16-GiB GPU or the previously failed 29.4-GiB-cgroup Kaggle host.

### Independent post-run correctness replay

Added `chowder_batch/reap_q4_first200_postrun_20261026.py` and adversarial tests. It loads only the original sealed receipts; independently calls `reap_gates._verify_new_q4_speed_session`; checks every one of the original 200 row IDs, expected native source/model hashes and runtime telemetry; loads canonical full 1,319-item GSM8K test source from the offline local Hugging Face cache and independently verifies its frozen full-source SHA; re-runs the *unchanged actual production* `reap_eval._score('gsm8k',...)` over the preserved 200 original model completion texts. No new model inference and **no generated Python execution** occurs. The scorer agreed on **all 200 rows**, reproducing **189/200**; output status is `ORIGINAL_WINDOWS_Q4_FIRST200_INDEPENDENTLY_RESCORED_CANDIDATE_ONLY`. The independent private receipt is `F:/reap-v2-q4-windows-200-20261026-phase33-original/independent-rescored-first200.private.json`, SHA256 `895b026255197218f962076882e319a9da9069bc8779f7312b1404592728d4da`. It still reports `chosen:null`, Gen0 `BLOCKED`.

### Matched original Q8 official-public AutomationBench preparation and CPU-only actual Kaggle check

New `reap_q8_automationbench_kaggle_20261026.py` prepares **two separate**, original-Q8 *private* Kaggle experiments matched to existing original Q4 evaluated task IDs and exact experimental serving flags, not the four-slot frozen Gen0 profile. The six-task 65,536-context profile and two-task 73,728-context profile have **different** generation/decoding flags and must not be pooled into one eight-task score. The preparer validates original Q4 result/model/native/protocol SHA, clean official upstream 1.0.6 Git SHA, complete 600 prompt set, exact selected task-contract hashes, original Q8 19,748,435,968 bytes SHA, expected native two-T4 41/41 GPU offload/no CPU experts, and pre-/post-load resource floors. Generated outputs and model contents are private; neither model is substituted or launched by merely preparing the notebook. Synthetic adversarial static tests and actual Q4 SHA checks passed.

Private official Git-tree archive `automationbench-v1.0.6-official.tar.bin` has 18,257,920 bytes and SHA256 **`0d9b2b3ea8d54250bf3f14a0ce1d0b7d766d445047cdd730c88575d19b57f278`**. Kaggle's normal `.tar` dataset upload **automatically unpacked** it and therefore lacked the file the protected notebook required. The source was preserved under `.tar.bin` and staged in a separate **PRIVATE, ready** Kaggle dataset `nikmarco/reap128v2-automationbench-sourcebin-20261027`, verified with Kaggle's actual file listing. Fresh private Q8 notebook packages reside in `F:/reap-v2-q8-automationbench-private-20261026/six-v2-sourcebin` and `.../guard2-v2-sourcebin`, with actual generated script hashes `887437535bea8bcaaa9dd08d3ce776b32a05bcb020b6f888a3082111ea90f1c2` and `bad3fd79841084d6e8a79ca523affb74d72eaef737afa5f84fec30eca4d1e1cb` respectively. Neither Q8 AutomationBench GPU experiment has been launched or scored.

An additional actual **CPU-only**, private, internet-disabled Kaggle probe was submitted and completed, at `nikmarco/reap-q8-automationbench-offline-cpu-preflight` (Kaggle-selected *actual notebook slug*, not merely the requested metadata ID). It SHA-verified the exact mounted original archive, confirmed `/usr/bin/python3.13` is installed and `uv` is installed, and tested importability of the official dependencies under `-E -s`. **Actual outcome: `BLOCKED_OFFLINE_PYTHON313_DEPENDENCIES_ABSENT`; `ModuleNotFoundError: No module named 'verifiers'`** on Python3.13 and available alternatives. Private pulled source receipt SHA256 `88a5169abe49168daa420976dd33728a81c5c7cb406600b899c2439096968d85`. This is a runtime admission failure **before Q8 GPU model load or benchmark inference**, not a model/task correctness failure. Do not launch the unchanged Q8 Kaggle GPU kernels until official versions and offline dependencies are independently staged and verified. A read-only, original `uv.lock`-based requirements export resolved 118 packages into an exclusive private prerequisites file, SHA256 `fbd77b926b40ec12d75501c1959851da9490688f116d4db89cd4f207cbfa24cd74d`; no Linux wheelhouse is certified or uploaded yet. The source model remains protected.

## Frozen statistical and release blockers

Original D0: 40 unique layers and 0<perplexity<20. Gate0: termination and degeneration rates versus original dense. Gate1 paired sample-Wald one-sided 95% LB(Q4−dense)>−0.03 on original GSM8K 1,319; Gate2 one-sided LB>−0.10 on HumanEval 164; random128 GSM8K paired LB(Q4−random)>0; same-source first-200 sustained Q4 speed ≥1.3× dense Q4. Original REAP160 fallback and original BF16 numerical parity remain unqualified.

Optional Amendment 4 v2-v1 paired McNemar **nonzero-margin risk-difference CI** lacks a uniquely specified inversion method. Already exposed data must not be retrospectively used to choose a passing CI. Require independent preregistration of one explicit algorithm and NEW locked disjoint confirmation cohorts, or keep `BLOCKED`. Independent HumanEval candidate-program sandbox has unset approved host-side capability/witness SHA; **no generated code may execute** until qualified disposable VM with outside observer and benign negative controls. The original dense Q4 on Kaggle T4×2 stopped at 9.992GiB unused cgroup (10GiB floor) before fully loaded; repeat only on a meaningfully larger admitted host.

## Reproduction and provenance

On this authorized Windows experimental worktree:

```powershell
python -m pytest -q tests/test_reap_automationbench_population_20261026.py tests/test_reap_q4_automationbench_20261026.py tests/test_reap_q4_windows_benchmark_watchdog_20261026.py
python -m chowder_batch.reap_q4_resource_admission --out F:\<exclusive-new-private-admission>.json
python -m chowder_batch.reap_q4_windows_benchmark_watchdog_20261026 --admission F:\<original-admission-file>.json --out F:\<exclusive-new-private-benchmark-dir>
kaggle kernels status nikmarco/reap128v2-q8-matched128-hle-textonly-20261026
```

Do not rerun the preceding original Windows experiment or the running Kaggle v4 as a duplicate. The exact private receipt SHA/files and verified final outcome must be read from their original exclusive directories, never constructed or substituted. Public benchmark references: [Terminal-Bench 4.0](https://www.tbench.ai/?version=4), [DeepSWE v1.1](https://deepswe.datacurve.ai/), [ProgramBench](https://programbench.com/), [NL2Repo-Bench](https://github.com/multimodal-art-projection/NL2RepoBench), [Agents' Last Exam](https://agents-last-exam.org/leaderboard), [HLE](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only).
