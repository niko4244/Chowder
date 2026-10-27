# REAP128_v2 phase34 — original Windows Q4 full-1319 source-bound result

**Date:** 2026-10-27 (America/Chicago). **Model:** original REAP128_v2 Q4_K_M, exact GGUF SHA-256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`. **Status:** `COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY`; frozen Gen-0 remains `BLOCKED`, `chosen:null`, `release_approved:false`.

## Actual original Windows inference and independent replay

The real source-bound local evaluator completed **all 1,319 unique original GSM8K test items**, with **1,224/1,319 correct (92.797574%)**, **1,281/1,319 naturally terminated (97.119030%)**, **zero reported trigram degeneration**, and **459,264 generated completion tokens**. It used the unchanged original `llama-server` flags `--fit on -np 4 -c 16384 --jinja`, native **41/41 decoder layers on CUDA** and **zero CPU-offloaded routed experts**. The original cached GSM8K canonical source SHA-256 is `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`.

Private immutable evidence is under `F:\reap-v2-q4-windows-full1319-phase34-original-20261027-0152` with the exact terminal `full1319-terminal.json`, original model output folder `reap128_v2_q4`, 1,319 original item completions, native serving log, and **822 append-only watchdog samples**. Source-bound full result rows SHA-256 `66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f`; terminal receipt SHA-256 `fabd6e8381ab68f73473b8a57bd80a94c2d2cf675968b7527fa5a37795bb0697`. Original child exited **0**, **native shutdown verified**, full wall time **2,515 seconds**.

An additional **read-only post-run process** independently invoked `complete_original_gsm8k` against the preserved original `serve.json`, full canonical source, saved raw completion text and original evaluator `_score`. It matched every field of the terminal `source_bound_scoring` and independently reproduced **1,224/1,319**; no further model inference or scorer changes. This reexecution verifies the source-bound correctness count, not the independent official frontier-benchmark judges.

At original prelaunch admission the host had **39.106 GiB available physical RAM**, **64.823 GiB available commit**, **14,823 MiB free VRAM**, and **34.614 GiB free F: disk**, satisfying the original source-bound start floors of respectively **29.315 GiB**, **34.315 GiB**, **12,288 MiB**, and **8 GiB**. Actual sampled runtime minima were **11.278 GiB free physical RAM**, **31.295 GiB free commit**, **34.325 GiB free F: disk**, and **179 MiB host-wide free WDDM VRAM**. The registered continuous host RAM/commit/disk floors (7/8/8 GiB) were respected. The low GPU-free host sample is **not** proof of the Q4 model's isolated peak usage: a separately owned Ollama runner briefly restarted before its supervisor was temporarily quiesced. No valid per-process peak allocation or uncontended Windows speed ratio is implied. The original model continued without a monitored resource-floor violation.

This **Windows Q4 candidate is distinct** from the previously completed Linux original Q4 GSM8K **1,229/1,319** research cohort. Do not pool samples across hosts or claim an unchanged-geometry dense-Q4 speed baseline. The original frozen Gate0/Gate1 still require their dense comparator and paired confidence-bound evidence, and Gate2 original HumanEval still lacks independently qualified generated-code sandbox execution. The original four-slot Windows Q4 full evaluation is **outside the eleven registered frontier suites** and cannot fill any one of them.

## Eleven original frontier suites — status, no substituted scores

| # | Registered original suite | Source-bound outcome or exact unmet requirement |
|---|---|---|
| 1 | Humanity's Last Exam, text/no tools | Actual Q4 original text-only 128/128 research outputs: 3/128 **nonofficial deterministic normalized matches**; Q8 actual 99/128 outputs, 0/99 heuristic matches, 29 missing; **neither is an official HLE judge result**. Requires genuine completion of the original Q8 matched ID set and original official judge to claim official score. |
| 2 | Terminal-Bench 4.0 | Requires original Harbor agent/terminal task images, tool protocol, eligible access, independent sandbox and official grader. No verified model-score claim. |
| 3 | DeepSWE v1.1 | Requires original mini-SWE agent/repository environments, task split, isolated execution and grader. No verified model-score claim. |
| 4 | ProgramBench | Requires original binary/documentation tasks, agent execution, hidden behavioral verification and safe sandbox. No verified model-score claim. |
| 5 | NL2Repo-Bench | Requires original task/repository set, OpenHands-compatible headless environments and full build/test grader. No verified model-score claim. |
| 6 | AutomationBench 1.0.6 | Official public source at Git `4a8e1061254004d9dac807054eed33fad7d1ff14`, 600 tasks (six domains × 100) with scoped source IDs. Original Q4 has 10 completed exploratory attempts / eight distinct tasks and zero strict binary passes on **non-poolable** native profiles; no 600-task final claim. The private source SHA-pinned Q8 Kaggle matched six+guard2 experiment is waiting for full original Python3.13 runtime/offline wheel and independent CPU source verification. |
| 7 | Agents' Last Exam v1.1 | Needs original task environments, agent/tool harness, official grading and isolated execution; no verified model-score claim. |
| 8 | ApexBench v1.1 | Exact named original official dataset/harness remains unverified; **do not substitute** a similarly named benchmark. |
| 9 | Chartography (with tools) | Original chart images + official with-tools and judging protocol needed; a **text-only** Q4 GGUF cannot legitimately consume the image inputs. |
| 10 | BabyVision (with tools) | Original vision inputs, tools and official grading needed; text-only Q4 invalid as a visual benchmark contestant. |
| 11 | ZeroBench-main (with tools) | Original visual main task set, PASS@5 generation and tool judge needed; text-only Q4 incompatible. |

All incomplete/blocked/tool-or-modality-ineligible suites remain **unqualified**, **not zero**, **not passing**, and never interchangeable with another model, screenshot OCR, synthetic questions, partial cohorts or substituted scorer. Their presence on this table is tracking, **not successful test execution**.

## Reversible host resource ownership

The user authorized temporary suspension of RAM/VRAM consuming services. The exact private pause and rollback evidence is `F:\reap-v2-phase34-quiesce-20261027-0141`: LiteLLM and PostgreSQL Podman containers were paused, original supervised Ollama model and launcher command recorded, and an Ollama-only Hermes `restart_policy: mode: never` was applied with an SHA-pinned original YAML backup during the local test. The source-bound local run has now exited with native shutdown verified. Restore the *exact* prior configuration and two paused containers when no other owned local original GPU benchmark is active; preserve independently owned tasks. Never leave an undocumented permanent disabling of Ollama, Hermes, WSL or LiteLLM.

**Scientific selection:** `BLOCKED`; PR #282 must remain **OPEN DRAFT, UNMERGED**. No Gen-1, new weights, scorer changes, or synthetic frontier scores authorized.
