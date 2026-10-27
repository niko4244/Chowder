# REAP128_v2 phase38 — authentic frontier execution receipts (October 27, 2026)

**Scientific decision: BLOCKED; chosen:null; release_approved:false. No PR merge, no Gen1 training.** This document records actual original model inferences, not completion of all eleven frontiers. It supplements the original Phase37 scorecard. The verified values below are an immutable **11:32 CDT snapshot**; the original AutomationBench600 evaluator remains active and subsequently advanced further.

## 1. Authentic original AutomationBench v1.0.6 Q4 source run

Original 600-task public population, 570 source groups, exact 600-task prompt SHA256: 1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5. Original git-clean Zapier official evaluator commit: 4a8e1061254004d9dac807054eed33fad7d1ff14. Original REAP128_v2 Q4_K_M GGUF bytes: 11,563,915,264; SHA256: 2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7.

The original public evaluator executes in Python3.13, one authentic original source index at a time, with the official API-worldstate toolset and original binary assertion verifier. An active Windows native llama-server was source-matched to the GGUF and served real model answers and tools. The distinct **NONCOMPARABLE exploratory** model flags were --fit on -np 1 -c 73728 --repeat-penalty 1.1 --dry-multiplier 0.8 --temperature 0.7 --jinja, through guard.v1. These cannot substitute for the frozen Gen0 four-slot flags --fit on -np 4 -c 16384 --jinja.

Private live receipt root: F:\reap-v2-q4-original-automationbench600-checkpointed-20261027-0953. Original owner SHA256: 849309859a5c1ef8736ee67668b33d8fbbf1dc3334b246451af4c4e5d46f220a. Actual native owner process PID40844 and llama-server PID33528 at 11:23 (point-in-time only; always revalidate PID/argv before acting).

The **independent frozen partial snapshot** at F:\reap-v2-original-q4-automationbench600-independently-verified-20261027-1136.json has SHA256 **480c98a857a347016348bd25af2048072cdec8ae23090db302a34516538cdd09**.

| Actual original metric at independent checkpoint | Value |
|---|---:|
| Registered public original items | 600 |
| Completed, independently rescored original source items | **16/600 (2.67%)** |
| Strict original complete-task WorldState passes | **0/16**, NOT 0/600 |
| Required positive WorldState assertions passed | **5/57** |
| Guard/negative/exclusion assertions passed | **95/95** |
| Genuine original-model calls | **549** |
| Genuine original tool calls | **706** |
| Evaluator-reported input tokens, including repeated context | **10,002,933** |
| Evaluator-reported output tokens | **203,872** |
| Aggregated model processing/generation seconds | **4,572.521** |
| 17th source item at checkpoint | index16, running UNSCORED |
| Official complete 600-population score | **UNKNOWN, IN PROGRESS** |

The new source-only live auditor, chowder_batch/reap_original600_live_independent_audit_20261027.py, independently reopens every immutable scored source JSON and receipt, checks source Git/Task/Model SHA and original native flags, grader pass and positive/exclusion assertion arithmetic, sample denominator and order, tokens, total native tool/model calls, and refuses forged or skipped checkpoints. The newly integrated batch executor, chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py, fixes an observed production problem: original evaluator save_results=False otherwise exports only when the entire 600 run finishes. The original monolithic run was stopped after 3/570 groups specifically to prevent **losing two hours of uncheckpointed partial execution**; original model inference from that first session **is NOT pooled** with this new source-bound run. An exact transition receipt preserves the old private evidence at F:\reap-v2-original-q4-automationbench-full600-live-20261027-0932.

Each real completed original task is now independently saved once with immutable raw original evaluator JSON and strict-score receipt; source-verified rows survive crashes and are not rerun on resume. An incomplete row remains UNKNOWN and is preserved under an exclusive quarantine identity before re-execution. Tests for forged scores, tampered sources, skip indexes, double use of interrupted rows, missing assertions, response model mismatch and fake runtime metrics are included.

**Observed weakness, not a zero score for all600:** several tasks issued extensive genuine tool interactions yet failed to cause required affirmative mutations. One task stopped via guard-labeled runaway truncated tool call; others reached agent step budget or ended without all required WorldState assertions. Only five mandatory positive assertions were met across the first16 tasks. This is authentic empirical evidence of Q4 tool workflow difficulty under an experimental profile; it does NOT establish causal model-vs-harness attribution or a full600 result.

### Measured local resource bottleneck

At 11:23 the source-owned Q4 llama-server held **24,023,855,104 bytes** Windows working set (~22.37GiB), including model tensors and runtime/KV. Windows physical free ~**13.86GiB**, available commit ~**31.53GiB**, NVIDIA 5060Ti total16,311MiB with 15,678MiB used / 373MiB free (HOST-WIDE). At 11:32 the host had ~21.97GiB physical and ~40.15GiB commit free, GPU 13,160MiB used / 2,891MiB free, after named supervised Ollama model unload. These are **point-in-time host resource observations, not model-exclusive peak VRAM or RAM allocations**. The source-pinned native server reported 41/41 CUDA layers and no CPU expert offload; no original admission or runtime reserve was lowered. Per-16-task repeated prompt usage exceeded 10 million counted input tokens, the principal directly observed inference cost. Do not disrupt the current original Q4 server with another GPU/model test.

## 2. Real Terminal-Bench 4.0 Q4 model-driven original task

Authenticated 66-task source terminal-bench/terminal-bench@4.0.0, official Harbor0.25.0/Terminus-2 and Podman task container. Original task interleaved-vigenere, task.toml SHA256 d2b9b4f98304d5458ccde463b9670a9ddd89cd4c01bbd03e93a4b92860bcab8a, instruction SHA256 d9a28686a9f18d5f76ef876f7b035d90bdede6d0625056e581fc65fba0e2f3cd, original REAP Q4 GGUF verified.

On separate isolated CPU-only 8192-context native Q4 server, model actually generated three agent responses and original shell tool actions, native prompt/completion token pairs (1245,474), (4221,804), (6193,521). The next 13,753-token request exceeded its 8192-token context. Fallback full and short summarization did not restore original verifier completion. Original task artifacts remained absent. One source-authenticated actual incomplete agent trial, **ZERO defensibly graded Terminal-Bench tasks**, NOT an official task score of zero. Official task and private Harbor receipt stored under evidence/reap_q4_terminalbench4_worker12_20261027; separate CPU model server stopped by worker. Require independently admitted >=32k/65k context, working Podman Compose cp and original verifier before another task. No duplicate GPU load while AutomationBench is active.

## 3. Original registered eleven-frontier status (not invented)

| Registered target | Q4 original authentic run | Q8 original authentic run | Strict evaluation status |
|---|---|---|---|
| 1. HLE no tools, text only | 128/128 responses; 3/128 **nonofficial heuristic**, no judge | 99/128 responses; 0/99 nonofficial heuristic; real missing29 cgroup mmap aborted with zero new rows | Official HLE judge and Q8 source-complete response set absent |
| 2. Terminal-Bench 4.0 | 3 real model-driven steps, **0 validly graded** | none | Context/Harbor verifier incomplete; full66 absent |
| 3. DeepSWE 1.1 | no genuine complete113 | none | Original independently safe code-edit agent and verifier absent |
| 4. ProgramBench | no genuine complete200 | none | Original binary/docs scorer/execution admission unavailable |
| 5. NL2Repo-Bench | no genuine complete104 | none | Original repo environment/OpenHands execution admission unavailable |
| 6. AutomationBench v1.0.6 | **real 16/600 original graded, 0/16 strict**, live run | no valid original full600 | Full600 original incomplete; Q4 experimental context profile |
| 7. Agents' Last Exam v1.1 | no complete122 | none | Real agent app/tool benchmark not executed |
| 8. ApexBench v1.1 | no original run | none | Exact registered v1.1 benchmark source unverified; no substitute |
| 9. Chartography with tools | no native image tasks | no native image tasks | Original text-only model lacks native vision input |
| 10. BabyVision with tools | no native image tasks | no native image tasks | Original text-only model lacks native vision input |
| 11. ZeroBench-main with tools | no native image tasks | no native image tasks | Original text-only model lacks native image input, no valid pass@5 |

Previously independently checked original Windows **GSM8K1319 Q4**: **1,224/1,319 (92.797574%)**, native original four-slot flags, genuine 41/41 CUDA and zero CPU expert offload; it is **outside the eleven**. Do not repeat. Previously original separately served Linux HumanEval164 candidate-response research Q4 **151/164** and Q8 **152/164** with their original cohort hashes; NOT a fresh independently admitted Windows disposable-generated-code Gate2. Separate v38 rootless source-reviewed research transport exists; current original v2 Gate2 approved-sandbox SHA and independent-witness SHA remain **None**, so Gen0 HumanEval Gate2 remains UNKNOWN. Last actual execution in this mission did NOT complete independent original164 program execution; do not promote historical scores.

Already completed BF16 all40 decoder-layer **isolated** CPU/CUDA original research and full26-shard hashes does not prove chained full-model logit parity. Original dense 1319 paired candidate, dense/candidate original 200 per-item same-host speed ≥1.3x, Gate0 EOS/trigram, frozen §4 Wald Gate1, original Gate2, REAP160 speed+quality, random128 paired native controls and Amendment4 prospective interval interpretation remain UNKNOWN or BLOCKED. Historical REAP128-v1 Gate1 LB **−0.03780741197 FAIL** is preserved; no fallback fabricated.

## 4. Validation, next execution and scientific decision

The three added source-preserving test files for checkpointing, independent scorer and research-only v38 source review passed **56/56 adversarial tests**, and Ruff passed all six scoped new module/test files as of 11:31. The independent scorer has successfully executed against real live evidence and saved the cited SHA-immutable F: JSON receipt. Running the scorer causes NO new model load or original evaluator mutation.

Reproduce independent live scoring in the authorized feature tree using:

    python -m chowder_batch.reap_original600_live_independent_audit_20261027 --original-run F:\reap-v2-q4-original-automationbench600-checkpointed-20261027-0953

**Do not launch a second full600 process while original PID40844/port8892 is active.** When genuinely inactive, inspect/check original process and source identity; then safely resume the same exact original private root and immutable score proofs using the original resumable runner. Do not duplicate the failed original Kaggle Q8 HLE missing29 job or retry the previously rate-limited Kaggle CPU receipt endpoint without controlled admission. The currently verified partial 16/600 is not success under the user's completion definition, and untested tasks remain unknown.

Final original scientific selection remains **BLOCKED / chosen:null / release_approved:false**. No PR merge, no deployment, no Gen1, and no private result disclosure.
