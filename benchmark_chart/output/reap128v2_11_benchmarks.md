# REAP128_v2 — 11 benchmark comparison

**Status:** no official same-protocol 11-suite REAP128_v2 run has been recorded. All *direct* cells are **N/T** (not tested), never zero. Published model-vendor references appear in separate columns and are not matched three-way trials.

| Screenshot category | REAP direct | Qwen direct | Opus direct | Qwen publisher ref† | Opus publisher ref† |
|:--|--:|--:|--:|--:|--:|
| HLE (no tools, text subset) | N/T | N/T | N/T | 30.8 | N/T |
| Terminal-Bench 4.0 | N/T | N/T | N/T | N/T | 66.4 |
| DeepSWE v1.1 | N/T | N/T | N/T | 42.2 | N/T |
| ProgramBench | N/T | N/T | N/T | N/T | N/T |
| NL2Repo-Bench | N/T | N/T | N/T | 42.3 | N/T |
| AutomationBench v1.0.6 | N/T | N/T | N/T | N/T | 40.0 |
| Agents' Last Exam | N/T | N/T | N/T | 20.4 | N/T |
| Apex Bench v1.1 | N/T | N/T | N/T | N/T | N/T |
| Chartography (with tools) | N/T | N/T | N/T | N/T | 89.0 |
| BabyVision (with tools) | N/T | N/T | N/T | N/T | N/T |
| ZeroBench-main (with tools) | N/T | N/T | N/T | N/T | N/T |

† Publisher-reported percentage, **not** a local score. A number may use a different cohort, judge, harness, version, metric, or tool set. Do not average or rank these columns. N/T means no verified value for that model and category.

## Published reference provenance and comparability

| Benchmark | Model | Publisher-reported | Exact limitation | Primary source |
|:--|:--|--:|:--|:--|
| HLE (no tools, text subset) | Qwen3.8-27B | 30.8% (accuracy) | different_cohort: Qwen reports HLE judged by GPT-4o; pure-text-only/no-tools cohort is not established | [Official publisher](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Terminal-Bench 4.0 | Claude Opus 5.5 | 66.4% (success_rate) | same_named_benchmark: Opus 5.5 xhigh effort, Claude Code harness, five trials/task; vendor result | [Official publisher](https://www.anthropic.com/claude-opus-5-5) |
| DeepSWE v1.1 | Qwen3.8-27B | 42.2% (resolved_rate) | different_harness: Qwen Claude Code harness; screenshot DeepSWE Mini-SWE-V2 | [Official publisher](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NL2Repo-Bench | Qwen3.8-27B | 42.3% (success_rate) | unverified_harness_match: Qwen Claude Code harness with repo download blocked; screenshot setup not fully disclosed | [Official publisher](https://huggingface.co/Qwen/Qwen3.8-27B) |
| AutomationBench v1.0.6 | Claude Opus 5.5 | 40.0% (success_rate) | unverified_version_match: Zapier evaluation reported by Anthropic; public page omits version v1.0.6 | [Official publisher](https://www.anthropic.com/claude-opus-5-5) |
| Agents' Last Exam | Qwen3.8-27B | 20.4% (pass@1) | unverified_metric_match: Qwen reports both Pass@1 20.4 and Score 42.9; this column displays Pass@1 only | [Official publisher](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Chartography (with tools) | Claude Opus 5.5 | 89.0% (accuracy) | unverified_tool_match: Anthropic with-tools result, unspecified common tool configuration | [Official publisher](https://www.anthropic.com/claude-opus-5-5) |

## Actual REAP measurements from immutable local audits (different benchmarks)

These scores are **Linux research evidence**, not any of the eleven screenshot categories and not an original Windows Gen-0 release gate.

| REAP checkpoint | Audited benchmark | Correct / total | Accuracy | Qualification |
|:--|:--|--:|--:|:--|
| REAP128_v2 Q4_K_M | GSM8K original Linux | 1229/1319 | 93.18% | source-audited provisional research |
| REAP128_v2 Q8_0 | GSM8K original Linux | 1220/1319 | 92.49% | source-audited provisional research |
| REAP128_v2 Q4_K_M | HumanEval original Linux | 151/164 | 92.07% | historical independently re-executed; strict sandbox attestation still limited |
| REAP128_v2 Q8_0 | HumanEval original Linux | 152/164 | 92.68% | original scorer only; separate independent replay PENDING |

The 16-item shared GSM8K pilot previously reported REAP Q4 and Q8 each 15/16, but Qwen3.8-27B and Opus 5.5 each **N/T** owing to rejected provider requests. That slice is insufficient for a ranking.

## Definitions and primary sources

- Eleven labels and special scoring constraints: [DeepSeek October 17 change log](https://api-docs.deepseek.com/updates/) (source of the screenshot *categories*, not scores for our three exact models).
- Exact Qwen model/official results: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Exact Opus model/official results: [Anthropic Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5).
- Benchmark definitions: [HLE (no tools, text subset)](https://github.com/centerforaisafety/hle), [Terminal-Bench 4.0](https://www.tbench.ai/run), [DeepSWE v1.1](https://deepswe.datacurve.ai), [ProgramBench](https://github.com/facebookresearch/ProgramBench), [NL2Repo-Bench](https://github.com/multimodal-art-projection/NL2RepoBench), [AutomationBench v1.0.6](https://github.com/zapier/AutomationBench), [Agents' Last Exam](https://agents-last-exam.org), [Apex Bench v1.1](https://api-docs.deepseek.com/updates/), [Chartography (with tools)](https://github.com/surge-ai/chartography), [BabyVision (with tools)](https://github.com/UniPat-AI/BabyVision), [ZeroBench-main (with tools)](https://zerobench.github.io).
- Local baseline commit: `e62cf4e2952bbd97b5b81ae9a5c81fbc2e90f6fb`. Source SHA-256 pins:

  - `benchmark_chart/references.json`: `8b7da2e68aafc769da5e47546183e04cadf654c1593df40394f6898f2c2bf4b1`
  - `docs/quals/reap-v2-gen0-integrated-scientific-state-20261026.md`: `177cabf3a73eeac23f879f4477916afa4da9c0705c9202701821858977f7dd0c`
  - `docs/quals/reap-v2-qwen38-opus55-common-test-20261026.md`: `29acb3dfa6aa65ec63a5c29db87ceff11906a132ecad4d5a7426744e2be508c5`

All source SHA-256 pins above hash bytes with CRLF normalized to LF, so the same Git evidence works on Windows and Linux.

Regenerate (standard-library Python only): `python -m benchmark_chart.generate` from the isolated worktree root. The generator refuses altered evidence, wrong model IDs, score/quote mismatches, unsupported category substitutions and protocol promotions. No GPU, API, generated-code execution, private output, or network call is involved.
