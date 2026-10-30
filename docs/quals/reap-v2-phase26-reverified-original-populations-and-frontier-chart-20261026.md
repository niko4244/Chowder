# REAP128_v2 phase 26 — independently reverified original populations and honest frontier scorecard

**2026-10-26.** This report is a new source-audit proof and benchmark
readiness record. It does **not** substitute different source suites, execute
generated HumanEval code, represent published API model numbers as our model's
scores, restart protected services, or change the frozen selector.

## Live source, runtime and independent original replay

Before audit: `feature/reap128v2-selection-20261025` / PR #282 both at
`d96d8df6dcc8d10bf5dfd9986677ca3b68e694a8`, clean HEAD/origin,
PR **OPEN, DRAFT, UNMERGED**. Separate experimental research worktree
`C:\Users\nikma\wt\chowder-reap128v2-lab-testing-20261026` has independent
`experiment/reap128v2-test-unblocked-20261026` branch at `e62cf4e2952bbd97b5b81ae9a5c81fbc2e90f6fb` before this phase. That
test-only marker is *not* scientific authorization or a runtime safety bypass.

The phase independently reran the full `chowder_batch.reap_external_full_audit`
on all **1,319 original GSM8K and 164 original HumanEval** rows **for each
quantized arm**, reloading the frozen local canonical sources and validating
all source SHA, per-item IDs, original scorer-derived outcomes, serving
sessions, 41/41 native CUDA placement, original model/source/native machine
receipt, full source file hashes, 4×4,096 / `-c 16384` context and 320 MiB
KV. This was an **audit of original recorded HumanEval scorer booleans**;
the generated programs were not independently executed in a v2 isolation
sandbox. Linux research only; Windows Gate0/Gate1/Gate2 are unqualified.

Original complete reference sources remained in protected private `F:`
research folders. Two **new**, separate, exclusive private output files:

| Arm | Private new independent audit | SHA-256 |
| --- | --- | --- |
| REAP128_v2 Q8_0 | `F:\reap-v2-phase26-original-q8-source-audit-20261026.json` | `107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683` |
| REAP128_v2 Q4_K_M | `F:\reap-v2-phase26-original-q4-source-audit-20261026.json` | `582e57557c40abad170592db1718f5aea3e61f8d3dae2db2e71a73c99ee19ea6` |

The Q8 digest exactly matches an independently preserved earlier run, not a
newly invented statistic. Distinct serving session IDs Q8
`95876063255e45159f57e29306e0e885`, Q4
`a2a6b83e345145839cc333561b2f3baf` were unchanged. Original full
weights Q8 SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`,
Q4 SHA `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.

| Original full Linux suite | Q8_0 | Q4_K_M |
| --- | ---: | ---: |
| GSM8K items | 1,319/1,319 | 1,319/1,319 |
| GSM8K correct | 1,220 (**92.49431%**) | 1,229 (**93.17665%**) |
| GSM8K natural completion | 1,280 | 1,281 |
| GSM8K generated tokens | 467,560 | 456,671 |
| GSM8K scoring wall | 5,726.947 s | 4,639.150 s |
| GSM8K source-bound generation speed | **81.6421 tokens/s** | **98.4385 tokens/s** |
| HumanEval original-scored items | 164/164 | 164/164 |
| HumanEval original scorer correct | 152 (**92.68293%**) | 151 (**92.07317%**) |
| HumanEval natural completion | 162 | 162 |
| HumanEval generation wall | 538.774 s | 457.068 s |
| HumanEval source-bound generation speed | **74.7642 tokens/s** | **89.2712 tokens/s** |
| Independent generated-code sandbox v2 | **NOT PROVEN** | **NOT PROVEN** |

Then the phase independently reran **original same-1,319-item Q8-vs-Q4
source-bound pairing**, written only to another new private file
`F:\reap-v2-phase26-original-q8-q4-paired-gsm8k-20261026.json` SHA
`ce44c3a1551c55aa91ef5f3d415c726af0448ca0c8eae61e2ebc145d01771802`:
1,198 both correct, 68 both wrong, 22 Q8-only and 31 Q4-only;
Q8−Q4 = **−0.0068233510235** (−0.682335 percentage points).
It is **descriptive**; no posthoc Wald/McNemar CI or noninferiority PASS.

BF16 GGUF bytes **37,122,439,488**; Q8_0 **19,748,435,968**;
Q4_K_M **11,563,915,264**. Architecture 40 layers, 128 routed experts,
top-8 per layer; **18,543,997,568 stored** and approximately
**3,444,503,168 active weight elements/token** (18.575%; 320 expert
selections/token). That is not a native measured FLOPs/token claim.

## Screenshot-specific chart integrity

The reference screenshot (DeepSeek V4.1 Flash 1017 vs its peers) contains
11 **different task environments**: Humanity's Last Exam (HLE, no tools),
Terminal-Bench 4.0, DeepSWE 1.1, ProgramBench, NL2Repo-Bench,
AutomationBench 1.0.6, Agents' Last Exam, Apex Bench v1.1,
Chartography (with tools), BabyVision (with tools), ZeroBench-main
(with tools). The original GSM8K and **HumanEval are not HLE** and cannot
legitimately fill that HLE table cell. Recorded HumanEval Python source
outcomes are not Terminal-Bench, DeepSWE or ProgramBench scores either.

Current independently sourced official published peer evidence:

- Alibaba Qwen3.8-27B exact card:
  https://huggingface.co/Qwen/Qwen3.8-27B ; HLE **30.8%**, DeepSWE 1.1
  **42.2%**, NL2Repo-Bench **42.3%**, Agents' Last Exam Pass@1 **20.4%**.
  Its Terminal-Bench figure **73.0% is Terminal-Bench 2.1**, *not*
  screenshot Terminal-Bench 4.0.
- Anthropic Claude **Opus 5.5** official Sep 22, 2026 release:
  https://www.anthropic.com/claude-opus-5-5 ; Terminal-Bench 4.0 **66.4%**
  (xhigh setting), AutomationBench **40.0%** (Zapier official test,
  not necessarily same 1.0.6 harness), Chartography with tools **89.0%**,
  HLE **67.7% WITH TOOLS**. That HLE number is **not eligible** for the
  screenshot's *without tools* row. The screenshot's `Opus 5` numbers
  describe **Opus 5**, not **Opus 5.5**.

The source/metric settings are not automatically harmonized across vendors;
published numbers are **reference scores**, not a new same-harness win/loss
against locally measured REAP. The separate experimental branch had a real
OpenRouter Qwen3.8-27B attempted original GSM8K API call refused with
HTTP **402**, and local Groq-Qwen and Opus aliases also refused their
arithmetic probes: **zero** genuine Qwen or Opus original-cohort answers.

## Actual admission and full benchmark gaps

Fresh read-only Windows host counters during this qualification: available
physical RAM approximately **10.18 GiB**, available commit approximately
**17.88 GiB**, free RTX 5060 Ti VRAM approximately **11,392 MiB**.
The unchanged original Windows Q4 first-200/full admission requires at least
**29.3146 GiB physical RAM**, **34.3146 GiB commit**, **12,288 MiB VRAM**
before launch (plus disk). Every sampled local resource class is below
threshold; the original 12-row low-RAM abort already demonstrated real
unsafe behavior. Do not stop unrelated services, weaken `--fit on -np 4
-c 16384 --jinja`, change KV/concurrency, rerun the registered 200-item
session or claim a hardware-executed new screenshot-category result.

Kaggle original random128 full remote kernel remains `RUNNING`; do not
duplicate. Private Q4 Kaggle T4×2 original GGUF is source-staged and
historically completed the full original native Linux cohorts, but no
**authorized, revision-pinned, access-accepted** official `cais/hle` dataset
or HLE-specific private task/judge artifact had been staged at the latest
read. The HLE source repository is *gated*, and the model is text-only,
so image-dependent HLE, Chartography and BabyVision tool-using official
scores need appropriate compatible multimodal/tool interfaces. Terminal,
SWE, ProgramBench and NL2Repo generated shell/Python require independently
qualified filesystem, network, PID/namespace, cgroup and teardown isolation;
the historical Kaggle CPU namespace probe **REFUSED** and the current WSL
fixed-benign child saw Windows drive mounts. None of these was converted
into an authorized untrusted-code host.

No new complete REAP score for **any of the 11 screenshot rows** has been
measured in this phase; each REAP cell is **N/T**, not zero or a guessed
percentage. Available original GSM8K/HumanEval scores are shown in a
separate, clearly labeled validation panel. Human interpretation of image
and coding agent tasks remains out of scope of this **text-only GGUF** until
verified supporting components and reproducing harnesses exist.

The *relevant* source-only scientific regression was independently rerun:
**482 PASS, 1 SKIP** across 18 targeted REAP files; no native model rerun,
generated Python execution or unsafe bench provider call. The sole warning
was pre-existing pytest unknown `asyncio_mode` under disabled plugin auto-load.

## Frozen Gen-0 selector and precise remaining execution

Original windows Gate0/1/2 **UNKNOWN**; random128 independent full control
**RUNNING/UNKNOWN**, original same-first-200 dense-vs-REAP Q4 required
**≥1.3×** matched native speed **UNKNOWN**, original REAP160 quality/speed
fallback **UNKNOWN**, genuine BF16/CUDA logits and cross-platform parity
**UNKNOWN**. Historical REAP128-v1 Gate1 **FAIL remains FAIL**. Optional
Wald-vs-paired-nonzero-margin McNemar method remains prospectively
**UNRESOLVED**, never chosen after viewing actual outcomes.

For screenshot HLE, provision a source-permitted official dataset with
explicit account consent, preserve a private pinned revision, officially
specified text-only filtering / answer judging / decoding, then admit a
dedicated CUDA machine and complete an immutable same-session cohort with
independent audit. For coding/agent tasks, add an independently verified
no-egress rootless microVM/container with independent descendant teardown;
only then enable model-generated programs. For BF16/dense/REAP160, obtain
a dedicated sufficiently provisioned CUDA host and original pinned models,
source and server geometry before benchmarking. Do **not** splice partial
remote serving sessions or reuse original Q8/Q4 results as foreign metrics.

**Frozen deterministic selector unchanged:**

```json
{"status":"BLOCKED","chosen":null,"release_approved":false}
```

No deployment, Gen-1 training, PR #282 merge, model release, raw prompts,
private outputs or credentials published.
