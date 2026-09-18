# Generation-0 Frontier Gap Report

Generated 2026-09-18T03:05:17.992190+00:00 from the frozen
eval report `C:\Users\nikma\Chowder-Protected\runs\2026-09-16-gen0-eval-freeze\freeze\eval-report.json` and the immutable
snapshot `gen0-frontier` (0 reference scores).
Context rows are drawn from every frozen snapshot: gen0-frontier-context-2026-09-17

Mechanical output of `gap_rows` (gate-eligible rows) followed by a
cited reference-context table (`context_rows`). No reference at a level
renders as `unavailable`; nothing is normalized into fake parity, and
context rows never enter a gap, a parity ratio, or a promotion input.

## math500@2024-04 — Chowder 0.0000

| Level | Reference | Gap | Comparability |
| --- | --- | --- | --- |
| LEVEL_0_FLOOR | unavailable | -- | unavailable |
| LEVEL_1_COMPARABLE_PEER | unavailable | -- | unavailable |
| LEVEL_2_OPEN_WEIGHT_FRONTIER | unavailable | -- | unavailable |
| LEVEL_3_STRETCH | unavailable | -- | unavailable |
| LEVEL_4_ABSOLUTE_FRONTIER | unavailable | -- | unavailable |

Reference context — cited published numbers that are **not** gate-eligible, listed for orientation only:

| Level | Model | Reference | Confidence | Harness | Blocked by |
| --- | --- | --- | --- | --- | --- |
| LEVEL_1_COMPARABLE_PEER | Meta-Llama-3.1-8B-Instruct | 0.346 | MEDIUM | lm-eval-harness hf backend, `minerva_math`, --num_fewshot 4, batch auto (v1 task) | comparability_confidence=MEDIUM, reasoning_setting='direct' |
| LEVEL_2_OPEN_WEIGHT_FRONTIER | DeepSeek-R1-Distill-Qwen-32B | 0.943 | LOW | DeepSeek official eval (MATH-500 pass@1) | comparability_confidence=LOW, reasoning_setting='extended-thinking' |
| LEVEL_3_STRETCH | DeepSeek-R1-Distill-Llama-70B | 0.945 | LOW | DeepSeek official eval (MATH-500 pass@1) | comparability_confidence=LOW, reasoning_setting='extended-thinking' |
| LEVEL_4_ABSOLUTE_FRONTIER | DeepSeek-R1 | 0.973 | LOW | DeepSeek official eval (MATH-500 pass@1) | comparability_confidence=LOW, reasoning_setting='extended-thinking' |

These rows do not contribute a gap, a parity ratio, or a promotion input; protocol divergence is named in the last column.

## mgsm@2022-11 — Chowder 0.0000

| Level | Reference | Gap | Comparability |
| --- | --- | --- | --- |
| LEVEL_0_FLOOR | unavailable | -- | unavailable |
| LEVEL_1_COMPARABLE_PEER | unavailable | -- | unavailable |
| LEVEL_2_OPEN_WEIGHT_FRONTIER | unavailable | -- | unavailable |
| LEVEL_3_STRETCH | unavailable | -- | unavailable |
| LEVEL_4_ABSOLUTE_FRONTIER | unavailable | -- | unavailable |

Reference context — cited published numbers that are **not** gate-eligible, listed for orientation only:

| Level | Model | Reference | Confidence | Harness | Blocked by |
| --- | --- | --- | --- | --- | --- |
| LEVEL_1_COMPARABLE_PEER | Qwen2.5-7B | 0.578 | LOW | Qwen official eval (MGSM 8-shot CoT, multilingual mean) | comparability_confidence=LOW, reasoning_setting='chain-of-thought' |
| LEVEL_2_OPEN_WEIGHT_FRONTIER | Qwen2.5-72B | 0.767 | LOW | Qwen official eval (MGSM 8-shot CoT, multilingual mean) | comparability_confidence=LOW, reasoning_setting='chain-of-thought' |
| LEVEL_3_STRETCH | Llama-3-70B | 0.671 | LOW | Qwen official eval (MGSM 8-shot CoT, multilingual mean) | comparability_confidence=LOW, reasoning_setting='chain-of-thought' |

These rows do not contribute a gap, a parity ratio, or a promotion input; protocol divergence is named in the last column.

**No protocol-comparable reference exists for any measured benchmark.** Every published number found for these benchmarks uses a different harness/shot/extraction protocol (see docs/FRONTIER_REFERENCE_SEED_2026-09-17.md); Chowder records that honestly instead of rendering fake gaps. The reference-context tables above name what exists and why it cannot be compared.
