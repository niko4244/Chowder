# Gen-2 preregistration — amendment 13

**Subject:** the evaluation *execution* configuration (`evaluation_execution.batch_size`)
**Date:** 2026-09-18
**Does this change a gate, a threshold, a benchmark set or the measurement protocol?** No.
`protection.decoding` is untouched: `{temperature: 0.0, do_sample: false, max_new_tokens: 512}`.

## Why this amendment exists

The Gen-2 declaration was not runnable in bounded time. Both evaluation arms and
the candidate arm decode **16 items × up to 512 tokens** per declared benchmark.
With the dense ~9B base offloaded (the only placement that fits this 16 GB card —
see `evaluators/placement.py`), every decode step re-streams the whole model over
PCIe, so a step costs ~2.6 s almost regardless of batch size.

Measured on the frozen Gen-0 base, 2026-09-18, over the frozen 16-item Math500
slice:

| configuration | per decode step | 16 rows × 512 tokens | outcome |
|---|---|---|---|
| one row per `generate` call | 2.60 s | ~5.9 h per suite, ~11.8 h for the arm pair | `measure-ancestor` **timed out at 7200 s** having written nothing (three attempts: two OOM/crash before the offload-buffer fix, one timeout) |
| sixteen rows per `generate` call | 2.77 s | ~24 min per suite, ~48 min for the arm pair | completes |

The re-stream is per *step*, not per row, so batching amortises it: 16 rows for
the price of one row's transfer. This is the difference between an arm that
cannot be measured at all and one that finishes inside the frozen
`timeout_seconds`.

## What is declared

`docs/gen2/gen2_campaign.json` now declares, alongside the protocol:

```json
"evaluation_execution": { "batch_size": 16 }
```

`CampaignManifest.evaluation_execution` is the **single owner** of this value.
The candidate evaluator (`campaign_runner.build_evaluator` →
`SubprocessEvaluationFn(batch_size=…)`) and both arm measurements
(`campaign_prepare.measure_arm`) read that one declared field, so the candidate
and the arms cannot be measured at different throughputs by accident. An
`evaluation_execution` block carrying a key nothing reads is refused at load
time (`EvaluationExecution.from_mapping`), and every arm row and candidate row
records `decoding.batch_size` in its own evidence, so a reader can tell how a
measurement was taken from the artifact alone.

The frozen judge is unaffected: `certification.protocol_problems` enforces the
keys the *declared protocol* names and ignores extra keys, and T20 compares
`protection.protocol.to_dict()` — which is unchanged — against its frozen
constant. No judge constant was edited.

## What was measured, and why this is declared rather than assumed neutral

Batching is **not** a no-op for generated tokens. On the frozen Gen-0 base,
identical prompts, greedy decoding, `max_new_tokens=4`, 16 items:

| comparison | rows with identical prediction, score, `generated_tokens`, `eos_terminated` |
|---|---|
| batch 1 vs batch 1 (control, independent runs) | **16 / 16** |
| batch 1 vs batch 16 | **14 / 16** |

Single-row decoding is reproducible run-to-run; batched decoding diverges from
it on 2 of 16 rows (this is bfloat16 reduction-order dependence on batch shape,
not run noise — the control rules noise out). Over 512 tokens a divergence at
step 4 cascades.

The consequence is a **comparability rule**, stated here because it cannot be
mechanised inside the frozen judge:

> Every arm this campaign compares — trusted ancestor, parent, candidate — must
> be measured at the same declared `batch_size`. A batch-16 arm may never be
> compared against a batch-1 arm. Gen-2 measures all three at batch 16.

The implementation enforces this by construction (one declared value, three
consumers) and records it in every row. Residual, explicitly not mechanised:
the frozen judge does not itself refuse a cross-throughput comparison, because
doing so would require editing a frozen judge to require evidence keys that
legacy rows do not carry. Recorded as an unresolved item rather than papered
over.

## Zero incremental campaign cost

The arms are referenced by the campaign at zero incremental cost (declared
`MEASURED_PARENT`, referenced by `baseline_eval_report_path` /
`parent_eval_report_path`), unchanged by this amendment. Batching cuts the wall
time the *candidate* evaluation occupies, which is the number that competes
against `wall_gpu_hours_ceiling_campaign`.

## What this amendment is not

* It is not a threshold change, and it does not make Gen-2 pass anything.
* It does not touch `n_samples`, `seed`, `shuffle`, `max_new_tokens`,
  `prompt_policy` or the instrument.
* It does not relabel any historical evidence: the Gen-1 arm measurements that
  exist remain what they were.
