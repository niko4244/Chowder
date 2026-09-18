# Gen-2 Preregistration — Amendment 8 (2026-09-18)

Written before any gen2 compute. It changes no threshold, no benchmark set, no
budget, no stopping rule and no verdict class. It moves one instrument into
`src/` so the frozen judge's instrument gates can be decided from evidence a
production run produced.

## A. What was missing

The frozen judge's T1–T10 read exactly one row: the campaign's declared target,
`generation-diagnostics@gen2-response-surface-v1`, carrying the completions of
the 16 frozen diagnostic prompts in `metadata.per_prompt` and five aggregates —
`eos_termination_rate`, `max_token_cap_rate`, `unclosed_think_rate`,
`obvious_loop_count`, `distinct_trigram_ratio_mean`.

Those definitions existed only in the historical Gen-1 driver
(`docs/gen1/run_gen1_cycle.py`). The production candidate evaluator measured the
right prompts through the right worker and wrote a real `predictions-<suite>.jsonl`
for them, but it recorded only the decoded text and a score — so the five
aggregates were absent, and T1–T10 came back `UNKNOWN` for every run a production
path could make. A verdict could only be reached by preparing that row outside the
runner, which is the hand-assembly the certification boundary exists to prevent.

## B. The port

| where | what now lives there |
| --- | --- |
| `chowder.growth.generation_diagnostics` | the frozen rules, computed from the worker's own per-item rows: EOS termination, cap-hit, unclosed `<think>`, three-identical-line loops, distinct word-trigram ratio |
| `chowder.evaluators.generation.observed_generation` | the two facts only the generating worker can observe — tokens produced and whether generation stopped on EOS — recorded beside every prediction by **both** text workers |
| `chowder.evaluators.scoring.observed_score` | the observation-defined scoring `eos_termination`: 1.0 for a generation that stopped on EOS, 0.0 for one that ran into the cap |
| `chowder.growth.evaluation_binding` | merges the diagnostics into the row's `metadata`, flat, where the judge reads them |

The scoring rule is deliberately the **same** rule the Gen-1 instrument used, and
it makes the row's own score its `eos_termination_rate` — the metric the catalog
declares for this benchmark. A row labeled `eos_termination_rate` whose score was
an exact-match mean would be the same class of defect as a copied parent score
wearing the candidate's generation label.

The rules are ports, not redesigns: same triggers, same denominators, same
denominator for the trigram mean (`1.0` for fewer than three words, exactly as the
Gen-1 driver returned). Changing one here would silently change the meaning of
every threshold the frozen judge compares against.

## C. It fails closed on the facts it does not have

The aggregates are defined over observations, so an unevaluated fact must not
become a zero:

* an item with no `generated_tokens` or no `eos_terminated` refuses with
  `GENERATION_DIAGNOSTICS_UNMEASURED`, naming the file and the item;
* a non-bool or negative observation refuses (a `False` read of an absent
  `eos_terminated` would count an unmeasured generation as a termination
  failure);
* `eos_terminated` together with a generation that reached the cap refuses: one
  of the two facts is wrong, and the instrument cannot know which.

The candidate-evaluation binding turns that refusal into a run refusal, so a
worker whose rows cannot be diagnosed produces no arm rather than a diagnosed one.

## D. The declared target is now a registered benchmark

`docs/gen2/gen2_campaign.json` names
`generation-diagnostics@gen2-response-surface-v1` as its target, and the frozen
judge pins the same id — but no row for that version existed in the benchmark
catalog, so a campaign could not plan against the target it declared (the metric
binder refuses a promotion set naming an unregistered benchmark, by design).

This amendment registers it: the same 16 prompts as the Gen-1 instrument, seed
1234, chat-template prompts, 512 max new tokens, primary metric
`eos_termination_rate`, `scorer=protocol_diagnostics`, tier 2,
`INTERNAL_REFERENCE_ONLY` / `ACTIVE_DIAGNOSTIC` — a behavioral instrument whose
score never enters capability-skill aggregation. Its `dataset_source` names the
frozen prompt list the campaign's `evaluation_material_path` must provide.

## E. The proof

`tests/test_growth_certification_coupling.py::test_the_runs_own_instrument_row_is_what_the_frozen_judge_scores`
drives a manifest through the real CLI with the **production** candidate evaluator
(only the child process is a recording worker), over the frozen diagnostic prompts
declared as the target, and then hands that same run root to the frozen judge. It
asserts `VERDICT: PROMOTED` — the judge's strongest output, returned only when no
threshold is `UNKNOWN` or `FAIL` — and names T1 through T10 individually, so a
regression says which gate stopped being decidable. The candidate arm's
`per_prompt` prompts, `eos_termination_rate`, `max_token_cap_rate` and
`obvious_loop_count` are asserted against the measured row before the judge reads
it.

`tests/test_growth_generation_diagnostics.py` pins the rules against
hand-computed values and every refusal above.

## F. What this amendment does not claim

Nothing here is a measured gen2 result and no gen2 compute was run to write it.

* The declaration still provides none of the inputs a run reads from disk; both
  entry points still refuse before compute, naming every missing input at once.
* T16 stays `UNKNOWN` until the declared Gen-0 trusted-ancestor arm is actually
  measured; the device-time instrumentation and the run root's manifest binding
  are untouched.
* The instrument is *ported*, not re-benchmarked: no gen0 or gen1 diagnostic
  number changes, and no historical artifact is rewritten.
