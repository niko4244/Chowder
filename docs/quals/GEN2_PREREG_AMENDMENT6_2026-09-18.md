# Gen-2 Preregistration — Amendment 6 (2026-09-18)

Written before any gen2 compute. It changes no threshold, no benchmark set, no
budget, no stopping rule and no verdict class. It supplies the instrument
Amendment 5 said was missing, and it states exactly how far that instrument goes.

## A. The candidate evaluator is now real

Amendment 5 established that the candidate arm is a run *output* and that the
runner must ask an evaluation seam to measure the artifact it selected. It also
recorded that no such seam was wired, so a gen2 run could not reach a verdict.

`chowder.growth.evaluation_binding.SubprocessEvaluationFn` is that seam, and
`chowder.growth.campaign_runner.build_evaluator` now builds it instead of
returning `None`. It reuses the production pieces rather than adding a second
evaluation framework:

* `chowder.evaluators.transformers_text_worker` through its canonical command
  line (`--spec/--result/--chowder-identity`), started by the same injectable
  process runner the training binding uses, so the worker verifies the checkout
  it imported before it reads its spec;
* `TransformersTextEvalSpec` as the spec, with `adapter_dir` the artifact the run
  selected and `max_new_tokens`, `seed` and prompt policy taken from the
  campaign's declared `protection.protocol`.

The manifest gains one key, `evaluation_material_path`. It is the evaluation
half of the sentence "the report is an output, the data is an input": a JSON
document declaring, for each declared benchmark, the dataset to measure and how
to score it (`benchmark_qualified_id`, `dataset`, `scoring`, `prompt_field`,
`expected_field`, `metric`). It is **required by the `run` phase**, so a campaign
that cannot measure its candidate refuses before any compute rather than after
its training budget is spent.

Four properties are what make the produced arm evidence:

1. **The slice is the declared protocol.** For every declared benchmark the
   evaluator writes the first `protection.n_samples` items, in dataset order,
   into the run root as the file it measures. The row's `sample_indices` are
   `0..N-1` by construction, the exact bytes measured stay in the run root, and a
   dataset shorter than the declared protocol refuses
   (`CANDIDATE_EVALUATION_SLICE_TOO_SHORT`) instead of measuring a smaller slice
   under the protocol's name.
2. **Identity is recomputed, not asserted.** The adapter artifact's digest is
   computed from its real bytes before anything loads; if it differs from the
   digest the run selected, the evaluation refuses
   (`CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH`). The report's
   `model_identity` carries that recomputed digest and the declared base digest,
   and the worker's `model_provenance` (whether an adapter was actually loaded)
   is recorded beside it.
3. **The row is bound to bytes that exist.** Each row's `raw_artifact_ref` is the
   `predictions-<suite>.jsonl` the worker wrote, `metadata.artifact_sha256` is
   that file's digest, `n_samples` is its item count, `score` is the mean of its
   item scores, and the worker's holdout fingerprint evidence is required and
   re-hashed — so a row cannot name one item set while carrying another's scores.
   Certification recomputes all of it; these are the same checks, applied where
   the arm is made rather than only where it is judged.
4. **The evaluation's compute is charged.** The binding reports the wall
   GPU-hours the evaluation actually occupied (process time × the worker's own
   accelerator count) as a `ComputeCost` with `device_measured=False`, which is
   the honest reading for a worker that does not separate device time, and which
   therefore cannot settle a device ceiling. The campaign charges it to the cycle
   ledger before settlement.

## B. What this amendment does *not* claim

The instrument measures the artifact; it does not yet compute the *target
instrument's* diagnostics. Gen2's target set is
`generation-diagnostics@gen2-response-surface-v1`, and the frozen judge's T1–T10
read one instrument row's `per_prompt` completions and its `eos_termination_rate`,
`max_token_cap_rate`, `obvious_loop_count`, `distinct_trigram_ratio_mean` and
`unclosed_think_rate`. Those definitions exist only in the historical
`docs/gen1/run_gen1_cycle.py`; no module under `src/` computes them, so the
evaluator writes the instrument's scored items but not its diagnostic metadata,
and T1–T10 remain `UNKNOWN` — which is a refusal to certify, not a pass. Porting
that instrument into production is the next build.

Nothing in this amendment is a measured gen2 result, and no gen2 compute was run
to write it. The declared inputs `docs/gen2/gen2_campaign.json` still lacks
(Amendment 5 §B, updated: `evaluation_material_path` joins the list of inputs
that have no artifact yet) are unchanged in status: the entry points refuse
before compute and name every missing input at once.
