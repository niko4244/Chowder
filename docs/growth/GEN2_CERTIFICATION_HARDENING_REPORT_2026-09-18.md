# Gen-2 certification hardening — pass report (2026-09-18)

Merged as PR #179 → protected main `824057e`; post-merge protected CI run
`35336860704` completed **success**, 6/6 jobs. No Gen-2 training, evaluation or
campaign compute has run. Nothing historical was rewritten: the Gen-1 ledger
record, its result artifacts and the frozen Gen-2 preregistration text are
byte-unchanged, and every semantic change is a dated amendment.

The pass was a correctness pass on the *certification boundary* — the code that
decides whether a Gen-2 artifact may be promoted. It exists because an audit of
`main` found the frozen judge could certify evidence the production path would
refuse, and implemented weaker approximations of rules the frozen prereg text
already stated.

## Defects fixed

| # | Defect on `main` | Fix | Owner of the fix |
|---|---|---|---|
| 1 | `EvalReport.load()` rebuilt each `BenchmarkRun` without `measurement_origin`, so `MEASURED_THIS_GENERATION` → save → load became `UNMEASURED` (usable candidate evidence destroyed; legacy rows must still read as `UNMEASURED`) | the loader restores the stored origin and defaults a row that never declared one to `UNMEASURED` | `src/chowder/evals/result.py` (landed #178, pinned by the new round-trip tests in `tests/test_growth_measurement_provenance.py`) |
| 2 | The judge read the wall ceiling straight from `cycle_compute_accounting.json` and ignored `device_measured` / `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`, so the same artifact could pass the judge and fail `chowder growth campaign settle` | the judge rebuilds a `ComputeCost` from the artifact's own fields and calls production `settle_campaign()`; any non-compliance is a hard FAIL with the production reason strings in the detail | `docs/gen2/judge_gen2.py` `_settlement_gates` |
| 3 | Protected coverage was "whatever rows exist": a missing `mgsm@2022-11` did not fail | required set is taken from the frozen campaign/prereg, not the artifact: each required slice present exactly once, candidate-measured, protocol-exact (16 items, indices 0–15, seed 1234, no shuffle, greedy, `max_new_tokens=512`, chat-template prompt, named raw artifact); an undeclared protected row cannot substitute | `docs/gen2/judge_gen2.py` `_protected_gates`, `_protocol_problems` |
| 4 | The candidate artifact supplied `parent_score` too, and the judge trusted it | three independently provenance-bound arms: `candidate_evaluation.json` (`MEASURED_THIS_GENERATION`), `parent_evaluation.json` (gen1, `MEASURED_PARENT`), `baseline_evaluation.json` (gen0, `MEASURED_PARENT`). The judge compares verified arms; a parent score asserted inside the candidate file gates nothing | `docs/gen2/judge_gen2.py` `Arm`, `_ancestor_gates` |
| 5 | An empty contamination manifest passed, because only existing entries were checked | required identities are derived from the frozen evaluated set (instrument + both protected slices) and training sources must be non-empty; COMPLETE + CLEAN is the only pass, KNOWN/POSSIBLE fail, missing/malformed is UNKNOWN | `docs/gen2/judge_gen2.py` `_contamination_gate` |
| 6 | Artifact identity checked only that a 64-char string was present | the digest is recomputed through production `training_binding.directory_digest` (directories) / `sha256_file` (files) and compared; missing, malformed, empty or mutated artifacts fail | `docs/gen2/judge_gen2.py` `_identity_gate`, `_digest_of` |
| 7 | Target adjudication used final rates, not the frozen paired per-prompt rule | the frozen rule is implemented mechanically: paired per-prompt comparison at the declared minimum effect **or** the absolute threshold crossed with ≥12/16 strictly better prompts; a tie fails | `docs/gen2/judge_gen2.py` `_target_gate`, using production `statistics.compare` |
| 8 | A Gen-2 candidate measured only against Gen-1 could promote while inheriting an unresolved Gen-1 regression | branch protection is judged against the trusted ancestor gen0 as well as the immediate parent (gen1). A Gen-1 that regressed against gen0 is named in the verdict and Gen-2 is adjudicated against gen0 | `docs/gen2/judge_gen2.py` `_ancestor_gates` |
| — | `gen2_campaign.json` held a `parent_model_path` (the dense base) beside `parent_model_digest` (the gen1 *adapter* digest): one field named two different objects, and the runner verified the base tree against the adapter's digest | identity is two typed pairs — `base_model_path`/`base_model_digest` and `parent_adapter_path`/`parent_adapter_digest` (omitted when parent == base) — each verified against its own tree before compute; the overloaded field is removed, not aliased | `src/chowder/growth/campaign.py`, `campaign_runner.py` |

The judge keeps only what a frozen policy should own: the benchmark set, the
thresholds, the branch rules and the verdict composition. Every foundational
operation is delegated to production code that already decides it (digest,
settlement, provenance, contamination, statistics, `EvalReport` parsing), so
`what Chowder runs` and `what the frozen judge thinks Chowder runs` cannot drift
apart silently.

## The one policy decision

The frozen manifest declared a **hard device settlement ceiling** while the
trainer reports only wall time, i.e. the executable path enforced the device
ceiling at admission and could never settle it. That contradiction was resolved
*in text, before compute*, in `docs/quals/GEN2_PREREG_AMENDMENT1_2026-09-18.md`
(design A):

- `budget.device_time_measured = false` → device ceilings are **admission**
  constraints on the projected plan; **wall** remains the post-run settlement
  gate; the judge records which ceiling is which rather than implying settlement.
- Setting it to `true` makes the device ceilings settle against measured device
  time, and an attempt reporting none refuses with
  `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` — unmeasured is never compliance.

The manifest, the preregistration, `settle_campaign`, the training binding and
the judge now say the same thing.

## Verification on the merged tree

Run on `origin/main` = `824057e` (detached checkout; nothing modified):

```
pytest -q tests/test_growth_measurement_provenance.py     18 passed
pytest -q tests/test_growth_gen2_judge.py                 49 passed
pytest -q tests/test_growth_campaign.py                   17 passed
pytest -q tests/test_growth_budget_settlement.py          16 passed
pytest -q tests/test_growth_promotion_settlement.py        4 passed
pytest -q tests/test_growth_metric_binding.py             50 passed
pytest -q tests/test_growth_ledger_corrections.py          6 passed
   focused total                                         160 passed
pytest -q                                    2304 passed, 77 skipped in 414.85s
ruff check src tests                                 All checks passed!
python -m build                          chowder_ai-0.3.0 sdist + wheel built
```

The 77 skips are the CPU-only matrix's GPU-gated tests. They are **skipped, not
passed**, and they are not evidence about anything this pass changed. The
figure of 2229 passed recorded in `INTEGRITY_PASS_REPORT_2026-09-17.md` was
measured before this pass existed; the delta is the tests added here.

## Adversarial review of the hardened boundary

Each question was re-attacked against the merged judge by building a clean run
root and mutating one thing, rather than by re-reading the code.

| Attack | Result | Evidence |
|---|---|---|
| Delete `mgsm@2022-11` from the candidate artifact | **refused** | probe: `T11 candidate mgsm@2022-11 measured + protocol-exact UNKNOWN`; `test_both_required_slices_present_and_protocol_exact` |
| Empty contamination manifest | **refused** | probe: all three required ids `UNKNOWN` plus `T12 training-source contamination CLEAN UNKNOWN`; `test_contamination_coverage_must_be_exact`, `test_an_empty_training_source_section_is_not_clean` |
| Insert `"a"*64` as the artifact digest | **refused** | probe: `T15 FAIL recorded aaaaaaaaaaaa vs recomputed 2e0380e59b74`; `test_a_fabricated_digest_refuses` |
| Put the parent score inside the candidate JSON and measure no parent | **refused** | probe: `T2`/`T3` UNKNOWN (parent per-prompt evidence missing) and `T16` UNKNOWN (ancestor arm missing); `test_the_parent_arm_must_be_independently_measured` |
| Change the seed or the sample indices | **refused** | probe: `T11 FAIL candidate protocol mismatch: sample_indices=[1..16]; seed=4321`; the parametrized protocol cases |
| A wall-only accounting artifact against a declared hard device settlement ceiling | **refused** | probe: `compliant=False reason=ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`; `test_a_device_settlement_ceiling_cannot_be_satisfied_by_an_unmeasured_device` |
| Save and reload candidate evidence, losing provenance | **refused / preserved** | probe: round-trip preserved all three origins; a stripped legacy file loaded as `UNMEASURED`×3; `test_eval_report_round_trips_every_origin`, `test_legacy_report_without_origin_field_loads_as_unmeasured` |
| Gen-2 matches a regressed Gen-1 and promotes anyway | **refused** | probe: `T16 FAIL math500 0.0000 vs gen0 0.5000` plus `INFO gen1 itself regressed … adjudicated against gen0`; `test_gen2_cannot_promote_by_inheriting_a_gen1_regression`, `test_gen2_promotes_when_the_whole_branch_holds` |
| The judge says PASS while `chowder growth campaign settle` says FAIL | **impossible** | probe: wall 0.2/1.1/1.5 → judge exit 0 and production compliant; wall 2.5 → judge exit 1 and production non-compliant, identical reasons; `test_the_judge_and_production_settlement_agree` |
| Aggregate target rates pass while the frozen paired rule would fail | **refused** | probe: rate at threshold with no paired signal → `FAIL paired=flat … strictly better on 2/16 (need 12)`; `test_aggregate_threshold_alone_does_not_pass_the_frozen_rule`, `test_a_claimed_aggregate_cannot_override_the_per_prompt_evidence` |

Two probes initially reported "bypassed" and both were probe bugs, recorded here
rather than quietly dropped: the settlement-agreement probe used a chained
comparison (`code == 0 == prod.compliant`) that compares `0` to a bool, and the
statistics probe accidentally constructed a genuine large paired improvement
(7/8 prompts flipped, delta −0.75 against a 0.25 minimum effect), which the
frozen rule is supposed to accept. Corrected, both hold.

## Is Gen-2 safe to execute?

**Yes, the certification path is hardened and the frozen policy text is
executable as written.** No Gen-2 compute has run, and the amendment that
resolved the device-ceiling contradiction was written before any result existed.

If a Gen-2 candidate's measured results pass every frozen gate — target rule,
format constraints, both candidate-measured protected slices, no regression
against gen1 *and* against gen0, exact CLEAN contamination coverage, production
campaign settlement within the wall ceiling, recomputed artifact identity — then
it is **eligible for a full `PROMOTED`**, because the branch-protection gate is
adjudicated against the trusted ancestor rather than against the unresolved
Gen-1 record. A Gen-2 that merely matches Gen-1 still cannot promote if Gen-1
regressed against Gen-0.

## Known limitations (unresolved, and deliberately not hidden)

- **Device time is not separated by any trainer, registry or evaluator path.**
  The Gen-2 device ceilings are therefore admission-only, recorded as such in
  the manifest (`device_time_measured: false`), the amendment and the judge's
  detail line. Post-run device settlement needs a device-reporting trainer;
  settling an unmeasured ceiling is refused rather than faked.
- **The Gen-2 protected and "broad" sets are the same two 16-item slices.**
  That is a regression screen, not a capability claim; breadth is not measured
  for Gen-2, and calibration/reliability sets are empty. A Gen-2 promotion is
  evidence about protocol compliance and about *not* regressing on those two
  slices.
- **`PromotionInput.actual_device_gpu_hours` is still a plain float** with an
  inline re-check rather than the validated `ComputeCost` object, because the
  recorded Gen-1 re-adjudication consumes that shape. Consolidating the
  promotion boundary onto `ComputeCost` is a later pass.
- The frozen instrument prompt list lives in the judge and is cross-checked
  against the gen1 driver's source (`test_the_frozen_instrument_matches_the_gen1_driver_source`);
  if a future driver changes the instrument, that check is what forces a new
  frozen instrument id rather than a silent re-definition.
