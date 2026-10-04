# GEN2 prereg amendment 14 — 2026-10-04: the declared inputs exist, and the protection is declared

Adds to the 2026-09-18 series (amendments 1–13); supersedes none of it. Two
things are declared here, both **before any Gen-2 candidate evaluation
exists**, which is the freeze's own admissibility test: no candidate result is
visible (`runs/2026-09-17-gen2-response-surface/attempts/*` hold training
evidence only, and the Gen-2 state root has no `candidate_evaluation.json` or
`chosen_candidate.json`; the run refused at the `candidate_evaluation` phase).

## A. The preregistered protection, declared mechanically

`retention_profile` (`gen2-protection`: max-regression 0.0625 — one 16th of a
16-item mini-slice — on `math500@2024-04` and `mgsm@2022-11`) and
`eval_tier_policy` (both protected benchmarks are `promotion-evidence`).

**This adds no threshold.** Both state in machine-readable form what
`protection.slice_regression_max = 0.0625` already stated in prose, on the same
two benchmarks. The loader parses both fail-closed (unknown fields, empty
profiles, malformed constraints, unmeasurable benchmarks, unknown tiers,
reserved-name and search-view demotions all refuse at load as
`CampaignManifestError`), and both promotion paths
(`GrowthCycle._apply_promotion_gates`, reached from `decide_promotion` and
`decide_promotion_from_runs`) enforce them per cycle, with both sides of a
constraint demanding earned provenance.

This supersedes the tag `GEN2_PREREG_AMENDMENT7_2026-10-04` written into
`docs/gen2/gen2_campaign.json`'s notes by commit `9080a33`, which collided with
the real `GEN2_PREREG_AMENDMENT7_2026-09-18.md`. The field now reads
`GEN2_PREREG_AMENDMENT14_2026-10-04`.

## B. The seven run-phase inputs, produced by production code

```
PYTHONPATH=src python -m chowder.cli growth campaign prepare \
  docs/gen2/gen2_campaign.json \
  --out-dir "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen2-response-surface/prepared-v10" \
  --parent-evidence "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen1-protocol-compliance" \
  --parent-measurement "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen2-response-surface/prepared-v9/parent-eval-report.json" \
  --write-declaration "…/prepared-v10/gen2_campaign.prepared.json"
```

A **new** directory (`prepared-v10`), never `prepared-v2`: preparation writes
its own `parent-eval-report.json`, and `prepared-v2` holds the preserved
~62-minute Gen-1 parent measurement that must not be overwritten.

What each declared input is, and where its evidence came from:

| declared input | produced from | evidence |
| --- | --- | --- |
| `hardware_budget_path` | `probe_hardware()` | real CUDA device probe on this machine (NVIDIA GeForce RTX 5060 Ti, 15.93 GB); bounded synthetic forward+backward step timings at 512/1024/2048; its own `measurement_method` says it is **not** a measurement of the campaign's model |
| `evaluation_material_path` | `_write_evaluation_material` + `load_pinned_slices` | the pinned offline caches `HuggingFaceH4/MATH-500` (test) and `juletxara/mgsm` (en/test), plus the in-repo frozen `generation_diagnostics.INSTRUMENT_PROMPTS` for the target |
| `parent_eval_report_path` | `_write_parent_evidence` reading the declared parent measurement | the **preserved** Gen-1 measurement of 2026-09-19, carried forward unchanged: three `MEASURED_PARENT` gen1 rows — `generation-diagnostics@gen2-response-surface-v1` 0.5625, `math500@2024-04` 0.0, `mgsm@2022-11` 0.0, 16 samples each, with the same `artifact_sha256` and `slice_sha256` as the preserved report. Not re-measured, and not a carried quote |
| `parent_profile_path` | `build_skill_profile` over exactly those rows | the attributed `SkillProfile` shape (`estimates`, one row per skill, attributed only to benchmarks that measure it); the stale legacy shape that blocked `prepared-v2` is not reproduced |
| `training_material_path` | `_write_corpus` after production planning | 5 curriculum items, 79,904 tokens, 10,340 examples over two admitted providers; `verifier_pass_rate` 1.0, `duplicate_rate` 0.0, contamination `CLEAN` |
| `data_registry_path` | `_write_corpus` | the admitted sources the corpus dispatches to |
| `project_template_path` | `_write_project_template` | the executor project, composed against the prepared parent arm's baseline metrics |
| `contamination_manifest_path` | `_write_contamination_manifest` | replaces a path that named a file which does not exist |

The declaration also now names the **planner's** recipe ids
(`recipe-00-lr5e-05`, `recipe-01-lr0.0001`) instead of `gen2-recipe-a`/`-b`,
which the planner never proposed and which `recipe_set` refuses.

## C. Readiness, measured

```
PYTHONPATH=src python -m chowder.cli growth campaign readiness \
  "…/prepared-v10/gen2_campaign.prepared.json"
```

`status: READY`, `reason_codes: []`, **all seventeen checks `ok`** — `schema`,
`declared_inputs`, `base_identity` (`59e767aab1da` over 10 payload files),
`parent_adapter_identity` (`ca8769c5e7e0`), `contamination`, `project_template`,
`training_material`, `data_registry`, `hardware_budget`, `parent_profile`,
`parent_arm` (3 rows, gen1), `ancestor_arm` (2 rows, gen0),
`protection_policy`, `plan` (5 items → 2 recipes), `recipe_set`,
`candidate_search` (no search declared: the recipes run once each — correct for
this first run), `campaign_projection` (0.000312 device / 0.001091 wall GPU-h
within the declared ceilings), `evaluator` (`SubprocessEvaluationFn` available),
`evaluator_coverage`. Elapsed 321 s, most of it the base model-content digest.

Nothing is skipped: a skipped check is not a passed check.

## D. What this amendment does not do

- It does not start the run. `chowder growth campaign run` remains a separate
  decision, and the campaign's own stopping rules are unchanged.
- It does not re-measure the Gen-1 parent arm (≈62 GPU-minutes) or the Gen-0
  baseline arm; both are declared measurements already on disk, with digests.
- It does not fix the judge's blindness to the run's decision
  (`docs/gen2/judge_gen2.py` remains frozen and unmodified; the minimal
  amendment is proposed for review in
  `docs/gen2/JUDGE_AMENDMENT_PROPOSAL_T21.md`, to be numbered amendment 15 if
  accepted).
- It does not add the target instrument's diagnostic metadata to a run output,
  which the judge's T1–T10 still lack.