# Gen-2 preregistration amendment 11 — preparing the declared inputs, and measuring the trusted-ancestor arm

**Date:** 2026-09-18
**Status:** pre-compute infrastructure amendment. **No Gen-2 candidate training has
been run.** No threshold, benchmark set, protocol, budget, stopping rule,
contamination requirement, provenance rule, trusted-ancestor rule or verdict
semantic is altered by this amendment.

## A. Why this amendment exists

Amendments 5–10 closed the *certification* side of Gen-2: the candidate arm is a
run output, the judge cannot be satisfied by a file sitting in the run root, and
the trusted ancestor is a declared, digest-bound input. What remained was the
**readiness** side. The declaration named seven inputs that no production code
produced, so `chowder growth campaign readiness docs/gen2/gen2_campaign.json`
refused with `READINESS_DECLARED_INPUT` and both `plan` and `run` refused before
compute. Hand-authoring those documents for every generation is the same class of
defect as a copied benchmark row: an artifact nobody can reproduce from evidence.

## B. `chowder growth campaign prepare`

One production command emits the declaration's inputs from evidence the
repository already holds. It is not a docs script and it invents nothing:

| Input | Source |
|---|---|
| `hardware_budget_path` | a **real device probe** (`campaign_prepare.probe_hardware`): CUDA identity and VRAM from torch, step timings measured by a bounded synthetic forward+backward at each declared sequence length. Explicitly labelled a *device* probe, not a measurement of the campaign's model; a real model-step measurement stays a separate pre-compute job. |
| `evaluation_material_path` | the **pinned local dataset caches**, offline (`HuggingFaceH4/MATH-500` test, `juletxara/mgsm` `en`/`test`), first `n_samples` items in dataset order. The generation-diagnostics instrument's 16 prompts are in-repo, owned in production by `chowder.growth.generation_diagnostics.INSTRUMENT_PROMPTS` (a test pins it byte-identical to the frozen judge's copy so the two cannot drift). |
| `parent_eval_report_path` | the **parent generation's own durable run root** (`--parent-evidence`). |
| `parent_profile_path` | built from the parent arm's own rows. |
| `project_template_path`, `training_material_path`, `data_registry_path` | derived from the manifest and the **production planner's own curriculum item ids**. |
| `contamination_manifest_path` | the real firewall, checked against the prepared protected slices. |

### Provenance rules this command obeys

* The **arm** is strict: a row is `MEASURED_PARENT` only when the parent's durable
  evidence holds a measurement of **that exact benchmark id**. A measurement made
  under another instrument version is *not* a measurement of this one, and a
  slice the parent carried from an earlier generation is not a measurement at
  all. Everything else is an honest `UNMEASURED` row naming why.
* The **profile** is a capability summary, so it keeps whatever the parent
  durably measured, under the instrument it measured it with. Arm and profile are
  deliberately different questions.
* A parent with no durable measurement of anything cannot produce a curriculum,
  and preparation refuses rather than planning from a fabricated zero.

For `docs/gen2/gen2_campaign.json` against the Gen-1 run root this yields:
`generation-diagnostics@gen2-response-surface-v1` **UNMEASURED** (Gen-1 measured
`generation-diagnostics@gen1-eval-protocol-v1`), and `math500@2024-04` /
`mgsm@2022-11` **UNMEASURED** (both carried from Gen-0, never measured on Gen-1).
That is the honest Gen-1 parent arm, and it is why the parent side of promotion
cannot currently demonstrate target improvement — see §E.

### What `prepare` also fixes

The recipe ids depend on the measured hardware and the parent profile, so they
were not knowable when the declaration was written. `--write-declaration` fills
`recipes` with the ids production actually proposes, so a run's recipe set is the
set the planner proposed rather than a guess.

## C. `chowder growth campaign measure-ancestor`

Amendment 2 requires a fresh 16-item measurement of the untouched dense Gen-0
base at the declared `baseline_eval_report_path`. This is now a production
command (`chowder growth campaign measure-ancestor`), not a bespoke script:

* it drives the **same production worker** the candidate evaluator uses
  (`chowder.evaluators.transformers_text_worker`), with **no adapter loaded**:
  the dense base is what is measured;
* rows carry `measurement_origin=MEASURED_PARENT`, the declared trusted-ancestor
  generation label, `n_samples=16`, `sample_indices=0..15`, the declared seed,
  decoding and prompt policy, and a digest-bound raw artifact each;
* the report names the `base_model_digest` it measured;
* it is a real prior evaluation job referenced at **zero incremental campaign
  cost** — not charged to the recipe envelope and not usable to claim budget
  compliance.

### One narrow evaluator change this required

`TransformersTextEvalSpec` required a non-empty `adapter_dir`, which made the
worker's own `if spec.adapter_dir is None: <load the base>` branch unreachable.
The spec now accepts `adapter_dir=None` as a **declared base-only measurement**;
an empty string stays refused. This completes existing intent rather than
loosening a gate: the worker already had the branch, and nothing else in the
candidate path passes `None`.

## D. What did NOT change

Thresholds (`duplication ≤ 0.125`, `echo ≤ 0.062`, format 8/8, correctness
≥ 15/16, EOS ≥ 0.900, cap < 0.100, loops ≤ 0, trigram ≥ 0.900, unclosed think
≤ 0.250, slice regression tolerance `0.0625`); the benchmark sets; the mini-slice
protocol (16 items, seed 1234, greedy, 512 tokens, chat template); the budgets;
the stopping rules; the verdict classes; the candidate-measured provenance rule;
the trusted-ancestor rule; the contamination requirement. Amendment 1's
device-ceiling policy is unchanged (`device_time_measured: false`).

## E. Known pre-compute blockers this amendment surfaces (not fixes)

1. **The parent target arm is unmeasured under the Gen-2 instrument.** Gen-1 was
   measured under `generation-diagnostics@gen1-eval-protocol-v1`; Gen-2's target
   is `generation-diagnostics@gen2-response-surface-v1`. Until the parent is
   measured under the Gen-2 instrument (or the campaign declares the parent's own
   instrument as its target), the promotion rule's target comparison has a
   missing parent arm and can only reach `INCONCLUSIVE`. This is a genuine design
   question for the campaign, and it is *reported* here rather than papered over
   by relabelling a Gen-1 number as a Gen-2 one.
2. **The Gen-0 arm must exist before readiness is green.** `measure-ancestor`
   writes it; readiness then reports `READINESS_ANCESTOR_ARM` only if it is still
   absent.

`readiness` after `prepare` on the committed declaration no longer reports
`READINESS_DECLARED_INPUT`; its remaining codes were `READINESS_ANCESTOR_ARM`
(now addressable by §C) and `READINESS_RECIPE_SET` (addressable by the
`--write-declaration` recipe fill in §B).
