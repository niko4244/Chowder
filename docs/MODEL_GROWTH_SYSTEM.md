# Model Growth System

The Autonomous Model Growth System sits above Chowder's training/evaluation
infrastructure. It turns the platform from *something that can safely run
training experiments* into *something that can systematically measure a model,
decide what it should learn next, source verified material for it, train
competing candidates, verify genuine improvement, preserve prior capability,
and iterate toward stronger generations*.

Everything here is evidence-driven: no decision is "autonomous" because an
LLM was asked what to do next. Every phase answers **"why did Chowder do
this?"** from measurements, explicit policies, uncertainty, budgets, and
provenance.

## The Model N → Model N+1 lifecycle

```
MODEL N
  → comprehensive evaluation        (evals/ adapters, tiers)
  → capability profile              (capability.py: skills + raw scores)
  → failure analysis                (failure_bank.py, failure_taxonomy.py)
  → skill-gap map                   (curriculum.py prioritize)
  → curriculum plan                 (curriculum.py plan, with decision traces)
  → data discovery / verification   (discovery.py, data_registry.py, contamination.py)
  → recipe search                   (recipe_planner.py, synthetic.py)
  → multiple candidates             (cycle.py train_candidates)
  → targeted + protected + frontier eval   (eval_tiers.py)
  → promotion / rejection           (promotion.py: predeclared, multi-objective)
  → MODEL N+1                       (lineage.py GenerationLedger)
  → repeat
```

`growth/cycle.py` orchestrates the phases and records machine-readable
provenance for each one. It never overrides hard constraints (contamination,
budgets, protected regressions) and never invents evidence.

## Module map

| Module | Responsibility |
| --- | --- |
| `capability.py` | Closed skill list (57 skills, 16 categories); profiles keep raw benchmark scores visible beside normalized skill estimates |
| `benchmark_registry.py` + `catalog.py` | 40 version-pinned benchmarks across the mandated categories, with status/lifecycle/split/adapter rules |
| `contamination.py` | The non-negotiable firewall: protected/development/training sets, MinHash + canary + substring + exact/normalized detection |
| `data_registry.py` | GOLD/SILVER/BRONZE/QUARANTINE trust classes with verification floors; licensing is required for registration |
| `discovery.py` | DISCOVER → INSPECT → REGISTER; candidates enter QUARANTINE, never trainable by default |
| `failure_bank.py`, `failure_taxonomy.py` | Durable, classified failure evidence feeding curriculum priority |
| `difficulty.py` | Evidence-based difficulty calibration into bands |
| `curriculum.py` | Weakness → prioritized plan with weighted decision traces and mixture (TARGET/PRESERVE/GENERAL/REPLAY/STRETCH) |
| `synthetic.py` | Generator → independent critic → objective verifier → dedup → contamination check pipeline |
| `recipe_planner.py` | Bounded candidate recipes within measured hardware envelopes |
| `eval_tiers.py` | Tier 0 smoke → Tier 4 frontier, bound by decision stakes and budget |
| `statistics.py` | Welch's t-test, Wilson CIs, Cohen's d, pass@k; a 1-question swing is not proof |
| `promotion.py` | The single predeclared promotion rule (see below) |
| `lineage.py` | Append-only generation ledger + regression-probe memory (anti-forgetting) |
| `frontier_reference.py` | Five reference levels, protocol-honest comparability, frozen snapshots |
| `research_kb.py` | Recorded research results screened for applicability to our 9B/local scale |
| `cycle.py` | The lifecycle orchestrator with per-phase provenance |

Companion package `chowder/evals/` normalizes Inspect AI, lm-eval-harness,
benchmark-native agent suites, and Chowder-native diagnostics into one
`EvalReport` schema with honest non-measurements.

## Promotion (never one number)

A candidate is **PROMOTED** only when *all* of the predeclared checks pass —
and since the integrity pass (2026-09-17), every check consumes only
evidence that can prove its provenance and its cost:

1. target improvement, statistically significant and ≥ the declared minimum,
   measured on the candidate (`measurement_origin=MEASURED_THIS_GENERATION`);
2. no protected regression beyond tolerance, from candidate-measured rows
   only — a carried parent row reads inconclusive, never "not-regressed";
3. no material broad-battery deterioration, from candidate-measured rows
   only;
4. no calibration/hallucination regression;
5. reliability (survives repeated samples where practical);
6. evidence integrity — no contamination (`TAINTED` otherwise);
7. resource envelope respected **by settlement as well as admission**:
   projected cost admits the run, actual measured cost (all recipes,
   evaluations, failed attempts — via the cycle compute accounting artifact)
   must stay under the frozen device/wall ceilings after execution or the
   cycle refuses. A ceiling is only settled against the unit it was declared
   in: wall and project ceilings are settled from the trainer's measurements,
   and a device ceiling is settled only when the run actually separated
   device time (`ComputeCost.device_measured`) — an unmeasured device figure
   fails closed with `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` rather than reading
   as "device free", so a device ceiling that a run cannot measure stays an
   admission constraint on the projected plan and the evidence says so
   (`ceiling_enforcement`). See
   `docs/growth/DEVICE_CEILING_FAIL_CLOSED_ADDENDUM_2026-09-18.md`.

A real target repair with incomplete full-promotion evidence is
**INCONCLUSIVE** with `target_repair_validated=true` — deliberately not a
new verdict class: PROMOTED must mean "complete evidence, all gates passed".
Historical verdicts are corrected only by append-only adjudication revisions
(`GenerationLedger.append_adjudication_revision`); original records never
change.

Frontier scores never decide promotion. A candidate below GPT-class can be a
legitimate generation if it beats its parent without regressions.

## Anti-forgetting

Convincingly repaired failure classes become protected regression probes
(`lineage.RegressionMemory`). Every future generation must keep passing them;
growth is cumulative, not one capability traded for another.

## The five frontier levels

Every benchmark can be viewed against: Generation-0 floor → parent →
comparable peer (7–9B class) → open-weight frontier → absolute frontier.
Comparisons are only emitted when benchmark version, tool setting, and
reasoning setting align; otherwise the report says
`NOT DIRECTLY COMPARABLE`. Snapshots are frozen at generation time
(`SnapshotStore`, never rewritten) so "Chowder improved" can be separated
from "the frontier moved faster."

## CLI

```text
chowder eval catalog          # the pinned benchmark catalog
chowder eval scoreboard FILE  # render an eval report honestly
chowder data audit            # registered sources + trainability
chowder data register ...     # the one mutating path; enters QUARANTINE
chowder data contamination    # firewall checks against protected material
chowder growth profile FILE   # capability profile
chowder growth curriculum FILE [--protected IDs]   # plan with decision traces
chowder growth status FILE    # cycle outcome summary
chowder growth campaign validate FILE   # is this a declaration this code can execute?
chowder growth campaign plan FILE       # the curriculum items and recipe ids a run will honor
chowder growth campaign run FILE        # execute the declared campaign end to end
chowder growth campaign settle FILE     # re-settle a campaign from its accounting artifact
```

## Gen-2 certification boundary

The frozen gen2 judge (`docs/gen2/judge_gen2.py`, frozen with
`docs/quals/GEN2_PREREG_2026-09-17.md` and
`GEN2_PREREG_AMENDMENT1/2/3_2026-09-18.md`) owns the frozen policy — benchmark
set, thresholds, branch rules, verdict composition — and no second
implementation of anything production already decides. It hashes artifacts
with `training_binding.directory_digest`, settles resources with
`campaign.settle_campaign`, reads provenance from `evals.result`, interprets
contamination with `metric_binding.MetricBinder`, compares with
`statistics.compare`, and parses arms with `evals.result.EvalReport`. Three
provenance-bound arms are judged: the gen2 candidate
(`MEASURED_THIS_GENERATION`), the gen1 parent and the trusted ancestor gen0
(`MEASURED_PARENT`), each measured on the same frozen 16-prompt instrument
and the same 16-item protected mini-slices. Promotion requires the target
rule *and* no protected regression against the parent *and* the ancestor, so
a candidate cannot promote by matching an unresolved parent. Missing
evidence is INCONCLUSIVE; contamination is TAINTED; a hard failure is
REJECTED. The pass that hardened this boundary before any Gen-2 compute,
with its adversarial re-attacks and its verification counts, is
`docs/growth/GEN2_CERTIFICATION_HARDENING_REPORT_2026-09-18.md`.

The run root *is* the judge's input. `campaign_runner` materialises the judged
evidence set from the run's own measurements: the three provenance-bound arms
(`candidate_evaluation.json` -- produced by evaluating the artifact the run
selected, see `GEN2_PREREG_AMENDMENT5_2026-09-18.md` -- `parent_evaluation.json`
from `parent_eval_report_path`, and `baseline_evaluation.json` from the declared
`baseline_eval_report_path` naming the gen0 arm branch protection is judged
against), the winner's identity in `chosen_candidate.json`, and the
contamination manifest the firewall bound. `chowder growth campaign run`
followed by `python docs/gen2/judge_gen2.py <state_root>` therefore reads one
directory. Provenance is invariant -- a row's `measurement_origin` is the
evaluator's declaration, the candidate arm's rows must be candidate-measured or
honestly unmeasured, and a declared arm is copied verbatim -- an input the
manifest does not declare produces no file rather than a placeholder, and
`tests/test_growth_certification_coupling.py` proves both directions: the
frozen judge certifies the root a CLI run just wrote, and absent or tampered
evidence refuses.

The candidate arm is the run's *output*, bound to the model it measured: the
runner asks an evaluation seam (`chowder.growth.candidate_evaluation`) to measure
the artifact it selected, and refuses with `CANDIDATE_EVALUATION_NOT_PRODUCED`
when no evaluator can be built rather than reading a report from a declared path
-- `candidate_eval_report_path` is retired and its presence refuses at load.
What the seam returns is checked before any verdict: generation (report and row),
`adapter_digest` equal to the selected artifact's digest, `base_model_digest`
equal to the declared base, scored rows `MEASURED_THIS_GENERATION`, coverage of
every declared benchmark, no duplicate measurement, and the evaluation's
measured cost charged to the cycle ledger before settlement.

The seam itself is real: `chowder.growth.evaluation_binding.SubprocessEvaluationFn`
(`docs/quals/GEN2_PREREG_AMENDMENT6_2026-09-18.md`) measures the selected adapter
through the production transformers-text worker -- the same
`--spec/--result/--chowder-identity` command line and the same injectable process
runner the trainer binding uses -- from the datasets the campaign declares in
`evaluation_material_path`. It writes the first `protection.n_samples` items of
each declared dataset into the run root as the slice it measures, verifies the
adapter's digest against the bytes before loading them, and returns rows whose
`raw_artifact_ref` is the `predictions-<suite>.jsonl` it produced, whose
`metadata.artifact_sha256` is that file's digest, and whose score is the mean of
its per-item scores -- the exact evidence certification recomputes.

The *target instrument's* diagnostic metadata (T1-T10) is produced by the same
row now (`docs/quals/GEN2_PREREG_AMENDMENT8_2026-09-18.md`):
`chowder.growth.generation_diagnostics` computes the frozen rules -- EOS
termination, cap-hit, unclosed `<think>`, three-in-a-row loops, distinct trigram
ratio -- from the per-item generations the worker recorded, and the binding
merges them into the row's `metadata` flat, where the frozen judge reads them.
The facts they are defined over are recorded by
`chowder.evaluators.generation.observed_generation` beside every prediction (a
decoded completion cannot say whether it stopped on EOS or ran into the cap), the
item score is the observation-defined `eos_termination` mode in
`chowder.evaluators.scoring` -- so the row's score *is* its
`eos_termination_rate` -- and an item with no observation refuses with
`GENERATION_DIAGNOSTICS_UNMEASURED` instead of counting an unmeasured generation
as a termination failure.

Three properties of that seam are enforced before a verdict, not asserted after
one (`docs/quals/GEN2_PREREG_AMENDMENT7_2026-09-18.md`). The evaluator is built
and admitted in a `readiness` phase *before* any recipe is admitted, so a run
cannot spend a training budget and only then discover that no instrument exists
or that the declared benchmarks are not covered; a refusal after training is
recorded (accounting, attempts, selection, reason) rather than raised. The
evaluation's cost is not optional: a bare `EvalReport`, a missing cost, or a zero
that names no measurement method refuses with `CANDIDATE_EVALUATION_COST_UNREPORTED`
/ `_COST_UNMEASURED` instead of settling as zero, so evaluation compute cannot
disappear from the campaign's ceiling. And the selected artifact's digest is
re-derived from the bytes on disk twice -- before it is measured, and before the
judged evidence set is written -- refusing with `CANDIDATE_ARTIFACT_DIGEST_STALE`,
so a recorded digest can never become a certified one.

`GEN2_PREREG_AMENDMENT9_2026-09-18.md` exposes that phase to operators as
`chowder growth campaign readiness <manifest>`: every pre-compute check, zero
compute, a machine-readable `status`/`checks[]`/`reason_codes[]` result and a
non-zero exit unless all pass. Its first real use found the base identity could
not verify: `base_model_digest` in the shipped manifests is the Gen-0 freeze's
semantic `model_content_digest` (`59e767aa…`, ten model files), but
`_verify_digest` used `training_binding.directory_digest`, which also hashes a
volatile HuggingFace `.cache/huggingface/**` (`8eb92aa6…`).
`GEN2_PREREG_AMENDMENT10_2026-09-18.md` reconciles the basis on the freeze's
side: `campaign_runner._verify_base_identity` verifies a base with
`local_model_manifest.model_content_digest`, a *model-content* digest over the
payload files only, so cache churn cannot move base identity and a substituted
payload file still refuses. The adapter keeps `directory_digest` as its own
field. Readiness against the committed declaration now reports
`base_identity: ok`; the remaining refusal is the undeclared inputs.

The production verdict and the frozen judge are held to one invariant by
`tests/test_growth_gen2_dry_run_matrix.py`: across clean promotion, an inherited
trusted-ancestor regression, an artifact mutated under its own measurement, an
evaluation overrun, an unwired evaluator, an unreported cost, a contamination
mismatch and base/ancestor identity mismatches, a manifest-driven run can never
record `PROMOTED` on a root the judge would refuse.

The gen0 arm itself is declared, not assumed:
`GEN2_PREREG_AMENDMENT2_2026-09-18.md` pins it in the manifest
(`baseline_eval_report_path`) as a fresh 16-item mini-slice measurement on the
untouched dense gen0 parent with `MEASURED_PARENT` rows, referenced at zero
incremental cost. Absent, off-protocol or wrongly originated, T16 stays
UNKNOWN/FAIL and the candidate cannot be promoted — an unresolved parent never
becomes the protection baseline by default.

Where a gate could have been satisfied by a document rather than by the
measurement it claims, `GEN2_PREREG_AMENDMENT3_2026-09-18.md` states the
verification rule. Contamination is judged from the artifact the campaign
**pinned** (`contamination_manifest_path`): no pin, a missing pin, or the pin
absent from the run root is UNKNOWN — the run root's own file is never a
fallback — and a run-root copy that differs from the pin is FAIL/T18
(`CONTAMINATION_EVIDENCE_NOT_PINNED`), so a clean file cannot certify a campaign
that pinned known contamination. Each protected measurement must name an
existing raw artifact (`raw_artifact_ref`, relative to the run root or
absolute), declare its sha256 in `metadata.artifact_sha256`, and carry exactly
`n_samples` per-sample values whose mean is its own score; the digest is
recomputed over the real bytes with the production helper, and the runner
carries those artifacts into the run root so the evidence set is self-contained.
Missing, unhashed, empty or self-contradictory measurement evidence refuses
(T11) rather than certifying.

## Campaign manifests: preregistration as configuration

A campaign manifest (`campaign.CampaignManifest`) is the preregistration a
run executes: parent identity and digest, promotion sets, recipe set, per
recipe and campaign ceilings in named units, stopping rules, promotion
policy version, and every input path the run reads. `campaign_runner`
turns it into a cycle by *composing* the pieces that already own each
decision — `GrowthCycle` for sequencing/curriculum/selection/promotion,
`SubprocessTrainingFn` for execution and projected-cost admission,
`MetricBinder` for declared metric semantics, `settle_cost`/
`settle_campaign` for post-run settlement — rather than adding a second
engine. No phase invents an input: a path the manifest did not declare is
a named refusal, and the campaign's resource gate is authoritative over
the lineage record, so a run that overran its own frozen envelope cannot
record a promoted generation. Every field the schema accepts is listed in
`campaign_runner.FIELD_ENFORCEMENT` with the behavior it drives, and
`assert_every_field_enforced` fails if the schema and that table diverge:
a declared key nobody acts on is refused rather than silently ignored.

## Status

Generation 0 is frozen: the dense Qwen3.8-9B abliterated parent
(content digest `59e767aa…7555f`) was evaluated evaluation-only at
`docs/gen0/GEN0_EVAL_RESULT_2026-09-17.md` (freeze digest `5c8b18ab…`,
immutable snapshot `gen0-frontier`, ledger record `gen0`). The cycle's
TrainingFn is bound to the real trainer and its promotion inputs to
declared metric semantics (#167), with the binding qualified in
`docs/GROWTH_TRAINFN_BINDING_QUALIFICATION_2026-09-16.md` and
`docs/GROWTH_PROMOTION_BINDING_QUALIFICATION_2026-09-16.md`. The first
real Model N → N+1 cycle has now **executed and closed: gen1 was recorded
PROMOTED, and after the integrity audit its effective verdict is
INCONCLUSIVE with target_repair_validated=true**
(`docs/growth/GEN1_READJUDICATION_ADDENDUM_2026-09-17.md`; original result
`docs/quals/GEN1_RESULT_2026-09-17.md`; prereg
`docs/quals/GEN1_PREREG_2026-09-17.md` under amendments 1–3 — both recipes
trained 30/30 steps through the production binding, and the frozen rule
adjudicated the candidate from independent evaluation). The proven,
repeatable path for every future cycle: prereg → measured preflight →
production train → independent evaluation → mechanical promotion → ledger.

Known limitations are listed in `docs/DATA_POLICY.md` and
`docs/BENCHMARK_CONTAMINATION_POLICY.md`.
