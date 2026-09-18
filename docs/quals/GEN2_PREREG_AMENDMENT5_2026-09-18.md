# Gen-2 Preregistration — Amendment 5 (2026-09-18)

Written before any gen2 compute. It changes no threshold, no benchmark set, no
budget, no stopping rule and no verdict class. It records two things about the
run path itself.

## A. The candidate arm is a run output, not an input

Amendment 2 declared the trusted-ancestor arm and PR #181 made the runner write
the judged evidence set. One arm was still an *input*: the candidate evaluation.
`candidate_eval_report_path` let a campaign name a report that was prepared
outside the run and hand it to the promotion path as the candidate side.

From this amendment the manifest key is **retired**: declaring it refuses at load
with that reason, and a run must instead be given an evaluation seam. After
selection the runner asks that seam to measure the artifact the run selected,
bound to:

* the artifact digest the binding measured over the winning attempt
  (`adapter_digest == selected artifact_sha256`),
* the declared dense base (`base_model_digest == base_model_digest`),
* the run's own candidate generation (report-level and row-level),
* `measurement_origin = MEASURED_THIS_GENERATION` for every scored row.

A row carrying `MEASURED_PARENT` or `CARRIED_REFERENCE` is refused by name; a row
that is honestly `UNMEASURED` is accepted as an unmeasured row and stays
unmeasured through the binder (unmeasured is not a pass). A declared benchmark
set with no row at all is a refusal — an unevaluated declaration is a missing
measurement, not an omission. The evaluation's measured cost is charged to the
cycle ledger before settlement, so evaluation compute counts against the
campaign's own declared ceilings.

This build wires **no** production evaluator: `campaign_runner.build_evaluator`
returns `None` and the run refuses with `CANDIDATE_EVALUATION_NOT_PRODUCED`
before any verdict. That is deliberate — the alternative is adjudicating on
evidence the run did not produce. The instrument that measures the selected
adapter under this campaign's declared protocol (the 16-item diagnostics slice
plus the protected mini-slices, per-sample) is the remaining build, and until it
exists a gen2 run cannot reach a verdict.

## B. Run readiness of `docs/gen2/gen2_campaign.json`

`run_campaign` requires six inputs from disk. As of this amendment the
checked-in gen2 declaration provides **none** of them as an existing artifact:

| required input | state |
| --- | --- |
| `project_template_path` | no parked file: the Gen-1 driver composed its project in process (`docs/gen1/run_gen1_cycle.py`) |
| `training_material_path` | no parked file: the driver authored its curriculum rows in source |
| `data_registry_path` | no parked file: the driver built its registry in process |
| `hardware_budget_path` | no parked file: the driver derived step cost from its own in-process probe |
| `parent_profile_path` | no parked file: the Gen-1 cycle produced no instrument-based capability profile, and the Gen-0 freeze profile is **not** the Gen-1 profile (declaring it would be a carried-evidence substitution) |
| `contamination_manifest_path` | declared, but produced by the run into its own state root |

`baseline_eval_report_path` (amendment 2) is declared and **not yet measured**.

Consequently `chowder growth campaign plan` and `... run` both refuse before
compute, naming every undeclared input at once. That refusal is the honest state:
it is not a defect in the gates, and no path is invented here to make the
declaration look runnable. The pre-compute build is: emit the four documents
above from production code (not from a `docs/` script), measure the Gen-0 arm at
the declared path under the frozen mini-slice protocol, and produce a Gen-1
parent profile from measured evidence.

## Unchanged

Every threshold, both benchmark sets, the mini-slice protocol, the budgets
(0.30/0.75 per recipe, 0.60/1.50 per campaign, `device_time_measured=false`),
the stopping rules, the verdict classes, the scoped-repair semantics, amendment
1's device-ceiling policy, amendment 2's ancestor arm and amendment 4's
certification-before-lineage ordering.
