# Reported-metric audit — 2026-10-08: every reported metric name against the quantity its code computes

**Status: audit record, decided.** R1–R6 are applied, each with a test and a
revert proof, and R7 is applied where it was contained: the calibration report an
operator reads is GiB-named, while the repo-wide `*_gb` capacity API is left
named as it is and recorded as an open rename with its measured blast radius
(578 sites across ~90 files). R3's decision also reached the source it was
about: `campaign_prepare` prepared a real material with the contradictory pair
this audit measured, and now derives the readout from the metric. This is
the sweep the T5 defect asks for: amendment 17 found one row whose *label* named
a quantity its code did not compute, `docs/gen2/GATE_LABEL_AUDIT_2026-10-08.md`
audited the frozen judge's own 24 rows, and
`docs/quals/GEN2_PREREG_AMENDMENT18_2026-10-08.md` applied those corrections.
The judge's rows are the **top** of a chain that reads the instrument's metadata
and the evaluation reports, so the same class has to be swept *downstream* —
where a wrong name is not read by a judge, but by an operator and by the next
gate.

## Why this audit exists

A label is a claim about a quantity. The failure mode is not a bug in arithmetic;
it is a name that says more, less, or something else than the number beside it,
so a reader acts on a fact nobody measured. T5 said "answer correctness" and
counted presence; T13 said "actual cost" for a device figure the run may not have
measured; the scoreboard below said "improved" for a metric that had got worse
(R1). None of these can certify anything by itself; all of them mislead a person
who is deciding whether to trust a model.

## Method

- Enumerate the surfaces where code reports a *named* value to a human: CLI
  payloads, rendered tables, report artifacts, registry and metadata
  declarations, and TUI panels. The judge's own row table is excluded here — it
  is the companion audit's subject.
- For each surface, read the label and the expression that produces the value,
  and compare the **name** with the **quantity**.
- Severity: **latent** (no current row moves, but the vocabulary admits a case
  that would misread), **refusing** (a wrong name makes a real measurement
  unusable), **disclosure-only** (a reader can misread; no decision moves).
- Findings are *measured* by running the code, not by reading it; each one below
  carries the command or the observed output that establishes it.

### The surfaces read

| surface | what a reader sees | verdict |
| --- | --- | --- |
| `src/chowder/evals/runner.py` (`Scoreboard`) | the eval scoreboard: category tables, the vs-parent delta table, frontier gaps | **R1** (applied) |
| `src/chowder/evals/result.py` | the `EvalReport`/`BenchmarkRun` artifact an operator and the binder read | **R5** |
| `src/chowder/evals/adapters.py` | the rows each harness adapter produces | **R4** |
| `src/chowder/growth/benchmark_registry.py`, `catalog.py` | the benchmark registry: per-benchmark scorer, metric, polarity, scale | **R2** |
| `src/chowder/growth/evaluation_binding.py`, `campaign_prepare.py`, `metric_binding.py` | the material a candidate is measured on, and the row it becomes | **R3**, **R5** |
| `src/chowder/growth/cli.py` | `chowder growth status`, `... campaign readiness`, `... campaign settle`, `... loop status` payloads | **R6** |
| `src/chowder/growth/generation_diagnostics.py` | the instrument's aggregates — the judge's T1–T10 inputs | clean (2 observations) |
| `src/chowder/growth/certification.py` | production's certification rows | clean |
| `src/chowder/campaign_scoreboard.py` | the public-benchmark and Fable scoreboards, efficiency metrics | clean |
| `src/chowder/parameter_accounting.py` | total/active/routed/MTP/vision parameter census | clean |
| `src/chowder/conditional_profile.py` | per-phase device/wall timings and memory peaks | clean |
| `src/chowder/calibration.py`, `hardware.py` | `chowder hardware-calibrate` / `hardware-detect` throughput and capacity | **R7** |
| `src/chowder/trainability.py`, `growth/frontier_reference.py`, `unsloth_env.py` | the trainability report, frontier gap table, doctor render | clean |
| `src/chowder/tui.py`, `tui_growth.py` | what the guided interface shows while a run is in flight | clean |

## Findings

| id | surface | what the name says | what the code computes | severity |
| --- | --- | --- | --- | --- |
| R1 | eval scoreboard delta table | ↑ improved / ↓ regressed | `compare()`'s higher-is-better verdict mapped without the registry's declared polarity | latent, **applied** |
| R2 | benchmark registry | `scorer` per benchmark ("judge", "unit_tests", "agent", …) | nothing: 42 declarations, 0 reads | disclosure-only, **declared** |
| R3 | evaluation material | `metric` names the measurement | the readout that produces the score is `scoring`, a separate declaration nothing compares it with | latent, **applied** |
| R4 | eval adapters | the row's `metric` | the harness's own first metric key (or a default), checked only for *equality* against the registry | refusing, **applied** |
| R5 | `EvalReport.load` | a row's `metric` | `"accuracy"` when the row declares none | disclosure-only, **applied** |
| R6 | `chowder growth campaign settle` | `"actual"` | the settled cost, whose device dimension may be unmeasured (`device_measured=false`) | disclosure-only, **applied** |
| R7 | `chowder hardware-calibrate` / `hardware-detect` | `*_gbps`, `*_gb` | GiB-based throughput and capacity (bytes ÷ 1024³), and a median of the timed passes | disclosure-only, **calibration applied, capacity API open** |

## Findings in detail

### R1 — the eval scoreboard's arrow language was direction-blind (applied)

`Scoreboard._render_deltas` mapped `compare(parent, candidate).verdict` to ↑/↓
in one direction only. `compare` answers in a **higher-is-better** vocabulary
(that is the same fact amendment 18's F2 found in the judge), while the registry
declares a polarity per benchmark and
`benchmark_registry.DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})`
admits the other one. Measured today: all 8 declared metrics and all 42 registry
entries are `higher_is_better`, so no shipped row was misread — the exposure is
latent, and the vocabulary is what makes it real.

The fix renders the arrow from the declaration and says so in the row: a
`lower_is_better` metric arrows **down** when its raw value rises, and its
Verdict cell carries `(lower is better)`. The revert proof is the measurement —
with the declaration ignored, the same row renders

```text
| latency_probe@2026-01 | 5.000 | 9.000 | +4.000 | ↑ |
```

for a metric that got **worse** (latency 5 → 9), and
`tests/test_growth_evals_and_cli.py::test_scoreboard_arrows_follow_the_declared_lower_is_better_polarity`
fails. The shipped higher-is-better path is pinned by the second test, so the
change cannot quietly invert today's arrows.

### R2 — the registry declares a scorer per benchmark that nothing reads

`BenchmarkEntry.scorer` is declared 42 times (`exact_match`, `multiple_choice`,
`unit_tests`, `judge`, `agent`, `protocol_diagnostics`) and read **zero** times in
`src/` or `tests/`:

```console
$ grep -rc "scorer=" src/chowder/growth/catalog.py
42
$ grep -rn "\.scorer\b" --include=*.py src tests | wc -l
0
```

So the registry's claim about *how each benchmark is scored* is metadata no code
consults; the executable declaration is the material's `scoring` (a different
vocabulary: `normalized_exact_match`, `final_number_match`, `eos_termination`),
and `docs/EVALUATION_MATRIX.md` reports the scorer as part of a benchmark's
evaluation posture. This is not a lie in either direction, and the registry marks
each entry's `implementation_source`, but a reader can reasonably believe the
scorer field is enforced. **Decision (applied): declare the posture, and pin the
separation.** `SCORER_POSTURES` is now the declared closed vocabulary, the field's
comment says it is descriptive and read by nothing, and a test pins the constant
to the set the catalog actually uses (so the two move together) as well as the
overlap between the two vocabularies -- which turned out to be exactly one word.
`exact_match` is a posture *and* a readout, and it means a different thing in
each, so the test states that rather than claiming a separation that does not
exist. Consuming the field was rejected: one posture is served by several
readouts, and inventing the mapping would refuse legitimate suites.

### R3 — the material declares the readout and the label separately

`SuiteMaterial` carries both `scoring` (what the worker scores with,
`normalized_exact_match` by default) and `metric` (the name the produced row
carries, `accuracy` by default). Nothing relates them, and the pair is accepted
even when it contradicts itself:

```console
$ python - <<'PY'
from chowder.growth.evaluation_binding import SuiteMaterial
s = SuiteMaterial.from_mapping(
    {"benchmark_qualified_id": "math500@2024-04", "dataset": "slice.jsonl",
     "scoring": "eos_termination", "metric": "accuracy"}, source="measure")
print("accepted:", s.scoring, "/", s.metric)
PY
accepted: eos_termination / accuracy
```

The two labels are then validated against *different* authorities: the worker
validates `scoring` against its own vocabulary, and `MetricBinder` validates
`metric` against the registry's `primary_metric`. A suite that declares a
text-scoring benchmark but an observed readout — or the reverse — is refused only
when the *names* happen to disagree. `campaign_prepare.py` builds the two from
different tables (`_SLICE_DATASETS[...]["scoring"]` and `_metric_for(qualified_id)`,
which returns `"accuracy"` for every id that is not `generation-diagnostics@…`),
so the pair is coupled by convention, not by construction. **Decision (applied):
refuse a pair that cannot both be true, and fix the source that produced one.**
The observed vocabulary has one owner (`evaluators.scoring.OBSERVED_SCORINGS`),
and `SuiteMaterial.from_mapping` now refuses both directions: an observed readout
must be labelled with the rate it reports (`eos_termination` ->
`eos_termination_rate`), and that rate name must come from an observed readout.
Building that rule surfaced the finding's real cost -- production itself was
preparing the contradictory pair, because `campaign_prepare` filled `scoring`
from the pinned slice table (defaulting to `normalized_exact_match`) and `metric`
from `_metric_for` (returning `eos_termination_rate` for the diagnostics
instrument). The metric is the authority for what the row measures, so the
readout now follows it (`_scoring_for`), and the campaign fixture that mirrored
the old shape was corrected with it.

### R4 — the adapters discover the metric name instead of declaring it

The row's `metric` is what the harness happened to call its number:
`LMEvalAdapter` takes the first float-valued key (`lm-eval` emits keys like
`acc,none`, split to `acc`), `NativeAgentBenchmarkAdapter` takes the suite's own
payload defaulting to `resolve_rate`, `InspectAdapter` and
`ChowderCustomEvalAdapter` default to `accuracy`. The only thing tying the name to
the quantity is an equality check in the binder — measured against the real
registry:

```console
metric='acc'      -> BindingRefusal: mmlu_pro@v2: the run's metric 'acc' is not
                     the registry's declared primary metric 'accuracy'; a declared
                     scale belongs to one metric, not to a benchmark's name
metric='accuracy' -> BoundMeasurement
```

Two consequences, both worth stating plainly. First, a row whose label is the
harness's own key is **unusable for promotion** even though it was measured
correctly — the refusal is loud and correct, and it means the lm-eval path cannot
contribute until the key is mapped onto the registry's name. Second, when the
harness's guess happens to equal the registry's name, the row binds whatever it
actually measured: nothing here checks that the adapter's number *is* the
registry's quantity, only that the two strings match. **Decision (applied): the
declared mapping, in one place.** `HARNESS_METRIC_ALIASES` maps the harness keys
that name the registry's quantity (`acc`, `acc_norm`, `exact_match`) onto
`accuracy`, the harness's own key is kept in `metadata["harness_metric"]` so the
relabelling is visible, and a key that is not declared is left exactly as the
harness reported it -- the binder still refuses it loudly, and the table never
guesses. A harness result with no numeric metric now names no metric and carries
no score, instead of claiming `acc` by default.

### R5 — an absent metric reads back as `"accuracy"`

`EvalReport.load` defaults `metric=run.get("metric", "accuracy")`. Measured: a
report row that declares no metric loads as `'accuracy'`. The repo refuses exactly
this shape elsewhere — `generation_diagnostics._observed_bool` refuses an absent
`eos_terminated` because "an unevaluated field would be read as False and turn an
unmeasured generation into a termination failure". Here an undeclared metric
becomes a *named* measurement. It is fail-closed downstream (the binder compares
names and refuses a mismatch), so the exposure is a wrong label on a row that
cannot bind, not a silent pass. **Decision (applied): an undeclared metric stays
undeclared, and the binder says so by name.** `BenchmarkRun.metric` defaults to
empty and `EvalReport.load` reads an absent metric back absent -- the rule
`measurement_origin` already follows by defaulting to `UNMEASURED`. The binder
refuses such a row with a message that names the absence and the quantity the
registry requires, rather than comparing an empty string against a name the row
never carried.

### R6 — the settle payload calls a possibly-unmeasured cost "actual"

`chowder growth campaign settle` prints `"actual": total.to_dict()` for the
settled cost. Amendment 1 declares `device_time_measured=false` — the device
dimension is an admission (projected) constraint while wall is the post-run
settlement — and `device_measured` travels inside the same object, so the reader
can see which dimension is which. The key name still says "actual" for a number
that may not be one; the judge's T13 row was corrected for exactly this wording
in amendment 18 (F4). **Decision (applied): rename the key.** The payload now
carries `"settled"` -- the word the command and the audit's own finding use --
with `device_measured` still travelling inside the settled object, so the
disclosure is unchanged and the key no longer claims the number is one. The
handler's docstring says "settled" too.

### R7 — throughput and capacity are GiB-based and named GB, and medians named as rates

`calibration.py` computes `(byte_count / _GIB) / seconds` (`_GIB = 1024 ** 3`) and
reports it as `read_gbps`, `write_gbps`, `copy_gbps`,
`host_to_device_gbps`, `device_to_host_gbps`; `total_vram_gb`/`free_vram_gb` are
`bytes / _GIB`; `hardware.py` parses `nvidia-smi` MiB as `memory_mib / 1024` and
names it `memory_gb`. The values are GiB and GiB/s, and `_median_gbps` takes a
median of the timed passes while the field names drop the statistic. The program's
own documented convention is the opposite: "Decimal GB uses 10^9 bytes; GiB uses
2^30" (`docs/EXPERIMENT_D_LOW_ACTIVE_HYBRID_LM.md`), which is the convention the
comparison tables in that document use. A reader mixing a model card's decimal GB
with these GiB readings overstates capacity and throughput by ~7.4%.
**Decision (applied where it was contained, open where it was not):**
`calibration.py` -- the payload `chowder hardware-calibrate` prints -- is renamed
to `*_gib_per_s_median` and `*_vram_gib`, so the reported number says both its
unit and its statistic, and a test pins the GiB arithmetic (10^9 bytes in a
second is 0.93 GiB/s, not 1.0). The repo-wide `*_gb` capacity API in
`hardware.py` (and `training_backends.py`'s `vram_gb`) is **not** renamed here:
that is 578 call sites across ~90 files -- memory preflight, every backend, the
TUI, the unsloth environment -- so it is recorded as a deliberate rename rather
than folded into a measurement fix. The unit is now stated at the definitions,
`AcceleratorLink.measured_bandwidth_gbps` is documented as written by nothing in
the repo (so its unit is whatever a caller meant), and the GiB arithmetic stays
pinned by `tests/test_hardware.py`.

## Checked and clean

- **The generation-diagnostics instrument** (the judge's T1–T10 inputs) computes
  each aggregate from the worker's own per-item rows, refuses an item that cannot
  report whether it terminated, refuses a termination after the cap, and computes
  distinct-trigram ratios and loop flags from decoded text with the gen1 rules
  ported verbatim. Two naming observations, neither a gate risk because T8's
  threshold is `<= 0`: `obvious_loop_count` counts *completions* containing a line
  repeated three times in a row (the judge's label reads "obvious loops"), and
  "distinct-trigram ratio" is a ratio over *word* trigrams (`text.split()`).
- **Parameter accounting** carries `active_definition` on the object, states
  `active = total − routed_experts − router` for sparse models, and refuses to
  build an "A4B" label without measured routing geometry.
- **`compute_cost` / campaign ceilings**: device and wall hours are separate
  fields, `device_measured` is explicit, and the settlement is a real comparison.
- **`campaign_scoreboard`**: an unmeasured benchmark renders `None`, never `0`;
  historical thresholds are recorded verbatim as exclusive; `EFFICIENCY_METRICS`
  names the four declared formulas and the two helpers match those names.
- **`conditional_profile`**: `p95_ms` is a nearest-rank 95th percentile
  (`min(n-1, ceil(0.95n)-1)`), `*_mean_inclusive` is the mean of that phase's own
  `cpu_wall_ms` samples, and the profiler's limits are stated in its docstring.
- **Production certification rows** (`certification.py`): each row names the
  requirement and the specific problem, and an unavailable arm is UNKNOWN.
- **Frontier gaps**: `gap = reference − score` with the sign documented on the
  field ("positive = Chowder behind"); the rendered table does not restate the
  sign, which is an observation rather than a finding.
- **Trainability / TUI / doctor**: the probe window, the wall unit
  (`expected cost … wall GPU-hours`) and each capability check are named
  explicitly.

## What the audit did not find

No reading here lets a **worse** result pass a gate. R4 refuses a correctly
measured row (never accepts a wrong one); R3 and R5 need a declaration to be wrong
before they can mislabel anything, and the binder still refuses the resulting row
when the names differ; R1 was latent for every shipped metric; R2, R6 and R7 are
display. Nothing in this audit changes a threshold, a normalization, a
comparison, or a gate.

## What this audit did not read

It covers the surfaces in the table above, not every f-string in the repo. Not
read: the historical Gen-1 driver's renderers (`docs/gen1/`), the Kaggle
transport's device-side payloads, `runtime_eval`'s task-harness reward reporting,
and the modules with no reporting path. There is no Gen-2 candidate evaluation in
this checkout, so no finding here is checked against a real candidate run — R4 is
measured against the real registry and a constructed row, which is the strongest
form available.

## Governance

R1 was applied with the audit, because it is a rendering bug in a reported
surface and it changes no shipped row. R2–R6 are applied in the follow-up change,
each with a test that fails without it and each revert-proven in place (ten
mutations, every pin caught, every file restored byte-for-byte):

| finding | change | pin |
| --- | --- | --- |
| R2 | `SCORER_POSTURES` declared; the field documented as a posture | constant == catalog's set, overlap with the readout vocabulary pinned to `exact_match` |
| R3 | two-way readout/label rule in `SuiteMaterial`; `_scoring_for` at the source | the measured contradiction refused, the admissible pair loads, the prepared readout follows the metric |
| R4 | `HARNESS_METRIC_ALIASES` + `metadata["harness_metric"]` | `acc` -> `accuracy` with the key kept; an undeclared key untouched; no numeric key names nothing |
| R5 | empty metric default, at construction and on load; binder refusal | the loaded row is empty and the binder names the absence |
| R6 | `"settled"` replaces `"actual"` | the payload key and the `device_measured` flag beside it |
| R7 | calibration renamed to GiB + median; unit documented elsewhere | the payload keys, and 10^9 bytes in a second is 0.93 GiB/s |

Two of the decisions changed more than a name. R3's rule exposed that production
prepared the contradictory pair it refused, so the source was fixed; and R2's
disjointness claim was wrong -- `exact_match` is in both vocabularies -- so the
claim was restated as the measured overlap instead of the assumption. What is
*not* applied is recorded too: the repo-wide `*_gb` rename (578 sites) and the
consumption of `scorer` (rejected on purpose). Nothing in this change moves a
threshold, a normalization, a comparison or a gate, and no verdict changes.
