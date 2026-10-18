# GEN2 prereg amendment 22 — 2026-10-18: the parent's raw predictions travel with the prepared parent arm

Adds to the 2026-09-17 prereg and amendments 1–21 and supersedes none of
them. **This is an evidence-plumbing fix, not a quality gate.** No slice,
threshold, retention bound, comparison, decoding parameter, recipe, step count
or compute ceiling moves.

## A. What run 8 did and why it was refused

Run 8 (launched 2026-10-18 03:18 from main `6c75b4b`, declaration
`prepared-v15`):

- attempt-26 (`recipe-00-lr5e-05`) and attempt-27 (`recipe-01-lr0.0001`) both
  SUCCEEDED and settled at 2.151 and 1.864 GPU-h, inside the 2.5 per-recipe
  ceiling;
- selection (`first_successful`) chose `recipe-00-lr5e-05`;
- the independent candidate arm measured the selected adapter (`fec4dba2…`).

Certification then refused the run (`refused_by: certification_evidence`,
verdict REFUSED):

> the parent arm's generation-diagnostics@gen2-response-surface-v1 row names
> raw artifact 'predictions-generation-diagnostics.jsonl', which does not
> exist next to …/prepared-v15/parent-eval-report.json

Certification resolves each declared row's relative `raw_artifact_ref` next to
the declared report. `prepare` had rewritten the parent rows into every
prepared directory since `prepared-v10`, but it never copied their raw files.
The only copies are in `<state_root>/parent-arm/`. No run could have
certified against any prepared directory.

## B. The fix (PRs #237, #238)

- `chowder growth campaign prepare --parent-artifacts <dir>` copies each
  relative raw file the parent rows name into the prepared directory. It
  refuses before any compute if a declared row's file cannot be found.
- #238 keeps the "carried" notes out of `parent-profile.json`'s `unmeasured`
  list. The first v16 prepare wrote them there, and that version is archived
  under `<state_root>/superseded-2026-10-18-v16-profile-notes/`.

`prepared-v16` was produced from main `bf6c898` with the same inputs as v15,
plus `--parent-artifacts <state_root>/parent-arm`. Compared field by field
with v15, ignoring timestamps, digests and the directory name:

- `contamination`, `corpus-quality`, `data-registry`, `training-material`,
  `evaluation-material`, `parent-eval-report`, `parent-profile` and
  `project-template` are identical. That means the same corpus, slices, rows
  and scores, and the same 140 steps × 8 examples.
- `hardware-budget` differs only in `measured_load_seconds` (probe noise; not
  read by the planner, which projects from `model_lifecycle`).
- The prepared declaration's `notes` differ only in prose already true since
  amendment 21. The text now states the 2.5/7.0 ceilings and their headroom
  rationale, where it previously quoted the pre-amendment-21 numbers.
- New: `predictions-generation-diagnostics.jsonl`, `predictions-math500.jsonl`
  and `predictions-mgsm.jsonl`, byte copies of `<state_root>/parent-arm/`.

The declaration's eight input paths now name `prepared-v16`.

## C. Admissibility, stated plainly

Unlike amendments 20 and 21, this one is written **after** a run produced
candidate measurements. Run 8's in-run evaluations (gen-diag 0.5 and
0.5625) and the candidate arm were visible before the rerun was decided.

The rerun is admissible for one reason: nothing a result could steer moves.

- The refusal was mechanical, a missing file, and independent of any score.
- The rerun repeats the same recipes, steps, slices, gates and judge.
- The fix can only make certification *possible*. It does not change what
  certification checks.

Run 8's `campaign-run.json`, `cycle_compute_accounting.json` and
`evaluation/` are archived under `<state_root>/superseded-2026-10-18-run8/`,
as run 7's were. Its attempts (`attempts/attempt-26`, `attempt-27`) stay in
place, and nothing is deleted. A verdict from the
rerun will be reported alongside run 8's refused record.
