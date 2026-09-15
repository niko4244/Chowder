# Qualification harness conventions

Every preregistration in this directory ships with a **mechanical judge
script** — a stdlib-only, read-only program that scores the run's durable
artifacts against each preregistered threshold, so the verdict is mechanical
when the run finishes. The pattern was established by
`judge_rung3c_2026-09-15.py` (rung 3c) and its shared machinery now lives in
`quals_harness.py`.

## The honest order

1. **Preregister** — write the prereg doc with thresholds, budget
   decomposition, and verdict logic. Commit and push it *before* any
   implementation or run.
2. **Write the judge** — one script per run, importing the shared harness.
   Commit it on the same branch as the prereg so instrument and contract are
   frozen together.
3. **Dry-run the judge** on the *previous* run's artifacts. It must refuse
   the old run for exactly the reasons the new prereg changes the contract —
   this validates discrimination, not just green-ness.
4. **Run.** Then run the judge against the new run root; its verdict table is
   the threshold-by-threshold judgment, quoted directly in the result doc.

## What lives where

- **`quals_harness.py`** — run-agnostic machinery only: the `Verdict` table
  and its finalization rules, artifact discovery, read-only registry access,
  hashing, JSON loading, and the report/exit discipline. It contains **no**
  run-specific pins. Its rules are pinned by `tests/test_quals_harness.py`.
- **`judge_<rung>_<date>.py`** — the run's thresholds, pins (hashes, budgets,
  ceilings), and threshold checks. Nothing here may be reused by editing it
  for a new run; copy it and change the pins, or extract into the harness
  when a check repeats across runs.

## Epistemics every judge inherits

- **UNKNOWN certifies nothing.** An artifact that is missing or unreadable is
  UNKNOWN, never an assumed pass. A judge that cannot decide a threshold
  refuses to certify.
- **Exit 0 only when every threshold is PASS.** Any FAIL refuses
  (exit 1); any UNKNOWN refuses to certify (exit 1). INFO rows are recorded
  but do not gate.
- **Read-only always.** The registry is opened with SQLite read-only URIs;
  corpora are hashed without writing; the run directory is never mutated.
- **A result that disagrees with its own ledger is a lie** — judges re-derive
  what they can (budget arithmetic, deltas) rather than trusting reported
  summary fields.

## Extracting into the harness

When a second judge needs a check, that is the signal to lift the check's
*shape* into `quals_harness.py` (keeping run-specific constants in the
judge) and pin the shape with a test in `tests/test_quals_harness.py`.
Harness changes run the full suite before commit: the harness is the
discipline every judge shares, so loosening it silently is the failure mode
these conventions exist to prevent.
