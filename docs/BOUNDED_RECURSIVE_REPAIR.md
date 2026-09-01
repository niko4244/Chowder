# Bounded Recursive Repair

Chowder's recursive repair controller is intentionally finite. It does not implement an open-ended `while not good_enough` loop.

## Preconditions

Every repair hop inherits the invariants established by the single-hop repair path:

- the target must be gate-rejected and successfully evaluated;
- the failure diagnosis must require an independent repair source;
- holdout fingerprints are reverified before repair-source retrieval;
- repair data is independently sourced and contamination checked;
- the exact parent adapter directory is SHA-256 bound and continued trainably;
- replay history is cumulative, normalized, deterministic, and SHA-256 bound;
- repair/replay bytes are reverified at controller and worker boundaries;
- compute reservations remain subject to the global GPU-hour budget.

## Stable failure signatures

`FailureRecord.failure_id` intentionally includes experiment/run identity, so it cannot detect the same underlying benchmark failure across generations.

Bounded recursion instead derives a stable internal signature from:

- evaluator;
- suite;
- evaluation protocol hash;
- failure source role;
- failure kind;
- benchmark row index;
- SHA-256(prompt);
- SHA-256(expected answer).

Experiment IDs, evaluation run IDs, predictions, and artifact IDs are excluded. Raw holdout text never crosses the repair-source provider boundary.

This allows Chowder to recognize that two different adapters are failing the same protected benchmark state even when their failure IDs differ.

## Target selection

Within each rejected generation Chowder walks tournament order and chooses the strongest independently repairable target whose stable failure signature has not exhausted its recurrence limit.

A repeated top-ranked failure therefore does not hide a lower-ranked novel repair opportunity.

## Progress rule

After the first repair hop, the score of the next selected rejected target must improve over the previous selected target by at least `min_score_improvement`.

This prevents tiny metric noise from justifying another training generation.

## Stop reasons

Every completed controller run terminates with one explicit reason:

- `PROMOTED` — the source generation was already promoted, or a repair hop produced a promoted candidate.
- `MAX_DEPTH` — the configured hard repair depth was reached.
- `NO_PROGRESS` — the best novel rejected target did not improve enough.
- `REPEATED_FAILURE` — all independently repairable failure signatures reached their recurrence limit.
- `BUDGET_EXHAUSTED` — no GPU-hour budget remains.
- `NO_ADMISSIBLE_CANDIDATE` — budget remains, but the next replay-adjusted repair population cannot fit/admit.
- `NO_REPAIRABLE_DIAGNOSTIC` — rejected candidates contain no independently repairable diagnostic evidence.

Unexpected integrity/provenance errors are not converted into benign stop states; they propagate and fail the run.

## Default policy

```python
RecursiveRepairPolicy(
    max_depth=3,
    min_score_improvement=1e-4,
    max_failure_signature_occurrences=1,
    replay_ratio=1.0,
)
```

The defaults permit a maximum of three repair hops, refuse to attack the same stable failure signature twice, require measurable score progress, and preserve cumulative rehearsal at a 1:1 replay ratio.

## Why promotion stops recursive repair

Recursive repair is a recovery path for rejected candidates. Once a candidate passes the promotion gate, it becomes the new accepted baseline. Further improvement should proceed through the broader experiment/evolution planner rather than treating an accepted model as though it were still a failed repair target.
