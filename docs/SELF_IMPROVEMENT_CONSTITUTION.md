# Chowder Self-Improvement Constitution

## Purpose

Chowder may search for better models and system configurations, but it is an
evidence-gated experiment engine rather than an unrestricted self-modifying
agent. The Python engine remains the authority for objectives, evaluation,
promotion, budgets, provenance, and recovery.

## Immutable objective identity

A run freezes an objective identity containing:

- goal version and canonical `Goal` digest;
- protected benchmark/dataset digest;
- evaluation-protocol digest; and
- constitution digest.

Every goal assessment and promotion decision must carry that identity. A
resume must use the exact same identity. If a goal, benchmark, dataset,
protocol, or constitution changes, the old run cannot continue; the operator
must start a new objective version.

## Protected surfaces

Autonomous workers may not modify any of these surfaces during an objective:

- goal thresholds and budgets;
- protected benchmark content;
- contamination rules;
- evaluation protocol fingerprints;
- promotion rules;
- independent-judge selection;
- budget ceilings;
- provenance and hashing code;
- sandbox/worktree boundaries;
- permission policy;
- this constitution; or
- evidence already recorded in the ledger.

A protected change requires both a new objective version and explicit human
approval. Approval does not rewrite or delete evidence from the old objective.
There are no autonomous commits or automatic protected-branch merges.

## Evidence and refusal rules

- Missing measurements are `UNKNOWN`, never zero and never success.
- Non-finite values, malformed evidence, digest mismatches, and incomplete
  evaluation identity are `INVALID`.
- A crashed, cancelled, or incomplete candidate cannot be promoted.
- The planner, implementer, and evaluator cannot be the sole judge of their
  own work.
- Budget or generation exhaustion is not success.
- Only a complete assessment in state `MET` is goal completion.
- Every refusal records machine-readable reason codes and the evidence
  references available at the time of refusal.

## Change-control protocol

The constitution implementation exposes a narrow change-control check. It does
not mutate state and cannot grant an autonomous protected change. It accepts a
protected change only when the proposed objective identity has a different
goal version and the caller supplies explicit human approval. Callers must
persist the approval and resulting objective as a separate, reviewable event;
the constitution does not erase the previous identity.

## Scope of this milestone

This milestone provides the immutable policy primitives and canonical goal
assessment contract. It does not yet run training, launch evaluation, own a
campaign lifecycle, or provide a desktop UI. Those layers must consume these
primitives rather than duplicate their decisions.
