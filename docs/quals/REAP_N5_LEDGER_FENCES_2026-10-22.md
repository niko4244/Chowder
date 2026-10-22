# REAP N5: ledger mutation and upward adjudication fences

Base: consolidated PR277 `a0a678d`. Scope is the public lineage writer,
adjudication revisions, and historical transaction-less lineage read path.
No frozen selector, campaign compute, model checkpoint or GPU changes.

## Reproduced before the patch

Five targeted CPU-only synthetic cases failed on the base commit:

1. An ordinary REJECTED REAP generation could be changed to effective
   `PROMOTED` using `append_adjudication_revision` alone.
2. A once-PROMOTED generation could be downgraded to INCONCLUSIVE and then
   silently upgraded to PROMOTED through a revision.
3. Direct `GenerationLedger.record` accepted `PROMOTED` with `{}` model identity.
4. The importable `_grant(..., judge_certified=True)` could mint a MAC-accepted
   REAP approval in-process without any actual frozen judge or human action.
5. A historical REAP ledger row marked PROMOTED with no promotion transaction
   remained visible to `get` / `effective_verdict`.

## Behavior after the patch

- An adjudication revision may preserve PROMOTED or downgrade it, but may not
  **introduce or restore PROMOTED** after a non-PROMOTED effective verdict.
  Both the writer and `effective_verdict` apply the fence, so a hand-edited
  revision file cannot bypass it. A new approval needs a separate, verified
  release transaction, not a handwritten adjudication row.
- A direct PROMOTED record must name a non-empty base model filesystem path.
  A bare revision string does not permit classification of a local REAP
  checkpoint. Duplicate generation IDs still fail first, as before.
- For classified REAP, the ledger verifies a candidate-bound owner grant **and**
  independently calls the frozen-judge precondition at mutation time. That
  predicate is deliberately always false in production until an independent
  frozen-judge receipt mechanism is available. A synthetic positive campaign
  test explicitly injects both independent judge seam results; no environment
  variable grants approval.
- Transaction-less REAP PROMOTED records are not readable as valid promotions,
  even if they predate this guard. They need an audited migration/release path.
  Non-REAP historical records retain their legacy transaction-free handling.

## Security limits

**This is a fail-closed production REAP fence, not a cryptographic owner
identity system.** `_grant` remains importable and can mint local MACs, but a
forged MAC *alone* no longer authorizes REAP ledger mutation while the frozen
judge predicate remains false. Arbitrary untrusted Python executing in the
same interpreter can still monkeypatch the judge predicate, validator or
ledger itself. A cryptographically signed, out-of-process frozen judge receipt
and owner release certificate with a pinned trust anchor is needed before
lifting this hold. A caller can also misrepresent an unidentified checkpoint
with a different path/config; the model-to-lineage classification is not a
signed immutable binding. Hash-verified checkpoint identity and immutable
lineage provenance remain separate work.

An adjudication document is a correction to evidence, not a substitute for
fresh REAP promotion. Candidate evaluation remains separate from release.
