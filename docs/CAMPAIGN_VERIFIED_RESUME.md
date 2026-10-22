# Verified campaign recovery

The campaign writer lock remains the sole admission authority for a state root.
After a process interruption, the runner can continue a campaign only when
every completed local training invocation has a durable, uniquely indexed
`campaign-attempt-receipts/*.json` record and a matching fsynced journal
`attempt_receipt` event. The recorded experiment's `attempt-XX` directory,
terminal `training-evidence.json`, measured or observed incremental cost,
worker source identity, and complete SHA256 artifact inventory are rechecked
before any saved attempt is replayed.

Recovery also verifies the exact frozen campaign declaration and hashes the
declared input files. Every recovered attempt is charged to a newly rendered
campaign accounting ledger **once**. Before a new attempt starts, the runner
checks its projected cost against the declared campaign budget **minus**
recovered actual spend. A declared candidate search reconstructs its
deterministic rounds from these attempts and continues survivors through the
existing checkpoint-aware training backend, preserving optimizer/scheduler/RNG
restoration and the backend's checkpoint manifest checks.

This mechanism **replays completed attempts**, rather than training them
again. It does not restart a half-executed trainer from an arbitrary or
unverified checkpoint. An `attempt started` without a matching, fully durable
receipt, a torn journal, an invalid cost, changed input bytes, a missing
artifact, a mismatched checkpoint, or a candidate evaluation that may already
have begun all block automatic recovery. The journal stays open for
operator-controlled forensic reconciliation; spent or unknown compute is
never silently treated as free. Training submitted through caller-supplied
or remote executor seams retains its existing accounting semantics and is not
automatically replayed.

The terminal campaign verdict and promoted lineage remain protected by the
separate PR273 promotion transaction. This recovery path does not change
human approval, promotion certification, or the frozen judge.
