# Atomic repair proposal persistence

Repair candidate admission spans three state domains:

1. generated repair files,
2. the in-memory experiment graph and GPU-hour reservations,
3. the persistent SQLite run registry.

Chowder treats these as one proposal transaction.

## Registry batches

`RunRegistry.record_experiments()` inserts an ordered experiment batch in one SQLite transaction. Parent rows may appear earlier in the same batch. A primary-key or foreign-key failure rolls back every insert.

`RunRegistry.record_failures()` likewise persists harvested failure evidence as one transaction, preventing partial diagnostic histories.

## Repair orchestration rollback

`prepare_and_propose_repair_population()` now executes:

```text
materialize repair data
        |
        v
build candidate population
        |
        v
EvolutionEngine.propose
        |
        v
RunRegistry.record_experiments
```

If any step raises before registry persistence completes:

- proposed engine candidates are withdrawn,
- their GPU-hour reservations are released without being charged,
- their graph nodes are removed,
- the generated repair directory is deleted,
- the SQLite batch is rolled back.

This prevents a split-brain state where the engine believes candidates exist but the durable registry does not, or where only a prefix of a candidate population is persisted.

## Boundary

This transaction covers proposal-time state only. Once a candidate enters `RUNNING`, rollback is no longer appropriate; training/evaluation compute is instead settled through the normal failure/accounting path.
