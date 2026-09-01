# Transactional experiment proposals

Autonomous experimentation must not leave half-applied graph or compute state.

## Invariant

`EvolutionEngine.propose()` now separates selection from mutation:

```text
candidate filtering
      |
      v
ordered batch preflight
  - duplicate IDs
  - parent availability
      |
      v
atomic graph add
      |
      v
budget reservations
```

If graph validation fails, no candidate from that proposal batch is added and no GPU-hour reservation is created.

Parents may still be earlier candidates in the same ordered batch, preserving parent→child experiment construction.

## Withdrawal

`withdraw_proposals()` removes still-PLANNED proposals and releases their reservations without charging spent compute. It refuses running/completed experiments and graph nodes with children outside the withdrawal batch.

This is intended for orchestration rollback when a post-proposal persistence step fails.

## Settlement

`fail()` and `adjudicate()` now require an active reservation. An unreserved result cannot inject compute spend, change a graph status, or enter the promotion tournament through the engine settlement API.

Duplicate results are rejected before any reservation is settled.
