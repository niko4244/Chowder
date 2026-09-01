# Single-hop autonomous repair coordinator

Chowder can now close one complete evidence-driven repair loop without exposing holdout examples to the repair source provider.

## Flow

```text
trained candidate
      |
      v
independent evaluation
      |
      v
gate rejects candidate
      |
      v
harvest item failures
      |
      v
cluster + repair plan
      |
      v
re-verify holdout fingerprint indexes
      |
      v
sanitized RepairRequest
      |
      v
RepairSourceProvider
      |
      v
source provenance + contamination audit
      |
      v
repair candidate population
      |
      v
train -> evaluate -> gate
      |
      v
promote / reject repairs
```

## Target policy

`run_single_hop_autonomous_repair()` only considers candidates that:

- completed training and evaluation successfully,
- were rejected by the promotion gate,
- have diagnostics with no harvesting error,
- produced at least one repair plan requiring independent source material.

Without an explicit candidate ID, the coordinator follows tournament order and chooses the strongest rejected candidate that is repairable. Within that candidate it chooses the largest failure cluster, using the plan ID as a deterministic tie-breaker.

## Holdout evidence

Before asking a provider for repair material, the coordinator reopens every holdout fingerprint index referenced by the source evaluation and verifies its SHA-256 against both evidence declarations. This protects against evidence mutation between evaluation and repair.

All evaluation-suite holdout indexes are passed to repair-data contamination admission, not only the failing suite.

## Deliberately single-hop

The coordinator does not recursively repair the repair generation. Recursive evolution needs explicit policies for:

- maximum repair depth,
- total repair GPU-hour budget,
- repeated-failure detection,
- no-progress termination,
- lineage diversity,
- escalation from data repair to optimizer/architecture hypotheses.

Those controls should be added before recursion is enabled.
