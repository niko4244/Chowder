# Leak-resistant repair provider boundary

Chowder may use evaluation failures to decide **what kind of capability needs repair**, but an autonomous data source/generator must not receive the protected holdout rows themselves.

## Provider-visible request

`RepairSourceProvider.propose()` accepts only `RepairRequest`. The request contains:

- opaque request / plan / cluster IDs;
- evaluator and suite names;
- broad failure class;
- controlled `RepairStrategy` enum;
- failure count;
- evaluation protocol SHA-256;
- opaque failure IDs.

It does not contain:

- prompt text;
- expected answers;
- model predictions;
- row indexes;
- free-form failure observations;
- free-form suspected-cause or intervention prose.

Free-form plan text is deliberately excluded because it could itself accidentally contain copied benchmark content.

## Controlled strategies

Failure classes are reduced to a small strategy vocabulary:

```text
empty_prediction     -> concise_answer
refusal_or_unknown   -> calibrated_answering
overlong_mismatch    -> format_control
other mismatch       -> near_neighbor_reasoning
```

A source provider therefore knows the class of repair needed without seeing the protected example that revealed the weakness.

## Provider output

A provider returns:

- provider name/version;
- declared independent sources;
- sourced repair examples.

Chowder validates the request ID, provider identity, unique sources/examples, and example→source references before accepting the proposal.

The proposal is then passed through the existing provenance and contamination pipeline. A provider cannot bypass the holdout guard merely because it did not receive the holdout directly: if it independently proposes an overlapping holdout prompt, materialization rejects the repair data.

## End-to-end boundary

```text
holdout evaluation
      |
      v
FailureRecord (private diagnostic evidence)
      |
      v
FailureCluster + RepairPlan
      |
      | strip raw rows + free-form prose
      v
RepairRequest
      |
      v
RepairSourceProvider
      |
      v
source proposal
      |
      +--> source provenance verification
      |
      +--> holdout contamination audit
      |
      v
verified repair dataset
      |
      v
repair candidate population
```

This makes the default autonomous path capability-directed rather than benchmark-answer-directed.
