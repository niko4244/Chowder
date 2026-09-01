# Cumulative adapter repair

A repair child is only a genuine repair of a rejected PEFT candidate if it starts from the exact adapter weights that were evaluated. Inheriting configuration alone is insufficient: reloading the base model and creating a fresh LoRA discards the rejected candidate's learned state.

Chowder therefore binds autonomous repairs to the rejected candidate's adapter artifact.

## Weight lineage

```text
base model
    |
    v
candidate adapter A
    |
    +--> independent evaluation --> rejected
    |
    +--> SHA-256(A) verified
             |
             v
       trainable PEFT load
       PeftModel.from_pretrained(
           base_model,
           adapter_A,
           is_trainable=True,
       )
             |
             + repair data
             + replay data
             v
       candidate adapter B
```

`adapter B` therefore contains the updated state of `adapter A`, rather than a new LoRA trained from the untouched base model.

## Provenance boundary

The training artifact for a rejected candidate already records `artifact_sha256`. Autonomous repair now:

1. resolves the candidate's `artifact_ref`,
2. requires it to be an adapter directory,
3. recomputes the deterministic directory SHA-256,
4. refuses repair if the bytes differ from the evaluated artifact,
5. embeds the verified parent adapter path/hash in every repair candidate,
6. includes the parent adapter SHA-256 in experiment identity and repair evidence.

The Transformers controller re-verifies the parent adapter immediately before launching the isolated worker and again after the worker exits. The worker independently verifies it before loading model dependencies and after training. Mutation at any boundary invalidates the run.

## LoRA topology freeze during continuation

A trainable parent adapter has an existing PEFT topology: rank, target modules, scaling configuration, and related adapter structure. Chowder does not silently reinterpret those weights under a different topology.

When a repair candidate has a parent adapter, `lora_patch` is rejected. Training hyperparameters such as learning rate or epochs may still vary.

Topology-changing experiments require a separate explicit mechanism such as adapter migration, distillation, merge/reinitialization, or another transformation whose provenance and evaluation semantics can be audited independently.

## Interaction with rehearsal

Cumulative repair and rehearsal solve different failure modes:

- **parent adapter continuation** preserves the learned weight state being repaired;
- **parent-data rehearsal** reduces forgetting of capabilities represented in earlier training data;
- **independent repair data** supplies new supervision for the diagnosed failure cluster.

All three identities are bound into the repair experiment before training.

## Recursion prerequisite

This mechanism is intentionally landed before recursive autonomous repair. Without weight continuation, a second repair hop would merely create another fresh adapter from the base model, making graph depth misleading.

Even with weight continuation, recursive repair still requires explicit depth, repeated-failure, no-progress, and budget stopping policies. It also requires careful replay-history handling so deeper repairs do not forget data rehearsed in earlier hops.
