# Custom Qwen MoE completion register

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Scope and lineage

The custom MoE in active engineering is Qwen3.6-35B-A3B REAP pruned to 128 routed experts, with 160 only if the frozen fallback criteria demand it. The separate Qwen3.8 Native Sparse Program retains its prohibition on distillation and its own checkpoint/ancestry; no evidence or adapters are transferred. Sources: `docs/REAP_MOE_LINEAGE.md`, `docs/REAP_QWEN36_PREREG.md`.

## Verified via committed code or PR evidence

- Main contains REAP observation/pruning, eval and serving programs, full REAP promotion judge specification, Gen-1 training-data provider, BF16 layer-bulk transfer code, and preservation of model/evidence lineage.
- #268 contains int8 fused expert loader and quantization, placement instrumentation, real REAP160 26-shard census, 20-step RTX 5060 Ti training history, throughput A/B, router/gradient/smoke witness collectors and Kaggle capacity/real-load attempts. Audit commit 24f2f886 adds early calibration/flag validation; full CI not yet reported complete.
- #268 reports 40 expert layers, top-8 routing, approximately 18.82 GiB int8 expert grid plus scales and 4.55 GiB BF16 non-expert parameters, but live full-file hashes cannot be verified here.

## Incomplete for finished model

1. Pre-registered choice among REAP128, permitted REAP160 fallback and dense/random control lacks fully inspected raw baseline measurements in this session.
2. The current int8 default-on retention config is not independently certified by the recorded sanctioned table; four required witness/operator gates remain unmeasured.
3. Real training throughput misses the existing 2.5 GPU-hour/1000-step budget by approximately 5.13 hours even before first-touch overhead.
4. Exact file hashes, tokenizer, complete model card, serving Q4 artifact, functional local loader reproduction and deployment tests are inaccessible/unverified.
5. Gen-0 trusted ancestor has not been proven frozen here. Gen-1 Fable/control experiments, protected benchmark results and signed promotion approval have not been established.

## Completion tests

- Hash all checkpoint shards/tokenizer/config; reinstantiate and compare architecture/parameters, routing, outputs, adapter save/reload and Q4 inference identity.
- Pass every frozen pruning and placement gate on same artifact/code/config/hardware; preserve exact result JSON and code SHA.
- Complete declared contamination audit for both Fable and Fable-free arms; resolve N2 or retain mandatory control and release restrictions.
- Execute authorized bounded Gen-1 under feasible preregistered compute, rescore raw benchmark items, use frozen judge, retain required human approval.

**Status: BLOCKED, not failed as an algorithm.** Insufficient approved compute/access and unpassed or unmeasured acceptance gates prevent a truthful final model release.
