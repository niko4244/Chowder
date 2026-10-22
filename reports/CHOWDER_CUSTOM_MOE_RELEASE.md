# Custom Qwen MoE release qualification (not a release)

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 fe94342c1073eb39e69be01ba224c63b38057b95; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

**Release verdict: NOT READY / BLOCKED.** Do not publish or advertise a certified Gen-0 / Gen-1 custom model on this evidence.

## Candidate inventory (not physically verified)

- Reported REAP160 BF16 checkpoint: `G:\\llm-cold\\bf16\\Qwen3.6-REAP160`, 26 shards, 46,039,573,000 total bytes, configuration sha256 `15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9` (from PR #268 prereg; not independently rehashed).
- Architecture evidence: Qwen3.6 / `qwen3_5_moe`-style text configuration, 40 layers, 160 experts per layer, 8 routed experts/token, fused expert matrices; quantized int8 CPU experts for LoRA training. Exact tokenizer and full sharded checkpoint digests unknown.
- Prior RTX 5060 Ti seq-2048 data: 20 training steps ran; peak and throughput numbers are historical and apply to the specified revision/config, not a final artifact.

## Required before release

1. Immutable identity manifest including **every shard** SHA256, config and tokenizer digest, parent commit and actual pruning calibration hash.
2. Independently tested BF16/int8 load, full forward, real LoRA backward, checkpoint/resume, adapter reload, architecture/router correctness.
3. Full registered quality comparisons with exact raw item hashes, paired statistics, protected regressions, termination, random128 negative control and Q4 inference speed/fit on the declared hardware.
4. Present-day current-code placement passes, including 14.5 GiB peak, WDDM no-spill witness, 20-step >=1% loss reduction, gradients and router entropy.
5. Legal review for all training-data/model weights and any Fable-derived output, including N2. Evaluate separate restrictions on redistribution of pretrained weights and adapters.
6. Frozen REAP judge certification, cost settlement and Nik's approval (N5). Promotion is not authorized by completing this document.

## Reproduction entry points (not executed here)

```shell
python chowder_batch/moe_int8_gates.py --base-model <checkpoint> --arms int8 --steps 3 --seq-len 2048 --out <new-result.json>
python chowder_batch/moe_int8_witness_run.py --base <checkpoint> --calib <approved-calibration.jsonl> --steps 24 --out <witness.json>
python chowder_batch/moe_int8_gates.py --witness <witness.json>
```

Do not run expensive GPU commands without authorization; verify run-time args against the checked-out branch, capture exact stdout/stderr/exit code and hashes. No validated packaged GGUF, adapter or training checkpoint was exported in this audit.
