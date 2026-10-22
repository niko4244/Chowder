# Custom Qwen MoE release qualification (not a release)

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

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


## Local verification addendum — 2026-10-22 (supersedes earlier GitHub-only findings)

The approved Windows workspace became accessible. The original main clone is a dirty rescue worktree and was not modified or reset. Isolated detached worktrees were created for testing. The physical REAP160 checkpoint at G:\llm-cold\bf16\Qwen3.6-REAP160 exists: 26 shards, 1,026 index-matched BF16 tensor headers, 46,039,444,192 payload bytes and 46,039,573,000 shard-file bytes; no header offset/index defects. Locally hashed config SHA256 matches 15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9. Full weight bodies remain unhashed.

Existing frozen experiment artifacts were reconciled. REAP128 fails GSM8K Gate 1 (paired bound -0.037807 versus -0.03), triggering the registered REAP160 fallback. REAP160 historical Gate 0, GSM8K, HumanEval and negative-control records PASS. A separate speed-resolution record reports 3.8523x Q4 speed versus dense with zero experts offloaded; all 12 raw comparison input files matched their recorded SHA256 hashes. This supports the fallback but does not constitute a new immutable Gen-0 freeze or a fresh CUDA run.

The local 24-step int8 witness SHA256 20e466d5362b8492fb9b55c312a57486283e641d46844e41c3ae4a85799617d7 records gradient PASS (19/19 LoRA B leaves), smoke PASS (mean loss 1.421 -> 0.551), router-balance PASS (worst entropy change 1.8%), WDDM spill NOT MEASURED. PR #268 revision 108f9322 fixes both the Markdown |g| table parsing defect and a witness --check that wrote despite claiming read-only; the real witness was imported through the validated generated-table mechanism. 7/8 placement gates now register measured values, with only WDDM unmeasured, and campaign admission still correctly refuses. Current-retention VRAM remeasurement and the 2.5 GPU-hour budget remain open.

Fresh Windows Python 3.11.9 focused CPU tests: recovery 90 PASS; int8/preflight 54 PASS; witness/gate/preflight 75 PASS; targeted Ruff clean. An initial 30-failure probe with artificially hidden CUDA was traced to a TorchAO import-time device query and succeeded when rerun with CHOWDER_NO_GPU=1 but CUDA device visibility restored, without GPU training.

The Gen-1 mix file checks passed against its current template-validated, 2048-token manifest: Fable 767 rows SHA256 384b6f10ae579af10ce36cbbe5278716b681d3223beab7343742b51c1857024d, Fable-free control 647 rows SHA256 681b2aa957d042028db8fa8710be6538b23f6563e088b9f752410bf5da174c9e. License decision N2 remains open. The card had approximately 12 GiB occupied and 4 GiB free, so no expensive CUDA tests or promotions were authorized or run.
