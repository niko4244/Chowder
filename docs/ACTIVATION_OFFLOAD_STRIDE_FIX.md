# activation_offload: expanded attention-bias stride corruption

Found during the Memory Fabric OOM-acceptance hardware investigation: with
`backend.training.activation_offload: "always"` on a real `Qwen/Qwen2.5-1.5B`
model, `batch_size=96`, `max_length=1024`, fp32, no quantization, LoRA r=8,
backward crashed with:

```
RuntimeError: attn_bias is not correctly aligned (strideM). attn_bias.stride(2) = 66,
and should be a multiple of 4.
```

originating inside `torch.nn.functional.silu` -> the SDPA/memory-efficient
-attention backward path, via `Variable._execution_engine.run_backward`.
Reproduced identically alone and combined with `optimizer_tiering` +
`frozen_layer_streaming` -- a real bug in `activation_offload` itself, not a
multi-mechanism interaction. Never caught by this project's existing tests
because they all use batch sizes of 2-8, which never selects the
memory-efficient/fused attention kernel that enforces this alignment
requirement.

## Root cause

`transformers.masking_utils.sdpa_mask` builds the 4D causal+padding
attention bias via `attention_mask.expand(batch_size, -1, q_length,
kv_length)` -- a broadcast view (stride 0 on the head dimension), never
materialized into real memory.

`torch.autograd.graph.saved_tensors_hooks` applies globally to *every*
tensor autograd decides to save for backward during the wrapped
forward+backward, not just a model's own per-layer activations -- including
this internal, transformers-constructed attention bias. The old pack/unpack
hooks did a naive `tensor.to("cpu", non_blocking=True)` / `.to(device,
non_blocking=True)` round trip. `Tensor.to()` only preserves a handful of
recognized memory formats (plain contiguous, channels-last, or a pure
transpose) -- for an arbitrary broadcast/sliced view it silently
materializes a *new*, differently-strided dense tensor with the same shape
and values. That's numerically harmless, but PyTorch's memory-efficient SDPA
backward kernel validates the exact stride layout of the `attn_bias` it
receives and rejects one that doesn't match what it expects.

Confirmed directly (`torch.nn.functional.scaled_dot_product_attention` under
`saved_tensors_hooks`, no model involved) with the *exact* reported error,
byte-for-byte, at `batch=96, heads=12, kv_heads=2, seq=66` (a real
dynamically-padded batch length; 66 isn't a multiple of 4, unlike a fixed
round `max_length`, which often coincidentally is):

```python
>>> mask.stride()          # (4356, 0, 66, 1) -- stride 0 on the head dim
>>> # baseline (no offload hooks): backward succeeds
>>> # with the OLD naive offload_pack/offload_unpack: backward raises
RuntimeError: attn_bias is not correctly aligned (strideM). attn_bias.stride(2) = 66,
and should be a multiple of 4.
```

## Fix

`chowder/backends/activation_offload_hooks.py` (new, shared by
`activation_offload_worker.py` and `transformers_worker.py`, which
previously duplicated this hand-rolled hook logic):

- A contiguous tensor (the overwhelming common case -- most saved
  activations are already contiguous) keeps the existing zero-overhead
  `.to("cpu")` / `.to(device)` path unchanged.
- A non-contiguous tensor moves only its real underlying storage (not a
  fully materialized copy of the broadcast/expanded shape, which would also
  needlessly inflate both the CPU-resident and transfer-back size) and
  reconstructs the exact original view with `as_strided()` on unpack, so the
  kernel sees back precisely what it saved.

Preserving strides was practical here (`tensor.untyped_storage()` for a
`.expand()`-produced view is exactly the small pre-expand storage, not the
broadcast-inflated one), so this is the narrow, value- and
stride-transparent fix rather than a scale/config guard.

Verified on real hardware (RTX 5060 Ti, torch 2.11.0+cu128, transformers
5.16.1, peft 0.20.0, in an isolated venv matching this project's
`pyproject.toml` `[train]`/`[dev]` version constraints) that the fix:

- Eliminates the crash at the exact repro scale above.
- Produces `output`/`grad_q`/`grad_k`/`grad_v` identical (`torch.allclose`)
  to a non-offloaded baseline run.
- Does not regress the fast (mocked-subprocess) suite: 775 passed, 0
  failed.

The exact reported full-scale scenario (`Qwen2.5-1.5B`, `batch_size=96`,
`max_length=1024`, fp32) was also attempted directly against the real model
on this machine's 16 GB card; the *baseline* (no offload) forward+backward
completed successfully (confirming the model/setup), but fp32 without
tensor-core acceleration on a consumer GPU is slow enough (~15 minutes for
one baseline step) that a second, concurrent process on the same device
during that run caused a real (unrelated) CUDA OOM rather than a clean
comparison -- this is a hardware/scheduling artifact of this specific
single-consumer-GPU machine, not a finding about the fix. The isolated SDPA
-level repro above reproduces the *exact* reported error message and stride
value at the *exact* reported `batch_size=96`, so the mechanism is
confirmed at full scale independent of that contention.

## Regression coverage

`tests/test_activation_offload.py`:

- `test_real_activation_offload_hooks_preserve_stride_of_expanded_attention_bias`
  -- direct, real-hardware reproduction against `scaled_dot_product_attention`
  and the real, shipped hooks, at `batch=96` (matching the real crash) with a
  `kv_length` deliberately not a multiple of 4. Confirmed to fail with the
  exact original error message when pointed at the pre-fix implementation,
  and to pass (with matching output/gradients) against the fix.
- `test_real_tiny_llama_trains_with_activation_offload_always_and_a_padded_batch`
  -- full production-path confirmation
  (`TransformersPeftExecutor.run` -> `transformers_worker.train`) that a
  real, genuinely variable-length batch (forcing the dynamic-padding
  collator to build a real `attention_mask`) trains successfully end to end
  with `activation_offload: "always"`.

Both require `CHOWDER_REAL_ML_SMOKE=1` and a real CUDA device, matching this
project's existing real-hardware test convention.

## How to run it

```bash
pip install -e ".[train,dev]"
CHOWDER_REAL_ML_SMOKE=1 python -m pytest -q tests/test_activation_offload.py -v
```

## Result

**PASSED for real**, 2026-09-04, on a local RTX 5060 Ti (16 GB) in an
isolated venv (`torch 2.11.0+cu128`, `transformers 5.16.1`, `peft 0.20.0`,
`accelerate 1.6.0`, matching `pyproject.toml`'s `[train]` constraints):

```
tests/test_activation_offload.py::test_real_activation_offload_hooks_preserve_stride_of_expanded_attention_bias PASSED
tests/test_activation_offload.py::test_real_tiny_llama_trains_with_activation_offload_always_and_a_padded_batch PASSED
```

Fast suite: `775 passed, 62 skipped` (no regressions from the refactor).
