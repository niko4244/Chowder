from __future__ import annotations

from typing import Any

# Shared by activation_offload_worker.py (the calibration/timing experiment)
# and transformers_worker.py (real training) -- both wrap a forward+backward
# pass in torch.autograd.graph.saved_tensors_hooks(offload_pack, offload_unpack)
# to move every tensor autograd saves for backward to CPU during forward and
# back to its original device when backward actually needs it.
#
# saved_tensors_hooks applies globally to *every* tensor autograd decides to
# save during the `with` block -- not just a model's own per-layer
# activations, but also small internal tensors PyTorch/transformers builds
# and also saves for backward, such as an expanded/broadcast 4D attention
# bias (transformers.masking_utils.sdpa_mask calls `.expand(batch_size, -1,
# q_length, kv_length)` on a per-position mask, which broadcasts across the
# head dimension via stride 0 rather than allocating real memory for it).
#
# A naive `tensor.to("cpu")` / `.to(device)` round trip on a non-contiguous
# tensor does not reproduce its original strides: `Tensor.to()` only
# preserves a handful of recognized memory formats (plain contiguous,
# channels-last, or a pure transpose) -- for an arbitrary broadcast/sliced
# view it silently materializes a *new*, differently-strided dense tensor
# with the same shape and values. That is harmless numerically, but PyTorch's
# memory-efficient/fused SDPA backward kernel validates the exact stride
# layout of a saved attn_bias it receives and rejects one that doesn't match
# what it expects (observed on real hardware: `RuntimeError: attn_bias is
# not correctly aligned (strideM). attn_bias.stride(2) = 66, and should be a
# multiple of 4` -- 66 being an ordinary dynamically-padded batch length,
# not anything unusual, and not a multiple of 4 the way a fixed max_length
# often coincidentally is). The bug never surfaces at the tiny, uniformly
# -padded batch sizes this project's existing tests use, only once a real
# batch produces a genuinely non-contiguous saved tensor.
#
# The fix: for a contiguous tensor (the overwhelming common case -- most
# saved activations are already contiguous), keep the existing zero-overhead
# path unchanged. For a non-contiguous tensor, move only its real underlying
# storage (not a fully materialized copy of the broadcast/expanded shape,
# which would also needlessly inflate both the CPU-resident and
# transfer-back size) and reconstruct the exact original view with
# `as_strided()` on unpack, so the kernel sees back precisely what it saved.
# Confirmed on real hardware (RTX 5060 Ti, torch 2.11.0+cu128, transformers
# 5.16.1) to eliminate the crash and produce identical loss/gradients to a
# non-offloaded run.


def offload_pack(tensor: Any) -> tuple[Any, int]:
    """Pack hook for saved_tensors_hooks. Returns (packed_value, bytes_moved)."""
    if not tensor.is_cuda:
        return tensor, 0
    if tensor.is_contiguous():
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        return (tensor.device, cpu_tensor), cpu_tensor.numel() * cpu_tensor.element_size()

    import torch

    storage = tensor.untyped_storage()
    n_elems = storage.nbytes() // tensor.element_size()
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(storage, 0, (n_elems,), (1,))
    flat_cpu = flat.to("cpu", non_blocking=True)
    packed = (tensor.device, flat_cpu, tensor.size(), tensor.stride(), tensor.storage_offset())
    return packed, flat_cpu.numel() * flat_cpu.element_size()


def offload_unpack(packed: Any) -> tuple[Any, int]:
    """Unpack hook for saved_tensors_hooks. Returns (tensor, bytes_moved)."""
    if not isinstance(packed, tuple):
        return packed, 0
    if len(packed) == 2:
        device, cpu_tensor = packed
        moved = cpu_tensor.to(device, non_blocking=True)
        return moved, cpu_tensor.numel() * cpu_tensor.element_size()
    device, flat_cpu, size, stride, storage_offset = packed
    flat_gpu = flat_cpu.to(device, non_blocking=True)
    restored = flat_gpu.as_strided(size, stride, storage_offset)
    return restored, flat_cpu.numel() * flat_cpu.element_size()
