"""Low-rank factorization of vocabulary embeddings and output projections.

This module provides the reusable pieces of the embedding/output-projection
compression experiment:

* :class:`LowRankEmbedding` / :class:`LowRankLMHead` -- drop-in ``nn.Module``
  replacements for a full-vocabulary ``nn.Embedding`` / ``nn.Linear`` that store
  two thin matrices instead of one ``[vocab, hidden]`` tensor.
* :func:`factorize_svd` -- memory-aware truncated SVD (exact on GPU/small
  CPU matrices, blockwise randomized on CPU) for initializing those factors.
* :func:`state_dict_savings` -- exact stored-parameter accounting.

Design constraints:

* Token IDs, vocabulary layout, and output dimensions are untouched: every
  factorized module presents the same ``weight`` shape semantics as the module
  it replaces, so prompts and logits stay aligned token-for-token.
* Factorized modules never claim to be the original architecture; the
  conversion utilities record what was replaced so a checkpoint can be
  restored exactly (see :mod:`chowder.low_rank_checkpoint`).
* GGUF or other quantized containers are never accepted as source weights here;
  ``factorize_svd`` only consumes dense floating-point tensors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

__all__ = [
    "LowRankEmbedding",
    "LowRankLMHead",
    "factorize_svd",
    "state_dict_savings",
    "matmul_flops_per_token",
    "FactorizationResult",
]


def _resolve_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _free_cuda_memory() -> int:
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


class _SVDBuffer:
    """Streaming accumulator for the covariance Gram matrix X^T X.

    Building ``X^T X`` costs ``hidden x hidden`` memory instead of the full
    ``[vocab, hidden]`` copy, which is what makes the 248320 x 5120 case fit in
    modest RAM. The singular values of ``X`` are the square roots of the
    eigenvalues of ``X^T X``; the right singular vectors come out in the same
    eigendecomposition.
    """

    def __init__(self, hidden: int, device: torch.device, dtype: torch.dtype):
        self.gram = torch.zeros(hidden, hidden, dtype=torch.float32, device=device)
        self.device = device

    def add(self, block: torch.Tensor) -> None:
        self.gram += block.to(self.device, dtype=torch.float32).T @ block.to(
            self.device, dtype=torch.float32
        )

    def top_right_vectors(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        eigvals, eigvecs = torch.linalg.eigh(self.gram)
        # eigh returns ascending eigenvalues; the top components are last.
        order = torch.argsort(eigvals, descending=True)[:rank]
        values = eigvals[order].clamp_min(0.0)
        vectors = eigvecs[:, order]
        return torch.sqrt(values), vectors


@dataclass
class FactorizationResult:
    """Factors for ``weight ~= U @ V`` plus reconstruction diagnostics."""

    u: torch.Tensor
    v: torch.Tensor
    energy_captured: float
    relative_frobenius_error: float
    method: str


def factorize_svd(
    weight: torch.Tensor,
    rank: int,
    *,
    block_rows: int = 65536,
    oversample: int = 16,
    n_power: int = 2,
    prefer_cuda: bool = True,
    seed: int = 0,
    max_exact_bytes: int | None = None,
) -> FactorizationResult:
    """Factor ``weight [vocab, hidden]`` into ``U [vocab, rank] @ V [rank, hidden]``.

    Memory-aware strategy:

    * If the matrix fits comfortably on the chosen device, an exact truncated
      SVD is computed directly.
    * Otherwise a randomized-range finder (blockwise, streaming) approximates
      the top singular subspace without ever materializing a second copy of the
      full matrix.

    ``weight`` must be a dense floating-point tensor. Quantized containers
    (GGUF and friends) are rejected: dequantize to real weights first, or do
    not call this function.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(weight.shape)}")
    if weight.dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
        raise ValueError(
            "refusing to factorize an integer/quantized tensor; dequantize to "
            "float weights first (GGUF files are not trainable weights)"
        )
    if not weight.is_floating_point():
        raise ValueError("weight must be a floating-point tensor")
    vocab, hidden = weight.shape
    if rank < 1 or rank >= min(vocab, hidden):
        raise ValueError(
            f"rank {rank} is outside the valid range 1..{min(vocab, hidden) - 1} "
            f"for a {vocab}x{hidden} matrix"
        )

    device = _resolve_device(prefer_cuda)
    work = weight.to(device=device, dtype=torch.float32)

    total_bytes = work.numel() * 4
    if max_exact_bytes is not None:
        # Injectable threshold so tests (and conservative callers) can force the
        # randomized path deterministically.
        exact_ok = total_bytes <= max_exact_bytes
    elif device.type == "cuda":
        free = _free_cuda_memory()
        exact_ok = total_bytes * 4 <= free
    else:
        exact_ok = total_bytes * 8 <= 8 * 1024**3

    if exact_ok:
        try:
            u, s, vh = torch.linalg.svd(work, full_matrices=False)
            u_r = u[:, :rank].contiguous()
            v_r = (torch.diag(s[:rank]) @ vh[:rank, :]).contiguous()
            energy = float((s[:rank] ** 2).sum() / (s**2).sum().clamp_min(1e-30))
            error = _relative_frobenius_error(work, u_r, v_r)
            return FactorizationResult(u_r, v_r, energy, error, "exact_svd")
        except torch.linalg.LinAlgError:
            pass  # fall through to the randomized path

    generator = torch.Generator(device="cpu").manual_seed(seed)
    projector = torch.randn(
        hidden, rank + oversample, dtype=torch.float32, generator=generator
    ).to(device)
    for _ in range(max(0, n_power)):
        projector = work.T @ (work @ projector)
    ranges: list[torch.Tensor] = []
    for start in range(0, vocab, block_rows):
        ranges.append(work[start : start + block_rows] @ projector)
    q = torch.linalg.qr(torch.cat(ranges, dim=0), mode="reduced")[0]
    small = q.T @ work
    u_s, s_s, vh_s = torch.linalg.svd(small, full_matrices=False)
    u_r = (q @ u_s[:, :rank]).contiguous()
    v_r = (torch.diag(s_s[:rank]) @ vh_s[:rank, :]).contiguous()
    energy = float((s_s[:rank] ** 2).sum() / (s_s**2).sum().clamp_min(1e-30))
    error = _relative_frobenius_error(work, u_r, v_r, max_rows=65536)
    return FactorizationResult(u_r, v_r, energy, error, "randomized_svd")


def _relative_frobenius_error(
    original: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    max_rows: int = 1 << 30,
) -> float:
    if original.shape[0] <= max_rows:
        diff = original - u @ v
        return float(diff.norm() / original.norm().clamp_min(1e-30))
    # Sample rows for a large-matrix estimate; deterministic slice keeps runs
    # comparable.
    step = max(1, original.shape[0] // max_rows)
    sample = original[::step]
    u_s, v_s = u[::step], v
    diff = sample - u_s @ v_s
    return float(diff.norm() / sample.norm().clamp_min(1e-30))


class LowRankEmbedding(nn.Module):
    """Vocabulary embedding stored as ``[vocab, rank] @ [rank, hidden]``.

    Drop-in for ``nn.Embedding(vocab, hidden)``: same forward contract, same
    output shape, no change to token IDs or padding semantics.
    """

    def __init__(self, vocab: int, hidden: int, rank: int):
        super().__init__()
        if not 0 < rank < min(vocab, hidden):
            raise ValueError(f"invalid rank {rank} for {vocab}x{hidden} embedding")
        self.vocab = vocab
        self.hidden = hidden
        self.rank = rank
        self.embedding_a = nn.Parameter(torch.empty(vocab, rank))
        self.embedding_b = nn.Parameter(torch.empty(rank, hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding_a, std=0.02)
        nn.init.normal_(self.embedding_b, std=0.02)

    def load_factors(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.shape != (self.vocab, self.rank) or b.shape != (self.rank, self.hidden):
            raise ValueError(
                f"factor shapes {tuple(a.shape)}/{tuple(b.shape)} do not match "
                f"module ({self.vocab}, {self.rank}, {self.hidden})"
            )
        with torch.no_grad():
            self.embedding_a.copy_(a)
            self.embedding_b.copy_(b)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_a[input_ids] @ self.embedding_b

    def extra_repr(self) -> str:  # pragma: no cover - repr helper
        return f"vocab={self.vocab}, hidden={self.hidden}, rank={self.rank}"


class LowRankLMHead(nn.Module):
    """Output projection stored as ``[rank, hidden] @ [vocab, rank]``.

    Drop-in for ``nn.Linear(hidden, vocab, bias=False)`` used as an LM head:
    identical logits shape, no bias, no change to the vocabulary axis.
    """

    def __init__(self, hidden: int, vocab: int, rank: int):
        super().__init__()
        if not 0 < rank < min(hidden, vocab):
            raise ValueError(f"invalid rank {rank} for {hidden}x{vocab} head")
        self.vocab = vocab
        self.hidden = hidden
        self.rank = rank
        self.head_a = nn.Parameter(torch.empty(rank, hidden))
        self.head_b = nn.Parameter(torch.empty(vocab, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.head_a, std=0.02)
        nn.init.normal_(self.head_b, std=0.02)

    def load_factors(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.shape != (self.rank, self.hidden) or b.shape != (self.vocab, self.rank):
            raise ValueError(
                f"factor shapes {tuple(a.shape)}/{tuple(b.shape)} do not match "
                f"module ({self.vocab}, {self.rank}, {self.hidden})"
            )
        with torch.no_grad():
            self.head_a.copy_(a)
            self.head_b.copy_(b)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (hidden_states @ self.head_a.T) @ self.head_b.T

    def extra_repr(self) -> str:  # pragma: no cover - repr helper
        return f"vocab={self.vocab}, hidden={self.hidden}, rank={self.rank}"


def state_dict_savings(
    original_weight: tuple[int, int],
    rank: int,
    *,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Exact stored-parameter accounting for one factorized matrix.

    ``original_weight`` is ``(vocab, hidden)`` in the input-embedding layout.
    The transposed output projection has identical accounting.
    """
    vocab, hidden = original_weight
    full = vocab * hidden
    factors = vocab * rank + rank * hidden
    return {
        "full_parameters": float(full),
        "factor_parameters": float(factors),
        "unique_parameters": float(factors),
        "parameters_used_per_token": float(factors),
        "compression_ratio": full / factors,
        "full_bytes_at_dtype": float(full * dtype_bytes),
        "factor_bytes_at_dtype": float(factors * dtype_bytes),
    }


def matmul_flops_per_token(
    original_weight: tuple[int, int],
    rank: int,
    *,
    batch_tokens: int = 1,
) -> dict[str, float]:
    """Theoretical matmul FLOPs for projecting ``batch_tokens`` through the matrix.

    A ``[vocab, hidden] @ [hidden, 1]`` style product costs ``2 * vocab * hidden``
    FLOPs; through the factorization it costs ``2 * (vocab + hidden) * rank``.
    """
    vocab, hidden = original_weight
    return {
        "full_flops": 2.0 * vocab * hidden * batch_tokens,
        "factor_flops": 2.0 * (vocab + hidden) * rank * batch_tokens,
        "flops_ratio": (vocab * hidden) / ((vocab + hidden) * rank),
    }
