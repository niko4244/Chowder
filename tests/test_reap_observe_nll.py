"""CPU-only equivalence tests for the full-census REAP observer's D0 loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from chowder_batch import reap_observe


class LocalRMSNorm(torch.nn.Module):
    """Match Qwen's per-token FP32 variance and output-dtype restoration."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.8, 1.2, width))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        original_dtype = states.dtype
        as_float = states.float()
        variance = as_float.square().mean(-1, keepdim=True)
        return ((as_float * torch.rsqrt(variance + 1e-6)).to(original_dtype)
                * self.weight.to(original_dtype))


def _reference(
    norm: torch.nn.Module, hidden: torch.Tensor, tokens: torch.Tensor, head: torch.Tensor,
) -> tuple[float, int]:
    """Original, unchunked lines 180-181 of reap_observe.py."""
    if tokens.numel() <= 1:
        return 0.0, 0
    with torch.inference_mode():
        logits = F.linear(norm(hidden)[0, :-1], head).float()
        loss = F.cross_entropy(logits, tokens[1:], reduction="sum").item()
    return loss, len(tokens) - 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("length", "chunk"), [
    (2, 1), (3, 2), (9, 4), (17, 16), (65, 8), (129, 64),
])
def test_chunked_matches_frozen_full_vocab_nll_and_perplexity(dtype, length, chunk):
    rng = torch.Generator().manual_seed(20261023 + length)
    width, vocab = 48, 1031
    hidden = torch.randn(1, length, width, generator=rng).to(dtype)
    head = (torch.randn(vocab, width, generator=rng) * 0.25).to(dtype)
    tokens = torch.randint(vocab, (length,), generator=rng)
    norm = LocalRMSNorm(width)
    expected, positions = _reference(norm, hidden, tokens, head)
    actual, counted = reap_observe._next_token_nll_chunked(
        norm, hidden, tokens, head, chunk_positions=chunk,
    )
    assert counted == positions == length - 1
    # FP32 CE's sum reduction is allowed to regroup across position chunks;
    # bfloat16 projection and the original final FP32 CE are unchanged.
    assert actual == pytest.approx(expected, rel=1e-6, abs=2e-3)
    assert math.exp(actual / counted) == pytest.approx(
        math.exp(expected / positions), rel=1e-6,
    )


def test_chunk_binds_logits_to_at_most_position_budget(monkeypatch):
    rng = torch.Generator().manual_seed(11)
    length, width, vocab, chunk = 142, 32, 8193, 16
    head = torch.randn(vocab, width, generator=rng)
    hidden = torch.randn(1, length, width, generator=rng)
    tokens = torch.randint(vocab, (length,), generator=rng)
    norm = LocalRMSNorm(width)
    expected, _ = _reference(norm, hidden, tokens, head)

    original_linear = reap_observe.F.linear
    positions_seen = []

    def inspect_linear(states, weight, *args, **kwargs):
        if weight is head:
            positions_seen.append(states.shape[0])
            assert states.shape == (min(states.shape[0], chunk), width)
        return original_linear(states, weight, *args, **kwargs)

    monkeypatch.setattr(reap_observe.F, "linear", inspect_linear)
    actual, count = reap_observe._next_token_nll_chunked(
        norm, hidden, tokens, head, chunk_positions=chunk,
    )
    assert actual == pytest.approx(expected, rel=1e-6)
    assert count == length - 1
    assert len(positions_seen) == math.ceil((length - 1) / chunk)
    assert max(positions_seen) == chunk
    assert positions_seen[-1] == (length - 1) % chunk


def test_full_vocabulary_projection_keeps_large_logit_stability():
    width, vocab, length = 8, 37, 11
    norm = LocalRMSNorm(width)
    hidden = torch.full((1, length, width), 10.0)
    head = torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width) / 2
    tokens = torch.arange(length) % vocab
    expected, _ = _reference(norm, hidden, tokens, head)
    actual, count = reap_observe._next_token_nll_chunked(
        norm, hidden, tokens, head, chunk_positions=3,
    )
    assert count == length - 1
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-6, abs=2e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_installed_qwen_rmsnorm_preserves_original_nll(dtype):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

    rng = torch.Generator().manual_seed(315)
    width, vocab, length = 48, 1297, 73
    hidden = torch.randn(1, length, width, generator=rng).to(dtype)
    head = (torch.randn(vocab, width, generator=rng) * 0.4).to(dtype)
    tokens = torch.randint(vocab, (length,), generator=rng)
    norm = Qwen3_5MoeRMSNorm(width, eps=1e-6)
    expected, count = _reference(norm, hidden, tokens, head)
    actual, got_count = reap_observe._next_token_nll_chunked(
        norm, hidden, tokens, head, chunk_positions=11,
    )
    assert got_count == count
    assert actual == pytest.approx(expected, rel=1e-6, abs=2e-3)


def test_nll_counts_and_per_source_perplexity_match_frozen_definition():
    rng = torch.Generator().manual_seed(744)
    vocab, width = 251, 16
    head = torch.randn(vocab, width, generator=rng)
    norm = LocalRMSNorm(width)
    totals = {"math": [0.0, 0], "code": [0.0, 0]}
    reference = {"math": [0.0, 0], "code": [0.0, 0]}
    for source, length in [("math", 15), ("code", 7), ("math", 11)]:
        hidden = torch.randn(1, length, width, generator=rng)
        targets = torch.randint(vocab, (length,), generator=rng)
        loss, count = reap_observe._next_token_nll_chunked(
            norm, hidden, targets, head, chunk_positions=4,
        )
        old_loss, old_count = _reference(norm, hidden, targets, head)
        totals[source][0] += loss
        totals[source][1] += count
        reference[source][0] += old_loss
        reference[source][1] += old_count
    for source in ("math", "code"):
        assert totals[source][1] == reference[source][1]
        assert totals[source][0] == pytest.approx(reference[source][0], rel=1e-6)
        assert math.exp(totals[source][0] / totals[source][1]) == pytest.approx(
            math.exp(reference[source][0] / reference[source][1]), rel=1e-6,
        )


@pytest.mark.parametrize("length", [0, 1])
def test_empty_and_one_token_sequences_have_no_next_token_loss(length):
    head = torch.randn(13, 4)
    hidden = torch.empty(1, length, 4)
    tokens = torch.arange(length)
    assert reap_observe._next_token_nll_chunked(
        LocalRMSNorm(4), hidden, tokens, head,
    ) == (0.0, 0)


@pytest.mark.parametrize("chunk", [0, -1, True, 2.0])
def test_invalid_chunk_budget_refused(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        reap_observe._next_token_nll_chunked(
            LocalRMSNorm(4), torch.randn(1, 5, 4), torch.arange(5),
            torch.randn(13, 4), chunk_positions=chunk,
        )


@pytest.mark.parametrize(("shape", "tokens"), [
    ((2, 5, 4), 5), ((1, 4, 4), 5), ((5, 4), 5),
])
def test_unaligned_hidden_or_token_sequence_refused(shape, tokens):
    with pytest.raises(ValueError, match="aligned token ids"):
        reap_observe._next_token_nll_chunked(
            LocalRMSNorm(4), torch.randn(*shape), torch.arange(tokens),
            torch.randn(13, 4),
        )


def test_invalid_next_token_target_is_not_silently_ignored():
    with pytest.raises((IndexError, RuntimeError)):
        reap_observe._next_token_nll_chunked(
            LocalRMSNorm(4), torch.randn(1, 4, 4),
            torch.tensor([0, 2, 13, 1]), torch.randn(13, 4),
            chunk_positions=2,
        )
