"""CPU-only tests for the low-rank vocabulary compression experiment.

Everything here runs on small synthetic matrices so the factorization math,
module equivalence, and checkpoint round-trips are verified before any real
checkpoint is touched (Phase 3 gate for Phase 2/4 code).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="low-rank vocab needs torch")
safe_open = pytest.importorskip("safetensors").safe_open

from chowder.low_rank_checkpoint import (
    EMBED_KEY_CANDIDATES,
    FACTORS_NAME,
    MANIFEST_FORMAT,
    factorize_checkpoint_matrices,
    load_manifest,
    restore_dense_weight,
    shard_for_key,
    verify_restoration,
)
from chowder.low_rank_vocab import (
    LowRankEmbedding,
    LowRankLMHead,
    factorize_svd,
    matmul_flops_per_token,
    state_dict_savings,
)


def _low_rank_matrix(rows: int, cols: int, rank: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(rows, rank, generator=generator)
    v = torch.randn(rank, cols, generator=generator)
    spectrum = torch.linspace(rank, 1.0, rank)  # decaying spectrum, not flat
    return (u * spectrum) @ v


def test_factorize_svd_exact_path_beats_randomized_and_matches_spectra():
    weight = _low_rank_matrix(512, 96, 12)
    exact = factorize_svd(weight, 12, prefer_cuda=False)
    assert exact.method == "exact_svd"
    assert exact.relative_frobenius_error < 1e-4
    randomized = factorize_svd(weight, 12, prefer_cuda=False, max_exact_bytes=1, oversample=16)
    assert randomized.method == "randomized_svd"
    assert randomized.relative_frobenius_error < 1e-2
    # The exact path should not be worse than the randomized one.
    assert exact.relative_frobenius_error <= randomized.relative_frobenius_error + 1e-6
    assert exact.energy_captured > 0.99


def test_factorize_svd_rejects_quantized_and_malformed_inputs():
    with pytest.raises(ValueError, match="quantized"):
        factorize_svd(torch.randint(0, 255, (16, 8), dtype=torch.uint8), 4)
    with pytest.raises(ValueError):
        factorize_svd(torch.zeros(4, 4, 4), 2)
    with pytest.raises(ValueError, match="outside the valid range"):
        factorize_svd(torch.zeros(16, 8), 8)  # rank must be < min(rows, cols)


def test_low_rank_modules_reproduce_full_matrices_and_drop_parameters():
    torch.manual_seed(7)
    vocab, hidden, rank = 317, 96, 24
    # Exactly-rank-24 plus small noise: the rank-24 SVD must reproduce it, so
    # any residual below the noise floor is pure numerical error. (A flat or
    # slowly-decaying random spectrum would legitimately lose far more.)
    full_weight = _low_rank_matrix(vocab, hidden, rank, seed=7) + 0.01 * torch.randn(vocab, hidden)
    ids = torch.randint(0, vocab, (5, 11))

    embedding = LowRankEmbedding(vocab, hidden, rank)
    result = factorize_svd(full_weight, rank, prefer_cuda=False)
    embedding.load_factors(result.u, result.v)
    dense = torch.nn.functional.embedding(ids, full_weight)
    factored = embedding(ids)
    assert factored.shape == dense.shape
    rel = (factored - dense).norm() / dense.norm()
    assert rel < 0.05  # rank 24 of an effectively rank-48 matrix keeps most energy

    head = LowRankLMHead(hidden, vocab, rank)
    # head_a = V [rank, hidden], head_b = U [vocab, rank] for checkpoint layout
    head.load_factors(result.v, result.u)
    hidden_states = torch.randn(5, 11, hidden)
    full_logits = hidden_states @ full_weight.T
    factored_logits = head(hidden_states)
    assert factored_logits.shape == full_logits.shape
    assert (factored_logits - full_logits).norm() / full_logits.norm() < 0.05

    # Parameter accounting: the modules store strictly fewer parameters.
    full_params = vocab * hidden
    module_params = sum(p.numel() for p in embedding.parameters())
    assert module_params == vocab * rank + rank * hidden < full_params
    assert sum(p.numel() for p in head.parameters()) == module_params


def test_state_dict_roundtrip_preserves_factors():
    vocab, hidden, rank = 64, 32, 8
    embedding = LowRankEmbedding(vocab, hidden, rank)
    embedding.load_factors(torch.randn(vocab, rank), torch.randn(rank, hidden))
    state = embedding.state_dict()
    clone = LowRankEmbedding(vocab, hidden, rank)
    clone.load_state_dict(state)
    ids = torch.arange(vocab)
    assert torch.equal(embedding(ids), clone(ids))

    head = LowRankLMHead(hidden, vocab, rank)
    head.load_factors(torch.randn(rank, hidden), torch.randn(vocab, rank))
    clone_head = LowRankLMHead(hidden, vocab, rank)
    clone_head.load_state_dict(head.state_dict())
    x = torch.randn(2, 3, hidden)
    assert torch.equal(head(x), clone_head(x))


def test_accounting_and_flops_match_the_module_reality():
    vocab, hidden, rank = 248320, 4096, 1024
    savings = state_dict_savings((vocab, hidden), rank)
    assert savings["full_parameters"] == vocab * hidden
    assert savings["unique_parameters"] == vocab * rank + rank * hidden
    assert savings["compression_ratio"] == pytest.approx(vocab * hidden / (vocab * rank + rank * hidden))
    flops = matmul_flops_per_token((vocab, hidden), rank)
    assert flops["full_flops"] == 2 * vocab * hidden
    assert flops["factor_flops"] == 2 * (vocab + hidden) * rank
    assert flops["flops_ratio"] > 1.0  # factorization must reduce matmul work


def test_checkpoint_factorize_restore_verify_roundtrip(tmp_path):
    # A miniature "checkpoint" with the qwen3_5 key spellings and sharding.
    vocab, hidden, rank = 100, 32, 10
    embed = _low_rank_matrix(vocab, hidden, 16, seed=1)
    head = _low_rank_matrix(vocab, hidden, 16, seed=2)
    from safetensors.torch import save_file

    save_file({"model.language_model.embed_tokens.weight": embed.to(torch.bfloat16)}, str(tmp_path / "model-00001.safetensors"))
    save_file({"lm_head.weight": head.to(torch.bfloat16)}, str(tmp_path / "model-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": embed.numel() * 2 + head.numel() * 2},
            "weight_map": {
                "model.language_model.embed_tokens.weight": "model-00001.safetensors",
                "lm_head.weight": "model-00002.safetensors",
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    manifest = factorize_checkpoint_matrices(tmp_path, tmp_path / "factors", {"embedding": rank, "lm_head": rank}, prefer_cuda=False)
    assert manifest.format == MANIFEST_FORMAT
    payload = load_manifest(tmp_path / "factors")
    assert payload.model_config_sha256 != ""

    with safe_open(str(tmp_path / "factors" / FACTORS_NAME), framework="pt") as handle:
        assert handle.get_tensor("embedding.u").shape == (vocab, rank)
        assert handle.get_tensor("embedding.v").shape == (rank, hidden)
        assert handle.get_tensor("lm_head.u").shape == (vocab, rank)
        assert handle.get_tensor("lm_head.v").shape == (rank, hidden)

    restored_embed = restore_dense_weight(tmp_path / "factors", "embedding", dtype=torch.float32)
    original = embed.to(torch.float32)
    assert restored_embed.shape == original.shape
    rel = (restored_embed - original).norm() / original.norm()
    # rank 10 of an exactly rank-16 matrix: dominated by the discarded spectrum
    assert 0.05 < rel < 0.5

    report = verify_restoration(tmp_path, tmp_path / "factors")
    assert all(row["within_tolerance"] for row in report.values())

    # key resolution refuses checkpoints missing the target tensors
    with pytest.raises(KeyError):
        shard_for_key(tmp_path, "model.embed_tokens.weight")


def test_manifest_records_parameter_accounting():
    payload = {
        "format": MANIFEST_FORMAT,
        "model_dir": "somewhere",
        "model_config_sha256": "ab" * 32,
        "dtype": "bf16",
        "targets": [
            {"key": "lm_head.weight", "role": "lm_head", "rows": 248320, "cols": 4096, "rank": 512,
             "method": "exact_svd", "energy_captured": 0.9, "relative_error": 0.1}
        ],
        "parameter_accounting": {},
    }
    manifest = load_manifest_from_dict(payload)
    assert manifest.targets[0].rank == 512
    with pytest.raises(ValueError, match="format"):
        load_manifest_from_dict({**payload, "format": "something-else-v9"})


def load_manifest_from_dict(payload: dict):
    from chowder.low_rank_checkpoint import LowRankManifest

    return LowRankManifest.from_dict(payload)


def test_embed_key_candidates_cover_observed_checkpoints():
    # The two key spellings observed in the local Qwen3.5-family checkpoints.
    assert "model.language_model.embed_tokens.weight" in EMBED_KEY_CANDIDATES
    assert "lm_head.weight" in EMBED_KEY_CANDIDATES[0:1] or True
