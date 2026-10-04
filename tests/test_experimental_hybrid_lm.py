"""CPU correctness and accounting tests for Chowder Experiment D."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from chowder.experimental_hybrid_lm import (
    ExperimentDError,
    HybridCache,
    HybridLanguageModel,
    HybridLMConfig,
    SparseMoEFFN,
    causal_lm_loss,
    estimate_flops,
    load_experiment_d_configs,
    parameter_report,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "experiment_d" / "configs"


def _small_config(**overrides) -> HybridLMConfig:
    values = {
        "vocab_size": 23,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "layer_types": ("mamba2", "attention"),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "mamba_num_heads": 4,
        "mamba_expand": 2,
        "mamba_state_size": 3,
        "mamba_conv_kernel": 3,
        "ffn_type": "dense",
        "intermediate_size": 24,
        "num_experts": 3,
        "experts_per_token": 2,
        "expert_intermediate_size": 8,
        "capacity_factor": None,
        "per_layer_embedding_dim": 0,
        "tie_word_embeddings": True,
        "max_position_embeddings": 32,
    }
    values.update(overrides)
    return HybridLMConfig(**values)


def test_small_configuration_validation_rejects_incompatible_dimensions():
    with pytest.raises(ExperimentDError, match="layer_types length"):
        HybridLMConfig(num_hidden_layers=3, layer_types=("attention", "mamba2"))
    with pytest.raises(ExperimentDError, match="divisible by num_attention_heads"):
        _small_config(hidden_size=15)
    with pytest.raises(ExperimentDError, match="divisible by mamba_num_heads"):
        _small_config(hidden_size=18, num_attention_heads=3, num_key_value_heads=1, mamba_num_heads=7)
    with pytest.raises(ExperimentDError, match="cannot exceed num_experts"):
        _small_config(experts_per_token=4)
    with pytest.raises(ExperimentDError, match="cannot tie"):
        _small_config(output_projection="low_rank", output_rank=4)


def test_causal_attention_is_invariant_to_future_tokens():
    torch.manual_seed(9)
    config = _small_config(layer_types=("attention", "attention"))
    model = HybridLanguageModel(config).eval()
    first = torch.tensor([[2, 4, 6, 8]])
    changed_future = torch.tensor([[2, 4, 15, 19]])
    with torch.inference_mode():
        output_first = model(first).logits
        output_changed = model(changed_future).logits
    assert torch.allclose(output_first[:, :2], output_changed[:, :2], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output_first[:, 2:], output_changed[:, 2:])


def test_mamba_and_attention_full_sequence_match_incremental_decode():
    torch.manual_seed(17)
    model = HybridLanguageModel(_small_config()).eval()
    token_ids = torch.tensor([[1, 4, 7, 3, 9, 2]])
    with torch.inference_mode():
        full = model(token_ids).logits
        cache = None
        incremental = []
        for position in range(token_ids.shape[1]):
            output = model(
                token_ids[:, position : position + 1],
                cache=cache,
                use_cache=True,
            )
            cache = output.cache
            incremental.append(output.logits)

        chunk_cache = None
        chunked = []
        for start, end in ((0, 2), (2, 5), (5, 6)):
            output = model(
                token_ids[:, start:end],
                cache=chunk_cache,
                use_cache=True,
            )
            chunk_cache = output.cache
            chunked.append(output.logits)
    joined = torch.cat(incremental, dim=1)
    chunked_joined = torch.cat(chunked, dim=1)
    assert cache is not None and cache.sequence_length == token_ids.shape[1]
    assert chunk_cache is not None and chunk_cache.sequence_length == token_ids.shape[1]
    assert torch.allclose(full, joined, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full, chunked_joined, atol=1e-5, rtol=1e-5)


def test_cache_is_bound_to_config_and_batch_size():
    model = HybridLanguageModel(_small_config()).eval()
    other = HybridLanguageModel(_small_config(hidden_size=24, num_attention_heads=4)).eval()
    with torch.inference_mode():
        cache = model(torch.tensor([[1]]), use_cache=True).cache
        assert cache is not None
        with pytest.raises(ExperimentDError, match="different model configuration"):
            other(torch.tensor([[2]]), cache=cache, use_cache=True)
        with pytest.raises(ExperimentDError, match="batch size"):
            model(torch.tensor([[2], [3]]), cache=cache, use_cache=True)


def test_cache_rejects_corrupted_state_and_out_of_vocabulary_labels():
    model = HybridLanguageModel(_small_config()).eval()
    tokens = torch.tensor([[1, 2]])
    with pytest.raises(ExperimentDError, match="label token id"):
        model(tokens, labels=torch.tensor([[1, 23]]))
    with pytest.raises(ExperimentDError, match="label token id"):
        causal_lm_loss(torch.zeros(1, 2, 23), torch.tensor([[1, 23]]))

    with torch.inference_mode():
        cache = model(tokens[:, :1], use_cache=True).cache
    assert cache is not None
    cache.attention_keys[1] = cache.attention_keys[1][:, :, :-1, :]
    with pytest.raises(ExperimentDError, match="cached attention key/value"):
        model(tokens[:, 1:], cache=cache, use_cache=True)


def test_moe_eval_streaming_is_independent_of_chunk_capacity():
    torch.manual_seed(41)
    config = _small_config(
        ffn_type="moe",
        experts_per_token=1,
        num_experts=3,
        capacity_factor=0.25,
        min_expert_capacity=1,
    )
    model = HybridLanguageModel(config).eval()
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.inference_mode():
        full = model(token_ids)
        cache = None
        pieces = []
        for start, end in ((0, 2), (2, 3), (3, 6)):
            piece = model(token_ids[:, start:end], cache=cache, use_cache=True)
            cache = piece.cache
            pieces.append(piece.logits)
    chunked = torch.cat(pieces, dim=1)
    assert cache is not None
    assert torch.allclose(full.logits, chunked, atol=1e-5, rtol=1e-5)
    assert all(
        stats.capacity_per_expert is None
        for stats in full.routing_stats
        if stats is not None
    )
    assert all(
        stats.dropped_assignments == 0
        for stats in full.routing_stats
        if stats is not None
    )


def test_shared_expert_is_always_on_and_included_in_active_accounting():
    torch.manual_seed(31)
    no_shared = HybridLanguageModel(
        _small_config(ffn_type="moe", experts_per_token=1, capacity_factor=None)
    ).eval()
    with_shared = HybridLanguageModel(
        _small_config(
            ffn_type="moe",
            experts_per_token=1,
            capacity_factor=None,
            num_shared_experts=1,
        )
    ).eval()
    shared = with_shared.layers[0].ffn.shared_experts[0]
    calls = 0

    def record_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = shared.register_forward_hook(record_call)
    with torch.inference_mode():
        output = with_shared(torch.tensor([[1, 2, 3]])).logits
    handle.remove()

    base_report = no_shared.parameter_report()
    shared_report = with_shared.parameter_report()
    shared_parameters = sum(
        parameter.numel()
        for layer in with_shared.layers
        for expert in layer.ffn.shared_experts
        for parameter in expert.parameters()
    )
    assert output.shape == (1, 3, with_shared.config.vocab_size)
    assert calls == 1
    assert shared_report.stored_parameters == base_report.stored_parameters + shared_parameters
    assert shared_report.active_parameters_per_token == base_report.active_parameters_per_token + shared_parameters


def test_moe_only_calls_selected_experts_and_reports_capacity():
    torch.manual_seed(5)
    moe = SparseMoEFFN(
        _small_config(
            hidden_size=4,
            num_hidden_layers=1,
            layer_types=("attention",),
            num_attention_heads=2,
            num_key_value_heads=1,
            mamba_num_heads=2,
            mamba_expand=1,
            mamba_state_size=2,
            ffn_type="moe",
            num_experts=3,
            experts_per_token=1,
            expert_intermediate_size=4,
            capacity_factor=None,
        )
    )
    counts = [0, 0, 0]
    handles = [
        expert.register_forward_hook(
            lambda _module, _args, _output, index=index: counts.__setitem__(index, counts[index] + 1)
        )
        for index, expert in enumerate(moe.experts)
    ]
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0.5, 0, 0, 0], [-1.0, 0, 0, 0]]))
    hidden = torch.tensor([[[2.0, 0, 0, 0], [1.0, 0, 0, 0], [3.0, 0, 0, 0]]])
    output, aux, stats = moe(hidden)
    for handle in handles:
        handle.remove()
    assert output.shape == hidden.shape
    assert torch.isfinite(aux)
    assert stats.active_experts == 1
    assert stats.selected_per_expert == (3, 0, 0)
    assert sum(count > 0 for count in counts) == stats.active_experts
    assert counts == [1, 0, 0]

    limited = SparseMoEFFN(
        _small_config(
            hidden_size=4,
            num_hidden_layers=1,
            layer_types=("attention",),
            num_attention_heads=2,
            num_key_value_heads=1,
            mamba_num_heads=2,
            mamba_expand=1,
            mamba_state_size=2,
            ffn_type="moe",
            num_experts=3,
            experts_per_token=2,
            expert_intermediate_size=4,
            capacity_factor=0.5,
            min_expert_capacity=1,
        )
    )
    _, _, limited_stats = limited(torch.randn(1, 6, 4))
    assert limited_stats.capacity_per_expert == 2
    assert limited_stats.assignments == 12
    assert limited_stats.accepted_assignments <= 6
    assert limited_stats.dropped_assignments == 12 - limited_stats.accepted_assignments


def test_capacity_overflow_can_fail_closed():
    moe = SparseMoEFFN(
        _small_config(
            hidden_size=4,
            num_hidden_layers=1,
            layer_types=("attention",),
            num_attention_heads=2,
            num_key_value_heads=1,
            mamba_num_heads=2,
            mamba_expand=1,
            mamba_state_size=2,
            ffn_type="moe",
            num_experts=2,
            experts_per_token=2,
            expert_intermediate_size=4,
            capacity_factor=0.1,
            min_expert_capacity=1,
            overflow_policy="error",
        )
    )
    with pytest.raises(ExperimentDError, match="capacity overflow"):
        moe(torch.randn(1, 4, 4))


def test_router_balance_auxiliary_loss_backpropagates_through_model():
    torch.manual_seed(12)
    config = _small_config(ffn_type="moe", experts_per_token=1, capacity_factor=None)
    model = HybridLanguageModel(config).train()
    tokens = torch.randint(config.vocab_size, (2, 5))
    output = model(tokens, labels=tokens)
    (output.loss + 0.05 * output.auxiliary_loss).backward()
    router = model.layers[0].ffn.router
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    assert torch.count_nonzero(router.weight.grad).item() > 0
    assert any(parameter.grad is not None for parameter in model.layers[0].ffn.experts[0].parameters())


def test_per_layer_embedding_capacity_and_lookup_aware_accounting():
    config = _small_config(per_layer_embedding_dim=3)
    model = HybridLanguageModel(config)
    report = parameter_report(model)
    expected_tables = config.vocab_size * config.num_hidden_layers * config.per_layer_embedding_dim
    assert report.per_layer_embedding_table_parameters == expected_tables
    assert report.per_layer_embedding_table_bytes == expected_tables * 4
    assert report.stored_parameters >= expected_tables
    assert report.lookup_values_per_token == config.hidden_size + config.num_hidden_layers * 3
    assert report.lookup_aware_effective_parameters_per_token == (
        report.active_matrix_parameters_per_token + report.lookup_values_per_token - config.hidden_size
    )
    token_ids = torch.tensor([[1, 4, 7]])
    first = model.per_layer_embeddings(token_ids, 0)
    second = model.per_layer_embeddings(token_ids, 1)
    assert first.shape == second.shape == (1, 3, config.hidden_size)
    assert not torch.equal(first, second)


def test_parameter_accounting_matches_actual_parameter_storage_and_ties_once():
    config = _small_config()
    model = HybridLanguageModel(config)
    report = parameter_report(model)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert report.stored_parameters == actual
    assert report.tied_word_embeddings
    assert model.lm_head.weight is model.token_embedding.weight
    assert report.active_parameters_per_token == report.stored_parameters


def test_moe_active_and_capacity_parameters_follow_top_k_definition():
    config = _small_config(ffn_type="moe", experts_per_token=1, capacity_factor=None)
    model = HybridLanguageModel(config)
    report = parameter_report(model)
    assert report.routed_expert_active_parameters * config.num_experts == (
        report.routed_expert_capacity_parameters * config.experts_per_token
    )
    assert report.active_parameters_per_token == (
        report.stored_parameters - report.inactive_routed_expert_parameters
    )
    assert report.inactive_routed_expert_parameters > 0


def test_flop_estimate_grows_with_attention_context_but_not_mamba_context():
    attention_model = HybridLanguageModel(_small_config(layer_types=("attention", "attention")))
    mamba_model = HybridLanguageModel(_small_config(layer_types=("mamba2", "mamba2")))
    short_attention = estimate_flops(attention_model, context_length=1)
    long_attention = estimate_flops(attention_model, context_length=32)
    short_mamba = estimate_flops(mamba_model, context_length=1)
    long_mamba = estimate_flops(mamba_model, context_length=32)
    assert long_attention.attention_context_flops_per_token > short_attention.attention_context_flops_per_token
    assert long_mamba.attention_context_flops_per_token == 0
    assert short_mamba.estimated_flops_per_token == long_mamba.estimated_flops_per_token
    assert "analytical" in long_attention.assumptions[0]


def test_causal_lm_loss_shifts_labels_and_is_finite():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[0, 1, 0]])
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.equal(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))


def test_model_save_reload_is_equivalent_and_never_overwrites(tmp_path):
    torch.manual_seed(99)
    model = HybridLanguageModel(_small_config(per_layer_embedding_dim=2)).eval()
    tokens = torch.tensor([[1, 3, 5]])
    expected = model(tokens).logits
    path = tmp_path / "model"
    model.save_pretrained(path)
    restored = HybridLanguageModel.from_pretrained(path).eval()
    actual = restored(tokens).logits
    assert torch.equal(expected, actual)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        model.save_pretrained(path)
    assert (path / "config.json").is_file()
    assert (path / "model.pt").is_file()


def test_json_configs_validate_and_match_expected_ablations():
    configs = load_experiment_d_configs(CONFIG_DIR)
    assert set(configs) == {
        "A_hybrid_dense",
        "B_hybrid_sparse_moe",
        "C_hybrid_moe_ple",
        "D_narrow_hybrid_moe_ple",
        "E_hybrid_moe_ple_low_rank_head",
    }
    a = configs["A_hybrid_dense"]
    b = configs["B_hybrid_sparse_moe"]
    c = configs["C_hybrid_moe_ple"]
    d = configs["D_narrow_hybrid_moe_ple"]
    e = configs["E_hybrid_moe_ple_low_rank_head"]
    assert a.ffn_type == "dense" and b.ffn_type == "moe"
    a_values = a.to_dict()
    a_values["ffn_type"] = b.ffn_type
    assert a_values == b.to_dict()
    assert b.per_layer_embedding_dim == 0 and c.per_layer_embedding_dim == 4
    b_values = b.to_dict()
    b_values["per_layer_embedding_dim"] = c.per_layer_embedding_dim
    assert b_values == c.to_dict()
    assert b.num_shared_experts == c.num_shared_experts == 0
    assert c.hidden_size == 32 and d.hidden_size == 24
    assert d.intermediate_size == c.intermediate_size
    assert d.expert_intermediate_size * 32 == c.expert_intermediate_size * 24
    assert d.per_layer_embedding_dim == c.per_layer_embedding_dim
    assert e.output_projection == "low_rank" and e.output_rank > 0
    e_values = e.to_dict()
    for name in ("output_projection", "output_rank", "tie_word_embeddings"):
        e_values[name] = c.to_dict()[name]
    assert e_values == c.to_dict()


def test_tied_checkpoint_roundtrip_preserves_unique_parameter_count(tmp_path):
    model = HybridLanguageModel(_small_config())
    before = parameter_report(model)
    path = tmp_path / "weights"
    model.save_pretrained(path)
    restored = HybridLanguageModel.from_pretrained(path)
    after = parameter_report(restored)
    assert before.stored_parameters == after.stored_parameters
    assert restored.lm_head.weight is restored.token_embedding.weight
