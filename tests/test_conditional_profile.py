"""CPU checks for the opt-in layer profiler and its evidence boundaries."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from chowder.conditional_profile import (
    ConditionalProfileError,
    profile_model_call,
    summarize_generation_profiles,
    write_profile_artifact,
)


class _ProfileAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


class _ProfileMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.relu(self.up_proj(hidden_states)))


class _ProfileLayer(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.self_attn = _ProfileAttention(hidden_size)
        self.mlp = _ProfileMLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states + self.self_attn(hidden_states)
        return residual + self.mlp(self.input_layernorm(residual))


class _ProfileModel(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_ProfileLayer(), _ProfileLayer()])
        self.model = inner
        self.config = type("Config", (), {"model_type": "cpu_test"})()

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_profile_counts_observed_projection_flops_by_layer_and_reports_limits():
    model = _ProfileModel()
    model.train()
    hidden = torch.randn(2, 5, 4)
    profile = profile_model_call(
        model,
        lambda: model(hidden),
        phase="prefill",
        warmup=1,
        iterations=2,
        use_torch_profiler=True,
        profile_cuda=False,
    )

    assert profile["model"]["decoder_layer_path"] == "model.layers"
    assert profile["model"]["decoder_layer_count"] == 2
    assert profile["model"]["layer_inventory"][0]["conditional_ffn_candidate"]
    assert profile["request"]["measured_iterations"] == 2
    assert profile["latency"]["max_ms"] >= profile["latency"]["min_ms"]
    # Per measured call, each layer costs 2*10 vectors* (4*4 + 4*8 + 8*4) FLOPs.
    expected_per_layer = 2 * (2 * 5) * (4 * 4 + 4 * 8 + 8 * 4)
    assert profile["linear_projection_flops_by_layer"]["prefill"]["0"] == expected_per_layer
    assert profile["linear_projection_flops_by_layer"]["prefill"]["1"] == expected_per_layer
    assert profile["linear_projection_flops_total"] == 2 * expected_per_layer
    assert profile["linear_projection_flops_by_layer_and_component"]["prefill"]["0"] == {
        "attention": 2 * (2 * 5) * 4 * 4,
        "ffn": 2 * (2 * 5) * (4 * 8 + 8 * 4),
    }
    assert profile["linear_projection_flops_by_component_and_phase"]["prefill"] == {
        "attention": 2 * (2 * (2 * 5) * 4 * 4),
        "ffn": 2 * (2 * (2 * 5) * (4 * 8 + 8 * 4)),
    }
    assert profile["model"]["layer_inventory"][0]["parameter_residency"]["cpu"][
        "logical_parameter_bytes"
    ] > 0
    assert profile["memory_bandwidth"]["measured"] is False
    assert profile["weight_residency"]["by_device"]["cpu"]["logical_parameter_bytes"] > 0
    assert profile["component_cpu_wall_ms_by_category"]["prefill"]["attention"] > 0
    assert profile["component_cpu_wall_ms_by_category"]["prefill"]["ffn"] > 0
    assert any(
        value["category"] == "attention"
        for value in profile["module_inclusive_timings"].values()
    )
    assert any(
        value["category"] == "ffn"
        for value in profile["module_inclusive_timings"].values()
    )
    # The profiling helper must restore each module's original train/eval state
    # and remove every installed hook after it finishes.
    assert model.training
    assert all(module.training for module in model.modules())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_profile_generation_splits_prefill_from_single_token_decode_calls():
    model = _ProfileModel()

    def generation():
        model(torch.randn(2, 3, 4))
        model(torch.randn(2, 1, 4))

    profile = profile_model_call(
        model,
        generation,
        call_kind="generation",
        warmup=0,
        iterations=1,
        use_torch_profiler=False,
        profile_cuda=False,
    )
    layer0 = profile["module_inclusive_timings"]
    assert "prefill:layer:model.layers.0" in layer0
    assert "decode:layer:model.layers.0" in layer0
    assert layer0["prefill:layer:model.layers.0"]["calls"] == 1
    assert layer0["decode:layer:model.layers.0"]["calls"] == 1
    assert profile["linear_projection_flops_by_phase"]["prefill"] > 0
    assert profile["linear_projection_flops_by_phase"]["decode"] > 0


def test_profile_refuses_ambiguous_or_missing_decoder_layers():
    with pytest.raises(ConditionalProfileError, match="cannot locate"):
        profile_model_call(nn.Linear(3, 3), lambda: None, use_torch_profiler=False)

    class _Ambiguous(nn.Module):
        def __init__(self):
            super().__init__()
            layer_a, layer_b = nn.Linear(3, 3), nn.Linear(3, 3)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer_a])
            self.layers = nn.ModuleList([layer_b])

        def forward(self, hidden_states):
            return self.model.layers[0](hidden_states)

    with pytest.raises(ConditionalProfileError, match="ambiguous"):
        profile_model_call(_Ambiguous(), lambda: None, use_torch_profiler=False)


def test_profile_artifact_is_exclusive_and_generation_summary_keeps_spread(tmp_path):
    artifact = {
        "latency": {"mean_ms": 4.0, "median_ms": 4.0, "p95_ms": 4.0, "max_ms": 4.0}
    }
    path = tmp_path / "nested" / "profile.json"
    write_profile_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_profile_artifact({"different": True}, path)

    summary = summarize_generation_profiles([
        {"latency": {"mean_ms": 2.0}},
        {"latency": {"mean_ms": 7.0}},
    ])
    assert summary["requests"] == 2
    assert summary["latency_ms"]["min_ms"] == 2.0
    assert summary["latency_ms"]["max_ms"] == 7.0
