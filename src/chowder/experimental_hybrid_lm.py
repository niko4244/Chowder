"""Small, isolated architecture generator for Chowder Experiment D.

This is a research reference, not a production backend and not a checkpoint
conversion layer. Its Mamba2 mixer is an explicit PyTorch recurrence with the
Mamba-2-style input-dependent delta, depthwise causal convolution, grouped
B/C state vectors, and per-head state; it favors testable CPU semantics over
fused-kernel performance. Attention is ordinary causal RoPE attention. MoE
expert modules are called only for accepted token assignments.

PyTorch is optional in Chowder; import this module only in an environment with
the ``train`` extra installed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ExperimentDError(ValueError):
    """Raised when an experimental architecture or execution is invalid."""


@dataclass(frozen=True)
class HybridLMConfig:
    """Validated dimensions and choices for a tiny hybrid decoder-only LM."""

    vocab_size: int = 128
    hidden_size: int = 64
    num_hidden_layers: int = 4
    layer_types: tuple[str, ...] = ("mamba2", "attention", "mamba2", "attention")
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    mamba_num_heads: int = 4
    mamba_expand: int = 2
    mamba_state_size: int = 8
    mamba_conv_kernel: int = 3
    ffn_type: str = "dense"
    intermediate_size: int = 128
    num_experts: int = 4
    experts_per_token: int = 2
    expert_intermediate_size: int = 64
    capacity_factor: float | None = 1.25
    min_expert_capacity: int = 1
    overflow_policy: str = "drop"
    num_shared_experts: int = 0
    per_layer_embedding_dim: int = 0
    output_projection: str = "standard"
    output_rank: int = 0
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    mamba_dt_min: float = 1e-3
    mamba_dt_max: float = 1.0

    def __post_init__(self) -> None:
        positive_ints = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "mamba_num_heads",
            "mamba_expand",
            "mamba_state_size",
            "mamba_conv_kernel",
            "intermediate_size",
            "num_experts",
            "experts_per_token",
            "expert_intermediate_size",
            "min_expert_capacity",
            "max_position_embeddings",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExperimentDError(f"{name} must be a positive integer")
        nonnegative_ints = ("num_shared_experts", "per_layer_embedding_dim", "output_rank")
        for name in nonnegative_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExperimentDError(f"{name} must be a nonnegative integer")

        if not isinstance(self.layer_types, (tuple, list)):
            raise ExperimentDError("layer_types must be a sequence")
        layer_types = tuple(self.layer_types)
        object.__setattr__(self, "layer_types", layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ExperimentDError("layer_types length must equal num_hidden_layers")
        if any(kind not in {"mamba2", "attention"} for kind in layer_types):
            raise ExperimentDError("layer_types entries must be 'mamba2' or 'attention'")

        if self.hidden_size % self.num_attention_heads:
            raise ExperimentDError("hidden_size must be divisible by num_attention_heads")
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2:
            raise ExperimentDError("attention head dimension must be even for RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ExperimentDError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size * self.mamba_expand) % self.mamba_num_heads:
            raise ExperimentDError("expanded Mamba width must be divisible by mamba_num_heads")
        if self.experts_per_token > self.num_experts:
            raise ExperimentDError("experts_per_token cannot exceed num_experts")
        if self.ffn_type not in {"dense", "moe"}:
            raise ExperimentDError("ffn_type must be 'dense' or 'moe'")
        if self.overflow_policy not in {"drop", "error"}:
            raise ExperimentDError("overflow_policy must be 'drop' or 'error'")
        if self.capacity_factor is not None and (
            isinstance(self.capacity_factor, bool)
            or not isinstance(self.capacity_factor, (int, float))
            or not math.isfinite(self.capacity_factor)
            or self.capacity_factor <= 0
        ):
            raise ExperimentDError("capacity_factor must be None or finite and positive")
        if self.output_projection not in {"standard", "low_rank"}:
            raise ExperimentDError("output_projection must be 'standard' or 'low_rank'")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ExperimentDError("tie_word_embeddings must be boolean")
        if self.output_projection == "low_rank":
            if self.tie_word_embeddings:
                raise ExperimentDError("low-rank output projection cannot tie the word embedding")
            if not 0 < self.output_rank <= min(self.hidden_size, self.vocab_size):
                raise ExperimentDError("output_rank must be in (0, min(hidden_size, vocab_size)]")
        elif self.output_rank != 0:
            raise ExperimentDError("output_rank must be zero for a standard output projection")
        if not isinstance(self.rms_norm_eps, (int, float)) or isinstance(self.rms_norm_eps, bool) or not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ExperimentDError("rms_norm_eps must be finite and positive")
        dt_limits = (self.mamba_dt_min, self.mamba_dt_max)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in dt_limits
        ) or self.mamba_dt_min <= 0 or self.mamba_dt_max < self.mamba_dt_min:
            raise ExperimentDError("Mamba dt limits must be finite, positive, and ordered")

    @property
    def mamba_inner_size(self) -> int:
        return self.hidden_size * self.mamba_expand

    @property
    def attention_head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_types"] = list(self.layer_types)
        result["model_type"] = "chowder_experiment_d_hybrid_lm"
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HybridLMConfig":
        values = dict(payload)
        values.pop("model_type", None)
        if "layer_types" in values:
            values["layer_types"] = tuple(values["layer_types"])
        try:
            return cls(**values)
        except TypeError as error:
            raise ExperimentDError(f"invalid configuration fields: {error}") from error

    @classmethod
    def from_json(cls, path: str | Path) -> "HybridLMConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExperimentDError("configuration JSON must contain an object")
        return cls.from_dict(payload)

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class HybridCache:
    """Mutable per-request KV and SSM state for causal incremental decoding."""

    config_digest: str
    sequence_length: int = 0
    batch_size: int | None = None
    attention_keys: dict[int, torch.Tensor] | None = None
    attention_values: dict[int, torch.Tensor] | None = None
    mamba_conv_states: dict[int, torch.Tensor] | None = None
    mamba_recurrent_states: dict[int, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_digest, str) or not self.config_digest:
            raise ExperimentDError("cache config_digest must be a nonempty string")
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 0
        ):
            raise ExperimentDError("cache sequence_length must be a nonnegative integer")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ExperimentDError("cache batch_size must be a positive integer or None")
        for name in (
            "attention_keys",
            "attention_values",
            "mamba_conv_states",
            "mamba_recurrent_states",
        ):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, {})
            elif not isinstance(value, dict):
                raise ExperimentDError(f"cache {name} must be a dictionary")



@dataclass(frozen=True)
class ExpertRoutingStats:
    tokens: int
    num_experts: int
    top_k: int
    capacity_per_expert: int | None
    assignments: int
    accepted_assignments: int
    dropped_assignments: int
    selected_per_expert: tuple[int, ...]
    accepted_per_expert: tuple[int, ...]

    @property
    def active_experts(self) -> int:
        return sum(count > 0 for count in self.accepted_per_expert)

    @property
    def max_capacity_fraction(self) -> float:
        if self.capacity_per_expert is None or self.capacity_per_expert == 0:
            return 0.0
        return max(self.accepted_per_expert, default=0) / self.capacity_per_expert


@dataclass(frozen=True)
class HybridLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    auxiliary_loss: torch.Tensor
    cache: HybridCache | None
    routing_stats: tuple[ExpertRoutingStats | None, ...]


@dataclass(frozen=True)
class ParameterReport:
    stored_parameters: int
    stored_parameter_bytes: int
    parameters_by_dtype: Mapping[str, int]
    parameter_bytes_by_dtype: Mapping[str, int]
    active_parameters_per_token: int
    active_matrix_parameters_per_token: int
    lookup_values_per_token: int
    lookup_aware_effective_parameters_per_token: int
    routed_expert_capacity_parameters: int
    routed_expert_active_parameters: int
    inactive_routed_expert_parameters: int
    per_layer_embedding_table_parameters: int
    per_layer_embedding_table_bytes: int
    tied_word_embeddings: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlopEstimate:
    context_length: int
    active_linear_weight_parameters: int
    linear_projection_flops_per_token: int
    mamba_convolution_flops_per_token: int
    mamba_state_flops_per_token: int
    attention_context_flops_per_token: int
    estimated_flops_per_token: int
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(hidden_states.dtype)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU FFN used by dense, routed, and shared experts."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Mamba2ReferenceMixer(nn.Module):
    """Unfused causal convolution + selective diagonal-state recurrence.

    The equations follow a CPU-testable Mamba-2-style parameterization; this
    is not claimed to reproduce any published checkpoint's exact implementation
    or kernel numerics.
    """

    def __init__(self, config: HybridLMConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.inner_size = config.mamba_inner_size
        self.num_heads = config.mamba_num_heads
        self.head_dim = self.inner_size // self.num_heads
        self.state_size = config.mamba_state_size
        self.kernel_size = config.mamba_conv_kernel
        self.conv_channels = self.inner_size + 2 * self.state_size
        projection_size = self.inner_size + self.conv_channels + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.conv_channels, self.kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_channels))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.inner_size))
        self.norm_eps = config.rms_norm_eps
        self.dt_min = config.mamba_dt_min
        self.dt_max = config.mamba_dt_max
        self.out_proj = nn.Linear(self.inner_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: HybridCache | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        gate, conv_inputs, dt_logits = self.in_proj(hidden_states).split(
            (self.inner_size, self.conv_channels, self.num_heads), dim=-1
        )
        conv_state = (
            cache.mamba_conv_states.get(self.layer_index)
            if cache is not None
            else None
        )
        recurrent_state = (
            cache.mamba_recurrent_states.get(self.layer_index)
            if cache is not None
            else None
        )
        if conv_state is None:
            conv_state = hidden_states.new_zeros(
                batch_size, self.conv_channels, self.kernel_size - 1
            )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.state_size,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        if conv_state.shape != (batch_size, self.conv_channels, self.kernel_size - 1):
            raise ExperimentDError("cached Mamba convolution state has an incompatible shape")
        expected_ssm = (batch_size, self.num_heads, self.head_dim, self.state_size)
        if recurrent_state.shape != expected_ssm:
            raise ExperimentDError("cached Mamba recurrent state has an incompatible shape")

        outputs: list[torch.Tensor] = []
        a = -torch.exp(self.A_log.float())
        for position in range(sequence_length):
            current = conv_inputs[:, position].unsqueeze(-1)
            window = torch.cat((conv_state, current), dim=-1)
            convolved = (window * self.conv_weight.unsqueeze(0)).sum(dim=-1)
            convolved = convolved + self.conv_bias
            conv_state = window[..., 1:]
            convolved = F.silu(convolved)
            x_t, b_t, c_t = convolved.split(
                (self.inner_size, self.state_size, self.state_size), dim=-1
            )
            x_t = x_t.float().reshape(batch_size, self.num_heads, self.head_dim)
            b_t = b_t.float()
            c_t = c_t.float()
            dt = F.softplus(dt_logits[:, position].float() + self.dt_bias.float())
            dt = dt.clamp(min=self.dt_min, max=self.dt_max)
            decay = torch.exp(dt * a.unsqueeze(0))
            recurrent_state = (
                recurrent_state * decay[:, :, None, None]
                + (dt[:, :, None] * x_t).unsqueeze(-1) * b_t[:, None, None, :]
            )
            y_t = (recurrent_state * c_t[:, None, None, :]).sum(dim=-1)
            y_t = y_t + self.D.float()[None, :, None] * x_t
            y_t = y_t.reshape(batch_size, self.inner_size)
            gated = y_t * F.silu(gate[:, position].float())
            gated = gated * torch.rsqrt(
                gated.square().mean(dim=-1, keepdim=True) + self.norm_eps
            )
            gated = gated * self.norm_weight.float()
            outputs.append(gated.to(hidden_states.dtype))

        if cache is not None:
            cache.mamba_conv_states[self.layer_index] = conv_state
            cache.mamba_recurrent_states[self.layer_index] = recurrent_state
        if not outputs:
            return hidden_states.new_empty(batch_size, 0, self.hidden_size)
        return self.out_proj(torch.stack(outputs, dim=1))


class CausalSelfAttention(nn.Module):
    """RoPE causal self-attention with grouped-query keys/values and KV cache."""

    def __init__(self, config: HybridLMConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    @staticmethod
    def _apply_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        head_dim = values.shape[-1]
        inverse_frequencies = 1.0 / (
            10_000.0
            ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim)
        )
        angles = positions.to(torch.float32)[:, None] * inverse_frequencies[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        even = values[..., 0::2].float()
        odd = values[..., 1::2].float()
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(values.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: HybridCache | None,
        position_offset: int,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        positions = torch.arange(
            position_offset,
            position_offset + query_length,
            device=hidden_states.device,
        )
        query = self._apply_rope(query, positions)
        key = self._apply_rope(key, positions)

        past_key = cache.attention_keys.get(self.layer_index) if cache is not None else None
        past_value = cache.attention_values.get(self.layer_index) if cache is not None else None
        if (past_key is None) != (past_value is None):
            raise ExperimentDError("attention KV cache is incomplete")
        if past_key is not None:
            if (
                past_key.shape[:2] != (batch_size, self.num_key_value_heads)
                or past_value.shape[:2] != (batch_size, self.num_key_value_heads)
                or past_key.shape[2] != position_offset
                or past_value.shape[2] != position_offset
                or past_key.shape[-1] != self.head_dim
                or past_value.shape[-1] != self.head_dim
                or past_key.device != key.device
                or past_value.device != value.device
            ):
                raise ExperimentDError("cached attention batch/head/position/device dimensions are incompatible")
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        if cache is not None:
            cache.attention_keys[self.layer_index] = key
            cache.attention_values[self.layer_index] = value

        repeat = self.num_heads // self.num_key_value_heads
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        key_positions = torch.arange(key.shape[2], device=hidden_states.device)
        query_positions = position_offset + torch.arange(query_length, device=hidden_states.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.num_heads * self.head_dim
        )
        return self.o_proj(attended)


class SparseMoEFFN(nn.Module):
    """Token-choice top-k MoE that never calls an unselected expert."""

    def __init__(self, config: HybridLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.capacity_factor = config.capacity_factor
        self.min_capacity = config.min_expert_capacity
        self.overflow_policy = config.overflow_policy
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_shared_experts)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, ExpertRoutingStats]:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        token_count = flat.shape[0]
        if token_count == 0:
            raise ExperimentDError("MoE cannot route an empty token batch")
        router_logits = self.router(flat)
        top_logits, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_logits.float(), dim=-1).to(hidden_states.dtype)

        # Capacity is a training-time batch constraint. During evaluation and
        # incremental decoding, dropping assignments based on the current
        # sequence/chunk size would make token outputs depend on chunking and
        # break full-sequence versus streaming equivalence.
        capacity = None
        if self.training and self.capacity_factor is not None:
            capacity = max(
                self.min_capacity,
                math.ceil(self.capacity_factor * token_count * self.top_k / self.num_experts),
            )
        accepted = torch.zeros_like(top_indices, dtype=torch.bool)
        selected_counts: list[int] = []
        accepted_counts: list[int] = []
        for expert_index in range(self.num_experts):
            locations = (top_indices == expert_index).nonzero(as_tuple=False)
            selected_count = int(locations.shape[0])
            selected_counts.append(selected_count)
            keep_count = selected_count if capacity is None else min(selected_count, capacity)
            if keep_count:
                kept = locations[:keep_count]
                accepted[kept[:, 0], kept[:, 1]] = True
            accepted_counts.append(keep_count)

        accepted_total = sum(accepted_counts)
        dropped_total = self.top_k * token_count - accepted_total
        if dropped_total and self.overflow_policy == "error":
            raise ExperimentDError(
                f"expert capacity overflow: dropped {dropped_total} of {token_count * self.top_k} assignments"
            )
        raw_weights = top_weights * accepted.to(top_weights.dtype)
        denominator = raw_weights.sum(dim=-1, keepdim=True)
        route_weights = torch.where(
            denominator > 0,
            raw_weights / denominator.clamp_min(torch.finfo(raw_weights.dtype).tiny),
            torch.zeros_like(raw_weights),
        )

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            # Locations are produced by nonzero in deterministic token/slot
            # order, giving reproducible first-come capacity admission.
            locations = (accepted & (top_indices == expert_index)).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token_indices, slot_indices = locations.unbind(dim=-1)
            active_rows = flat.index_select(0, token_indices)
            expert_output = expert(active_rows)
            weighted = expert_output * route_weights[token_indices, slot_indices].unsqueeze(-1)
            output = output.index_add(0, token_indices, weighted)

        for expert in self.shared_experts:
            output = output + expert(flat)

        probabilities = F.softmax(router_logits.float(), dim=-1)
        raw_load = torch.tensor(selected_counts, device=flat.device, dtype=torch.float32)
        raw_load = raw_load / max(1, token_count * self.top_k)
        auxiliary_loss = self.num_experts * torch.sum(raw_load * probabilities.mean(dim=0))
        stats = ExpertRoutingStats(
            tokens=token_count,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_per_expert=capacity,
            assignments=token_count * self.top_k,
            accepted_assignments=accepted_total,
            dropped_assignments=dropped_total,
            selected_per_expert=tuple(selected_counts),
            accepted_per_expert=tuple(accepted_counts),
        )
        return output.view(original_shape), auxiliary_loss, stats


class PerLayerTokenEmbeddings(nn.Module):
    """One token-identity table and hidden projection per decoder layer.

    The embedding tables are separate modules so only the current layer's
    lookup is materialized during the forward pass. Total table capacity is
    exactly vocab_size * number_of_layers * embedding_dim.
    """

    def __init__(self, config: HybridLMConfig) -> None:
        super().__init__()
        self.tables = nn.ModuleList(
            nn.Embedding(config.vocab_size, config.per_layer_embedding_dim)
            for _ in range(config.num_hidden_layers)
        )
        self.projections = nn.ModuleList(
            nn.Linear(config.per_layer_embedding_dim, config.hidden_size, bias=False)
            for _ in range(config.num_hidden_layers)
        )

    def forward(self, input_ids: torch.Tensor, layer_index: int) -> torch.Tensor:
        return self.projections[layer_index](self.tables[layer_index](input_ids))


class HybridDecoderLayer(nn.Module):
    def __init__(self, config: HybridLMConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "mamba2":
            self.mixer = Mamba2ReferenceMixer(config, layer_index)
        else:
            self.mixer = CausalSelfAttention(config, layer_index)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.ffn_type == "dense":
            self.ffn: nn.Module = SwiGLU(config.hidden_size, config.intermediate_size)
        else:
            self.ffn = SparseMoEFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: HybridCache | None,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ExpertRoutingStats | None]:
        normalized = self.input_norm(hidden_states)
        if isinstance(self.mixer, CausalSelfAttention):
            mixer_output = self.mixer(
                normalized, cache=cache, position_offset=position_offset
            )
        else:
            mixer_output = self.mixer(normalized, cache=cache)
        hidden_states = hidden_states + mixer_output
        normalized = self.ffn_norm(hidden_states)
        if isinstance(self.ffn, SparseMoEFFN):
            ffn_output, auxiliary_loss, stats = self.ffn(normalized)
        else:
            ffn_output = self.ffn(normalized)
            auxiliary_loss, stats = None, None
        return hidden_states + ffn_output, auxiliary_loss, stats


class LowRankLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


class HybridLanguageModel(nn.Module):
    """Decoder-only hybrid LM builder, with no Chowder campaign integration."""

    def __init__(self, config: HybridLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.per_layer_embeddings = (
            PerLayerTokenEmbeddings(config)
            if config.per_layer_embedding_dim > 0
            else None
        )
        self.layers = nn.ModuleList(
            HybridDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.output_projection == "standard":
            self.lm_head: nn.Module = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.tie_word_embeddings:
                self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = LowRankLMHead(
                config.hidden_size, config.vocab_size, config.output_rank
            )

    def _validate_cache_state(self, cache: HybridCache, batch_size: int) -> None:
        if (
            isinstance(cache.sequence_length, bool)
            or not isinstance(cache.sequence_length, int)
            or cache.sequence_length < 0
        ):
            raise ExperimentDError("cache sequence_length must be a nonnegative integer")
        maps = (
            cache.attention_keys,
            cache.attention_values,
            cache.mamba_conv_states,
            cache.mamba_recurrent_states,
        )
        if any(not isinstance(state_map, dict) for state_map in maps):
            raise ExperimentDError("cache state maps must be dictionaries")

        attention_layers = {
            index
            for index, layer in enumerate(self.layers)
            if isinstance(layer.mixer, CausalSelfAttention)
        }
        mamba_layers = {
            index
            for index, layer in enumerate(self.layers)
            if isinstance(layer.mixer, Mamba2ReferenceMixer)
        }
        expected_keys = (
            (cache.attention_keys, attention_layers),
            (cache.attention_values, attention_layers),
            (cache.mamba_conv_states, mamba_layers),
            (cache.mamba_recurrent_states, mamba_layers),
        )
        for state_map, expected in expected_keys:
            if cache.sequence_length == 0:
                if state_map:
                    raise ExperimentDError("empty cache cannot contain layer state")
            elif set(state_map) != expected:
                raise ExperimentDError("cache has incomplete or unexpected layer state")

        if cache.sequence_length == 0:
            return
        expected_device = self.token_embedding.weight.device
        for index in attention_layers:
            mixer = self.layers[index].mixer
            assert isinstance(mixer, CausalSelfAttention)
            expected_shape = (
                batch_size,
                mixer.num_key_value_heads,
                cache.sequence_length,
                mixer.head_dim,
            )
            key = cache.attention_keys[index]
            value = cache.attention_values[index]
            if (
                not isinstance(key, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or key.shape != expected_shape
                or value.shape != expected_shape
                or key.device != expected_device
                or value.device != expected_device
                or key.dtype != self.token_embedding.weight.dtype
                or value.dtype != self.token_embedding.weight.dtype
                or not key.is_floating_point()
                or key.dtype != value.dtype
                or not value.is_floating_point()
            ):
                raise ExperimentDError("cached attention key/value tensors have incompatible shape, dtype, or device")

        for index in mamba_layers:
            mixer = self.layers[index].mixer
            assert isinstance(mixer, Mamba2ReferenceMixer)
            conv_state = cache.mamba_conv_states[index]
            recurrent_state = cache.mamba_recurrent_states[index]
            if (
                not isinstance(conv_state, torch.Tensor)
                or conv_state.shape != (batch_size, mixer.conv_channels, mixer.kernel_size - 1)
                or conv_state.device != expected_device
                or conv_state.dtype != self.token_embedding.weight.dtype
                or not conv_state.is_floating_point()
                or not isinstance(recurrent_state, torch.Tensor)
                or recurrent_state.shape
                != (batch_size, mixer.num_heads, mixer.head_dim, mixer.state_size)
                or recurrent_state.device != expected_device
                or recurrent_state.dtype != torch.float32
            ):
                raise ExperimentDError("cached Mamba states have incompatible shape, dtype, or device")

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        cache: HybridCache | None = None,
        use_cache: bool = False,
    ) -> HybridLMOutput:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ExperimentDError("input_ids must be a [batch, sequence] tensor")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise ExperimentDError("input_ids must use int32 or int64 for embedding lookup")
        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ExperimentDError("input batch and sequence lengths must be positive")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocab_size).any()):
            raise ExperimentDError("input token id is outside the configured vocabulary")
        if labels is not None and (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 2
            or labels.shape != input_ids.shape
            or labels.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        ):
            raise ExperimentDError("labels must be an integer [batch, sequence] tensor matching input_ids")
        if input_ids.device != self.token_embedding.weight.device:
            raise ExperimentDError("input_ids must be on the same device as the model")
        if labels is not None and (
            bool((labels < 0).any())
            or bool((labels >= self.config.vocab_size).any())
        ):
            raise ExperimentDError("label token id is outside the configured vocabulary")
        if cache is not None and not isinstance(cache, HybridCache):
            raise ExperimentDError("cache must be a HybridCache instance")
        if cache is not None and not use_cache:
            raise ExperimentDError("pass use_cache=True when supplying a mutable decode cache")
        if use_cache:
            if cache is None:
                cache = HybridCache(self.config.digest(), batch_size=batch_size)
            elif cache.config_digest != self.config.digest():
                raise ExperimentDError("decode cache belongs to a different model configuration")
            elif cache.batch_size is not None and (
                isinstance(cache.batch_size, bool)
                or not isinstance(cache.batch_size, int)
                or cache.batch_size != batch_size
            ):
                raise ExperimentDError("decode cache batch size does not match input")
            else:
                cache.batch_size = batch_size
        if cache is not None:
            self._validate_cache_state(cache, batch_size)
        position_offset = cache.sequence_length if cache is not None else 0
        if position_offset + sequence_length > self.config.max_position_embeddings:
            raise ExperimentDError("input exceeds max_position_embeddings")
        hidden_states = self.token_embedding(input_ids)
        auxiliary_losses: list[torch.Tensor] = []
        routing_stats: list[ExpertRoutingStats | None] = []
        for index, layer in enumerate(self.layers):
            if self.per_layer_embeddings is not None:
                hidden_states = hidden_states + self.per_layer_embeddings(input_ids, index)
            hidden_states, auxiliary_loss, stats = layer(
                hidden_states,
                cache=cache,
                position_offset=position_offset,
            )
            if auxiliary_loss is not None:
                auxiliary_losses.append(auxiliary_loss)
            routing_stats.append(stats)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels) if labels is not None else None
        auxiliary_loss = (
            torch.stack(auxiliary_losses).mean()
            if auxiliary_losses
            else logits.new_zeros(())
        )
        if cache is not None:
            cache.sequence_length += sequence_length
        return HybridLMOutput(
            logits=logits,
            loss=loss,
            auxiliary_loss=auxiliary_loss,
            cache=cache,
            routing_stats=tuple(routing_stats),
        )

    def parameter_report(self) -> ParameterReport:
        return parameter_report(self)

    def flop_estimate(self, *, context_length: int = 1) -> FlopEstimate:
        return estimate_flops(self, context_length=context_length)

    def save_pretrained(self, path: str | Path) -> str:
        """Write config and safe weights-only state to a new directory."""
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite Experiment D model: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        (target / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
        }
        torch.save(state, target / "model.pt")
        return str(target)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> "HybridLanguageModel":
        source = Path(path)
        config = HybridLMConfig.from_json(source / "config.json")
        state = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ExperimentDError("saved model state is empty or malformed")
        dtype = next(
            (tensor.dtype for tensor in state.values() if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()),
            torch.float32,
        )
        model = cls(config).to(device=device, dtype=dtype)
        model.load_state_dict(state, strict=True)
        return model


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy over a matching token-id sequence."""
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ExperimentDError("labels must be a [batch, sequence] tensor")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ExperimentDError("logits and labels batch/sequence dimensions must match")

    if labels.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ExperimentDError("labels must use an integer dtype")
    if bool((labels < 0).any()) or bool((labels >= logits.shape[-1]).any()):
        raise ExperimentDError("label token id is outside the logits vocabulary")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]).float(),
        labels[:, 1:].contiguous().view(-1).to(device=logits.device, dtype=torch.long),
    )


def _unique_parameters(model: nn.Module) -> tuple[list[nn.Parameter], int, int, dict[str, int], dict[str, int]]:
    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    by_dtype: dict[str, int] = {}
    bytes_by_dtype: dict[str, int] = {}
    total_bytes = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameters.append(parameter)
        dtype = str(parameter.dtype)
        count = parameter.numel()
        nbytes = count * parameter.element_size()
        by_dtype[dtype] = by_dtype.get(dtype, 0) + count
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + nbytes
        total_bytes += nbytes
    return parameters, sum(parameter.numel() for parameter in parameters), total_bytes, by_dtype, bytes_by_dtype


def parameter_report(model: HybridLanguageModel) -> ParameterReport:
    """Exact unique stored count plus explicit sparse and lookup conventions."""
    config = model.config
    _parameters, stored, stored_bytes, by_dtype, bytes_by_dtype = _unique_parameters(model)
    routed_capacity = 0
    routed_active = 0
    for layer in model.layers:
        if isinstance(layer.ffn, SparseMoEFFN):
            expert_size = sum(parameter.numel() for parameter in layer.ffn.experts[0].parameters())
            routed_capacity += expert_size * config.num_experts
            routed_active += expert_size * config.experts_per_token
    inactive_routed = routed_capacity - routed_active
    active = stored - inactive_routed

    token_table_capacity = model.token_embedding.weight.numel()
    ple_table_capacity = 0
    if model.per_layer_embeddings is not None:
        ple_table_capacity = sum(table.weight.numel() for table in model.per_layer_embeddings.tables)
    lookup_only_token_table = 0 if config.tie_word_embeddings else token_table_capacity
    matrix_active = (
        stored
        - inactive_routed
        - lookup_only_token_table
        - ple_table_capacity
    )
    lookup_values = config.hidden_size + (
        config.num_hidden_layers * config.per_layer_embedding_dim
    )
    # A tied word table is also the full output matrix, so its row lookup is
    # already contained in matrix-active parameters. PLE rows are separate.
    lookup_overlap = config.hidden_size if config.tie_word_embeddings else 0
    lookup_aware = matrix_active + lookup_values - lookup_overlap
    table_parameters = config.vocab_size * config.num_hidden_layers * config.per_layer_embedding_dim
    if table_parameters != ple_table_capacity:
        raise ExperimentDError("PLE table storage differs from the declared vocabulary/layer/dimension product")
    table_bytes = sum(
        table.weight.numel() * table.weight.element_size()
        for table in (model.per_layer_embeddings.tables if model.per_layer_embeddings is not None else ())
    )
    return ParameterReport(
        stored_parameters=stored,
        stored_parameter_bytes=stored_bytes,
        parameters_by_dtype=by_dtype,
        parameter_bytes_by_dtype=bytes_by_dtype,
        active_parameters_per_token=active,
        active_matrix_parameters_per_token=matrix_active,
        lookup_values_per_token=lookup_values,
        lookup_aware_effective_parameters_per_token=lookup_aware,
        routed_expert_capacity_parameters=routed_capacity,
        routed_expert_active_parameters=routed_active,
        inactive_routed_expert_parameters=inactive_routed,
        per_layer_embedding_table_parameters=table_parameters,
        per_layer_embedding_table_bytes=table_bytes,
        tied_word_embeddings=config.tie_word_embeddings,
    )


def _linear_weight_count(module: nn.Module) -> int:
    return sum(child.weight.numel() for child in module.modules() if isinstance(child, nn.Linear))


def estimate_flops(model: HybridLanguageModel, *, context_length: int = 1) -> FlopEstimate:
    """Analytical per-token estimate, not a hardware measurement.

    Includes active linear multiply-adds, depthwise Mamba convolution and
    diagonal-state update/readout, and causal attention score/value products.
    Norms, activations, softmax, residual adds, routing comparisons, padding,
    backward work, and kernel overhead are excluded and called out below.
    """
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ExperimentDError("context_length must be a positive integer")
    config = model.config
    active_linear_weights = 0
    attention_layers = 0
    mamba_layers = 0
    mamba_conv_flops = 0
    mamba_state_flops = 0

    for layer in model.layers:
        if isinstance(layer.mixer, CausalSelfAttention):
            attention_layers += 1
            active_linear_weights += _linear_weight_count(layer.mixer)
        else:
            mamba_layers += 1
            mixer = layer.mixer
            active_linear_weights += mixer.in_proj.weight.numel() + mixer.out_proj.weight.numel()
            # A depthwise length-K convolution costs approximately K MACs per channel.
            mamba_conv_flops += 2 * mixer.conv_channels * mixer.kernel_size
            # Per state: decay*state + dt*x*B; output reads state*C and adds D*x.
            mamba_state_flops += (
                mixer.num_heads * mixer.head_dim * (5 * mixer.state_size + 1)
            )
        if isinstance(layer.ffn, SparseMoEFFN):
            moe = layer.ffn
            active_linear_weights += moe.router.weight.numel()
            active_linear_weights += config.experts_per_token * _linear_weight_count(moe.experts[0])
            active_linear_weights += sum(_linear_weight_count(expert) for expert in moe.shared_experts)
        else:
            active_linear_weights += _linear_weight_count(layer.ffn)

    if model.per_layer_embeddings is not None:
        active_linear_weights += sum(
            projection.weight.numel() for projection in model.per_layer_embeddings.projections
        )
    if isinstance(model.lm_head, LowRankLMHead):
        active_linear_weights += model.lm_head.down.weight.numel() + model.lm_head.up.weight.numel()
    else:
        # Includes a tied token embedding used as the full vocabulary output matrix.
        active_linear_weights += model.lm_head.weight.numel()

    linear_flops = 2 * active_linear_weights
    # Each query attends to the current token plus its causal history; QK and
    # AV products each cost roughly 2*hidden_size*context_length FLOPs.
    attention_flops = 4 * attention_layers * config.hidden_size * context_length
    return FlopEstimate(
        context_length=context_length,
        active_linear_weight_parameters=active_linear_weights,
        linear_projection_flops_per_token=linear_flops,
        mamba_convolution_flops_per_token=mamba_conv_flops,
        mamba_state_flops_per_token=mamba_state_flops,
        attention_context_flops_per_token=attention_flops,
        estimated_flops_per_token=linear_flops + mamba_conv_flops + mamba_state_flops + attention_flops,
        assumptions=(
            "analytical operation estimate for the implemented architecture; not a profiler or measured FLOP counter",
            "linear weights count one multiply-add as two FLOPs; the estimate assumes all top-k routes execute and does not discount capacity-dropped assignments",
            "attention uses the supplied decode context length; prefill FLOPs grow quadratically and are not represented by one decode-token estimate",
            "excludes RMSNorm, activations, softmax, residual adds, router top-k comparisons, padding, and training backward work",
        ),
    )


def load_experiment_d_configs(directory: str | Path) -> dict[str, HybridLMConfig]:
    """Load and validate a directory of named JSON configuration files."""
    root = Path(directory)
    configs: dict[str, HybridLMConfig] = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "registry.json":
            continue
        config = HybridLMConfig.from_json(path)
        configs[path.stem] = config
    return configs
