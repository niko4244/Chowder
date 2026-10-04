"""Isolated primitives for the conditional-compute research experiment.

This module is deliberately not wired into Chowder's production model path.
Only token-local FFNs are ready bypass candidates; generic attention, cache,
and recurrent-state surgery is refused unless an explicit adapter is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import math
from pathlib import Path
import statistics
from itertools import chain
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalComputeError(RuntimeError):
    """Raised when a requested conditional path cannot preserve semantics."""


@dataclass(frozen=True)
class RoutePlan:
    """Per-token hard route plus optional differentiable gate scores."""

    keep: torch.Tensor
    scores: torch.Tensor | None = None
    straight_through: torch.Tensor | None = None
    valid_tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class RoutingStats:
    """Routing counts; selected/executed fractions are normalized by valid rows."""

    tokens: int
    selected_tokens: int
    executed_tokens: int
    selected_fraction: float
    executed_fraction: float
    dense_fallback: bool = False
    fallback_reason: str | None = None
    valid_tokens: int | None = None


class AlwaysOnRouter(nn.Module):
    """Unconditional control: route every valid token through the wrapped block."""

    def forward(self, hidden_states, position_ids=None, token_mask=None) -> RoutePlan:
        del position_ids
        valid = _coerce_token_mask(token_mask, hidden_states.shape[:-1], hidden_states.device)
        scores = torch.ones(valid.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        return RoutePlan(valid, scores, scores, valid)

    def auxiliary_loss(self, hidden_states, token_mask=None) -> torch.Tensor:
        del token_mask
        return hidden_states.new_zeros(())


class FixedSkipRouter(nn.Module):
    """Periodic token skip control; single-token decode defaults to dense without IDs."""

    def __init__(self, skip_every: int = 4, skip_phase: int = 0) -> None:
        super().__init__()
        if isinstance(skip_every, bool) or not isinstance(skip_every, int) or skip_every < 1:
            raise ValueError("skip_every must be a positive integer")
        if isinstance(skip_phase, bool) or not isinstance(skip_phase, int) or not 0 <= skip_phase < skip_every:
            raise ValueError("skip_phase must be in [0, skip_every)")
        self.skip_every = skip_every
        self.skip_phase = skip_phase

    def forward(self, hidden_states, position_ids=None, token_mask=None) -> RoutePlan:
        shape = hidden_states.shape[:-1]
        valid = _coerce_token_mask(token_mask, shape, hidden_states.device)
        if position_ids is None and shape and shape[-1] == 1:
            keep = valid
        else:
            if position_ids is None:
                if not shape:
                    raise ValueError("hidden states require a token dimension")
                positions = torch.arange(shape[-1], device=hidden_states.device)
                positions = positions.reshape((1,) * (len(shape) - 1) + (shape[-1],)).expand(shape)
            else:
                if position_ids.is_floating_point() and (
                    not bool(torch.isfinite(position_ids).all())
                    or not bool((position_ids == position_ids.round()).all())
                ):
                    raise ValueError("position_ids must contain finite integer values")
                try:
                    positions = torch.broadcast_to(position_ids.to(hidden_states.device, torch.long), shape)
                except RuntimeError as error:
                    raise ValueError("position_ids must broadcast to token dimensions") from error
            keep = (positions.remainder(self.skip_every) != self.skip_phase) & valid
        scores = keep.to(hidden_states.dtype)
        return RoutePlan(keep, scores, scores, valid)

    def auxiliary_loss(self, hidden_states, token_mask=None) -> torch.Tensor:
        del token_mask
        return hidden_states.new_zeros(())


class FixedLayerRouter(nn.Module):
    """Deterministic whole-layer skip pattern for a fixed-depth control."""

    def __init__(self, layer_index: int, *, skip_layers: Sequence[int]) -> None:
        super().__init__()
        if isinstance(skip_layers, (str, bytes)):
            raise TypeError("skip_layers must be a sequence of integer indices")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("layer_index must be a nonnegative integer")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in skip_layers):
            raise ValueError("skip_layers must contain nonnegative integers")
        self.layer_index = layer_index
        self.skip_layers = frozenset(skip_layers)

    def forward(self, hidden_states, position_ids=None, token_mask=None) -> RoutePlan:
        del position_ids
        valid = _coerce_token_mask(token_mask, hidden_states.shape[:-1], hidden_states.device)
        keep = valid if self.layer_index not in self.skip_layers else torch.zeros_like(valid)
        scores = keep.to(hidden_states.dtype)
        return RoutePlan(keep, scores, scores, valid)

    def auxiliary_loss(self, hidden_states, token_mask=None) -> torch.Tensor:
        del token_mask
        return hidden_states.new_zeros(())


class BudgetTokenRouter(nn.Module):
    """Learned stable top-k router with per-sequence or global token budgets."""

    def __init__(self, hidden_size: int, *, keep_rate: float = 0.75, minimum_tokens: int = 1,
                 temperature: float = 1.0, budget_scope: str = "per_sequence") -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(keep_rate, bool) or not isinstance(keep_rate, (int, float)) or not math.isfinite(keep_rate) or not 0 < keep_rate <= 1:
            raise ValueError("keep_rate must be finite and in (0, 1]")
        if budget_scope not in {"per_sequence", "global"}:
            raise ValueError("budget_scope must be 'per_sequence' or 'global'")
        if isinstance(minimum_tokens, bool) or not isinstance(minimum_tokens, int) or minimum_tokens < 0:
            raise ValueError("minimum_tokens must be a nonnegative integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.projection = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.projection.bias)
        self.keep_rate = float(keep_rate)
        self.minimum_tokens = minimum_tokens
        self.temperature = float(temperature)
        self.budget_scope = budget_scope

    def probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(hidden_states).squeeze(-1) / self.temperature)

    def forward(self, hidden_states, position_ids=None, token_mask=None) -> RoutePlan:
        del position_ids
        if hidden_states.ndim not in {2, 3}:
            raise ValueError("budget routing expects [tokens, hidden] or [batch, tokens, hidden]")
        shape = hidden_states.shape[:-1]
        valid = _coerce_token_mask(token_mask, shape, hidden_states.device)
        probs = self.probabilities(hidden_states)
        keep = torch.zeros(shape, dtype=torch.bool, device=hidden_states.device)

        def choose(candidates: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
            count = min(candidates.numel(), max(self.minimum_tokens, math.ceil(candidates.numel() * self.keep_rate)))
            result = torch.zeros_like(row, dtype=torch.bool)
            if count:
                order = torch.argsort(row.index_select(0, candidates), descending=True, stable=True)
                result[candidates.index_select(0, order[:count])] = True
            return result

        if self.budget_scope == "global":
            candidates = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            keep = choose(candidates, probs.reshape(-1)).reshape(shape)
        else:
            rows = probs.reshape(-1, probs.shape[-1])
            valid_rows = valid.reshape_as(rows)
            kept = keep.reshape_as(rows)
            for index, row in enumerate(rows):
                candidates = valid_rows[index].nonzero(as_tuple=False).squeeze(-1)
                kept[index] = choose(candidates, row)
            keep = kept.reshape(shape)
        hard = keep.to(probs.dtype)
        straight = hard.detach() - probs.detach() + probs
        return RoutePlan(keep, probs, straight, valid)

    def auxiliary_loss(self, hidden_states, token_mask=None) -> torch.Tensor:
        probs = self.probabilities(hidden_states)
        valid = _coerce_token_mask(token_mask, probs.shape, hidden_states.device)
        chosen = probs[valid]
        return (chosen.mean() - self.keep_rate).square() if chosen.numel() else probs.sum() * 0


class LearnedTokenRouter(nn.Module):
    """Sigmoid hard gate with a separate differentiable budget regularizer."""

    def __init__(self, hidden_size: int, *, target_keep_rate: float = 0.75,
                 threshold: float | None = None, temperature: float = 1.0) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(target_keep_rate, bool) or not isinstance(target_keep_rate, (int, float)) or not math.isfinite(target_keep_rate) or not 0 < target_keep_rate < 1:
            raise ValueError("target_keep_rate must be finite and in (0, 1)")
        route_threshold = target_keep_rate if threshold is None else threshold
        if isinstance(route_threshold, bool) or not isinstance(route_threshold, (int, float)) or not math.isfinite(route_threshold) or not 0 < route_threshold < 1:
            raise ValueError("threshold must be finite and in (0, 1)")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.projection = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.projection.weight, mean=0, std=0.02)
        nn.init.constant_(self.projection.bias, math.log(target_keep_rate / (1 - target_keep_rate)))
        self.target_keep_rate = float(target_keep_rate)
        self.threshold = float(route_threshold)
        self.temperature = float(temperature)

    def probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(hidden_states).squeeze(-1) / self.temperature)

    def forward(self, hidden_states, position_ids=None, token_mask=None) -> RoutePlan:
        del position_ids
        probs = self.probabilities(hidden_states)
        valid = _coerce_token_mask(token_mask, probs.shape, hidden_states.device)
        keep = (probs >= self.threshold) & valid
        hard = keep.to(probs.dtype)
        straight = hard.detach() - probs.detach() + probs
        return RoutePlan(keep, probs, straight, valid)

    def auxiliary_loss(self, hidden_states, token_mask=None) -> torch.Tensor:
        probs = self.probabilities(hidden_states)
        valid = _coerce_token_mask(token_mask, probs.shape, hidden_states.device)
        chosen = probs[valid]
        return (chosen.mean() - self.target_keep_rate).square() if chosen.numel() else probs.sum() * 0


def _coerce_token_mask(token_mask: torch.Tensor | None, token_shape, device: torch.device) -> torch.Tensor:
    if token_mask is None:
        return torch.ones(token_shape, dtype=torch.bool, device=device)
    if not isinstance(token_mask, torch.Tensor):
        raise TypeError("token_mask must be a tensor")
    if token_mask.dtype is not torch.bool:
        if not (token_mask.is_floating_point() or token_mask.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}):
            raise TypeError("token_mask must be boolean or numeric zero/one")
        if token_mask.is_floating_point() and not bool(torch.isfinite(token_mask).all()):
            raise ValueError("token_mask must be finite zero/one")
        if not bool(((token_mask == 0) | (token_mask == 1)).all()):
            raise ValueError("token_mask must contain only zero/one")
    try:
        return torch.broadcast_to(token_mask.to(device=device, dtype=torch.bool), token_shape)
    except RuntimeError as error:
        raise ValueError("token_mask must broadcast to token dimensions") from error


def _validated_route(router, hidden_states, position_ids, token_mask) -> RoutePlan:
    plan = router(hidden_states, position_ids, token_mask=token_mask)
    if not isinstance(plan, RoutePlan):
        raise TypeError("router must return a RoutePlan")
    if not isinstance(plan.keep, torch.Tensor):
        raise TypeError("router keep mask must be a tensor")
    if (
        tuple(plan.keep.shape) != tuple(hidden_states.shape[:-1])
        or plan.keep.dtype is not torch.bool
        or plan.keep.device != hidden_states.device
    ):
        raise ValueError("router keep mask must match token dimensions/device and be boolean")
    for name, score in (("scores", plan.scores), ("straight-through", plan.straight_through)):
        if score is not None and (tuple(score.shape) != tuple(plan.keep.shape) or score.device != hidden_states.device or not score.is_floating_point() or not bool(torch.isfinite(score).all())):
            raise ValueError(f"router {name} must be finite floating point matching mask/device")
    valid = _coerce_token_mask(token_mask, plan.keep.shape, hidden_states.device)
    if plan.valid_tokens is not None:
        # Do not mutate the caller's mask or a broadcast/expanded view.
        valid = valid & _coerce_token_mask(plan.valid_tokens, plan.keep.shape, hidden_states.device)
    return RoutePlan(plan.keep & valid, plan.scores, plan.straight_through, valid)


def _routing_stats(plan: RoutePlan, executed_tokens: int, *, fallback=False, reason=None) -> RoutingStats:
    total = plan.keep.numel()
    selected = int(plan.keep.sum().item())
    valid = int(plan.valid_tokens.sum().item()) if plan.valid_tokens is not None else total
    if not 0 <= executed_tokens <= total:
        raise ValueError("executed_tokens outside token row count")
    return RoutingStats(total, selected, executed_tokens, selected / valid if valid else 0.0,
                        executed_tokens / valid if valid else 0.0, fallback, reason, valid)


_STATE_CONTEXT_KEYS = (
    "cache",
    "past_key_value",
    "past_key_values",
    "cache_params",
    "inference_params",
    "inference_context",
    "cache_position",
    "recurrent_state",
    "ssm_state",
    "state_cache",
    "state",
)


def _has_mutable_cache_state(context: Mapping[str, Any]) -> bool:
    """Conservatively recognize common mutable attention/SSM state contracts."""
    return context.get("use_cache") is True or any(
        context.get(key) is not None for key in _STATE_CONTEXT_KEYS
    )


class ConditionalFFN(nn.Module):
    """Gather selected rows into a token-local FFN and return its residual delta."""

    def __init__(self, ffn: nn.Module, router: nn.Module) -> None:
        super().__init__()
        self.ffn = ffn
        self.router = router
        self.last_routing_stats: RoutingStats | None = None

    def forward(self, hidden_states, *, position_ids=None, return_routing_stats=False, token_mask=None):
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must have token and hidden dimensions")
        plan = _validated_route(self.router, hidden_states, position_ids, token_mask)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        selected = plan.keep.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if selected.numel():
            active = flat.index_select(0, selected)
            delta = self.ffn(active)
            if (
                not isinstance(delta, torch.Tensor)
                or delta.shape != active.shape
                or delta.device != active.device
                or delta.dtype != active.dtype
            ):
                raise ConditionalComputeError(
                    "conditional FFN must preserve selected input shape/device/dtype"
                )
            if plan.straight_through is not None:
                scale = plan.straight_through.reshape(-1).index_select(0, selected).to(delta.dtype).unsqueeze(-1)
                delta = delta * scale
            out = torch.zeros_like(flat).index_copy(0, selected, delta)
        else:
            out = torch.zeros_like(flat)
        result = out.reshape_as(hidden_states)
        stats = _routing_stats(plan, int(selected.numel()))
        self.last_routing_stats = stats
        return (result, stats) if return_routing_stats else result

    def auxiliary_loss(self, hidden_states, token_mask=None):
        fn = getattr(self.router, "auxiliary_loss", None)
        return fn(hidden_states, token_mask=token_mask) if callable(fn) else hidden_states.new_zeros(())


@dataclass(frozen=True)
class SparseDepthResult:
    hidden_states: torch.Tensor
    cache_updated: bool = False


class SparseDepthAdapter(Protocol):
    supports_sparse_token_execution: bool
    supports_selective_cache: bool

    def forward_selected_tokens(self, layer, hidden_states, selected_indices, **context) -> SparseDepthResult: ...


class TokenwiseResidualAdapter(nn.Module):
    """CPU reference for independent token-local residual blocks, not attention."""

    supports_sparse_token_execution = True
    supports_selective_cache = False

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward_selected_tokens(self, layer, hidden_states, selected_indices, **context):
        del layer, context
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        active = flat.index_select(0, selected_indices)
        delta = self.transform(active)
        if (
            not isinstance(delta, torch.Tensor)
            or delta.shape != active.shape
            or delta.device != active.device
            or delta.dtype != active.dtype
        ):
            raise ConditionalComputeError(
                "tokenwise transform must preserve hidden shape/device/dtype"
            )
        return SparseDepthResult(flat.index_copy(0, selected_indices, active + delta).reshape_as(hidden_states))


class ConditionalDepthLayer(nn.Module):
    """Adapter-based sparse depth with fail-closed dense fallback."""

    def __init__(self, layer, router, *, adapter=None, fallback="dense"):
        super().__init__()
        if fallback not in {"dense", "error"}:
            raise ValueError("fallback must be 'dense' or 'error'")
        self.layer, self.router, self.adapter, self.fallback = layer, router, adapter, fallback
        self.last_routing_stats: RoutingStats | None = None

    def _dense(self, hidden, context):
        result = self.layer(hidden, **context)
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != hidden.shape
            or result.device != hidden.device
            or result.dtype != hidden.dtype
        ):
            raise ConditionalComputeError(
                "dense layer must preserve hidden-state shape/device/dtype"
            )
        return result

    @staticmethod
    def _scatter(hidden, result, plan, selected):
        flat = hidden.reshape(-1, hidden.shape[-1])
        if not selected.numel():
            return hidden
        active = flat.index_select(0, selected)
        routed = result.reshape_as(flat).index_select(0, selected)
        if plan.straight_through is not None:
            scale = (
                plan.straight_through.reshape(-1)
                .index_select(0, selected)
                .to(routed.dtype)
                .unsqueeze(-1)
            )
            routed = active + (routed - active) * scale
        return flat.index_copy(0, selected, routed).reshape_as(hidden)

    def forward(self, hidden_states, *, position_ids=None, return_routing_stats=False, token_mask=None, **context):
        plan = _validated_route(self.router, hidden_states, position_ids, token_mask)
        layer_context = dict(context)
        if position_ids is not None:
            layer_context["position_ids"] = position_ids
        selected = plan.keep.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        total = plan.keep.numel()
        has_cache = _has_mutable_cache_state(context)
        supports_sparse = (
            self.adapter is not None
            and getattr(self.adapter, "supports_sparse_token_execution", False) is True
        )
        if selected.numel() == total:
            output = self._scatter(hidden_states, self._dense(hidden_states, layer_context), plan, selected)
            stats = _routing_stats(plan, total)
        elif not selected.numel() and not has_cache and supports_sparse:
            output = hidden_states
            stats = _routing_stats(plan, 0)
        else:
            supported = supports_sparse and (
                not has_cache
                or getattr(self.adapter, "supports_selective_cache", False) is True
            )
            if not supported:
                reason = "no verified sparse adapter for this layer/cache contract"
                if self.fallback == "error":
                    raise ConditionalComputeError(reason)
                # Dense fallback is a real full-layer execution: keep its
                # output for every row rather than scattering only routed ones.
                output = self._dense(hidden_states, layer_context)
                stats = _routing_stats(plan, total, fallback=True, reason=reason)
            else:
                result = self.adapter.forward_selected_tokens(
                    self.layer, hidden_states, selected, **layer_context
                )
                if not isinstance(result, SparseDepthResult) or not isinstance(result.hidden_states, torch.Tensor):
                    raise ConditionalComputeError("sparse adapter returned invalid hidden-state result")
                if (
                    result.hidden_states.shape != hidden_states.shape
                    or result.hidden_states.device != hidden_states.device
                    or result.hidden_states.dtype != hidden_states.dtype
                ):
                    raise ConditionalComputeError("sparse adapter returned mismatched hidden-state shape/device/dtype")
                if not isinstance(result.cache_updated, bool):
                    raise ConditionalComputeError("sparse adapter cache-updated status must be boolean")
                if has_cache and not result.cache_updated:
                    raise ConditionalComputeError("sparse adapter did not confirm cache/state update")
                output = self._scatter(hidden_states, result.hidden_states, plan, selected)
                stats = _routing_stats(plan, int(selected.numel()))
        self.last_routing_stats = stats
        return (output, stats) if return_routing_stats else output

    def auxiliary_loss(self, hidden_states, token_mask=None):
        fn = getattr(self.router, "auxiliary_loss", None)
        return fn(hidden_states, token_mask=token_mask) if callable(fn) else hidden_states.new_zeros(())


@dataclass(frozen=True)
class ExitStats:
    tokens: int
    early_exit_tokens: int
    verification_tokens: int
    deep_tokens: int
    mean_score: float
    threshold: float


@dataclass(frozen=True)
class ExitCalibration:
    threshold: float
    accuracy: float
    full_accuracy: float
    early_exit_fraction: float
    fallback_fraction: float
    mean_compute_per_token: float
    full_compute_per_token: float
    candidate_thresholds: int


@dataclass(frozen=True)
class GateTaskBatch:
    task_loss: torch.Tensor
    router_inputs: Mapping[nn.Module, torch.Tensor]
    token_masks: Mapping[nn.Module, torch.Tensor] | None = None
    routing_stats: Mapping[nn.Module, RoutingStats] | None = None


@dataclass(frozen=True)
class GateTrainingReport:
    steps: int
    mean_task_loss: float
    mean_gate_budget_loss: float
    mean_total_loss: float
    router_health: Mapping[str, Mapping[str, Any]]
    backbone_frozen: bool
    seed: int


def train_frozen_gate_control(model, routers, batches, *, step_fn, learning_rate=1e-3, budget_loss_weight=1.0, seed=0):
    if not callable(step_fn):
        raise ValueError("step_fn must be callable")
    routers = tuple(routers)
    if not routers:
        raise ValueError("routers must be nonempty")
    if any(not callable(getattr(router, "auxiliary_loss", None)) for router in routers):
        raise ValueError("every trained router must expose auxiliary_loss")
    try:
        batch_iterator = iter(batches)
        first_batch = next(batch_iterator)
    except (TypeError, StopIteration) as error:
        raise ValueError("batches must be a nonempty iterable") from error
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    numeric_options = (learning_rate, budget_loss_weight)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in numeric_options
    ):
        raise ValueError("learning_rate and budget_loss_weight must be finite numeric values")
    if learning_rate <= 0 or budget_loss_weight < 0:
        raise ValueError("learning_rate must be positive and budget_loss_weight nonnegative")
    gate_parameters = freeze_backbone_train_gates(model, routers)
    gate_ids = {id(p) for p in gate_parameters}
    optimizer = torch.optim.AdamW(gate_parameters, lr=learning_rate)
    metrics: dict[str, list[float]] = {"task": [], "budget": [], "total": []}
    route_stats: dict[str, list[RoutingStats]] = {f"gate_{i}": [] for i in range(len(routers))}
    modules = list(model.modules())
    modes = [m.training for m in modules]
    cuda_devices = {
        p.device.index if p.device.index is not None else torch.cuda.current_device()
        for p in gate_parameters
        if p.device.type == "cuda"
    }
    devices = sorted(cuda_devices)
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for batch in chain((first_batch,), batch_iterator):
                result = step_fn(batch)
                if not isinstance(result, GateTaskBatch):
                    raise TypeError("step_fn must return GateTaskBatch")
                if (
                    not isinstance(result.task_loss, torch.Tensor)
                    or result.task_loss.ndim != 0
                    or not bool(torch.isfinite(result.task_loss))
                ):
                    raise ValueError("task_loss must be a finite scalar tensor")
                router_ids = {id(router) for router in routers}
                if {id(router) for router in result.router_inputs} != router_ids:
                    raise ValueError("router_inputs must map exactly the trained routers")
                if result.token_masks is not None and any(
                    id(router) not in router_ids for router in result.token_masks
                ):
                    raise ValueError("token_masks contains an untrained router")
                if result.routing_stats is not None and any(
                    id(router) not in {id(trained) for trained in routers}
                    for router in result.routing_stats
                ):
                    raise ValueError("routing_stats contains an untrained router")
                budget_losses = [
                    router.auxiliary_loss(
                        result.router_inputs[router],
                        token_mask=(result.token_masks or {}).get(router),
                    )
                    for router in routers
                ]
                if any(
                    not isinstance(loss, torch.Tensor)
                    or loss.ndim != 0
                    or not bool(torch.isfinite(loss))
                    for loss in budget_losses
                ):
                    raise ConditionalComputeError(
                        "router auxiliary losses must be finite scalars"
                    )
                budget = torch.stack(budget_losses).mean()
                total = result.task_loss + budget_loss_weight * budget
                if not bool(torch.isfinite(total)):
                    raise ValueError("total loss must be finite")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if any(p.grad is not None for p in model.parameters() if id(p) not in gate_ids):
                    raise ConditionalComputeError("frozen backbone received gradients")
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in gate_parameters):
                    raise ConditionalComputeError("gate gradients missing or non-finite")
                optimizer.step()
                metrics["task"].append(float(result.task_loss.detach()))
                metrics["budget"].append(float(budget.detach()))
                metrics["total"].append(float(total.detach()))
                for i, router in enumerate(routers):
                    stat = (result.routing_stats or {}).get(router)
                    if stat is not None:
                        route_stats[f"gate_{i}"].append(stat)
    finally:
        for module, training in zip(modules, modes):
            module.train(training)
    return GateTrainingReport(
        len(metrics["task"]), statistics.fmean(metrics["task"]), statistics.fmean(metrics["budget"]),
        statistics.fmean(metrics["total"]), {key: routing_health(value) for key, value in route_stats.items()},
        all(not p.requires_grad for p in model.parameters() if id(p) not in gate_ids), seed,
    )


def confidence_scores(logits, *, kind="margin"):
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("logits require at least two classes")
    flat = logits.reshape(-1, logits.shape[-1]).float()
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("logits must be finite")
    if kind == "margin":
        top = torch.topk(flat, 2, dim=-1).values
        scores = top[:, 0] - top[:, 1]
    elif kind == "max_probability":
        scores = flat.softmax(dim=-1).amax(dim=-1)
    else:
        raise ValueError("unsupported confidence kind")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("confidence score must be finite")
    return scores


class SelectiveDecodeVerifier(Protocol):
    supports_selective_decode: bool
    def __call__(self, hidden_states, selected_indices, **context): ...


class AdaptiveEarlyExit(nn.Module):
    """Early exit only for independent one-token rows without mutable cache state."""

    def __init__(self, threshold, *, score_kind="margin"):
        super().__init__()
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or math.isnan(threshold) or threshold == -math.inf:
            raise ValueError("threshold must be numeric or +infinity")
        if score_kind not in {"margin", "max_probability"}:
            raise ValueError("unsupported score kind")
        self.threshold = float(threshold)
        self.score_kind = score_kind
        self.last_exit_stats = None

    def forward(self, early_logits, hidden_states, *, verifier, single_token_decode, cache=None, return_stats=False, **context):
        if not single_token_decode:
            raise ConditionalComputeError("prefill/sequence early exit has no verified causal adapter")
        if early_logits.ndim not in {2, 3} or hidden_states.ndim not in {2, 3}:
            raise ConditionalComputeError("early exit needs independent rows or one-token sequences")
        if (early_logits.ndim == 3 and early_logits.shape[1] != 1) or (hidden_states.ndim == 3 and hidden_states.shape[1] != 1):
            raise ConditionalComputeError("cannot flatten multi-token causal sequences")
        if cache is not None or _has_mutable_cache_state(context):
            raise ConditionalComputeError("cache/Mamba state has no selective-update adapter")
        if not early_logits.is_floating_point():
            raise ConditionalComputeError("early logits must be floating point")
        if early_logits.device != hidden_states.device:
            raise ValueError("logits and hidden states must be on the same device")
        logits = early_logits.reshape(-1, early_logits.shape[-1])
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if logits.shape[0] != hidden.shape[0]:
            raise ValueError("logit/hidden row counts differ")
        if logits.shape[0] == 0:
            raise ValueError("early exit requires at least one row")
        scores = confidence_scores(logits, kind=self.score_kind)
        exits = scores >= self.threshold
        indices = (~exits).nonzero(as_tuple=False).squeeze(-1)
        output = logits.clone()
        if indices.numel():
            if getattr(verifier, "supports_selective_decode", False) is not True:
                raise ConditionalComputeError("verifier lacks selective-decode support")
            deep = verifier(hidden.index_select(0, indices), indices, cache=cache, **context)
            if not isinstance(deep, torch.Tensor) or deep.shape != (indices.numel(), logits.shape[-1]) or deep.device != logits.device or deep.dtype != logits.dtype:
                raise ConditionalComputeError("verifier returned invalid shape/device/dtype")
            if not bool(torch.isfinite(deep).all()):
                raise ConditionalComputeError("verifier returned non-finite logits")
            output = output.index_copy(0, indices, deep)
        stats = ExitStats(logits.shape[0], int(exits.sum()), int(indices.numel()), int(indices.numel()), float(scores.mean()), self.threshold)
        self.last_exit_stats = stats
        result = output.reshape_as(early_logits)
        return (result, stats) if return_stats else result


class IntermediatePredictionHead(nn.Module):
    """Auxiliary vocab head; logits are not calibrated automatically."""

    def __init__(self, hidden_size, vocab_size, *, shared_output_projection=None):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or hidden_size <= 0 or vocab_size <= 1:
            raise ValueError("invalid head dimensions")
        self.projection = shared_output_projection if shared_output_projection is not None else nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size, self.vocab_size = hidden_size, vocab_size

    def forward(self, hidden_states):
        logits = self.projection(hidden_states)
        if logits.shape[-1] != self.vocab_size:
            raise ConditionalComputeError("head returned wrong vocabulary dimension")
        return logits


def intermediate_head_loss(logits, labels, *, ignore_index=-100):
    if logits.ndim < 2 or logits.shape[:-1] != labels.shape:
        raise ValueError("labels shape mismatch")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating point")
    if labels.is_complex() or labels.dtype is torch.bool or (labels.is_floating_point() and (not bool(torch.isfinite(labels).all()) or not bool((labels == labels.round()).all()))):
        raise ValueError("labels must contain integer class ids")
    target = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    active = target != ignore_index
    if not bool(active.any()):
        return logits.float().sum() * 0.0
    if bool(((target[active] < 0) | (target[active] >= logits.shape[-1])).any()):
        raise ValueError("label outside vocabulary range")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target, ignore_index=ignore_index)


def freeze_backbone_train_gates(model, routers):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    routers = tuple(routers)
    if not routers or any(
        not isinstance(router, nn.Module) for router in routers
    ):
        raise ValueError("routers must contain model-owned modules")
    router_ids = {id(router) for router in routers}
    if len(router_ids) != len(routers):
        raise ValueError("routers must not contain duplicates")
    model_module_ids = {id(module) for module in model.modules()}
    if any(id(router) not in model_module_ids for router in routers):
        raise ValueError("every router must already belong to the supplied model")
    gate_ids = {id(parameter) for router in routers for parameter in router.parameters()}
    params = tuple(parameter for parameter in model.parameters() if id(parameter) in gate_ids)
    if not params:
        raise ValueError("routers have no parameters")
    # Validate everything before changing requires_grad on the caller's model.
    model.requires_grad_(False)
    for router in routers:
        router.requires_grad_(True)
    return params


def calibrate_exit_threshold(early_logits, deep_logits, labels, *, early_compute_per_token, full_compute_per_token, minimum_accuracy, score_kind="margin"):
    if (
        early_logits.ndim != 2
        or deep_logits.ndim != 2
        or early_logits.shape != deep_logits.shape
    ):
        raise ValueError("expected matching 2D logits")
    if early_logits.shape[0] == 0 or early_logits.shape[-1] < 2:
        raise ValueError("calibration requires examples and at least two classes")
    if early_logits.device != deep_logits.device:
        raise ValueError("early and deep logits must be on the same device")
    if not early_logits.is_floating_point() or not deep_logits.is_floating_point() or not bool(torch.isfinite(early_logits).all()) or not bool(torch.isfinite(deep_logits).all()):
        raise ValueError("logits must contain finite floating-point values")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.numel() != early_logits.shape[0]
        or labels.numel() == 0
    ):
        raise ValueError("one nonempty label per example is required")
    calibration_values = (minimum_accuracy, early_compute_per_token, full_compute_per_token)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in calibration_values
    ):
        raise ValueError("calibration constraints must be finite numeric values")
    if (
        not 0 <= minimum_accuracy <= 1
        or early_compute_per_token <= 0
        or full_compute_per_token < early_compute_per_token
    ):
        raise ValueError("invalid calibration constraints")
    scores = confidence_scores(early_logits, kind=score_kind)
    early_pred, deep_pred = early_logits.argmax(-1), deep_logits.argmax(-1)
    if labels.is_complex() or labels.dtype is torch.bool or (labels.is_floating_point() and (not bool(torch.isfinite(labels).all()) or not bool((labels == labels.round()).all()))):
        raise ValueError("labels must be integer class ids")
    target = labels.to(early_logits.device).long()
    if bool(((target < 0) | (target >= early_logits.shape[-1])).any()):
        raise ValueError("labels outside logits classes")
    candidates = torch.cat((torch.unique(scores, sorted=True), scores.new_tensor([float("inf")])))
    best = None
    full_accuracy = float((deep_pred == target).float().mean())
    for candidate in candidates:
        exits = scores >= candidate
        accuracy = float((torch.where(exits, early_pred, deep_pred) == target).float().mean())
        fraction = float(exits.float().mean())
        compute = early_compute_per_token + (1 - fraction) * (full_compute_per_token - early_compute_per_token)
        row = (compute, float(candidate), accuracy, fraction)
        if accuracy + 1e-12 >= minimum_accuracy and (best is None or row[:2] < best[:2]):
            best = row
    if best is None:
        raise ConditionalComputeError("no threshold meets dev accuracy floor")
    compute, threshold, accuracy, fraction = best
    return ExitCalibration(threshold, accuracy, full_accuracy, fraction, 1 - fraction, compute, full_compute_per_token, len(candidates))


def pareto_frontier(points: Sequence[Mapping[str, float]], *, compute_key="compute_per_token", quality_key="quality"):
    normalized = []
    for point in points:
        compute, quality = float(point[compute_key]), float(point[quality_key])
        if not math.isfinite(compute) or not math.isfinite(quality) or compute < 0:
            raise ValueError("Pareto metrics must be finite with nonnegative compute")
        normalized.append({**dict(point), compute_key: compute, quality_key: quality})
    ordered = sorted(normalized, key=lambda row: (row[compute_key], -row[quality_key]))
    frontier, best = [], -math.inf
    for row in ordered:
        if row[quality_key] > best:
            frontier.append(row)
            best = row[quality_key]
    return frontier


def render_pareto_svg(points, path, *, compute_key="compute_per_token", quality_key="quality", width=720, height=440):
    if (
        not points
        or isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width < 320
        or height < 240
    ):
        raise ValueError("points and integer SVG dimensions of at least 320x240 are required")
    frontier = pareto_frontier(points, compute_key=compute_key, quality_key=quality_key)
    xs, ys = [float(p[compute_key]) for p in points], [float(p[quality_key]) for p in points]
    left, right, top, bottom = 78, width - 28, 36, height - 70
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax = xmin + 1
    if ymin == ymax: ymax = ymin + 1
    def project(x, y):
        return (left + (x - xmin) / (xmax - xmin) * (right - left), bottom - (y - ymin) / (ymax - ymin) * (bottom - top))
    compute_label = html.escape(str(compute_key))
    quality_label = html.escape(str(quality_key))
    dots = []
    for point in points:
        x, y = float(point[compute_key]), float(point[quality_key])
        px, py = project(x, y)
        label = html.escape(str(point.get("variant", "measurement")), quote=True)
        dots.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="#276ef1"><title>{label}: compute={x:g}, quality={y:g}</title></circle>')
    coords = " ".join(f"{project(float(p[compute_key]), float(p[quality_key]))[0]:.2f},{project(float(p[compute_key]), float(p[quality_key]))[1]:.2f}" for p in frontier)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#253044"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#253044"/>'
        f'<text x="{(left + right)/2:.1f}" y="{height-20}" text-anchor="middle">{compute_label}</text>'
        f'<text x="18" y="{(top+bottom)/2:.1f}" transform="rotate(-90 18 {(top+bottom)/2:.1f})" text-anchor="middle">{quality_label}</text>'
        f'<polyline points="{coords}" fill="none" stroke="#d04a02" stroke-width="2"/>' + "".join(dots) + "</svg>\n"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(svg)
    return str(output)


def routing_health(stats: Sequence[RoutingStats], *, collapse_floor=0.01):
    if (
        isinstance(collapse_floor, bool)
        or not isinstance(collapse_floor, (int, float))
        or not math.isfinite(collapse_floor)
        or not 0 <= collapse_floor < 0.5
    ):
        raise ValueError("collapse_floor must be finite and in [0, 0.5)")
    tokens = valid = selected = executed = 0
    for stat in stats:
        if not isinstance(stat, RoutingStats):
            raise TypeError("stats must contain RoutingStats values")
        vt = stat.valid_tokens if stat.valid_tokens is not None else stat.tokens
        counts = (stat.tokens, vt, stat.selected_tokens, stat.executed_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("routing counts must be integers")
        if min(counts) < 0 or vt > stat.tokens or stat.selected_tokens > vt or stat.executed_tokens > stat.tokens:
            raise ValueError("invalid routing counts")
        fractions = (stat.selected_fraction, stat.executed_fraction)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in fractions
        ):
            raise ValueError("routing fractions must be finite numbers")
        if not 0 <= stat.selected_fraction <= 1 or stat.executed_fraction < 0:
            raise ValueError("routing fractions are outside valid ranges")
        if not isinstance(stat.dense_fallback, bool):
            raise ValueError("dense_fallback must be boolean")
        if stat.fallback_reason is not None and not isinstance(stat.fallback_reason, str):
            raise ValueError("fallback_reason must be a string or null")
        expected_selected = stat.selected_tokens / vt if vt else 0.0
        expected_executed = stat.executed_tokens / vt if vt else 0.0
        if not math.isclose(stat.selected_fraction, expected_selected, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
            stat.executed_fraction, expected_executed, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("routing fractions differ from counts")
        tokens += stat.tokens
        valid += vt
        selected += stat.selected_tokens
        executed += stat.executed_tokens
    fraction = selected / valid if valid else 0.0
    collapsed = bool(valid and (fraction <= collapse_floor or fraction >= 1 - collapse_floor))
    return {
        "calls": len(stats),
        "tokens": tokens,
        "valid_tokens": valid,
        "selected_tokens": selected,
        "executed_tokens": executed,
        "selected_fraction": fraction,
        "executed_fraction": executed / valid if valid else 0.0,
        "dense_fallback_calls": sum(stat.dense_fallback for stat in stats),
        "gate_collapsed": collapsed,
        "collapse_mode": (
            "always_off"
            if valid and fraction <= collapse_floor
            else "always_on"
            if valid and fraction >= 1 - collapse_floor
            else None
        ),
    }
