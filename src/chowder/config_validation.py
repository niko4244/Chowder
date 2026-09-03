from __future__ import annotations

import math
from typing import Any, Mapping


TRANSFORMERS_BACKEND_SCHEMA_VERSION = 1


class ConfigValidationError(ValueError):
    """Raised when a resolved execution config is structurally invalid."""


def _mapping(value: Any, *, path: str, allow_none: bool = False) -> Mapping[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{path} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ConfigValidationError(
            f"{path} contains unsupported key(s): {', '.join(unknown)}"
        )


def _finite_nonnegative(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{path} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ConfigValidationError(f"{path} must be finite and non-negative")
    return number


def _accelerator_count(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{path} must be an integer")
    if value < 0:
        raise ConfigValidationError(f"{path} cannot be negative")
    return value


def validate_transformers_backend_config(config: Mapping[str, Any]) -> None:
    """Strictly validate the resolved Transformers/PEFT backend namespace.

    The rest of a Chowder experiment may legitimately contain research metadata
    (`repair`, evaluator state, provenance, etc.), so this function intentionally
    validates only `backend`. Within that namespace unknown keys fail closed: a
    typo must never silently fall through to a default training parameter.
    """

    if not isinstance(config, Mapping):
        raise ConfigValidationError("resolved config must be a mapping")
    backend = _mapping(config.get("backend"), path="backend")
    _reject_unknown(
        backend,
        {
            "schema_version",
            "type",
            "base_model",
            "dataset",
            "dataset_sha256",
            "revision",
            "dataset_format",
            "text_field",
            "messages_field",
            "max_length",
            "quantization",
            "precision",
            "trust_remote_code",
            "offline",
            "training",
            "lora",
            "runtime",
            "replay",
            "parent_adapter",
            "profile",
            "resume_from_checkpoint",
            "min_free_disk_gb",
            "memory_preflight",
        },
        path="backend",
    )

    if "min_free_disk_gb" in backend:
        _finite_nonnegative(backend["min_free_disk_gb"], path="backend.min_free_disk_gb")

    schema_version = backend.get("schema_version", TRANSFORMERS_BACKEND_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ConfigValidationError("backend.schema_version must be an integer")
    if schema_version != TRANSFORMERS_BACKEND_SCHEMA_VERSION:
        raise ConfigValidationError(
            f"unsupported backend.schema_version {schema_version}; "
            f"expected {TRANSFORMERS_BACKEND_SCHEMA_VERSION}"
        )

    backend_type = backend.get("type", "transformers-peft")
    if backend_type != "transformers-peft":
        raise ConfigValidationError(
            f"unsupported backend.type for TransformersPeftExecutor: {backend_type!r}"
        )

    training = _mapping(backend.get("training", {}), path="backend.training")
    _reject_unknown(
        training,
        {
            "epochs",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "warmup_steps",
            "weight_decay",
            "max_grad_norm",
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "gradient_checkpointing",
            "activation_offload",
            "optimizer_tiering",
            "frozen_layer_streaming",
            "detailed_timing_telemetry",
            "save_strategy",
            "save_steps",
            "save_total_limit",
        },
        path="backend.training",
    )

    lora = _mapping(backend.get("lora", {}), path="backend.lora")
    _reject_unknown(
        lora,
        {"r", "alpha", "dropout", "target_modules", "target_preset", "use_rslora"},
        path="backend.lora",
    )
    if "target_modules" in lora:
        modules = lora["target_modules"]
        if not isinstance(modules, (list, tuple)) or not modules:
            raise ConfigValidationError(
                "backend.lora.target_modules must be a non-empty list/tuple"
            )
        if any(not isinstance(item, str) or not item.strip() for item in modules):
            raise ConfigValidationError(
                "backend.lora.target_modules entries must be non-empty strings"
            )

    runtime = _mapping(backend.get("runtime", {}), path="backend.runtime")
    _reject_unknown(
        runtime,
        {"timeout_seconds", "active_accelerator_count"},
        path="backend.runtime",
    )
    if "active_accelerator_count" in runtime:
        _accelerator_count(
            runtime["active_accelerator_count"],
            path="backend.runtime.active_accelerator_count",
        )

    replay = _mapping(backend.get("replay", {}), path="backend.replay", allow_none=True)
    _reject_unknown(
        replay,
        {"dataset", "sha256", "ratio", "manifest", "manifest_sha256"},
        path="backend.replay",
    )

    parent_adapter = _mapping(
        backend.get("parent_adapter", {}),
        path="backend.parent_adapter",
        allow_none=True,
    )
    _reject_unknown(
        parent_adapter,
        {"path", "sha256"},
        path="backend.parent_adapter",
    )

    profile = _mapping(backend.get("profile", {}), path="backend.profile", allow_none=True)
    _reject_unknown(
        profile,
        {
            "estimated_steps",
            "seconds_per_step",
            "peak_vram_gb",
            "source",
            "active_accelerator_count",
        },
        path="backend.profile",
    )
    if "active_accelerator_count" in profile:
        _accelerator_count(
            profile["active_accelerator_count"],
            path="backend.profile.active_accelerator_count",
        )
    for key in ("estimated_steps", "seconds_per_step", "peak_vram_gb"):
        if key in profile:
            _finite_nonnegative(profile[key], path=f"backend.profile.{key}")
