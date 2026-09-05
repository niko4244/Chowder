from __future__ import annotations

from typing import Any, Mapping

from .executors import TrainingExecutor


TRANSFORMERS_ENGINE = "transformers"
UNSLOTH_ENGINE = "unsloth"
SUPPORTED_PEFT_ENGINES = frozenset({TRANSFORMERS_ENGINE, UNSLOTH_ENGINE})


class BackendSelectionError(ValueError):
    """Raised when a project does not select a supported training engine safely."""


def _backend(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise BackendSelectionError("resolved config must be a mapping")
    backend = config.get("backend")
    if not isinstance(backend, Mapping):
        raise BackendSelectionError("resolved config must contain a backend mapping")
    return backend


def resolve_training_engine(config: Mapping[str, Any]) -> str:
    """Resolve the training implementation without changing project semantics.

    ``backend.type='transformers-peft'`` is the historical public spelling and
    remains an alias for the Transformers engine. New projects may use the
    framework-neutral ``backend.type='peft'`` plus an explicit engine. Unsloth
    is recognized here so schema/dispatch evolution has one stable seam, but
    its executor is intentionally not constructed until the isolated runtime
    slice lands.
    """

    backend = _backend(config)
    backend_type = str(backend.get("type", "transformers-peft")).strip().lower()
    raw_engine = backend.get("engine")

    if backend_type == "transformers-peft":
        if raw_engine is None:
            return TRANSFORMERS_ENGINE
        engine = str(raw_engine).strip().lower()
        if engine != TRANSFORMERS_ENGINE:
            raise BackendSelectionError(
                "legacy backend.type='transformers-peft' can only use "
                "backend.engine='transformers'; use backend.type='peft' for other engines"
            )
        return TRANSFORMERS_ENGINE

    if backend_type != "peft":
        raise BackendSelectionError(f"unsupported backend.type: {backend_type!r}")

    if raw_engine is None:
        raise BackendSelectionError(
            "backend.engine is required when backend.type='peft'; "
            "choose 'transformers' or 'unsloth' explicitly"
        )
    engine = str(raw_engine).strip().lower()
    if engine not in SUPPORTED_PEFT_ENGINES:
        raise BackendSelectionError(
            f"unsupported PEFT training engine {engine!r}; expected one of "
            f"{sorted(SUPPORTED_PEFT_ENGINES)}"
        )
    return engine


def normalize_training_config_for_executor(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt canonical PEFT selection to an executor's existing config contract.

    The Transformers executor predates ``backend.engine`` and still validates
    ``backend.type='transformers-peft'`` strictly. Keep that mature executor
    unchanged for the first engine-selection slice: canonical
    ``type='peft', engine='transformers'`` is normalized at the orchestration
    boundary. Unsloth configs are deliberately left untouched for its future
    executor, which must see and record its real engine identity.
    """

    engine = resolve_training_engine(config)
    normalized = dict(config)
    backend = dict(_backend(config))
    if engine == TRANSFORMERS_ENGINE:
        backend["type"] = "transformers-peft"
        backend.pop("engine", None)
    normalized["backend"] = backend
    return normalized


def create_training_executor(config: Mapping[str, Any]) -> TrainingExecutor:
    """Construct the selected built-in executor.

    Engine selection is explicit; there is intentionally no ``auto`` mode
    until Chowder has apples-to-apples evidence for both implementations.
    """

    engine = resolve_training_engine(config)
    if engine == TRANSFORMERS_ENGINE:
        from .backends.transformers_peft import TransformersPeftExecutor

        return TransformersPeftExecutor()
    if engine == UNSLOTH_ENGINE:
        from .backends.unsloth_peft import UnslothPeftExecutor

        return UnslothPeftExecutor()
    raise AssertionError(f"unhandled training engine: {engine}")
