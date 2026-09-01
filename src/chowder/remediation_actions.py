from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class RemediationActionKind(str, Enum):
    RETRY_DOWNLOAD = "retry_download"
    PIN_DEPENDENCY = "pin_dependency"
    SET_ENVIRONMENT = "set_environment"
    SET_RUNTIME_OPTION = "set_runtime_option"
    SET_TRAINING_PARAMETER = "set_training_parameter"
    RESOLVE_ARTIFACT_PATH = "resolve_artifact_path"
    REPROVISION_HARDWARE = "reprovision_hardware"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    RESYNC_ARTIFACT = "resync_artifact"


@dataclass(frozen=True)
class RemediationAction:
    kind: RemediationActionKind
    parameters: Mapping[str, Any]
    source_patch: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("remediation action parameters cannot be empty")
        if not self.source_patch:
            raise ValueError("remediation action source_patch cannot be empty")


class UnsupportedRemediationAction(ValueError):
    pass


@runtime_checkable
class RemediationAdapter(Protocol):
    name: str
    capabilities: frozenset[RemediationActionKind]

    def apply(
        self,
        action: RemediationAction,
        *,
        resolved_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


def action_from_config_patch(config_patch: Mapping[str, Any]) -> RemediationAction:
    """Compile one legacy Investigator patch into a typed action.

    The current Investigator generator is intentionally frozen for benchmark
    comparability, so this translation happens *after* hypothesis generation.
    Unknown or multi-action patches fail closed instead of being applied as
    arbitrary dictionaries.
    """

    if not isinstance(config_patch, Mapping) or not config_patch:
        raise UnsupportedRemediationAction("remediation patch must be a non-empty mapping")
    if len(config_patch) != 1:
        raise UnsupportedRemediationAction(
            "remediation patches must resolve to exactly one typed action"
        )
    key, value = next(iter(config_patch.items()))
    if not isinstance(key, str) or not key:
        raise UnsupportedRemediationAction("remediation patch key must be a non-empty string")

    if key == "resume_download" and bool(value):
        return RemediationAction(
            RemediationActionKind.RETRY_DOWNLOAD,
            {"resume": True},
            dict(config_patch),
        )
    if key == "transformers_version":
        return RemediationAction(
            RemediationActionKind.PIN_DEPENDENCY,
            {"package": "transformers", "version": str(value)},
            dict(config_patch),
        )
    if key == "kernel_metadata.machine_shape":
        return RemediationAction(
            RemediationActionKind.REPROVISION_HARDWARE,
            {"machine_shape": str(value)},
            dict(config_patch),
        )
    if key == "allocator_conf":
        return RemediationAction(
            RemediationActionKind.SET_ENVIRONMENT,
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": str(value)},
            dict(config_patch),
        )
    if key in {"cudnn_enabled", "device_map"}:
        return RemediationAction(
            RemediationActionKind.SET_RUNTIME_OPTION,
            {"name": key, "value": value},
            dict(config_patch),
        )
    if key in {"max_length", "batch_size", "gradient_accumulation_steps"}:
        return RemediationAction(
            RemediationActionKind.SET_TRAINING_PARAMETER,
            {"name": key, "value": value},
            dict(config_patch),
        )
    if key == "dataset_path_resolution":
        return RemediationAction(
            RemediationActionKind.RESOLVE_ARTIFACT_PATH,
            {"strategy": str(value)},
            dict(config_patch),
        )
    if key == "resync_kernel_dataset":
        return RemediationAction(
            RemediationActionKind.RESYNC_ARTIFACT,
            {"enabled": bool(value)},
            dict(config_patch),
        )
    if key == "checkpoint_ref":
        return RemediationAction(
            RemediationActionKind.RESTORE_CHECKPOINT,
            {"checkpoint_ref": str(value)},
            dict(config_patch),
        )

    raise UnsupportedRemediationAction(
        f"no typed remediation action is registered for patch key {key!r}"
    )


def require_capability(
    adapter: RemediationAdapter,
    action: RemediationAction,
) -> None:
    if action.kind not in adapter.capabilities:
        raise UnsupportedRemediationAction(
            f"adapter {adapter.name!r} does not support remediation action {action.kind.value!r}"
        )


@dataclass(frozen=True)
class ConfigRemediationAdapter:
    """Pure config transformer for remediation actions Chowder can apply locally.

    It deliberately cannot install packages, redownload files, restore external
    checkpoints, or reprovision cloud accelerators. Those operations require a
    provider/runtime-specific adapter instead of being faked as config edits.
    """

    name: str = "config-remediation"
    capabilities: frozenset[RemediationActionKind] = frozenset(
        {
            RemediationActionKind.SET_ENVIRONMENT,
            RemediationActionKind.SET_RUNTIME_OPTION,
            RemediationActionKind.SET_TRAINING_PARAMETER,
            RemediationActionKind.RESOLVE_ARTIFACT_PATH,
        }
    )

    def apply(
        self,
        action: RemediationAction,
        *,
        resolved_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        require_capability(self, action)
        config = _deep_copy(resolved_config)
        backend = config.setdefault("backend", {})
        if not isinstance(backend, dict):
            raise ValueError("backend config must be a mapping")

        if action.kind is RemediationActionKind.SET_ENVIRONMENT:
            runtime = backend.setdefault("runtime", {})
            if not isinstance(runtime, dict):
                raise ValueError("backend.runtime must be a mapping")
            env = runtime.setdefault("environment", {})
            if not isinstance(env, dict):
                raise ValueError("backend.runtime.environment must be a mapping")
            env[str(action.parameters["name"])] = action.parameters["value"]
        elif action.kind is RemediationActionKind.SET_RUNTIME_OPTION:
            runtime = backend.setdefault("runtime", {})
            if not isinstance(runtime, dict):
                raise ValueError("backend.runtime must be a mapping")
            runtime[str(action.parameters["name"])] = action.parameters["value"]
        elif action.kind is RemediationActionKind.SET_TRAINING_PARAMETER:
            name = str(action.parameters["name"])
            value = action.parameters["value"]
            if name == "max_length":
                backend["max_length"] = value
            else:
                training = backend.setdefault("training", {})
                if not isinstance(training, dict):
                    raise ValueError("backend.training must be a mapping")
                training[name] = value
        elif action.kind is RemediationActionKind.RESOLVE_ARTIFACT_PATH:
            runtime = backend.setdefault("runtime", {})
            if not isinstance(runtime, dict):
                raise ValueError("backend.runtime must be a mapping")
            runtime["dataset_path_resolution"] = action.parameters["strategy"]
        return config


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy(item) for item in value)
    return value
