"""Campaign manifest for the Qwen3.8 native-sparse program (Track F).

Binds the program's fixed identities (primary parent, control, comparison
parents, protected suite version, lineage policy, sparse/MoE target) and its
promotion-grade run policy (training engine, recursive-repair bounds,
promotion rules, GPU-hour budget) into one hashed, versioned object -- so a
real campaign run can prove, after the fact, exactly which parents/suite/
policy it used, and cannot silently start with recursive repair disabled or
left at accidental defaults.

This module does not reimplement anything ``ProjectSpec``,
``RecursiveRepairPolicy``, or ``backend_selection`` already validate; it
binds their real instances together and adds the identity/lineage fields
those types have no reason to know about.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .backend_selection import SUPPORTED_PEFT_ENGINES
from .recursive_repair import RecursiveRepairPolicy


class CampaignManifestError(ValueError):
    """Raised when a campaign manifest does not describe a safe, real run."""


def _non_empty_str(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignManifestError(f"{label} must be a non-empty string")
    return value


def _pinned_revision(value: str, *, label: str) -> str:
    value = _non_empty_str(value, label=label)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise CampaignManifestError(
            f"{label} must be a full 40-character git commit SHA, not a "
            f"branch name or short hash: {value!r}"
        )
    return value


def _sha256_hex(value: str, *, label: str) -> str:
    value = _non_empty_str(value, label=label)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise CampaignManifestError(f"{label} must be a 64-character SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class ParentPin:
    """One parent checkpoint, pinned to an exact, non-moving revision."""

    repo: str
    revision: str
    role: str

    def __post_init__(self) -> None:
        _non_empty_str(self.repo, label="parent repo")
        _pinned_revision(self.revision, label=f"parent {self.repo} revision")
        _non_empty_str(self.role, label="parent role")

    def to_dict(self) -> dict[str, str]:
        return {"repo": self.repo, "revision": self.revision, "role": self.role}


@dataclass(frozen=True)
class SuiteVersionPin:
    """The frozen protected-evaluation-suite content this campaign scores against."""

    name: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _non_empty_str(self.name, label="suite version name")
        _sha256_hex(self.manifest_sha256, label="suite manifest_sha256")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class Qwen38CampaignManifest:
    """The full binding for one Qwen3.8 native-sparse campaign generation.

    Deliberately requires a non-empty repair corpus and repair variant list:
    a campaign manifest with no repair configuration would let
    ``run_project`` silently degrade into a plain train -> evaluate -> stop
    run the first time a candidate is rejected, which is exactly the
    failure mode the Qwen3.8 program directive named as unacceptable.
    """

    primary_parent: ParentPin
    native_control: ParentPin
    comparison_parents: tuple[ParentPin, ...]
    suite_version: SuiteVersionPin
    training_engine: str
    repair_policy: RecursiveRepairPolicy
    repair_corpus_files: tuple[str, ...]
    repair_variant_names: tuple[str, ...]
    gpu_hour_budget: float
    minimum_promotion_gain: float
    program: str = "chowder-qwen3.8-native-sparse"
    target_architecture: str = "sparse_moe"
    desired_active_parameters_min_b: float = 3.0
    desired_active_parameters_max_b: float = 4.0
    native_qwen3_8_required: bool = True
    distillation_parent_allowed: bool = False
    require_protocol_match: bool = True
    manifest_schema_version: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.native_qwen3_8_required:
            raise CampaignManifestError(
                "native_qwen3_8_required is a hard program rule and cannot be disabled"
            )
        if self.distillation_parent_allowed:
            raise CampaignManifestError(
                "distillation into a different student architecture is never "
                "allowed for this program; distillation_parent_allowed must be False"
            )
        if not self.comparison_parents:
            raise CampaignManifestError("campaign requires at least one comparison parent")
        if self.training_engine not in SUPPORTED_PEFT_ENGINES:
            raise CampaignManifestError(
                f"unsupported training_engine {self.training_engine!r}; expected one of "
                f"{sorted(SUPPORTED_PEFT_ENGINES)}"
            )
        if not self.repair_corpus_files:
            raise CampaignManifestError(
                "campaign manifest requires a non-empty repair_corpus_files -- a "
                "campaign with no repair corpus would silently run as "
                "train -> evaluate -> stop the first time a candidate is rejected"
            )
        if not self.repair_variant_names:
            raise CampaignManifestError(
                "campaign manifest requires at least one repair_variant_names entry"
            )
        if len(self.repair_variant_names) != len(set(self.repair_variant_names)):
            raise CampaignManifestError("repair_variant_names must be unique")
        if not self.require_protocol_match:
            raise CampaignManifestError(
                "a promotion-grade campaign must require_protocol_match=True; "
                "use a plain ProjectSpec run for protocol-relaxed experimentation"
            )
        if self.gpu_hour_budget <= 0:
            raise CampaignManifestError("campaign gpu_hour_budget must be positive")
        if self.minimum_promotion_gain < 0:
            raise CampaignManifestError("campaign minimum_promotion_gain cannot be negative")
        min_b = float(self.desired_active_parameters_min_b)
        max_b = float(self.desired_active_parameters_max_b)
        if not (min_b > 0 and max_b >= min_b):
            raise CampaignManifestError(
                "desired_active_parameters_min_b/max_b must describe a real, "
                "positive, non-inverted range"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "program": self.program,
            "primary_parent": self.primary_parent.to_dict(),
            "native_control": self.native_control.to_dict(),
            "comparison_parents": [pin.to_dict() for pin in self.comparison_parents],
            "lineage_policy": {
                "native_qwen3_8_required": self.native_qwen3_8_required,
                "distillation_parent_allowed": self.distillation_parent_allowed,
            },
            "target": {
                "architecture": self.target_architecture,
                "desired_active_parameters_min_b": self.desired_active_parameters_min_b,
                "desired_active_parameters_max_b": self.desired_active_parameters_max_b,
            },
            "suite_version": self.suite_version.to_dict(),
            "training_engine": self.training_engine,
            "repair_policy": {
                "max_depth": self.repair_policy.max_depth,
                "min_score_improvement": self.repair_policy.min_score_improvement,
                "max_failure_signature_occurrences": (
                    self.repair_policy.max_failure_signature_occurrences
                ),
                "replay_ratio": self.repair_policy.replay_ratio,
            },
            "repair_corpus_files": list(self.repair_corpus_files),
            "repair_variant_names": list(self.repair_variant_names),
            "promotion_rules": {
                "gpu_hour_budget": self.gpu_hour_budget,
                "minimum_promotion_gain": self.minimum_promotion_gain,
                "require_protocol_match": self.require_protocol_match,
            },
            "extra": dict(self.extra),
        }

    def manifest_sha256(self) -> str:
        raw = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def default_qwen38_campaign_manifest(
    *,
    repair_corpus_files: tuple[str, ...],
    repair_variant_names: tuple[str, ...],
    gpu_hour_budget: float,
    minimum_promotion_gain: float = 0.0,
    training_engine: str = "unsloth",
    repair_policy: RecursiveRepairPolicy | None = None,
) -> Qwen38CampaignManifest:
    """The real, current identity binding for this program.

    Revisions and the suite manifest digest are the exact values recorded in
    docs/QWEN38_SPARSE_PROGRAM.md's pinned-revision table and verified suite
    freeze -- not placeholders. Training engine defaults to Unsloth per the
    program directive's own stated primary blocker (make Unsloth able to
    execute the real recursive-repair semantics, not just a standalone LoRA).

    ``repair_corpus_files``, ``repair_variant_names``, and
    ``gpu_hour_budget`` are required keyword arguments rather than fields
    this factory could default to something empty -- a manifest is only
    ever valid with real, non-empty repair configuration (see
    ``Qwen38CampaignManifest.__post_init__``), so there is no safe default
    to fall back to here.
    """

    return Qwen38CampaignManifest(
        primary_parent=ParentPin(
            repo="orcarouter/Qwen3.8-27B-Uncensored",
            revision="404ea47aaa5d8a8b00049c9e9750089aca011ab2",
            role="primary",
        ),
        native_control=ParentPin(
            repo="Qwen/Qwen3.8-27B",
            revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            role="control",
        ),
        comparison_parents=(
            ParentPin(
                repo="OBLITERATUS/Qwen3.8-27B-OBLITERATED",
                revision="a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8",
                role="comparison",
            ),
            ParentPin(
                repo=(
                    "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-"
                    "Heretic-Uncensored-NM-DAU"
                ),
                revision="81c73940f94023f7d64e3ae6abcc653fc837d415",
                role="comparison",
            ),
        ),
        suite_version=SuiteVersionPin(
            name="v1",
            manifest_sha256=(
                "7946d8c9b14356b5f99c4de3a6a32aef8dbcd7c0ede7180a3bf569bb2ba8473c"
            ),
        ),
        training_engine=training_engine,
        repair_policy=repair_policy if repair_policy is not None else RecursiveRepairPolicy(),
        repair_corpus_files=repair_corpus_files,
        repair_variant_names=repair_variant_names,
        gpu_hour_budget=gpu_hour_budget,
        minimum_promotion_gain=minimum_promotion_gain,
    )
