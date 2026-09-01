from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contamination import ContaminationAudit, repair_dataset_index_digest
from .failures import RepairPlan
from .models import Experiment, Hypothesis
from .provenance import sha256_file
from .repair_sources import verify_source_manifest


@dataclass(frozen=True)
class VerifiedRepairDataset:
    path: str
    sha256: str
    contamination_audit: ContaminationAudit
    source_manifest_path: str | None = None
    source_manifest_sha256: str | None = None

    def verify(self, *, require_source_manifest: bool = False) -> Path:
        if not self.contamination_audit.clean:
            raise ValueError("repair dataset contamination audit is not clean")
        if len(self.sha256) != 64:
            raise ValueError("repair dataset SHA-256 is invalid")
        if len(self.contamination_audit.repair_index_sha256) != 64:
            raise ValueError(
                "repair contamination audit is missing repair-example identity"
            )
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"repair dataset not found: {path}")
        actual = sha256_file(path)
        if actual != self.sha256:
            raise ValueError("repair dataset content changed after verification")
        actual_repair_index = repair_dataset_index_digest(path)
        if actual_repair_index != self.contamination_audit.repair_index_sha256:
            raise ValueError(
                "repair dataset examples do not match contamination audit"
            )

        has_path = self.source_manifest_path is not None
        has_digest = self.source_manifest_sha256 is not None
        if has_path != has_digest:
            raise ValueError(
                "repair source manifest path and SHA must be supplied together"
            )
        if require_source_manifest and not has_path:
            raise ValueError(
                "autonomous repair requires an independent-source provenance manifest"
            )
        if has_path:
            assert self.source_manifest_path is not None
            assert self.source_manifest_sha256 is not None
            actual_manifest = verify_source_manifest(
                self.source_manifest_path,
                dataset_sha256=self.sha256,
                contamination_audit=self.contamination_audit,
            )
            if actual_manifest != self.source_manifest_sha256:
                raise ValueError(
                    "repair source manifest content changed after verification"
                )
        return path


@dataclass(frozen=True)
class VerifiedReplayDataset:
    """Immutable rehearsal source inherited from an already-trained parent."""

    path: str
    sha256: str
    ratio: float = 1.0

    def verify(self) -> Path:
        if len(self.sha256) != 64:
            raise ValueError("replay dataset SHA-256 is invalid")
        ratio = float(self.ratio)
        if not math.isfinite(ratio) or ratio <= 0 or ratio > 10:
            raise ValueError("replay ratio must be finite and in (0, 10]")
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"replay dataset not found: {path}")
        if sha256_file(path) != self.sha256:
            raise ValueError("replay dataset content changed after verification")
        return path


@dataclass(frozen=True)
class RepairVariant:
    name: str
    estimated_gpu_hours: float
    training_patch: Mapping[str, Any] = field(default_factory=dict)
    lora_patch: Mapping[str, Any] = field(default_factory=dict)
    expected_deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("repair variant name is required")
        if self.estimated_gpu_hours <= 0:
            raise ValueError("repair variant estimated_gpu_hours must be positive")


def replay_compute_multiplier(replay: VerifiedReplayDataset | None) -> float:
    """Conservative compute multiplier for budget admission.

    The worker caps replay rows at the available parent rows, so ``1 + ratio``
    is an upper bound on row-count growth relative to repair-only training.
    Reserving the upper bound is preferable to allowing default rehearsal to
    create a known GPU-hour budget underestimate.
    """

    if replay is None:
        return 1.0
    replay.verify()
    return 1.0 + float(replay.ratio)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_repair_candidate(
    *,
    parent_id: str,
    plan: RepairPlan,
    dataset: VerifiedRepairDataset,
    variant: RepairVariant,
    require_source_manifest: bool = False,
    replay: VerifiedReplayDataset | None = None,
) -> Experiment:
    """Build one deterministic repair child without changing evaluation semantics."""

    if not parent_id.strip():
        raise ValueError("repair candidate parent_id is required")
    dataset_path = dataset.verify(require_source_manifest=require_source_manifest)

    replay_path: Path | None = None
    replay_multiplier = replay_compute_multiplier(replay)
    if replay is not None:
        replay_path = replay.verify()
        if replay_path == dataset_path:
            raise ValueError("repair and replay datasets must be different files")

    training_patch = dict(variant.training_patch)
    lora_patch = dict(variant.lora_patch)
    for mapping, label in ((training_patch, "training"), (lora_patch, "lora")):
        if any(not isinstance(key, str) or not key.strip() for key in mapping):
            raise ValueError(
                f"repair {label} override keys must be non-empty strings"
            )

    backend_patch: dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset.sha256,
    }
    if replay is not None and replay_path is not None:
        backend_patch["replay"] = {
            "dataset": str(replay_path),
            "sha256": replay.sha256,
            "ratio": float(replay.ratio),
        }
    if training_patch:
        backend_patch["training"] = training_patch
    if lora_patch:
        backend_patch["lora"] = lora_patch

    effective_estimate = variant.estimated_gpu_hours * replay_multiplier
    audit = dataset.contamination_audit
    repair_evidence = {
        "plan_id": plan.plan_id,
        "cluster_id": plan.cluster_id,
        "repair_dataset_sha256": dataset.sha256,
        "repair_index_sha256": audit.repair_index_sha256,
        "holdout_index_sha256": list(audit.holdout_index_sha256),
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "source_failure_ids": list(plan.source_failure_ids),
        "replay_dataset_sha256": replay.sha256 if replay is not None else None,
        "replay_ratio": float(replay.ratio) if replay is not None else 0.0,
        "repair_only_estimated_gpu_hours": variant.estimated_gpu_hours,
        "replay_adjusted_estimated_gpu_hours": effective_estimate,
        "variant": variant.name,
    }
    config_patch = {
        "backend": backend_patch,
        "repair": repair_evidence,
    }

    identity = _canonical_digest(
        {
            "parent_id": parent_id,
            "plan_id": plan.plan_id,
            "dataset_sha256": dataset.sha256,
            "repair_index_sha256": audit.repair_index_sha256,
            "holdout_index_sha256": list(audit.holdout_index_sha256),
            "source_manifest_sha256": dataset.source_manifest_sha256,
            "replay_dataset_sha256": replay.sha256 if replay is not None else None,
            "replay_ratio": float(replay.ratio) if replay is not None else 0.0,
            "variant": variant.name,
            "training_patch": training_patch,
            "lora_patch": lora_patch,
            "expected_deltas": dict(variant.expected_deltas),
        }
    )
    experiment_id = f"repair-{identity[:16]}"
    hypothesis = Hypothesis(
        observation=plan.observation,
        suspected_cause=plan.suspected_cause,
        intervention=f"{plan.intervention}; candidate variant={variant.name}",
        expected_deltas=dict(variant.expected_deltas),
    )
    return Experiment(
        experiment_id=experiment_id,
        parent_id=parent_id,
        hypothesis=hypothesis,
        config_patch=config_patch,
        estimated_gpu_hours=effective_estimate,
        tags=("repair", f"plan:{plan.plan_id[:12]}", f"variant:{variant.name}"),
    )


def build_repair_population(
    *,
    parent_id: str,
    plan: RepairPlan,
    dataset: VerifiedRepairDataset,
    variants: Iterable[RepairVariant],
    require_source_manifest: bool = False,
    replay: VerifiedReplayDataset | None = None,
) -> tuple[Experiment, ...]:
    variants = tuple(variants)
    if not variants:
        raise ValueError("at least one repair variant is required")
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("repair variant names must be unique")
    candidates = tuple(
        build_repair_candidate(
            parent_id=parent_id,
            plan=plan,
            dataset=dataset,
            variant=variant,
            require_source_manifest=require_source_manifest,
            replay=replay,
        )
        for variant in variants
    )
    ids = [candidate.experiment_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("repair variants collapsed to duplicate experiment IDs")
    return candidates


def build_autonomous_repair_population(
    *,
    parent_id: str,
    plan: RepairPlan,
    dataset: VerifiedRepairDataset,
    variants: Iterable[RepairVariant],
    replay: VerifiedReplayDataset | None = None,
) -> tuple[Experiment, ...]:
    """Strict autonomous path: source provenance is mandatory."""

    return build_repair_population(
        parent_id=parent_id,
        plan=plan,
        dataset=dataset,
        variants=variants,
        require_source_manifest=True,
        replay=replay,
    )
