from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contamination import ContaminationAudit
from .failures import RepairPlan
from .models import Experiment, Hypothesis
from .provenance import sha256_file


@dataclass(frozen=True)
class VerifiedRepairDataset:
    path: str
    sha256: str
    contamination_audit: ContaminationAudit

    def verify(self) -> Path:
        if not self.contamination_audit.clean:
            raise ValueError("repair dataset contamination audit is not clean")
        if len(self.sha256) != 64:
            raise ValueError("repair dataset SHA-256 is invalid")
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"repair dataset not found: {path}")
        actual = sha256_file(path)
        if actual != self.sha256:
            raise ValueError("repair dataset content changed after verification")
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


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_repair_candidate(
    *,
    parent_id: str,
    plan: RepairPlan,
    dataset: VerifiedRepairDataset,
    variant: RepairVariant,
) -> Experiment:
    """Build one deterministic repair child without changing evaluation semantics.

    Only `backend.dataset`, `backend.training`, and `backend.lora` are patched.
    Base model, revision, precision, quantization, and the entire evaluation
    section therefore remain inherited from the parent lineage.
    """

    if not parent_id.strip():
        raise ValueError("repair candidate parent_id is required")
    dataset_path = dataset.verify()

    training_patch = dict(variant.training_patch)
    lora_patch = dict(variant.lora_patch)
    for mapping, label in ((training_patch, "training"), (lora_patch, "lora")):
        if any(not isinstance(key, str) or not key.strip() for key in mapping):
            raise ValueError(f"repair {label} override keys must be non-empty strings")

    backend_patch: dict[str, Any] = {"dataset": str(dataset_path)}
    if training_patch:
        backend_patch["training"] = training_patch
    if lora_patch:
        backend_patch["lora"] = lora_patch

    repair_evidence = {
        "plan_id": plan.plan_id,
        "cluster_id": plan.cluster_id,
        "repair_dataset_sha256": dataset.sha256,
        "holdout_index_sha256": list(dataset.contamination_audit.holdout_index_sha256),
        "source_failure_ids": list(plan.source_failure_ids),
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
        estimated_gpu_hours=variant.estimated_gpu_hours,
        tags=("repair", f"plan:{plan.plan_id[:12]}", f"variant:{variant.name}"),
    )


def build_repair_population(
    *,
    parent_id: str,
    plan: RepairPlan,
    dataset: VerifiedRepairDataset,
    variants: Iterable[RepairVariant],
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
        )
        for variant in variants
    )
    ids = [candidate.experiment_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("repair variants collapsed to duplicate experiment IDs")
    return candidates
