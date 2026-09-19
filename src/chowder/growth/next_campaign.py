"""A selected target becomes a frozen campaign, without a human composing it.

The single-generation engine already refuses to run a declaration that is
missing, malformed or self-contradictory. What it never had was a *producer*:
every manifest so far was written by hand, so "autonomous growth" stopped at
"a person writes the next campaign".

This module is that producer, and it is deliberately small. It composes the
per-generation fields (which generation, which target, which budget, which
state root) from a :class:`TargetProposal` and an *immutable* :class:`LoopPolicy`,
and carries everything durable about the parent (dense base identity, the
trusted-ancestor arm, the contamination policy's shape, the promotion policy
version) forward unchanged.

Two properties matter more than the field list:

* **The policy is the ceiling, not a suggestion.** A policy value is never
  taken from the previous campaign's manifest, so a target selector -- or a
  hand-edited declaration -- cannot enlarge the budget, shrink the protected
  set, move the trusted ancestor, or add a training type the policy does not
  allow. A template whose protection or execution configuration disagrees with
  the policy is refused rather than merged: a divergence means the template was
  not the policy's own output.
* **Freezing happens before compute and is write-once.** The declaration and
  its preregistration are written with exclusive create, carrying the policy
  digest, the proposal it came from and a digest over both artifacts. A second
  build into the same generation directory refuses, so thresholds, benchmark
  sets, budgets, the ancestor and the recipe rule cannot change after candidate
  results exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import (
    CampaignBudget,
    CampaignManifest,
    CampaignManifestError,
    EvaluationExecution,
    ProtectionDeclaration,
)
from .campaign_prepare import prepared_input_paths
from .target_selection import TargetProposal

#: Named refusals. Each names the thing that was asked for and why it cannot be.
NEXT_CAMPAIGN_POLICY_DRIFT = "NEXT_CAMPAIGN_POLICY_DRIFT"
NEXT_CAMPAIGN_TARGET_UNUSABLE = "NEXT_CAMPAIGN_TARGET_UNUSABLE"
NEXT_CAMPAIGN_TREATMENT_NOT_ALLOWED = "NEXT_CAMPAIGN_TREATMENT_NOT_ALLOWED"
NEXT_CAMPAIGN_TARGET_TOO_EXPENSIVE = "NEXT_CAMPAIGN_TARGET_TOO_EXPENSIVE"
NEXT_CAMPAIGN_ALREADY_FROZEN = "NEXT_CAMPAIGN_ALREADY_FROZEN"
NEXT_CAMPAIGN_PARENT_IDENTITY = "NEXT_CAMPAIGN_PARENT_IDENTITY"
NEXT_CAMPAIGN_SCHEMA = "NEXT_CAMPAIGN_SCHEMA"

POLICY_SCHEMA = "growth-loop-policy/1"
PREREGISTRATION_SCHEMA = "growth-preregistration/1"


class NextCampaignRefusal(RuntimeError):
    """A next campaign that cannot be built, with the reason named."""


@dataclass(frozen=True)
class LoopPolicy:
    """The immutable envelope a loop may not widen, however well a run is going.

    Loaded from a document the loop only reads. Every ceiling, the protected
    set, the trusted ancestor, the declared protocol, the execution throughput,
    the stopping rules and the allowed treatments live here, so the components
    that *choose* things (target selection, campaign composition) hold no
    authority over them.
    """

    maximum_generations: int
    maximum_total_wall_gpu_hours: float
    maximum_consecutive_non_promotions: int
    maximum_same_target_attempts: int
    maximum_candidates: int
    plateau_epsilon: float
    allowed_training_types: tuple[str, ...]
    protected_benchmarks: tuple[str, ...]
    broad_benchmarks: tuple[str, ...]
    campaign_budget: CampaignBudget
    protection: ProtectionDeclaration
    evaluation_execution: EvaluationExecution
    candidate_selection_policy: str
    stopping_rules: tuple[str, ...]
    promotion_policy_version: str
    human_review_triggers: tuple[str, ...] = ()
    calibration_benchmarks: tuple[str, ...] = ()
    reliability_benchmarks: tuple[str, ...] = ()
    #: Skills the policy declares structurally out of reach for this training
    #: path. The loop routes them to human review rather than spending an
    #: envelope proving again that the same intervention does not work.
    structural_skills: tuple[str, ...] = ()

    #: Every key a policy document may carry. A key nothing reads would be a
    #: limit that looks enforced and is not.
    _KEYS = (
        "maximum_generations",
        "maximum_total_wall_gpu_hours",
        "maximum_consecutive_non_promotions",
        "maximum_same_target_attempts",
        "maximum_candidates",
        "plateau_epsilon",
        "allowed_training_types",
        "protected_benchmarks",
        "broad_benchmarks",
        "campaign_budget",
        "protection",
        "evaluation_execution",
        "candidate_selection_policy",
        "stopping_rules",
        "promotion_policy_version",
        "human_review_triggers",
    )
    #: Declared but not required: a policy that pins no calibration or
    #: reliability sets declares empty ones, which is not the same as forgetting.
    _OPTIONAL_KEYS = (
        "calibration_benchmarks",
        "reliability_benchmarks",
        "structural_skills",
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("maximum_generations", self.maximum_generations),
            ("maximum_consecutive_non_promotions", self.maximum_consecutive_non_promotions),
            ("maximum_same_target_attempts", self.maximum_same_target_attempts),
            ("maximum_candidates", self.maximum_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NextCampaignRefusal(f"{POLICY_SCHEMA}: {label} must be a positive integer")
        if isinstance(self.maximum_total_wall_gpu_hours, bool) or not isinstance(
            self.maximum_total_wall_gpu_hours, (int, float)
        ):
            raise NextCampaignRefusal(
                f"{POLICY_SCHEMA}: maximum_total_wall_gpu_hours must be a number"
            )
        if float(self.maximum_total_wall_gpu_hours) <= 0:
            raise NextCampaignRefusal(
                f"{POLICY_SCHEMA}: maximum_total_wall_gpu_hours must be positive"
            )
        if float(self.plateau_epsilon) < 0:
            raise NextCampaignRefusal(f"{POLICY_SCHEMA}: plateau_epsilon cannot be negative")
        if not self.allowed_training_types:
            raise NextCampaignRefusal(
                f"{POLICY_SCHEMA}: allowed_training_types cannot be empty; a loop "
                "that may train nothing cannot improve anything"
            )
        if not self.protected_benchmarks:
            raise NextCampaignRefusal(
                f"{POLICY_SCHEMA}: protected_benchmarks cannot be empty; a policy "
                "that protects nothing cannot certify anything"
            )

    @classmethod
    def from_mapping(
        cls, document: Mapping[str, Any], *, source: str = "<memory>"
    ) -> "LoopPolicy":
        if not isinstance(document, Mapping):
            raise NextCampaignRefusal(
                f"{source}: {POLICY_SCHEMA}: the loop policy must be an object"
            )
        unknown = sorted(set(document) - set(cls._KEYS) - set(cls._OPTIONAL_KEYS) - {"schema"})
        if unknown:
            # Named, not ignored: a declared limit no component reads is exactly
            # the kind of configuration that looks enforced and is not.
            raise NextCampaignRefusal(
                f"{source}: {POLICY_SCHEMA}: unknown loop-policy fields {unknown}; a "
                "limit nothing reads is not a limit"
            )
        missing = sorted(set(cls._KEYS) - set(document))
        if missing:
            raise NextCampaignRefusal(
                f"{source}: {POLICY_SCHEMA}: the loop policy is missing {missing}"
            )

        def _strings(key: str, *, optional: bool = False) -> tuple[str, ...]:
            value = document.get(key, [] if optional else None)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise NextCampaignRefusal(f"{source}: {key} must be a list of strings")
            return tuple(value)

        budget = CampaignBudget(**dict(document["campaign_budget"]))
        protection = ProtectionDeclaration.from_mapping(
            document["protection"], source=f"{source}: protection"
        )
        execution = EvaluationExecution.from_mapping(
            document["evaluation_execution"], source=f"{source}: evaluation_execution"
        )
        return cls(
            maximum_generations=document["maximum_generations"],
            maximum_total_wall_gpu_hours=document["maximum_total_wall_gpu_hours"],
            maximum_consecutive_non_promotions=document["maximum_consecutive_non_promotions"],
            maximum_same_target_attempts=document["maximum_same_target_attempts"],
            maximum_candidates=document["maximum_candidates"],
            plateau_epsilon=document["plateau_epsilon"],
            allowed_training_types=_strings("allowed_training_types"),
            protected_benchmarks=_strings("protected_benchmarks"),
            broad_benchmarks=_strings("broad_benchmarks"),
            campaign_budget=budget,
            protection=protection,
            evaluation_execution=execution,
            candidate_selection_policy=str(document["candidate_selection_policy"]),
            stopping_rules=_strings("stopping_rules"),
            promotion_policy_version=str(document["promotion_policy_version"]),
            human_review_triggers=_strings("human_review_triggers", optional=True),
            calibration_benchmarks=_strings("calibration_benchmarks", optional=True),
            reliability_benchmarks=_strings("reliability_benchmarks", optional=True),
            structural_skills=_strings("structural_skills", optional=True),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "LoopPolicy":
        path = Path(path)
        if not path.is_file():
            raise NextCampaignRefusal(f"{POLICY_SCHEMA}: loop policy not found: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise NextCampaignRefusal(
                f"{POLICY_SCHEMA}: loop policy {path} is not JSON: {error}"
            ) from error
        return cls.from_mapping(document, source=str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "maximum_generations": self.maximum_generations,
            "maximum_total_wall_gpu_hours": self.maximum_total_wall_gpu_hours,
            "maximum_consecutive_non_promotions": self.maximum_consecutive_non_promotions,
            "maximum_same_target_attempts": self.maximum_same_target_attempts,
            "maximum_candidates": self.maximum_candidates,
            "plateau_epsilon": self.plateau_epsilon,
            "allowed_training_types": list(self.allowed_training_types),
            "protected_benchmarks": list(self.protected_benchmarks),
            "broad_benchmarks": list(self.broad_benchmarks),
            "campaign_budget": self.campaign_budget.to_dict()
            if hasattr(self.campaign_budget, "to_dict")
            else {
                "device_gpu_hours_ceiling_per_recipe": self.campaign_budget.device_gpu_hours_ceiling_per_recipe,
                "wall_gpu_hours_ceiling_per_recipe": self.campaign_budget.wall_gpu_hours_ceiling_per_recipe,
                "device_gpu_hours_ceiling_campaign": self.campaign_budget.device_gpu_hours_ceiling_campaign,
                "wall_gpu_hours_ceiling_campaign": self.campaign_budget.wall_gpu_hours_ceiling_campaign,
                "device_time_measured": self.campaign_budget.device_time_measured,
            },
            "protection": self.protection.to_dict(),
            "evaluation_execution": self.evaluation_execution.to_dict(),
            "candidate_selection_policy": self.candidate_selection_policy,
            "stopping_rules": list(self.stopping_rules),
            "promotion_policy_version": self.promotion_policy_version,
            "human_review_triggers": list(self.human_review_triggers),
            "calibration_benchmarks": list(self.calibration_benchmarks),
            "reliability_benchmarks": list(self.reliability_benchmarks),
            "structural_skills": list(self.structural_skills),
        }

    def digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def allows_treatment(self, treatment: str) -> bool:
        return str(treatment) in set(self.allowed_training_types)


@dataclass(frozen=True)
class FrozenCampaign:
    """A declaration and its preregistration, written once and never edited."""

    cycle_id: str
    candidate_version: str
    directory: Path
    manifest_path: Path
    preregistration_path: Path
    manifest: CampaignManifest
    target: TargetProposal
    policy_digest: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "candidate_version": self.candidate_version,
            "directory": str(self.directory),
            "manifest_path": str(self.manifest_path),
            "preregistration_path": str(self.preregistration_path),
            "policy_digest": self.policy_digest,
            "target": self.target.to_dict(),
            "frozen_digest": self.digest,
        }


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


class NextCampaignBuilder:
    """Composes one generation's declaration from a target and a policy."""

    def __init__(self, *, policy: LoopPolicy) -> None:
        self.policy = policy

    # -- composition -------------------------------------------------------

    def build(
        self,
        *,
        parent: CampaignManifest,
        target: TargetProposal,
        generation_root: str | Path,
        attempt: int = 1,
        parent_identity: tuple[str, str] | None = None,
    ) -> FrozenCampaign:
        """Write the frozen declaration for one campaign attempt.

        ``parent`` is the last trusted declaration: the durable facts about the
        dense base, the trusted-ancestor arm and the promotion policy are
        carried from it verbatim, because they are properties of the lineage and
        not of this generation's target. ``parent_identity`` is the promoted
        adapter (path, digest) the next generation trains from; ``None`` means
        the parent generation is the dense base itself.

        ``generation_root`` is the directory generations live *under*; the
        attempt's own directory is ``generation_root / cycle_id``, derived here
        rather than by the caller. Two callers naming the same attempt
        differently is how one attempt's frozen declaration gets overwritten by
        another's -- so the name has exactly one owner, and it is this one.
        """
        self._require_policy_conformant(parent)
        candidate_version = _next_version(parent.resolved_candidate_version())

        self._require_target_usable(target)
        treatments = target.suggested_training_type
        if not self.policy.allows_treatment(treatments):
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_TREATMENT_NOT_ALLOWED}: the target asks for "
                f"{treatments!r}, and this policy allows only "
                f"{list(self.policy.allowed_training_types)}; a loop cannot widen "
                "its own allowed treatments"
            )
        ceiling = float(self.policy.campaign_budget.wall_gpu_hours_ceiling_campaign)
        if float(target.expected_cost_gpu_hours) > ceiling:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_TARGET_TOO_EXPENSIVE}: the target expects "
                f"{target.expected_cost_gpu_hours} wall GPU-hours and the policy's "
                f"campaign ceiling is {ceiling}; a campaign that cannot fit its own "
                "envelope is not a campaign"
            )
        adapter_path = ""
        adapter_digest = ""
        if parent_identity is not None:
            adapter_path, adapter_digest = (str(parent_identity[0]), str(parent_identity[1]))
            if not adapter_path.strip() or len(adapter_digest) != 64:
                raise NextCampaignRefusal(
                    f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent identity "
                    f"{parent_identity!r} is not a (path, sha256) pair, so the next "
                    "generation would train from an unbound adapter"
                )

        cycle_id = f"{candidate_version}-a{int(attempt)}-{_slug(target.target_skill)}"
        directory = Path(generation_root) / cycle_id
        directory.mkdir(parents=True, exist_ok=True)
        inputs = prepared_input_paths(directory)

        document: dict[str, Any] = {
            "cycle_id": cycle_id,
            "parent_version": parent.resolved_candidate_version(),
            "base_model_path": parent.base_model_path,
            "base_model_digest": parent.base_model_digest,
            "parent_adapter_path": adapter_path,
            "parent_adapter_digest": adapter_digest,
            "state_root": str(directory),
            "target_benchmarks": list(target.target_benchmarks),
            "protected_benchmarks": list(self.policy.protected_benchmarks),
            "broad_benchmarks": list(self.policy.broad_benchmarks),
            "calibration_benchmarks": list(self.policy.calibration_benchmarks),
            "reliability_benchmarks": list(self.policy.reliability_benchmarks),
            "budget": {
                "device_gpu_hours_ceiling_per_recipe": self.policy.campaign_budget.device_gpu_hours_ceiling_per_recipe,
                "wall_gpu_hours_ceiling_per_recipe": self.policy.campaign_budget.wall_gpu_hours_ceiling_per_recipe,
                "device_gpu_hours_ceiling_campaign": self.policy.campaign_budget.device_gpu_hours_ceiling_campaign,
                "wall_gpu_hours_ceiling_campaign": self.policy.campaign_budget.wall_gpu_hours_ceiling_campaign,
                "device_time_measured": self.policy.campaign_budget.device_time_measured,
            },
            "recipes": [
                f"{candidate_version}-recipe-{index}"
                for index in range(1, int(self.policy.maximum_candidates) + 1)
            ],
            "candidate_selection_policy": self.policy.candidate_selection_policy,
            "stopping_rules": list(self.policy.stopping_rules),
            "promotion_policy_version": self.policy.promotion_policy_version,
            "protection": self.policy.protection.to_dict(),
            "evaluation_execution": self.policy.evaluation_execution.to_dict(),
            # The trusted ancestor is the lineage's, not this generation's: a
            # generation that just promoted does not silently become the floor.
            "baseline_eval_report_path": parent.baseline_eval_report_path,
            **inputs,
            "notes": (
                f"composed by NextCampaignBuilder from target {target.target_skill!r} "
                f"(priority {target.priority:.4f}, treatment {treatments!r}) under "
                f"policy {self.policy.digest()[:16]}; {target.treatment_reason}"
            ),
        }
        try:
            manifest = CampaignManifest.from_mapping(document, source=cycle_id)
        except CampaignManifestError as error:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_SCHEMA}: the composed declaration is not loadable: {error}"
            ) from error

        return self._freeze(
            manifest=manifest,
            document=document,
            directory=directory,
            cycle_id=cycle_id,
            candidate_version=candidate_version,
            target=target,
        )

    # -- the policy boundary ----------------------------------------------

    def _require_policy_conformant(self, parent: CampaignManifest) -> None:
        """The template must be the policy's own output, not a hand-edit of it."""
        declared_protection = parent.protection.to_dict()
        policy_protection = self.policy.protection.to_dict()
        if declared_protection != policy_protection:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_POLICY_DRIFT}: the parent declaration protects "
                f"with {declared_protection} and the policy with {policy_protection}; "
                "merging the two would let a manifest move the trusted ancestor or "
                "the tolerance it is held to"
            )
        if parent.evaluation_execution.to_dict() != self.policy.evaluation_execution.to_dict():
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_POLICY_DRIFT}: the parent declaration executes "
                f"evaluations at {parent.evaluation_execution.to_dict()} and the "
                f"policy at {self.policy.evaluation_execution.to_dict()}; the arms "
                "and the candidate must share one declared throughput"
            )
        if parent.budget.wall_gpu_hours_ceiling_campaign > self.policy.campaign_budget.wall_gpu_hours_ceiling_campaign:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_POLICY_DRIFT}: the parent declaration declares a "
                "campaign ceiling above the policy's; the policy is the ceiling"
            )
        if not parent.baseline_eval_report_path.strip():
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_POLICY_DRIFT}: the parent declaration names no "
                "trusted-ancestor arm, so the next generation could not be held "
                "against the floor it must not regress through"
            )

    def _require_target_usable(self, target: TargetProposal) -> None:
        if not target.target_benchmarks:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_TARGET_UNUSABLE}: the target {target.target_skill!r} "
                "names no benchmark, so nothing would measure whether it improved"
            )
        overlap = sorted(set(target.target_benchmarks) & set(self.policy.protected_benchmarks))
        if overlap:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_TARGET_UNUSABLE}: the target {target.target_skill!r} "
                f"names protected benchmarks {overlap}; protected sets are gates, "
                "never optimization targets"
            )
        if not target.target_skill.strip():
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_TARGET_UNUSABLE}: the target names no skill"
            )

    # -- freezing ----------------------------------------------------------

    def _freeze(
        self,
        *,
        manifest: CampaignManifest,
        document: Mapping[str, Any],
        directory: Path,
        cycle_id: str,
        candidate_version: str,
        target: TargetProposal,
    ) -> FrozenCampaign:
        manifest_path = directory / "campaign.json"
        preregistration_path = directory / "preregistration.json"
        policy_digest = self.policy.digest()
        payload = {
            "schema": PREREGISTRATION_SCHEMA,
            "cycle_id": cycle_id,
            "candidate_version": candidate_version,
            "policy": self.policy.to_dict(),
            "policy_digest": policy_digest,
            "target": target.to_dict(),
            "manifest": dict(document),
            "statement": (
                "frozen before any candidate compute: the target, thresholds, "
                "protected set, budget ceilings, trusted ancestor and recipe rule "
                "in this declaration may not change once written. A later policy "
                "change is a new generation attempt, not an edit."
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        payload["frozen_digest"] = digest
        for path, content in (
            (manifest_path, json.dumps(dict(document), indent=2, sort_keys=True) + "\n"),
            (preregistration_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        ):
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(content)
            except FileExistsError as error:
                raise NextCampaignRefusal(
                    f"{NEXT_CAMPAIGN_ALREADY_FROZEN}: {path} already exists; a frozen "
                    "declaration is never rewritten -- a new attempt is a new "
                    "generation directory"
                ) from error
        return FrozenCampaign(
            cycle_id=cycle_id,
            candidate_version=candidate_version,
            directory=directory,
            manifest_path=manifest_path,
            preregistration_path=preregistration_path,
            manifest=manifest,
            target=target,
            policy_digest=policy_digest,
            digest=digest,
        )


def _next_version(version: str) -> str:
    if not version.startswith("gen") or not version[3:].isdigit():
        raise NextCampaignRefusal(
            f"{NEXT_CAMPAIGN_SCHEMA}: the parent generation {version!r} does not "
            "follow gen<N>, so the next generation cannot be derived from it"
        )
    return f"gen{int(version[3:]) + 1}"
