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
#: A declaration was about to be frozen while it still carried the placeholder
#: recipe identities a draft uses only to size the planner's proposal. Freezing
#: those would preregister a recipe set no planner proposed.
NEXT_CAMPAIGN_RECIPE_SET_UNPLANNED = "NEXT_CAMPAIGN_RECIPE_SET_UNPLANNED"
#: Preparation ran but reported no recipe set, so the exact recipes the run
#: would execute are unknown and nothing may be frozen.
NEXT_CAMPAIGN_RECIPE_SET_ABSENT = "NEXT_CAMPAIGN_RECIPE_SET_ABSENT"

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


#: The prefix a draft's placeholder recipe ids carry. A frozen declaration may
#: never contain one, and :meth:`NextCampaignBuilder.freeze` refuses if it does.
UNPLANNED_RECIPE_PREFIX = "unplanned-recipe-"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


@dataclass(frozen=True)
class ParentEvidenceRef:
    """Which model the next generation learns from, named rather than inferred.

    Every field is a *declared* fact about one generation, not a guess from the
    shape of a directory tree.  The loop used to find the parent's evidence with
    an expression like ``state_root.parent.parent``, which happens to point at
    the right place only while one layout holds: a sibling campaign, a moved run
    root or a re-run generation silently redirects the whole lineage to another
    model's numbers.

    The ref is carried forward only by a promotion, and it is carried *whole*:
    the adapter digest, the base identity and the path to the arm that measured
    the promoted model all travel together, so a resume cannot pair one
    generation's adapter with another's measurement.
    """

    generation: str
    run_root: str
    base_model_path: str
    base_model_digest: str
    adapter_path: str = ""
    adapter_digest: str = ""
    #: The report whose rows are this model's measurements under the campaign's
    #: declared protocol. Defaults to the run root's own candidate arm; a
    #: generation measured separately (an arm re-measured under a newer
    #: instrument) names that file here instead, so preparation reads the fresh
    #: measurement rather than an older instrument that merely shares the root.
    candidate_evaluation_path: str = ""
    #: The attributed capability profile this generation produced, when it was
    #: written to disk. Empty means "derive it from the arm", never "assume".
    capability_profile_ref: str = ""

    @property
    def identity(self) -> tuple[str, str] | None:
        """The (path, sha256) this ref pins as the model, or ``None`` for base."""
        if not self.adapter_path.strip():
            return None
        return (self.adapter_path, self.adapter_digest)

    @property
    def measured_arm_path(self) -> str:
        """The file holding this model's measurements, by declared convention."""
        return self.candidate_evaluation_path or str(
            Path(self.run_root) / "candidate_evaluation.json"
        )

    @classmethod
    def from_declaration(
        cls,
        manifest: CampaignManifest,
        *,
        candidate_evaluation_path: str = "",
        capability_profile_ref: str = "",
    ) -> "ParentEvidenceRef":
        """Read a generation's lineage facts from its own declaration.

        A declaration states its generation, its run root, its dense base and the
        adapter it trained from -- so the parent's evidence ref is a *projection*
        of declared fields rather than an inference about layout.
        """
        return cls(
            generation=str(manifest.resolved_candidate_version()),
            run_root=str(manifest.state_root),
            base_model_path=str(manifest.base_model_path),
            base_model_digest=str(manifest.base_model_digest),
            adapter_path=str(manifest.parent_adapter_path),
            adapter_digest=str(manifest.parent_adapter_digest),
            candidate_evaluation_path=str(candidate_evaluation_path),
            capability_profile_ref=str(capability_profile_ref),
        )

    def verify(self, manifest: CampaignManifest) -> None:
        """Refuse a ref that does not describe the declaration's own lineage.

        Cheap and structural on purpose: byte-level digest verification is the
        readiness gate's job and runs against the real artifact. What is checked
        here is that the ref and the declaration cannot describe two different
        models, which is the failure that makes the *wrong* adapter the parent.
        """
        if not self.generation.strip():
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent evidence ref names no "
                "generation, so the lineage it describes cannot be attributed"
            )
        if not self.run_root.strip():
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent evidence ref for "
                f"{self.generation!r} names no run root, so its measurements "
                "cannot be located"
            )
        if self.generation != str(manifest.resolved_candidate_version()):
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent evidence ref measures "
                f"{self.generation!r} and the declaration's candidate is "
                f"{manifest.resolved_candidate_version()!r}; the next generation "
                "would train from one model while recording another as its parent"
            )
        if self.run_root != str(manifest.state_root):
            # The ref must name *this* generation's own run root. The defect this
            # replaces located it with ``state_root.parent.parent``, which points
            # at the right place only while one directory layout holds -- a
            # sibling campaign or a moved root silently redirects the lineage to
            # another model's numbers.
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent evidence ref reads "
                f"{self.run_root!r} and the declaration's run root is "
                f"{manifest.state_root!r}; the evidence the next generation would "
                "learn from is not the evidence this generation produced"
            )
        # The adapter is deliberately *not* compared against the declaration's
        # ``parent_adapter_*``: those name the adapter that declaration trained
        # *from*, while this ref names the model the declaration produced. For a
        # derived ref the two coincide; for a promoted one they are one
        # generation apart, and conflating them would make a promotion's own
        # evidence look like a mismatch.
        if self.adapter_path.strip() and len(self.adapter_digest) != 64:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_PARENT_IDENTITY}: the parent evidence ref pins "
                "an adapter path without a sha256, so the bytes it names are "
                "unverified"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "run_root": self.run_root,
            "base_model_path": self.base_model_path,
            "base_model_digest": self.base_model_digest,
            "adapter_path": self.adapter_path,
            "adapter_digest": self.adapter_digest,
            "candidate_evaluation_path": self.candidate_evaluation_path,
            "capability_profile_ref": self.capability_profile_ref,
        }

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "ParentEvidenceRef":
        return cls(
            generation=str(document.get("generation", "")),
            run_root=str(document.get("run_root", "")),
            base_model_path=str(document.get("base_model_path", "")),
            base_model_digest=str(document.get("base_model_digest", "")),
            adapter_path=str(document.get("adapter_path", "")),
            adapter_digest=str(document.get("adapter_digest", "")),
            candidate_evaluation_path=str(document.get("candidate_evaluation_path", "")),
            capability_profile_ref=str(document.get("capability_profile_ref", "")),
        )


@dataclass(frozen=True)
class CampaignDraft:
    """A composed declaration *before* anyone knows which recipes will run.

    A draft is deliberately not a preregistered campaign.  Its recipe entries are
    placeholders whose only job is to tell the production planner how many
    candidates the policy allows; the real identities come back from planning and
    only :meth:`NextCampaignBuilder.freeze` may write them into a declaration.

    Nothing is written to disk by composition, so a draft can be prepared,
    planned against, refused and discarded without leaving a frozen artifact
    that a later attempt would collide with.
    """

    cycle_id: str
    candidate_version: str
    generation_root: Path
    directory: Path
    attempt: int
    manifest: CampaignManifest
    document: Mapping[str, Any]
    target: TargetProposal
    policy_digest: str
    placeholder_recipe_ids: tuple[str, ...]
    parent_evidence: ParentEvidenceRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "candidate_version": self.candidate_version,
            "directory": str(self.directory),
            "attempt": self.attempt,
            "target": self.target.to_dict(),
            "policy_digest": self.policy_digest,
            "placeholder_recipe_ids": list(self.placeholder_recipe_ids),
            "parent_evidence": self.parent_evidence.to_dict(),
            "frozen": False,
        }


class NextCampaignBuilder:
    """Composes one generation's declaration from a target and a policy."""

    def __init__(self, *, policy: LoopPolicy) -> None:
        self.policy = policy

    # -- composition -------------------------------------------------------

    def draft(
        self,
        *,
        parent: CampaignManifest,
        target: TargetProposal,
        generation_root: str | Path,
        attempt: int = 1,
        parent_evidence: ParentEvidenceRef | None = None,
    ) -> CampaignDraft:
        """Compose one campaign attempt, leaving the recipe set to be planned.

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
        # The parent's evidence ref is stated, not discovered: it is either the
        # caller's (the promoted run the loop remembered) or a projection of this
        # declaration's own lineage fields. Never a search of the filesystem.
        evidence = parent_evidence or ParentEvidenceRef.from_declaration(parent)
        evidence.verify(parent)
        # The adapter the next generation trains from *is* the ref's identity:
        # there is one owner of "what model is the parent", so a caller cannot
        # hand in a promoted identity alongside a ref describing another model.
        adapter_path = evidence.adapter_path
        adapter_digest = evidence.adapter_digest

        cycle_id = f"{candidate_version}-a{int(attempt)}-{_slug(target.target_skill)}"
        directory = Path(generation_root) / cycle_id
        directory.mkdir(parents=True, exist_ok=True)
        inputs = prepared_input_paths(directory)
        # Placeholders, not a recipe set: their count is the authority the
        # planner reads, their names are never frozen.
        placeholder_recipe_ids = tuple(
            f"{candidate_version}-{UNPLANNED_RECIPE_PREFIX}{index}"
            for index in range(1, int(self.policy.maximum_candidates) + 1)
        )

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
            "recipes": list(placeholder_recipe_ids),
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

        return CampaignDraft(
            cycle_id=cycle_id,
            candidate_version=candidate_version,
            generation_root=Path(generation_root),
            directory=directory,
            attempt=int(attempt),
            manifest=manifest,
            document=document,
            target=target,
            policy_digest=self.policy.digest(),
            placeholder_recipe_ids=placeholder_recipe_ids,
            parent_evidence=evidence,
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

    def freeze(
        self, draft: CampaignDraft, *, recipe_ids: Sequence[str]
    ) -> FrozenCampaign:
        """Write the frozen declaration, now that the exact recipes are known.

        ``recipe_ids`` are the identities the *production* planner proposed for
        this draft. They are passed in rather than inferred so the recipe set has
        one owner: preparation, which ran the planner. A placeholder id, an empty
        set or a set larger than the policy allows is refused here -- the
        declaration is the last point at which a wrong recipe set can still be
        stopped, and after the freeze the run can only refuse it.

        This is the only writer of a frozen declaration. The moment it returns,
        the target, thresholds, benchmark sets, recipes, budgets, trusted
        ancestor and parent identity are fixed; a later change is a new attempt
        in a new directory, never an edit.
        """
        ids = tuple(str(value) for value in recipe_ids)
        placeholders = set(draft.placeholder_recipe_ids)
        if not ids:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_RECIPE_SET_ABSENT}: preparation reported no "
                "recipe set for this draft, so the recipes the run would execute "
                "are unknown; nothing is frozen until the production planner has "
                "proposed them"
            )
        if placeholders & set(ids):
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_RECIPE_SET_UNPLANNED}: the recipe set still "
                f"carries placeholder ids {sorted(placeholders & set(ids))}; a "
                "draft's placeholders size the planner's proposal and are never "
                "a frozen recipe set"
            )
        if len(ids) > int(self.policy.maximum_candidates):
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_SCHEMA}: {len(ids)} recipes were planned and this "
                f"policy allows at most {self.policy.maximum_candidates}; a "
                "campaign cannot run more candidates than its policy admits"
            )
        document = dict(draft.document)
        document["recipes"] = list(ids)
        try:
            manifest = CampaignManifest.from_mapping(document, source=draft.cycle_id)
        except CampaignManifestError as error:
            raise NextCampaignRefusal(
                f"{NEXT_CAMPAIGN_SCHEMA}: the composed declaration is not loadable: {error}"
            ) from error
        return self._write(
            manifest=manifest,
            document=document,
            directory=draft.directory,
            cycle_id=draft.cycle_id,
            candidate_version=draft.candidate_version,
            target=draft.target,
        )

    def _write(
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
