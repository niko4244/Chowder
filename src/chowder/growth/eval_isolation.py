"""The evidence-trust wall between search and promotion.

Where ``eval_tiers.py`` bounds evaluation *cost* by decision stakes, this
module bounds evaluation *trust*: which measurements a candidate search is
allowed to see at all.

Three tiers of evidence:

- Search evidence (tier 1) is cheap, training-side, and never contains a
  protected or final benchmark. Allowed to eliminate candidates.
- Survivor evidence (tier 2) is the larger development evaluation a survivor
  earns. Read after the search narrows.
- Promotion evidence (tier 3) is the frozen target benchmark, the protected
  retained-capability suite, and the regression gates -- the only tier a
  promotion may be decided on.

The wall is structural, not advisory: every declared benchmark is classified
exactly once, anything the search controller or the selection policy can read
is checked against that classification, and a Tier-3 benchmark appearing in
any search-readable surface refuses the campaign at plan time -- before a
single GPU-hour is spent. A future developer who adds a protected benchmark
to a search surface fails here, not after the winner is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - type-only, keeps the wall import-light
    from .retention import RetentionProfile

__all__ = [
    "EvalTier",
    "EvalTierPolicy",
    "SearchIsolationRefusal",
    "assert_promotion_gate_isolation",
    "assert_search_isolation",
    "classify_benchmarks",
]


class EvalTier(str, Enum):
    """How much trust a measurement is allowed to carry, and for whom."""

    #: Cheap search evidence: training stability, loss, artifact validity,
    #: non-protected development probes. Allowed to eliminate candidates.
    SEARCH_EVIDENCE = "search-evidence"
    #: Survivor evidence: larger development evaluations, capability targets,
    #: behavior diagnostics. Read after the search narrows.
    SURVIVOR_EVIDENCE = "survivor-evidence"
    #: Promotion evidence: frozen target benchmark, protected retained-
    #: capability suite, regression and safety gates. Never search-readable.
    PROMOTION_EVIDENCE = "promotion-evidence"


class SearchIsolationRefusal(RuntimeError):
    """A search surface reached for evidence it must not see."""


#: A benchmark whose *name* matches one of these patterns is promotion
#: evidence by construction, whatever a declaration claims: the protected
#: suites and frozen final gates are not re-classifiable downward by config.
_RESERVED_PROMOTION_PATTERNS: tuple[str, ...] = (
    "protected",
    "retained-capabilities",
    "final",
    "frozen",
    "gsm8k-heldout",
    "contamination",
)


@dataclass(frozen=True)
class EvalTierPolicy:
    """The campaign's declared classification of its own benchmarks."""

    #: benchmark name -> declared tier.
    classification: Mapping[str, EvalTier]

    def tier_of(self, benchmark: str) -> EvalTier:
        tier = self.classification.get(benchmark)
        if tier is not None:
            return _raise_if_reserved_downgrade(benchmark, tier)
        # Unclassified benchmarks are promotion evidence by default: the
        # classification must *earn* a lower tier explicitly, never inherit
        # one from a missing entry.
        return EvalTier.PROMOTION_EVIDENCE

    def benchmarks_of(self, tier: EvalTier) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, t in self.classification.items() if t is tier)
        )


def _raise_if_reserved_downgrade(benchmark: str, tier: EvalTier) -> EvalTier:
    lowered = benchmark.lower()
    reserved = any(pattern in lowered for pattern in _RESERVED_PROMOTION_PATTERNS)
    if reserved and tier is not EvalTier.PROMOTION_EVIDENCE:
        raise SearchIsolationRefusal(
            f"benchmark {benchmark!r} matches a reserved promotion-evidence "
            f"name and cannot be classified downward to {tier.value}; the "
            "protected suites are not re-classifiable by configuration"
        )
    return tier


def classify_benchmarks(
    declarations: Mapping[str, str],
    *,
    source: str = "<memory>",
) -> EvalTierPolicy:
    """Build the policy from a declaration's ``eval_tiers`` section.

    Unknown tier names refuse rather than defaulting: a typo'd tier would
    otherwise silently place a benchmark in whichever tier the default
    protects -- or, worse, in a lower one.
    """
    classification: dict[str, EvalTier] = {}
    for benchmark, tier_name in declarations.items():
        try:
            tier = EvalTier(tier_name)
        except ValueError:
            raise SearchIsolationRefusal(
                f"{source}: benchmark {benchmark!r} declares tier "
                f"{tier_name!r}, which is not one of "
                f"{sorted(t.value for t in EvalTier)}"
            ) from None
        # Refuse the downgrade at declaration time, not first read: a config
        # that demotes a reserved benchmark is refused at plan time even if
        # nothing ever consults it.
        _raise_if_reserved_downgrade(benchmark, tier)
        classification[benchmark] = tier
    return EvalTierPolicy(classification=classification)


def assert_search_isolation(
    *,
    policy: EvalTierPolicy,
    search_readable_benchmarks: Sequence[str],
    selection_policies: Sequence[str] = (),
) -> None:
    """Refuse any search-readable surface that can see promotion evidence.

    ``search_readable_benchmarks`` is every benchmark name reachable from the
    candidate search's screen or the candidate-selection policy -- the
    campaign declaration's search-section metrics, the screen's evidence
    fields, anything a `run_attempt` result could be ranked on. A Tier-3 name
    here is a wall breach, refused before compute.
    """
    for benchmark in search_readable_benchmarks:
        tier = policy.tier_of(benchmark)
        if tier is EvalTier.PROMOTION_EVIDENCE:
            raise SearchIsolationRefusal(
                f"the candidate search can read {benchmark!r}, which is "
                "promotion evidence (tier 3); search must never optimize "
                "directly against the promotion gate -- reclassify it as "
                "search evidence only if it genuinely is a non-protected "
                "development probe, or remove it from the search surface"
            )
    for policy_name in selection_policies:
        if "protected" in policy_name.lower() or "final" in policy_name.lower():
            raise SearchIsolationRefusal(
                f"selection policy {policy_name!r} names protected evidence; "
                "selection over the final round reads training-side fields only"
            )


def assert_promotion_gate_isolation(
    *,
    policy: EvalTierPolicy,
    retention_profile: "RetentionProfile",
) -> None:
    """Refuse a promotion gate measured on evidence the search can see.

    A retention constraint names the benchmark its measurement must come
    from. If that benchmark classifies below promotion evidence, the gate is
    readable by the very search that produced the candidate -- the search
    could then shape its own gate. This is wiring, not a measured outcome, so
    it refuses outright instead of downgrading a verdict: fixing the
    classification or moving the measurement is the only way past it.
    """
    for constraint in retention_profile.constraints:
        tier = policy.tier_of(constraint.benchmark)
        if tier is not EvalTier.PROMOTION_EVIDENCE:
            raise SearchIsolationRefusal(
                f"retention constraint {constraint.dimension!r} is measured on "
                f"{constraint.benchmark!r}, which the campaign classified as "
                f"{tier.value}; a promotion gate on search-readable evidence "
                "lets the search shape its own gate -- measure the constraint "
                "on promotion evidence, or reclassify the benchmark honestly"
            )
