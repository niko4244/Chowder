"""Synthetic data pipeline.

SEED -> GENERATOR -> SOLUTION -> INDEPENDENT CRITIC -> OBJECTIVE VERIFIER
-> DEDUP -> CONTAMINATION CHECK -> DIFFICULTY ESTIMATION -> QUALITY SCORE
-> ACCEPT/REJECT.

The generating model never certifies its own output: acceptance requires an
independent critic and, where the domain allows, an objective verifier.
Output lands in QUARANTINE and only graduates to SILVER/GOLD trust when the
required verification evidence exists (the data registry enforces this).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .contamination import ContaminationFirewall
from .difficulty import estimate_difficulty

VERIFICATION_BY_DOMAIN: Mapping[str, tuple[str, ...]] = {
    "coding": ("executable_tests",),
    "math": ("symbolic_numeric", "authoritative_key"),
    "factual": ("citation_supported",),
    "reasoning": ("multi_judge",),
    "research": ("citation_supported", "multi_judge"),
}


@dataclass
class SyntheticCandidate:
    """One synthetic item moving through the pipeline."""

    candidate_id: str
    seed_id: str
    skill: str
    domain: str
    text: str
    solution: str
    generator_id: str
    critics: tuple[dict[str, Any], ...] = ()
    verifier: dict[str, Any] | None = None
    difficulty_band: str = ""
    quality_score: float = 0.0
    contamination_verdict: str = "UNKNOWN"
    accepted: bool = False
    rejection_reason: str = ""

    def digest(self) -> str:
        return hashlib.sha256(
            (self.text + "\x00" + self.solution).encode("utf-8")
        ).hexdigest()


@dataclass
class PipelineConfig:
    """Acceptance thresholds -- declared before generation, not tuned after."""

    min_critic_agreement: float = 2  # critics approving (of those run)
    min_quality: float = 0.6
    require_verifier_domains: frozenset[str] = frozenset({"coding", "math", "factual"})
    max_per_seed: int = 8


class SyntheticPipeline:
    """Runs the staged pipeline over seeds with injected callables.

    ``generate``/``critique``/``verify`` are injected so tests can drive the
    pipeline deterministically and production can plug real models; the
    pipeline itself owns the ORDER, the thresholds, and the refusals.
    """

    def __init__(
        self,
        *,
        firewall: ContaminationFirewall,
        config: PipelineConfig | None = None,
        generate: Callable[[str, str], tuple[str, str]] | None = None,
        critique: Callable[[str, str, str], dict[str, Any]] | None = None,
        verify: Callable[[str, str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.firewall = firewall
        self.config = config or PipelineConfig()
        self._generate = generate
        self._critique = critique
        self._verify = verify

    def _critique_one(self, candidate: SyntheticCandidate, critic_id: str) -> dict[str, Any]:
        if self._critique is None:
            return {"critic": critic_id, "approve": False, "reason": "no critic available"}
        result = dict(self._critique(candidate.text, candidate.solution, critic_id))
        result.setdefault("critic", critic_id)
        result.setdefault("approve", False)
        return result

    def _verify_one(self, candidate: SyntheticCandidate) -> dict[str, Any] | None:
        required = self.config.require_verifier_domains
        if candidate.domain in required and self._verify is None:
            return {
                "verifier": "none",
                "pass": False,
                "reason": f"domain {candidate.domain} requires objective verification",
            }
        if self._verify is None:
            return None
        result = dict(self._verify(candidate.text, candidate.solution, candidate.domain))
        result.setdefault("verifier", "objective")
        result.setdefault("pass", False)
        return result

    def process_seed(
        self,
        *,
        seed_id: str,
        seed_text: str,
        skill: str,
        domain: str,
        generator_id: str,
        variants: int = 3,
        critic_ids: Sequence[str] = ("critic-a", "critic-b"),
    ) -> tuple[list[SyntheticCandidate], list[SyntheticCandidate]]:
        """Generate variants for one seed and run the full gate chain.

        Returns (accepted, rejected) with rejection reasons recorded.
        """
        if self._generate is None:
            raise ValueError("no generator configured")
        accepted: list[SyntheticCandidate] = []
        rejected: list[SyntheticCandidate] = []
        seen_digests: set[str] = set()

        for variant in range(min(variants, self.config.max_per_seed)):
            text, solution = self._generate(seed_text, f"{seed_id}-v{variant}")
            candidate = SyntheticCandidate(
                candidate_id=f"{seed_id}-v{variant}",
                seed_id=seed_id,
                skill=skill,
                domain=domain,
                text=text,
                solution=solution,
                generator_id=generator_id,
            )
            digest = candidate.digest()

            # 1. DEDUP (exact + normalized within this seed's variants)
            normalized_key = hashlib.sha256(
                "".join(text.lower().split()).encode()
            ).hexdigest()
            if digest in seen_digests or normalized_key in seen_digests:
                candidate.rejection_reason = "duplicate"
                rejected.append(candidate)
                continue

            # 2. CONTAMINATION CHECK against protected benchmarks
            check = self.firewall.check_text(text)
            candidate.contamination_verdict = check.verdict
            if check.verdict != "CLEAN":
                candidate.rejection_reason = f"contamination:{check.verdict}"
                rejected.append(candidate)
                continue

            # 3. INDEPENDENT CRITICS
            candidate.critics = [self._critique_one(candidate, c) for c in critic_ids]
            approvals = sum(1 for c in candidate.critics if c.get("approve"))
            if approvals < self.config.min_critic_agreement:
                candidate.rejection_reason = (
                    f"critic_agreement:{approvals}/{len(candidate.critics)}"
                )
                rejected.append(candidate)
                continue

            # 4. OBJECTIVE VERIFIER where the domain demands it
            verifier_result = self._verify_one(candidate)
            candidate.verifier = verifier_result
            if verifier_result is not None and not verifier_result.get("pass"):
                candidate.rejection_reason = (
                    f"verifier:{verifier_result.get('reason', 'failed')}"
                )
                rejected.append(candidate)
                continue

            # 5. DIFFICULTY ESTIMATION (fall back to solution length when the
            # verifier reports no step count; a band of UNKNOWN is refused --
            # evidence, not assertion)
            reasoning_steps = (
                verifier_result.get("reasoning_steps") if verifier_result else None
            )
            if reasoning_steps is None:
                difficulty = estimate_difficulty(
                    solution_length_tokens=max(1, len(solution.split()))
                )
            else:
                difficulty = estimate_difficulty(reasoning_steps=int(reasoning_steps))
            candidate.difficulty_band = difficulty.band

            # 6. QUALITY SCORE (critic scores + verifier bonus, no self-cert)
            critic_scores = [float(c.get("score", 0.0)) for c in candidate.critics]
            base = sum(critic_scores) / len(critic_scores) if critic_scores else 0.0
            bonus = 0.1 if verifier_result and verifier_result.get("pass") else 0.0
            candidate.quality_score = round(min(1.0, base + bonus), 4)
            if candidate.quality_score < self.config.min_quality:
                candidate.rejection_reason = (
                    f"quality:{candidate.quality_score:.2f}<{self.config.min_quality}"
                )
                rejected.append(candidate)
                continue

            seen_digests.add(digest)
            seen_digests.add(normalized_key)
            candidate.accepted = True
            accepted.append(candidate)

        return accepted, rejected


def trust_class_for(candidate: SyntheticCandidate) -> str:
    """The trust class an accepted candidate earns -- QUARANTINE until its
    verification evidence is registered in the data registry, per domain."""
    required = VERIFICATION_BY_DOMAIN.get(candidate.domain, ("multi_judge",))
    if candidate.verifier and candidate.verifier.get("pass"):
        if any(v in {"executable_tests", "symbolic_numeric", "authoritative_key"} for v in required):
            return "GOLD"
        return "SILVER"
    if any(c.get("approve") for c in candidate.critics) and len(candidate.critics) >= 2:
        return "SILVER"
    return "QUARANTINE"
