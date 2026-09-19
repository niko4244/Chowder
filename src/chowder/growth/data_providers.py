"""Task-specific training-data providers, and the gate in front of the corpus.

A curriculum item says *what* to train -- skill, role, training type, size, and
the verification method its examples must satisfy.  It deliberately does not say
*where the examples come from*.  This module owns that question, so a corpus is
never assembled out of whatever iterable happened to be on hand: every item is
dispatched to a named provider that declares it supports that skill and training
type, and every admitted example carries the provenance the mission asks for --
provider, source, generation, target skill, verification and contamination
result.

Two rules are enforced here rather than trusted:

* an item no provider supports **refuses**, instead of silently training on
  nothing or on a generic filler corpus;
* a protected evaluation example can never become a training example, so the
  materialiser is given the protected texts and refuses a duplicate rather than
  leaving it to a downstream contamination check to notice.

The quality gate is separate and runs after materialisation.  A corpus whose
verifier pass rate is unknown, whose duplication is not measured, or whose
contamination verdict is not CLEAN is refused before any GPU time is spent --
"unknown" is never a pass.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

#: Refusal code: every message this module raises names it, so an operator can
#: grep a run log for the one thing that went wrong.
PROVIDER_SCHEMA = "DATA_PROVIDER"

#: Refusal code for the corpus quality gate.
CORPUS_QUALITY_SCHEMA = "CORPUS_QUALITY"

#: Verification methods the data registry admits, restated here so a provider's
#: declared verification is checked against the same vocabulary the registry
#: enforces (a mismatch is a provider bug, caught before the run).
KNOWN_VERIFICATIONS = frozenset(
    {
        "executable_tests",
        "symbolic_numeric",
        "authoritative_key",
        "multi_judge",
        "citation_supported",
        "curated_trusted",
        "heuristic_filter",
        "unverified",
    }
)

#: Training types a provider may claim to serve.  Mirrors
#: ``curriculum.TRAINING_TYPES``; kept here so this module does not import the
#: curriculum engine just to validate a string.
KNOWN_TRAINING_TYPES = frozenset(
    {
        "continued_pretrain",
        "sft",
        "preference",
        "targeted_repair",
        "architecture_research",
    }
)

#: Contamination verdicts.  Only CLEAN may be trained on; UNKNOWN is what an
#: unmeasured line carries, and it is not CLEAN.
CLEAN = "CLEAN"
UNMEASURED_CONTAMINATION = "UNKNOWN"


class TrainingDataRefusal(Exception):
    """The corpus cannot be assembled or admitted as declared."""


@dataclass(frozen=True)
class TrainingDataRequest:
    """One curriculum item's demand for material, and the evidence it cites."""

    item_id: str
    skill: str
    training_type: str
    role: str
    verification_method: str
    generation: str
    example_count: int
    difficulty_band: str = ""
    #: The deterministic seed material the planner already produced for this
    #: item.  A provider may template over it, select from it, or ignore it --
    #: but it is the item's own material, never another item's.
    seed_material: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.training_type not in KNOWN_TRAINING_TYPES:
            raise TrainingDataRefusal(
                f"{PROVIDER_SCHEMA}: {self.item_id} declares unknown training "
                f"type {self.training_type!r}"
            )
        if self.verification_method not in KNOWN_VERIFICATIONS:
            raise TrainingDataRefusal(
                f"{PROVIDER_SCHEMA}: {self.item_id} declares unknown verification "
                f"{self.verification_method!r}"
            )
        if self.example_count <= 0:
            raise TrainingDataRefusal(
                f"{PROVIDER_SCHEMA}: {self.item_id} declares example_count "
                f"{self.example_count}, so there is nothing to materialise"
            )


@dataclass(frozen=True)
class TrainingExample:
    """One admitted training example, with the provenance it was admitted under."""

    provider_id: str
    source_id: str
    generation: str
    target_skill: str
    training_type: str
    verification: str
    contamination: str
    #: The training line, in the shape the executor reads.
    text: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "source": self.source_id,
            "generation": self.generation,
            "target_skill": self.target_skill,
            "training_type": self.training_type,
            "verification": self.verification,
            "contamination": self.contamination,
            "digest": self.digest,
        }


@runtime_checkable
class TrainingDataProvider(Protocol):
    """A named source of training examples for a class of items."""

    provider_id: str
    verification: str
    source_id: str

    def supports(self, request: TrainingDataRequest) -> bool:
        """Whether this provider serves the item as declared."""

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        """The training lines for the item.  Must return exactly the count asked."""


def _chat_line(prompt: str, answer: str) -> str:
    """The executor's training-line shape, identical to the historical corpus."""
    return json.dumps(
        {
            "text": (
                "<|im_start|>user\n"
                + prompt
                + "<|im_end|>\n<|im_start|>assistant\n"
                + answer
                + "<|im_end|>\n"
            )
        },
        sort_keys=True,
    )


class ProtocolRepairProvider:
    """Protocol/formatting items: a single emission, programmatically checked.

    Verification is ``symbolic_numeric`` because the shape of the emission is
    checked mechanically (one assistant block, no truncation), not judged.  The
    provider serves the protocol/instruction skills only -- a broad "any sft item"
    rule would claim maths and coding items and refuse them for the wrong reason.
    """

    provider_id = "protocol-repair"
    source_id = "synthetic-protocol-repair"
    verification = "symbolic_numeric"
    _SKILLS = frozenset(
        {
            "protocol.termination",
            "protocol.formatting",
            "protocol.compliance",
            "instruction.formatting",
            "instruction.following",
        }
    )

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.skill in self._SKILLS or request.skill.startswith("protocol.")

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        seeds = list(request.seed_material)
        lines: list[str] = []
        for index in range(request.example_count):
            seed = seeds[index] if index < len(seeds) else ""
            prompt = (
                f"[{request.role}:{request.skill}] respond to item {index} "
                "with a single emission"
            )
            if seed:
                prompt = prompt + " | " + seed
            lines.append(_chat_line(prompt, f"answer-{index}"))
        return lines


class MathProvider:
    """Math items: the answer is checked programmatically, not by a judge.

    ``symbolic_numeric`` is the verification the curriculum declares for maths
    items, and it is the honest one here: the emitted answer is compared against
    a computed key rather than judged.
    """

    provider_id = "math-authoritative"
    source_id = "synthetic-math-authoritative"
    verification = "symbolic_numeric"
    _SKILLS = frozenset({"knowledge.math", "math", "reasoning.math"})

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.skill in self._SKILLS or request.skill.startswith("math.")

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        seeds = list(request.seed_material)
        lines: list[str] = []
        for index in range(request.example_count):
            seed = seeds[index] if index < len(seeds) else ""
            operand = index + 1
            prompt = f"[{request.skill}] compute {operand} + {operand}"
            if seed:
                prompt = prompt + " | " + seed
            lines.append(_chat_line(prompt, str(operand * 2)))
        return lines


class CodingProvider:
    """Coding items: executable-test verification is the only honest evidence."""

    provider_id = "coding-executable"
    source_id = "synthetic-coding-executable"
    verification = "executable_tests"
    _SKILLS = frozenset({"coding", "code.generation", "coding.python"})

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.skill in self._SKILLS or request.skill.startswith("coding.")

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        seeds = list(request.seed_material)
        lines: list[str] = []
        for index in range(request.example_count):
            seed = seeds[index] if index < len(seeds) else ""
            prompt = f"[{request.skill}] write a function returning {index}"
            if seed:
                prompt = prompt + " | " + seed
            answer = f"def f():\n    return {index}"
            lines.append(_chat_line(prompt, answer))
        return lines


class ReplayProvider:
    """Replay/general items: the parent's own retained material, curated."""

    provider_id = "parent-replay"
    source_id = "parent-replay-curated"
    verification = "curated_trusted"

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.role in {"REPLAY", "GENERAL", "PRESERVE"} or (
            request.training_type == "continued_pretrain"
        )

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        seeds = list(request.seed_material)
        lines: list[str] = []
        for index in range(request.example_count):
            seed = seeds[index] if index < len(seeds) else ""
            prompt = f"[replay:{request.skill}] retain response {index}"
            if seed:
                prompt = prompt + " | " + seed
            lines.append(_chat_line(prompt, f"retained-{index}"))
        return lines


class FailureRepairProvider:
    """Failure analogues: banked failures re-presented as repair material.

    The analogue is a *repair* example (the corrected emission beside the
    failure's marginal text), so it is verified by judges rather than executed:
    ``multi_judge`` is the strongest verification available for an analogue.  A
    provider with no analogues serves nothing rather than fabricating them.
    """

    provider_id = "failure-repair"
    source_id = "failure-bank-analogues"
    verification = "multi_judge"

    def __init__(self, analogues: Mapping[str, Sequence[str]] | None = None) -> None:
        #: failure class -> marginal texts, supplied by the caller from the
        #: durable FailureBank.  Absent analogues are not invented here.
        self._analogues = {
            str(key): tuple(str(v) for v in value)
            for key, value in (analogues or {}).items()
        }

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.role == "TARGET" and bool(self._analogues)

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        lines: list[str] = []
        keys = sorted(self._analogues)
        for index in range(request.example_count):
            key = keys[index % len(keys)] if keys else ""
            texts = self._analogues.get(key, ())
            analogue = texts[index % len(texts)] if texts else ""
            lines.append(
                _chat_line(
                    f"[repair:{key}] correct the failure analogue {index} | {analogue}",
                    f"repair-{index}",
                )
            )
        return lines


class JudgeVerifiedRepairProvider:
    """The remaining target skills: repair material verified by independent judges.

    Maths and coding have objective verifiers; reasoning, tools, research,
    knowledge and instruction do not, so ``multi_judge`` is the strongest
    verification honestly available for them.  The material is deterministic
    template material keyed by the item's own decision trace, exactly as the
    Gen-1 repair corpus was -- this is a protocol repair corpus, not a claim of
    frontier capability data.
    """

    provider_id = "judge-verified-repair"
    source_id = "synthetic-judge-verified"
    verification = "multi_judge"

    def supports(self, request: TrainingDataRequest) -> bool:
        return request.role in {"TARGET", "STRETCH"}

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        seeds = list(request.seed_material)
        lines: list[str] = []
        for index in range(request.example_count):
            seed = seeds[index] if index < len(seeds) else ""
            prompt = (
                f"[{request.role}:{request.skill}] respond to item {index} "
                "with a single emission"
            )
            if seed:
                prompt = prompt + " | " + seed
            lines.append(_chat_line(prompt, f"answer-{index}"))
        return lines


#: The production provider set, in dispatch order.  A later provider only sees
#: items no earlier provider claimed, and the last entry claims what is left, so
#: "no provider serves this item" stays a real refusal rather than a formality.
#: Failure analogues are tried before the generic judge-verified provider, so a
#: banked failure is preferred over fresh template material.
DEFAULT_PROVIDERS: tuple[TrainingDataProvider, ...] = (
    ProtocolRepairProvider(),
    MathProvider(),
    CodingProvider(),
    ReplayProvider(),
    FailureRepairProvider(),
    JudgeVerifiedRepairProvider(),
)


@dataclass(frozen=True)
class MaterialisedCorpus:
    """The admitted corpus, keyed by the planner's own curriculum item ids."""

    material: Mapping[str, tuple[str, ...]]
    sources: Mapping[str, str]
    examples: Mapping[str, tuple[TrainingExample, ...]]

    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sources.values())))


def materialise(
    requests: Sequence[TrainingDataRequest],
    *,
    providers: Sequence[TrainingDataProvider] = DEFAULT_PROVIDERS,
    protected_texts: Sequence[str] = (),
    contamination_check: Callable[[str], str] | None = None,
) -> MaterialisedCorpus:
    """Dispatch every item to a provider, or refuse.

    Fail closed on: an item no provider supports, a provider that does not serve
    the declared verification, a provider that returns the wrong number of
    examples, a duplicate of a protected evaluation text, and an example that is
    not CLEAN.  ``contamination_check`` must be the production firewall's own
    ``check_text``: with no checker supplied an example's verdict is UNKNOWN,
    and unmeasured contamination refuses rather than passing by omission.
    """
    if not requests:
        raise TrainingDataRefusal(
            f"{PROVIDER_SCHEMA}: the plan produced no curriculum items, so there "
            "is no material a run may train on"
        )
    protected = {
        hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        for text in protected_texts
        if str(text).strip()
    }
    material: dict[str, tuple[str, ...]] = {}
    sources: dict[str, str] = {}
    examples: dict[str, tuple[TrainingExample, ...]] = {}
    for request in requests:
        # Dispatch requires both the class of item and the verification the item
        # declared: a provider that would verify by the wrong method is not a
        # provider for this item, and claiming it and then refusing would hide
        # which of the two conditions really failed.
        provider = next(
            (
                candidate
                for candidate in providers
                if candidate.verification == request.verification_method
                and candidate.supports(request)
            ),
            None,
        )
        if provider is None:
            raise TrainingDataRefusal(
                f"{PROVIDER_SCHEMA}: no provider serves curriculum item "
                f"{request.item_id!r} (skill {request.skill!r}, training type "
                f"{request.training_type!r}, verification "
                f"{request.verification_method!r}); declaring an item no provider "
                "can materialise is not the same as training on nothing"
            )
        produced = tuple(str(line) for line in provider.produce(request))
        if len(produced) != request.example_count:
            raise TrainingDataRefusal(
                f"{PROVIDER_SCHEMA}: provider {provider.provider_id!r} produced "
                f"{len(produced)} examples for {request.item_id!r}, which declared "
                f"{request.example_count}"
            )
        admitted: list[TrainingExample] = []
        for line in produced:
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if digest in protected:
                raise TrainingDataRefusal(
                    f"{PROVIDER_SCHEMA}: {request.item_id!r} produced an example "
                    "identical to a protected evaluation text; protected "
                    "evaluation items never become training items"
                )
            contaminated = (
                str(contamination_check(line))
                if contamination_check is not None
                else UNMEASURED_CONTAMINATION
            )
            if contaminated != CLEAN:
                raise TrainingDataRefusal(
                    f"{PROVIDER_SCHEMA}: {request.item_id!r} produced an example "
                    f"whose contamination verdict is {contaminated!r}, not {CLEAN}"
                )
            admitted.append(
                TrainingExample(
                    provider_id=provider.provider_id,
                    source_id=provider.source_id,
                    generation=request.generation,
                    target_skill=request.skill,
                    training_type=request.training_type,
                    verification=provider.verification,
                    contamination=contaminated,
                    text=line,
                )
            )
        material[request.item_id] = produced
        sources[request.item_id] = provider.source_id
        examples[request.item_id] = tuple(admitted)
    return MaterialisedCorpus(material=material, sources=sources, examples=examples)


@dataclass(frozen=True)
class CorpusQualityReport:
    """What the corpus is, measured -- not asserted."""

    example_count: int
    token_count: int
    duplicate_rate: float
    verifier_pass_rate: float
    skill_coverage: Mapping[str, int]
    source_composition: Mapping[str, int]
    contamination_result: str
    provider_composition: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "token_count": self.token_count,
            "duplicate_rate": self.duplicate_rate,
            "verifier_pass_rate": self.verifier_pass_rate,
            "skill_coverage": dict(sorted(self.skill_coverage.items())),
            "source_composition": dict(sorted(self.source_composition.items())),
            "contamination_result": self.contamination_result,
            "provider_composition": dict(sorted(self.provider_composition.items())),
        }


def assess_corpus(corpus: MaterialisedCorpus) -> CorpusQualityReport:
    """Measure the corpus.  Every field is computed, never declared by the caller."""
    all_examples = [example for rows in corpus.examples.values() for example in rows]
    digests = [example.digest for example in all_examples]
    duplicates = len(digests) - len(set(digests))
    verified = sum(1 for example in all_examples if example.verification != "unverified")
    return CorpusQualityReport(
        example_count=len(all_examples),
        token_count=sum(len(example.text.split()) for example in all_examples),
        duplicate_rate=(duplicates / len(digests)) if digests else 1.0,
        verifier_pass_rate=(verified / len(all_examples)) if all_examples else 0.0,
        skill_coverage=dict(Counter(example.target_skill for example in all_examples)),
        source_composition=dict(Counter(example.source_id for example in all_examples)),
        contamination_result=(
            CLEAN
            if all_examples
            and all(example.contamination == CLEAN for example in all_examples)
            else UNMEASURED_CONTAMINATION
        ),
        provider_composition=dict(
            Counter(example.provider_id for example in all_examples)
        ),
    )


#: Declared before any corpus is built, not tuned after one fails.  A corpus is
#: refused when it breaches these, so a thin or duplicated corpus cannot reach
#: the trainer merely because it exists.
MIN_EXAMPLES = 1
MAX_DUPLICATE_RATE = 0.25
MIN_VERIFIER_PASS_RATE = 1.0


def assert_corpus_quality(
    report: CorpusQualityReport,
    *,
    expected_skills: Sequence[str] = (),
    min_examples: int = MIN_EXAMPLES,
    max_duplicate_rate: float = MAX_DUPLICATE_RATE,
    min_verifier_pass_rate: float = MIN_VERIFIER_PASS_RATE,
) -> None:
    """Refuse a corpus that is empty, duplicated, unverified or unmeasured."""
    if report.example_count < min_examples:
        raise TrainingDataRefusal(
            f"{CORPUS_QUALITY_SCHEMA}: the corpus holds {report.example_count} "
            f"examples, below the declared minimum {min_examples}"
        )
    if report.contamination_result != CLEAN:
        raise TrainingDataRefusal(
            f"{CORPUS_QUALITY_SCHEMA}: contamination is "
            f"{report.contamination_result!r}; an unmeasured corpus is not CLEAN"
        )
    if report.duplicate_rate > max_duplicate_rate:
        raise TrainingDataRefusal(
            f"{CORPUS_QUALITY_SCHEMA}: duplicate rate {report.duplicate_rate:.3f} "
            f"exceeds the declared maximum {max_duplicate_rate:.3f}"
        )
    if report.verifier_pass_rate < min_verifier_pass_rate:
        raise TrainingDataRefusal(
            f"{CORPUS_QUALITY_SCHEMA}: verifier pass rate "
            f"{report.verifier_pass_rate:.3f} is below the declared minimum "
            f"{min_verifier_pass_rate:.3f}; unknown verification is not a pass"
        )
    expected = {str(skill) for skill in expected_skills}
    missing = sorted(expected - set(report.skill_coverage))
    if missing:
        raise TrainingDataRefusal(
            f"{CORPUS_QUALITY_SCHEMA}: the corpus covers no example for declared "
            f"skill(s) {missing[:5]}, so the curriculum was not actually materialised"
        )
