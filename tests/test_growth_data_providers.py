"""Every curriculum item is materialised by a named provider, or it refuses.

The point of this layer is that a corpus is assembled by dispatch, not by
whatever iterable was on hand, and that the mission's two rules hold: an item no
provider serves refuses, and a protected evaluation example can never become a
training example.  The tests below fail safe in both directions -- the wrong
provider, the wrong count, an unmeasured contamination verdict and a thin or
duplicated corpus all refuse rather than reaching the trainer.
"""

from __future__ import annotations

import pytest

from chowder.growth.data_providers import (
    CLEAN,
    CORPUS_QUALITY_SCHEMA,
    DEFAULT_PROVIDERS,
    PROVIDER_SCHEMA,
    CodingProvider,
    FailureRepairProvider,
    JudgeVerifiedRepairProvider,
    MathProvider,
    ProtocolRepairProvider,
    ReplayProvider,
    TrainingDataRefusal,
    TrainingDataRequest,
    assert_corpus_quality,
    assess_corpus,
    materialise,
)


def _request(
    item_id: str = "item-1",
    *,
    skill: str = "protocol.termination",
    training_type: str = "targeted_repair",
    role: str = "TARGET",
    verification: str = "symbolic_numeric",
    example_count: int = 3,
) -> TrainingDataRequest:
    return TrainingDataRequest(
        item_id=item_id,
        skill=skill,
        training_type=training_type,
        role=role,
        verification_method=verification,
        generation="gen2",
        example_count=example_count,
    )


def _clean(_text: str) -> str:
    return CLEAN


def test_each_skill_class_is_served_by_its_own_provider() -> None:
    requests = (
        _request("a", skill="protocol.termination", verification="symbolic_numeric"),
        _request("b", skill="math.algebra", verification="symbolic_numeric"),
        _request("c", skill="coding.generation", verification="executable_tests"),
        _request("d", skill="reasoning.causal", verification="multi_judge"),
    )
    corpus = materialise(requests, contamination_check=_clean)
    assert corpus.sources == {
        "a": ProtocolRepairProvider.source_id,
        "b": MathProvider.source_id,
        "c": CodingProvider.source_id,
        "d": JudgeVerifiedRepairProvider.source_id,
    }
    # Provenance is recorded per example, not per corpus.
    example = corpus.examples["d"][0]
    assert example.provider_id == JudgeVerifiedRepairProvider.provider_id
    assert example.generation == "gen2"
    assert example.target_skill == "reasoning.causal"
    assert example.training_type == "targeted_repair"
    assert example.verification == "multi_judge"
    assert example.contamination == CLEAN
    assert len(example.digest) == 64


def test_a_math_item_is_not_claimed_by_the_protocol_provider() -> None:
    """The old broad rule would have claimed this item and refused it wrongly."""
    corpus = materialise(
        (_request("m", skill="math.arithmetic", verification="symbolic_numeric"),),
        contamination_check=_clean,
    )
    assert corpus.sources["m"] == MathProvider.source_id


def test_an_item_no_provider_serves_refuses() -> None:
    with pytest.raises(TrainingDataRefusal) as error:
        materialise(
            (_request("x", skill="astrology.natal", verification="symbolic_numeric"),),
            contamination_check=_clean,
        )
    assert PROVIDER_SCHEMA in str(error.value)
    assert "no provider serves curriculum item" in str(error.value)


def test_an_item_whose_verification_no_provider_matches_refuses() -> None:
    """A provider that would verify by the wrong method is not a provider here.

    Coding material is verified by executable tests; declaring it symbolic is a
    mis-declaration, and no provider may quietly serve it anyway.
    """
    with pytest.raises(TrainingDataRefusal) as error:
        materialise(
            (
                _request(
                    "x",
                    skill="coding.generation",
                    verification="symbolic_numeric",
                ),
            ),
            contamination_check=_clean,
        )
    assert PROVIDER_SCHEMA in str(error.value)
    assert "symbolic_numeric" in str(error.value)


def test_a_provider_that_returns_the_wrong_count_refuses() -> None:
    class Short(ProtocolRepairProvider):
        provider_id = "short"

        def produce(self, request: TrainingDataRequest):
            return ["one"]

    with pytest.raises(TrainingDataRefusal) as error:
        materialise(
            (_request("x", example_count=3),),
            providers=(Short(),),
            contamination_check=_clean,
        )
    assert PROVIDER_SCHEMA in str(error.value)
    assert "which declared 3" in str(error.value)


def test_a_protected_evaluation_text_never_becomes_training_material() -> None:
    """The mission's protected-leak case, refused at materialisation."""
    protected = DEFAULT_PROVIDERS[0].produce(_request("x", example_count=1))[0]
    with pytest.raises(TrainingDataRefusal) as error:
        materialise(
            (_request("x", example_count=1),),
            protected_texts=(protected,),
            contamination_check=_clean,
        )
    assert PROVIDER_SCHEMA in str(error.value)
    assert "protected evaluation text" in str(error.value)


def test_without_a_contamination_checker_the_verdict_is_unknown_and_refuses() -> None:
    """Unmeasured contamination is not CLEAN -- omission must not pass."""
    with pytest.raises(TrainingDataRefusal) as error:
        materialise((_request("x"),))
    assert PROVIDER_SCHEMA in str(error.value)
    assert "'UNKNOWN'" in str(error.value)


def test_a_possible_contamination_verdict_refuses() -> None:
    with pytest.raises(TrainingDataRefusal) as error:
        materialise((_request("x"),), contamination_check=lambda _text: "POSSIBLE")
    assert "'POSSIBLE'" in str(error.value)


def test_failure_analogues_are_preferred_over_generic_material() -> None:
    provider = FailureRepairProvider({"protocol.looping": ["a marginal emission"]})
    corpus = materialise(
        (_request("x", skill="reasoning.causal", verification="multi_judge"),),
        providers=(provider, JudgeVerifiedRepairProvider()),
        contamination_check=_clean,
    )
    assert corpus.sources["x"] == FailureRepairProvider.source_id


def test_a_failure_provider_with_no_analogues_serves_nothing() -> None:
    """An empty analogue set must not be turned into fabricated material."""
    corpus = materialise(
        (_request("x", skill="reasoning.causal", verification="multi_judge"),),
        providers=(FailureRepairProvider(), JudgeVerifiedRepairProvider()),
        contamination_check=_clean,
    )
    assert corpus.sources["x"] == JudgeVerifiedRepairProvider.source_id


def test_a_replay_role_is_served_by_the_replay_provider() -> None:
    corpus = materialise(
        (
            _request(
                "r",
                skill="knowledge.factuality",
                role="REPLAY",
                verification="curated_trusted",
            ),
        ),
        contamination_check=_clean,
    )
    assert corpus.sources["r"] == ReplayProvider.source_id


def test_an_unknown_training_type_refuses_at_construction() -> None:
    with pytest.raises(TrainingDataRefusal) as error:
        _request("x", training_type="fine_tune_vibes")
    assert PROVIDER_SCHEMA in str(error.value)


def test_a_zero_example_count_refuses_at_construction() -> None:
    with pytest.raises(TrainingDataRefusal) as error:
        _request("x", example_count=0)
    assert PROVIDER_SCHEMA in str(error.value)


def test_no_items_refuses() -> None:
    with pytest.raises(TrainingDataRefusal) as error:
        materialise((), contamination_check=_clean)
    assert PROVIDER_SCHEMA in str(error.value)


def _corpus_with(providers, requests):
    return materialise(requests, providers=providers, contamination_check=_clean)


def test_assess_corpus_measures_rather_than_asserts() -> None:
    class Duplicating(ProtocolRepairProvider):
        provider_id = "duplicating"

        def produce(self, request: TrainingDataRequest):
            return ["same line"] * request.example_count

    corpus = _corpus_with((Duplicating(),), (_request("x", example_count=4),))
    report = assess_corpus(corpus)
    assert report.example_count == 4
    assert report.duplicate_rate == pytest.approx(0.75)
    assert report.verifier_pass_rate == 1.0
    assert report.contamination_result == CLEAN
    assert report.skill_coverage == {"protocol.termination": 4}
    assert report.provider_composition == {"duplicating": 4}


def test_the_quality_gate_refuses_a_corpus_above_the_duplicate_ceiling() -> None:
    class Duplicating(ProtocolRepairProvider):
        provider_id = "duplicating"

        def produce(self, request: TrainingDataRequest):
            return ["same line"] * request.example_count

    report = assess_corpus(
        _corpus_with((Duplicating(),), (_request("x", example_count=4),))
    )
    with pytest.raises(TrainingDataRefusal) as error:
        assert_corpus_quality(report)
    assert CORPUS_QUALITY_SCHEMA in str(error.value)
    assert "duplicate rate" in str(error.value)


def test_the_quality_gate_refuses_an_unverified_corpus() -> None:
    """A corpus whose verification is unknown is not a pass."""
    report = assess_corpus(_corpus_with((ProtocolRepairProvider(),), (_request("x"),)))
    unverified = type(report)(
        example_count=report.example_count,
        token_count=report.token_count,
        duplicate_rate=report.duplicate_rate,
        verifier_pass_rate=0.0,
        skill_coverage=report.skill_coverage,
        source_composition=report.source_composition,
        contamination_result=report.contamination_result,
        provider_composition=report.provider_composition,
    )
    with pytest.raises(TrainingDataRefusal) as error:
        assert_corpus_quality(unverified)
    assert CORPUS_QUALITY_SCHEMA in str(error.value)
    assert "verifier pass rate" in str(error.value)


def test_the_quality_gate_refuses_a_corpus_missing_a_declared_skill() -> None:
    report = assess_corpus(
        _corpus_with((MathProvider(),), (_request("x", skill="math.algebra"),))
    )
    with pytest.raises(TrainingDataRefusal) as error:
        assert_corpus_quality(report, expected_skills=("protocol.termination",))
    assert CORPUS_QUALITY_SCHEMA in str(error.value)
    assert "covers no example" in str(error.value)


def test_the_quality_gate_admits_a_healthy_corpus() -> None:
    report = assess_corpus(_corpus_with((ProtocolRepairProvider(),), (_request("x"),)))
    assert_corpus_quality(report, expected_skills=("protocol.termination",))


def test_a_clean_corpus_reaches_the_shape_the_binding_reads() -> None:
    corpus = materialise(
        (
            _request("a", skill="math.algebra", verification="symbolic_numeric"),
            _request("b", skill="coding.generation", verification="executable_tests"),
        ),
        contamination_check=_clean,
    )
    assert "math" not in corpus.sources["b"]
    assert set(corpus.material) == {"a", "b"}
    assert set(corpus.sources) == {"a", "b"}
    assert len(corpus.source_ids()) == 2
    for lines in corpus.material.values():
        assert all(line.startswith('{"text":') for line in lines)
