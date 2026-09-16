"""The contamination firewall and data registry refuse the forbidden paths.

The critical negative test: a protected benchmark test split offered as
training data MUST be refused -- not warned about, refused. These tests pin
that plus the trust-class/verification floor and the licensing gate.
"""

from __future__ import annotations

import hashlib

import pytest

from chowder.growth.contamination import ContaminationFirewall, MinHashIndex, _shingles
from chowder.growth.data_registry import (
    DataSource,
    admit,
    quarantine,
    seed_registry,
)
from chowder.growth.discovery import Candidate, DataDiscovery


PROTECTED_TEXTS = (
    "Question 41: What is the half-life of dubnium-268 in seconds?",
    "Question 42: Prove that the braid group B_4 embeds in Aut(F_2).",
    "Question 43: Compute the third-order correction to the anomalous magnetic moment.",
)

PARAPHRASED = "Compute the third order correction to the anomalous magnetic moment please."


def _firewall() -> ContaminationFirewall:
    firewall = ContaminationFirewall()
    firewall.register_protected(
        "example_frontier_bench@2025-02",
        PROTECTED_TEXTS,
        canaries=("BRAID-CANARY-7f3a",),
    )
    return firewall


# ---------------- contamination detection ----------------


def test_exact_protected_text_is_flagged():
    firewall = _firewall()
    result = firewall.check_text(PROTECTED_TEXTS[0])
    assert not result.clean
    assert result.verdict in {"POSSIBLE", "KNOWN_CONTAMINATION"}


def test_normalized_reformatting_is_detected():
    firewall = _firewall()
    mangled = "  ".join(PROTECTED_TEXTS[1].upper().split())
    result = firewall.check_text(mangled)
    assert not result.clean


def test_canary_inside_training_text_is_detected():
    firewall = _firewall()
    result = firewall.check_text(
        "training passage with embedded BRAID-CANARY-7f3a marker"
    )
    assert not result.clean
    assert any(m.detector == "canary" for m in result.matches)


def test_paraphrase_over_threshold_is_flagged_and_below_is_clean():
    firewall = _firewall()
    assert not firewall.check_text(PARAPHRASED).clean
    assert firewall.check_text("an ordinary sentence about gardening in spring.").clean


def test_shingle_fingerprints_do_not_depend_on_the_process_salt():
    """Guard fingerprints must be reproducible across processes.

    The builtin ``hash()`` is salted per process, so shingle ids -- and with
    them the LSH banding and the Jaccard estimate -- used to differ between
    runs. The same candidate text could be CLEAN in one process and POSSIBLE
    in the next, which makes a leak verdict unverifiable after the fact.
    """
    expected = int.from_bytes(
        hashlib.blake2b(
            b"alpha beta gamma delta epsilon zeta eta theta", digest_size=8
        ).digest(),
        "big",
    )
    assert expected in _shingles("alpha beta gamma delta epsilon zeta eta theta")


def test_a_banding_miss_cannot_silently_pass_benchmark_derived_text(monkeypatch):
    """Candidate retrieval is an accelerator; it must not be the decision.

    With 4 rows per band, a real ~50%-overlapping paraphrase of protected
    text misses every band a large fraction of the time. When that happened,
    the overlap fallback sat inside ``if candidates:`` and never ran, so a
    copy was reported CLEAN. Force the miss and require the refusal anyway.
    """
    firewall = _firewall()
    monkeypatch.setattr(MinHashIndex, "candidates", lambda self, text: set())
    result = firewall.check_text(PARAPHRASED)
    assert not result.clean
    assert any(m.detector == "substring" for m in result.matches)


def test_known_contamination_is_declared_not_discovered():
    firewall = _firewall()
    firewall.declare_known_contamination(
        "legacy_corpus@2024-01", "training corpus contains the benchmark's source crawl"
    )
    manifest = firewall.manifest(
        evaluated_benchmarks=("legacy_corpus@2024-01", "example_frontier_bench@2025-02")
    )
    benchmarks = manifest["benchmarks"]
    assert benchmarks["legacy_corpus@2024-01"] == {
        "status": "KNOWN_CONTAMINATION",
        "reason": "training corpus contains the benchmark's source crawl",
    }
    # The other benchmark was fingerprinted but not checked against samples;
    # the manifest must say UNKNOWN rather than pretending CLEAN.
    assert benchmarks["example_frontier_bench@2025-02"]["status"] == "UNKNOWN"


def test_manifest_reports_clean_for_fingerprinted_uncontaminated_samples():
    firewall = _firewall()
    manifest = firewall.manifest(
        evaluated_benchmarks=("example_frontier_bench@2025-02",),
        source_samples={
            "example_frontier_bench@2025-02": (
                "wholly unrelated prose about alpine botany",
            )
        },
    )
    status = manifest["benchmarks"]["example_frontier_bench@2025-02"]["status"]
    assert status == "CLEAN"


# ---------------- data registry rules ----------------


def _source(**overrides):
    base = dict(
        source_id="sample-math",
        dataset_name="Sample Math Corpus",
        revision="2025-03",
        url="https://example/math",
        license="Apache-2.0",
        permitted_training_use=True,
        domain="math",
        language="en",
        source_type="synthetic",
        verification="executable_tests",
        trust_class="GOLD",
        example_count=10_000,
        token_estimate=50_000_000,
        provenance="generated and unit-tested locally",
        acquisition_timestamp="2026-09-15T00:00:00Z",
        source_hash="sha256:" + "0" * 64,
        contamination_relationship="CLEAN",
        quality_score=0.9,
    )
    base.update(overrides)
    return DataSource(**base)


def test_revision_latest_is_refused():
    with pytest.raises(ValueError, match="revision must be pinned"):
        _source(revision="latest")


def test_unknown_license_is_refused_for_registration():
    with pytest.raises(ValueError, match="license must be recorded"):
        _source(license="unknown")


def test_trust_class_requires_matching_verification_floor():
    with pytest.raises(ValueError, match="requires verification"):
        _source(trust_class="GOLD", verification="heuristic_filter")
    with pytest.raises(ValueError, match="requires verification"):
        _source(trust_class="QUARANTINE", verification="executable_tests")


def test_gold_with_clean_contamination_is_trainable_once_included():
    source = admit(_source(), decision="included", reason="verified pipeline")
    assert source.trainable


def test_unknown_contamination_blocks_trainability_even_when_included():
    source = admit(
        _source(contamination_relationship="UNKNOWN"),
        decision="included",
        reason="checks pending",
    )
    assert not source.trainable


def test_admit_cannot_include_a_source_whose_license_forbids_training():
    with pytest.raises(ValueError, match="license does not permit"):
        admit(_source(permitted_training_use=False), decision="included")


def test_quarantine_downgrades_and_blocks_training():
    source = quarantine(_source(trust_class="GOLD", verification="executable_tests"))
    assert source.trust_class == "QUARANTINE"
    assert not source.trainable


def test_seed_registry_sources_start_untrainable_until_admitted():
    registry = seed_registry()
    assert registry.sources()
    for source in registry.sources():
        assert not source.trainable, (
            f"{source.source_id} must not be trainable before an explicit admit() "
            "and a firewall check"
        )


# ---------------- discovery refusal ----------------


def test_benchmark_shaped_candidate_is_refused():
    discovery = DataDiscovery(registry=seed_registry())
    discovery.discover(
        Candidate(
            candidate_id="hle-test-split",
            name="HLE test split",
            origin="huggingface",
            url="https://example/hle",
            revision="2025-02",
            license_declared="MIT",
            domain="reasoning",
            language=("en",),
            source_type="research",
            example_count=2500,
            token_estimate=1_000_000,
        )
    )
    with pytest.raises(PermissionError, match="failed inspection"):
        discovery.register(
            "hle-test-split",
            pii_reviewed=True,
            secrets_reviewed=True,
            quality_score=0.9,
            contamination_status="CLEAN",
            inclusion_reason="obviously wrong",
        )


def test_registration_requires_pii_and_secret_review():
    discovery = DataDiscovery(registry=seed_registry())
    discovery.discover(
        Candidate(
            candidate_id="web-sample",
            name="Web Sample",
            origin="huggingface",
            url="https://example/web",
            revision="v1",
            license_declared="ODC-BY-1.0",
            domain="knowledge",
            language=("en",),
            source_type="web",
            example_count=100,
            token_estimate=200_000,
        )
    )
    with pytest.raises(PermissionError, match="PII and secret review"):
        discovery.register(
            "web-sample",
            pii_reviewed=False,
            secrets_reviewed=True,
            quality_score=0.7,
            contamination_status="CLEAN",
            inclusion_reason="not reviewed yet",
        )


def test_registered_candidate_enters_as_quarantine_pending():
    discovery = DataDiscovery(registry=seed_registry())
    discovery.discover(
        Candidate(
            candidate_id="code-sample",
            name="Code Sample",
            origin="github",
            url="https://github.com/example/repo",
            revision="abc1234",
            license_declared="MIT",
            domain="coding",
            language=("python",),
            source_type="code",
            example_count=500,
            token_estimate=1_000_000,
        )
    )
    source = discovery.register(
        "code-sample",
        pii_reviewed=True,
        secrets_reviewed=True,
        quality_score=0.8,
        contamination_status="UNKNOWN",
        inclusion_reason="awaiting firewall check",
    )
    assert source.trust_class == "QUARANTINE"
    assert source.inclusion_decision == "pending"
    assert not source.trainable
