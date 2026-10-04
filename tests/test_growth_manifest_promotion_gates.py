"""The manifest loader closes the promotion-gate seam: a campaign declares its
retention profile and eval tier policy, and the cycle's promotion gates bind
from that configuration — the same objects, the same enforcement, the same
fail-closed refusals a programmatic construction gets.

Three contracts are pinned here:

- **parse-valid binds.** A well-formed declaration parses into the domain
  types, reaches ``CycleConfig`` through the production builder, and a
  manifest-declared cycle's ``decide_promotion`` downgrades a promotion the
  predeclared rule alone would have granted.
- **malformed refuses at load.** An unknown field, an unmeasurable
  constraint, an unknown tier, or a gate demoted into the search's view is a
  ``CampaignManifestError`` from ``from_mapping`` — never a silently dropped
  section.
- **declaring neither changes nothing.** Every manifest predating the fields
  loads with both unset and promotes exactly as it did before.

The attribution contract rides along: a declared gate's reasons reach the
record whenever it fires, not only when it was the sole cause of the verdict.
"""

from __future__ import annotations

import pytest

from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION
from chowder.growth.campaign import CampaignManifest, CampaignManifestError
from chowder.growth.campaign_runner import _build_cycle
from chowder.growth.contamination import ContaminationFirewall
from chowder.growth.eval_isolation import EvalTier, SearchIsolationRefusal
from chowder.growth.promotion import BenchmarkResult
from chowder.growth.retention import RetentionConstraint, RetentionProfile

import test_growth_campaign_runner as campaign_fixture
from growth_gate_fixtures import _result, _samples

TARGET = campaign_fixture.TARGET_ID
CONSTRAINED = campaign_fixture.PROTECTED_ID

#: The constrained benchmark is declared as a plain *target* here, so the
#: predeclared promotion rule alone does not gate its regression: a target win
#: over a regressed co-target promotes, and only the retention profile can
#: stop it. That makes the downgrade below attributable to the declaration.
_UNGATED_SETS = {
    "target_benchmarks": [TARGET, CONSTRAINED],
    "protected_benchmarks": [],
    "broad_benchmarks": [TARGET],
}

#: The constrained benchmark is a *protected* benchmark here, so the predeclared
#: rule alone already rejects a regression on it. The declared profile measures
#: the same dimension, so both gates fire on the same candidate -- the shape
#: where the record used to name only the predeclared rule. The broad battery is
#: empty so the verdict comes from the protected arithmetic, not from an
#: unrelated unmeasured arm.
_GATED_SETS = {
    "target_benchmarks": [TARGET],
    "protected_benchmarks": [CONSTRAINED],
    "broad_benchmarks": [],
}

PROFILE_SECTION = {
    "profile_id": "gen2-retention",
    "constraints": [
        {
            "dimension": "protected-math",
            "kind": "max-regression",
            "value": 0.0,
            "benchmark": CONSTRAINED,
        }
    ],
}

TIER_SECTION = {
    "classification": {
        CONSTRAINED: "promotion-evidence",
        TARGET: "promotion-evidence",
    }
}


def _manifest(tmp_path, **overrides) -> CampaignManifest:  # noqa: ANN001
    """A real manifest through the production fixture, with overrides."""
    manifest, _runner, _document = campaign_fixture._campaign(tmp_path, **overrides)
    return manifest


def _cycle(tmp_path, manifest: CampaignManifest):  # noqa: ANN001
    """The production builder: the loader's output reaches the cycle's config."""
    return _build_cycle(
        manifest,
        executor=lambda recipe, items: {},  # noqa: ARG005
        firewall=ContaminationFirewall(),
        root=tmp_path / "state",
    )


def _decide(cycle, *, candidate_target, candidate_constrained, parent_target, parent_constrained):  # noqa: ANN001
    candidate = {
        TARGET: _result(TARGET, candidate_target, origin=MEASURED_THIS_GENERATION),
        CONSTRAINED: _result(
            CONSTRAINED, candidate_constrained, origin=MEASURED_THIS_GENERATION
        ),
    }
    parent = {
        TARGET: _result(TARGET, parent_target, origin=MEASURED_PARENT),
        CONSTRAINED: _result(CONSTRAINED, parent_constrained, origin=MEASURED_PARENT),
    }
    return cycle.decide_promotion(
        candidate_results=candidate,
        parent_results=parent,
        device_gpu_hours=0.01,
    )


# --------------------------------------------------------------------------
# contract 1: parse-valid binds, and enforces identically
# --------------------------------------------------------------------------


def test_a_declared_retention_profile_parses_into_the_domain_type(tmp_path) -> None:  # noqa: ANN001
    manifest = _manifest(tmp_path, retention_profile=PROFILE_SECTION)

    profile = manifest.retention_profile
    assert isinstance(profile, RetentionProfile)
    assert profile.profile_id == "gen2-retention"
    (constraint,) = profile.constraints
    assert constraint.dimension == "protected-math"
    assert constraint.kind == "max-regression"
    assert constraint.value == 0.0
    assert constraint.benchmark == CONSTRAINED


def test_a_declared_eval_tier_policy_parses_into_the_domain_type(tmp_path) -> None:  # noqa: ANN001
    manifest = _manifest(tmp_path, eval_tier_policy=TIER_SECTION)

    policy = manifest.eval_tier_policy
    assert policy is not None
    assert policy.tier_of(CONSTRAINED) is EvalTier.PROMOTION_EVIDENCE
    # Unclassified benchmarks stay promotion evidence by default.
    assert policy.tier_of("unclassified@2026-01") is EvalTier.PROMOTION_EVIDENCE


def test_a_manifest_declared_profile_binds_and_downgrades_a_promotion(tmp_path) -> None:  # noqa: ANN001
    """The loader seam's load-bearing proof.

    The same regression that the predeclared rule alone lets promote is
    REJECTED once the manifest declares the retention profile: the
    configuration reaches the cycle, and the cycle's promotion path enforces
    it through the same gate code a programmatic construction uses.
    """
    manifest = _manifest(tmp_path, **_UNGATED_SETS, retention_profile=PROFILE_SECTION)
    assert manifest.retention_profile is not None

    cycle = _cycle(tmp_path, manifest)
    assert cycle.config.retention_profile is manifest.retention_profile

    decision = _decide(
        cycle,
        candidate_target=0.45,        # the target improved
        candidate_constrained=0.05,   # the constrained dimension regressed
        parent_target=0.28,
        parent_constrained=0.31,
    )

    assert decision.verdict == "REJECTED"
    assert any(
        reason.startswith("RETENTION_REGRESSION") and "protected-math" in reason
        for reason in decision.reasons
    )
    # The predeclared rule itself had passed: the declaration is what turned
    # this promotion into a rejection.
    assert "all predeclared promotion checks passed" in decision.reasons


def test_a_manifest_declared_tier_policy_refuses_a_search_readable_gate_at_load(
    tmp_path,  # noqa: ANN001
) -> None:
    """Both sections declared with the gate demoted into the search's view.

    The tier wall is checked when the campaign is declared, not when a
    promotion first reads it: wiring errors surface before compute.
    """
    demoted = {
        "classification": {CONSTRAINED: "search-evidence", TARGET: "promotion-evidence"}
    }
    with pytest.raises(CampaignManifestError) as error:
        _manifest(
            tmp_path,
            **_UNGATED_SETS,
            retention_profile=PROFILE_SECTION,
            eval_tier_policy=demoted,
        )
    assert "shape its own gate" in str(error.value)


def test_a_declared_gate_is_attributed_even_when_the_rule_already_rejected(
    tmp_path,  # noqa: ANN001
) -> None:
    """A declared gate that fires on an already-rejected candidate is recorded.

    Enforcement was never in question here -- the predeclared protected
    arithmetic had already rejected. What was missing was the attribution: the
    record named the predeclared rule and stayed silent about the declared
    gate that fired on the same candidate, which reads as a gate that passed.
    """
    manifest = _manifest(tmp_path, **_GATED_SETS, retention_profile=PROFILE_SECTION)
    cycle = _cycle(tmp_path, manifest)

    decision = _decide(
        cycle,
        candidate_target=0.45,        # the target improved
        candidate_constrained=0.05,   # the protected benchmark regressed hard
        parent_target=0.28,
        parent_constrained=0.31,
    )

    # The predeclared rule's own verdict and its reason survive untouched.
    assert decision.verdict == "REJECTED"
    assert decision.checks["protected_regression"] == "violated"
    assert "1 protected regression(s)" in decision.reasons

    # And the declared gate's breach is attributed to the same candidate.
    assert any(
        reason.startswith("RETENTION_REGRESSION") and "protected-math" in reason
        for reason in decision.reasons
    ), decision.reasons


def test_a_declared_gate_annotates_but_does_not_strengthen_an_inconclusive_verdict(
    tmp_path,  # noqa: ANN001
) -> None:
    """Annotating an INCONCLUSIVE decision records the breach without
    upgrading the verdict.

    The predeclared rule found the evidence too thin to decide. A declared
    breach is a real fact about that candidate and belongs in the record, but
    it must not manufacture a REJECTED the rule's own evidence never earned --
    a strong verdict over thin evidence is the same category of mistake as a
    promotion over a breached gate.
    """
    manifest = _manifest(tmp_path, **_GATED_SETS, retention_profile=PROFILE_SECTION)
    cycle = _cycle(tmp_path, manifest)

    decision = _decide(
        cycle,
        candidate_target=0.45,
        # Inside the predeclared 0.02 tolerance, outside the declared 0.0 one.
        candidate_constrained=0.30,
        parent_target=0.28,
        parent_constrained=0.31,
    )

    assert decision.checks["protected_regression"] != "violated"
    assert decision.verdict == "INCONCLUSIVE"
    assert any(
        reason.startswith("RETENTION_REGRESSION") and "protected-math" in reason
        for reason in decision.reasons
    ), decision.reasons


# --------------------------------------------------------------------------
# contract 2: malformed declarations fail loudly at load time
# --------------------------------------------------------------------------


def _base_document(tmp_path) -> dict:  # noqa: ANN001
    """One valid manifest document, parsed once, reused for every refusal."""
    _manifest, _runner, document = campaign_fixture._campaign(tmp_path)
    assert "retention_profile" not in document
    assert "eval_tier_policy" not in document
    return document


def test_malformed_declarations_refuse_at_load(tmp_path) -> None:  # noqa: ANN001
    document = _base_document(tmp_path)

    bad_section = {"profile_id": "", "constraints": PROFILE_SECTION["constraints"]}
    bad_constraint = {
        "profile_id": "p",
        "constraints": [{"dimension": "d", "kind": "max-regression", "value": 0.0}],
    }
    unmeasurable = {
        "profile_id": "p",
        "constraints": [
            {
                "dimension": "d",
                "kind": "max-regression",
                "value": 0.0,
                "benchmark": "unmeasured@2026-01",
            }
        ],
    }
    demoted_reserved = {
        "classification": {"protected_suite@2026-01": "search-evidence"}
    }

    cases = [
        (
            {"retention_profile": {**PROFILE_SECTION, "extra": 1}},
            "unknown retention_profile fields",
        ),
        ({"retention_profile": bad_section}, "profile_id"),
        (
            {"retention_profile": {"profile_id": "p", "constraints": []}},
            "non-empty list",
        ),
        ({"retention_profile": bad_constraint}, "must declare exactly"),
        (
            {
                "retention_profile": {
                    "profile_id": "p",
                    "constraints": [
                        {
                            "dimension": "d",
                            "kind": "no-worse",
                            "value": 0.0,
                            "benchmark": CONSTRAINED,
                        }
                    ],
                }
            },
            # The kind rule's owner is the domain type; the loader wraps the
            # domain error with source context.
            "has unknown kind 'no-worse'",
        ),
        (
            {
                "retention_profile": {
                    "profile_id": "p",
                    "constraints": [
                        {
                            "dimension": "d",
                            "kind": "max-regression",
                            "value": "high",
                            "benchmark": CONSTRAINED,
                        }
                    ],
                }
            },
            "finite number",
        ),
        (
            {
                "retention_profile": {
                    "profile_id": "p",
                    "constraints": [
                        {
                            "dimension": "d",
                            "kind": "max-regression",
                            "value": 0.0,
                            "benchmark": "math500@latest",
                        }
                    ],
                }
            },
            "pinned benchmark@version",
        ),
        ({"retention_profile": unmeasurable}, "never cover"),
        ({"eval_tier_policy": {"tiers": TIER_SECTION["classification"]}}, "unknown eval_tier_policy fields"),
        ({"eval_tier_policy": {"classification": {}}}, "non-empty"),
        (
            {
                "eval_tier_policy": {
                    "classification": {CONSTRAINED: "best-evidence"}
                }
            },
            "tier",
        ),
        ({"eval_tier_policy": demoted_reserved}, "reserved promotion-evidence"),
    ]

    for section_override, expected in cases:
        manifest_error = None
        try:
            CampaignManifest.from_mapping(
                {**document, **section_override}, source="<gates-test>"
            )
        except CampaignManifestError as error:
            manifest_error = error
        assert manifest_error is not None, f"expected refusal for {section_override}"
        assert expected in str(manifest_error), (
            f"{expected!r} not named for {section_override}: {manifest_error}"
        )


def test_the_kind_rule_has_one_owner_and_the_loader_wraps_it(tmp_path) -> None:  # noqa: ANN001
    """A change to the kind rule lands in the domain, not in two places.

    The domain type raises its own named error; the loader's refusal carries
    that error with the manifest's source context, so load-time behavior is
    fail-closed without re-implementing the rule.
    """
    with pytest.raises(ValueError) as domain_error:
        RetentionConstraint(dimension="d", kind="no-worse", value=0.0, benchmark=CONSTRAINED)
    assert "has unknown kind" in str(domain_error.value)

    document = _base_document(tmp_path)
    with pytest.raises(CampaignManifestError) as load_error:
        CampaignManifest.from_mapping(
            {
                **document,
                "retention_profile": {
                    "profile_id": "p",
                    "constraints": [
                        {
                            "dimension": "d",
                            "kind": "no-worse",
                            "value": 0.0,
                            "benchmark": CONSTRAINED,
                        }
                    ],
                },
            },
            source="<owner-test>",
        )
    assert "<owner-test>" in str(load_error.value)
    assert str(domain_error.value) in str(load_error.value)


def test_an_undeclared_top_level_key_still_refuses(tmp_path) -> None:  # noqa: ANN001
    """The sections are new schema keys, not an open door: typos still refuse."""
    document = _base_document(tmp_path)
    with pytest.raises(CampaignManifestError) as error:
        CampaignManifest.from_mapping(
            {**document, "retention_profiles": PROFILE_SECTION}, source="<gates-test>"
        )
    assert "unknown manifest fields" in str(error.value)


# --------------------------------------------------------------------------
# contract 3: declaring neither changes nothing
# --------------------------------------------------------------------------


def test_a_manifest_without_the_sections_loads_with_both_unset(tmp_path) -> None:  # noqa: ANN001
    manifest = _manifest(tmp_path)
    assert manifest.retention_profile is None
    assert manifest.eval_tier_policy is None


def test_without_the_sections_the_same_regression_still_promotes(tmp_path) -> None:  # noqa: ANN001
    """The historical behavior, unchanged: no declaration, no gate.

    Identical measurements to the downgrade test above — the only difference
    is the absence of the section, so this pair is the proof that the loader
    added a declaration, not a default.
    """
    manifest = _manifest(tmp_path, **_UNGATED_SETS)
    assert manifest.retention_profile is None

    cycle = _cycle(tmp_path, manifest)
    decision = _decide(
        cycle,
        candidate_target=0.45,
        candidate_constrained=0.05,
        parent_target=0.28,
        parent_constrained=0.31,
    )

    assert decision.verdict == "PROMOTED"
    assert not any(
        reason.startswith("RETENTION_") for reason in decision.reasons
    )
