from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from chowder.provenance import sha256_file
from chowder.qwen38_campaign import (
    CampaignManifestError,
    ParentPin,
    Qwen38CampaignManifest,
    SuiteVersionPin,
    default_qwen38_campaign_manifest,
)
from chowder.recursive_repair import RecursiveRepairPolicy

_REAL_REVISION = "404ea47aaa5d8a8b00049c9e9750089aca011ab2"
_REAL_SUITE_SHA = "7946d8c9b14356b5f99c4de3a6a32aef8dbcd7c0ede7180a3bf569bb2ba8473c"


def _real_manifest(**overrides) -> Qwen38CampaignManifest:
    kwargs = dict(
        repair_corpus_files=("repair_corpus.jsonl",),
        repair_variant_names=("more-steps",),
        gpu_hour_budget=100.0,
        minimum_promotion_gain=0.02,
    )
    kwargs.update(overrides)
    return default_qwen38_campaign_manifest(**kwargs)


def test_default_manifest_binds_the_real_program_identities():
    manifest = _real_manifest()
    assert manifest.primary_parent.repo == "orcarouter/Qwen3.8-27B-Uncensored"
    assert manifest.primary_parent.revision == _REAL_REVISION
    assert manifest.native_control.repo == "Qwen/Qwen3.8-27B"
    assert len(manifest.comparison_parents) == 2
    assert manifest.suite_version.manifest_sha256 == _REAL_SUITE_SHA
    assert manifest.training_engine == "unsloth"
    assert manifest.native_qwen3_8_required is True
    assert manifest.distillation_parent_allowed is False
    assert manifest.require_protocol_match is True


def test_manifest_sha256_is_a_real_deterministic_content_hash():
    a = _real_manifest()
    b = _real_manifest()
    assert a.manifest_sha256() == b.manifest_sha256()
    assert len(a.manifest_sha256()) == 64

    expected = hashlib.sha256(
        __import__("json")
        .dumps(a.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()
    assert a.manifest_sha256() == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: replace(m, primary_parent=replace(m.primary_parent, revision="a" * 40)),
        lambda m: replace(m, training_engine="transformers"),
        lambda m: replace(m, repair_corpus_files=("other.jsonl",)),
        lambda m: replace(m, repair_variant_names=("other-variant",)),
        lambda m: replace(m, gpu_hour_budget=200.0),
        lambda m: replace(m, minimum_promotion_gain=0.5),
        lambda m: replace(m, repair_policy=RecursiveRepairPolicy(max_depth=5)),
        lambda m: replace(
            m, suite_version=SuiteVersionPin(name="v2", manifest_sha256="b" * 64)
        ),
    ],
)
def test_manifest_sha256_changes_with_any_real_input_change(mutate):
    baseline = _real_manifest()
    mutated = mutate(baseline)
    assert mutated.manifest_sha256() != baseline.manifest_sha256()


def test_distillation_parent_allowed_is_a_hard_rejected_rule():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="distillation"):
        replace(baseline, distillation_parent_allowed=True)


def test_native_qwen3_8_required_cannot_be_disabled():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="native_qwen3_8_required"):
        replace(baseline, native_qwen3_8_required=False)


def test_empty_comparison_parents_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="comparison parent"):
        replace(baseline, comparison_parents=())


def test_unsupported_training_engine_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="training_engine"):
        replace(baseline, training_engine="bagel")


def test_empty_repair_corpus_files_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="repair_corpus_files"):
        replace(baseline, repair_corpus_files=())


def test_empty_repair_variant_names_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="repair_variant_names"):
        replace(baseline, repair_variant_names=())


def test_duplicate_repair_variant_names_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="unique"):
        replace(baseline, repair_variant_names=("more-steps", "more-steps"))


def test_require_protocol_match_false_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="require_protocol_match"):
        replace(baseline, require_protocol_match=False)


def test_non_positive_gpu_hour_budget_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="gpu_hour_budget"):
        replace(baseline, gpu_hour_budget=0.0)


def test_negative_minimum_promotion_gain_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="minimum_promotion_gain"):
        replace(baseline, minimum_promotion_gain=-0.1)


def test_inverted_active_parameter_range_rejected():
    baseline = _real_manifest()
    with pytest.raises(CampaignManifestError, match="desired_active_parameters"):
        replace(
            baseline,
            desired_active_parameters_min_b=5.0,
            desired_active_parameters_max_b=3.0,
        )


def test_short_revision_rejected():
    with pytest.raises(CampaignManifestError, match="40-character"):
        ParentPin(repo="Qwen/Qwen3.8-27B", revision="404ea47a", role="control")


def test_non_hex_revision_rejected():
    with pytest.raises(CampaignManifestError, match="40-character"):
        ParentPin(repo="Qwen/Qwen3.8-27B", revision="g" * 40, role="control")


def test_short_suite_manifest_sha_rejected():
    with pytest.raises(CampaignManifestError, match="SHA-256"):
        SuiteVersionPin(name="v1", manifest_sha256="abc123")


def test_suite_manifest_digest_matches_the_real_frozen_file_on_this_machine():
    """Regression guard against digest drift: if the real frozen suite file
    at C:\\Users\\nikma\\Chowder-Protected\\suites\\v1\\manifest.json is
    present on this machine, the hardcoded digest in
    default_qwen38_campaign_manifest must still match it exactly. Skips
    cleanly on any other machine (including CI), which has no reason to
    have this machine-local frozen artifact."""

    manifest_path = Path(
        r"C:\Users\nikma\Chowder-Protected\suites\v1\manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip(f"frozen suite manifest not present at {manifest_path}")
    assert sha256_file(manifest_path) == _REAL_SUITE_SHA
