"""Adversarial CPU-only original cross-platform native prompt witness tests."""

from __future__ import annotations

from collections import Counter

import pytest

from chowder_batch import reap_original_cross_platform_gsm8k_prompt_witness_20261029 as witness


def _canonical_maps():
    # The real original map is deliberately not fabricated as a positive test.
    # It must come from the SHA-pinned native logs in the integration test.
    return {arm: {f"gsm8k-{i}": 100 + i % 37 for i in range(witness.N)}
            for arm in witness.EXPECTED_ARMS}


def test_original_three_session_cross_platform_witness():
    if not witness.windows.ROOT.is_dir():
        pytest.skip("Original immutable F: Windows Q4 evidence unavailable")
    if any(not witness.native.ARMS[arm]["log"].is_file() for arm in ("q4", "q8")):
        pytest.skip("Original immutable F: Linux native evidence unavailable")
    result = witness.verify_original_cross_platform_prompts()
    assert result["canonical_gsm8k_items_per_arm"] == 1319
    assert result["cross_platform_same_canonical_gsm8k_prompt_lengths"] is True
    assert result["per_item_prompt_length_map_sha256"] == witness.EXPECTED_PROMPT_MAP_SHA256
    assert result["declared_prompt_tokens_per_arm"] == 137469
    assert result["prompt_text_authenticated"] is False
    assert result["native_response_text_authenticated"] is False
    assert result["model_weights_authenticated_at_original_execution"] is False
    assert result["frozen_gate1_qualified"] is False
    assert result["frozen_gate2_qualified"] is False
    assert (result["status"], result["chosen"], result["release_approved"]) == (
        "BLOCKED", None, False
    )


def test_missing_original_arm_is_refused():
    maps = _canonical_maps()
    del maps["linux_q8"]
    with pytest.raises(witness.CrossPlatformPromptError, match="missing original"):
        witness.compare_original_prompt_maps(maps)


def test_swapped_prompt_lengths_refused_despite_equal_multiset():
    maps = _canonical_maps()
    linux = maps["linux_q8"]
    before = Counter(linux.values())
    linux["gsm8k-0"], linux["gsm8k-1"] = linux["gsm8k-1"], linux["gsm8k-0"]
    assert Counter(linux.values()) == before
    with pytest.raises(witness.CrossPlatformPromptError, match="per-item prompts differ"):
        witness.compare_original_prompt_maps(maps)


def test_incomplete_or_duplicate_identity_refused():
    maps = _canonical_maps()
    del maps["windows_q4"]["gsm8k-1318"]
    maps["windows_q4"]["gsm8k-0-copy"] = 101
    with pytest.raises(witness.CrossPlatformPromptError, match="incomplete"):
        witness.compare_original_prompt_maps(maps)


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.0, "101", None])
def test_invalid_native_prompt_length_refused(invalid):
    maps = _canonical_maps()
    maps["linux_q4"]["gsm8k-0"] = invalid
    with pytest.raises(witness.CrossPlatformPromptError, match="incomplete"):
        witness.compare_original_prompt_maps(maps)


def test_consistent_but_forged_three_arm_map_cannot_pass_frozen_digest():
    maps = _canonical_maps()
    with pytest.raises(witness.CrossPlatformPromptError, match="mapping digest differs"):
        witness.compare_original_prompt_maps(maps)
