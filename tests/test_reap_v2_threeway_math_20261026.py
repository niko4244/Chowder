"""No external API access: original-source and hosted-comparison integrity."""

import copy
import hashlib

import pytest

from chowder_batch import reap_v2_threeway_math_20261026 as c


def fixture_response():
    return {"id": "fake-private-call", "model": c.MODELS["qwen38_27b"],
            "choices": [{"finish_reason": "stop", "message": {"content": "The answer is 12"}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 50, "cost": 0.00025}}


def fixture_item():
    return {"id": "gsm8k-17", "prompt": "What is six times two?",
            "expected": "12"}


def test_sample_is_reproducible_unique_independent_of_outcomes():
    first = c.selected_indices()
    assert first == c.selected_indices()
    assert len(first) == 16 == len(set(first))
    assert first == tuple(sorted(first))
    assert min(first) >= 0 and max(first) < 1319
    with pytest.raises(ValueError):
        c.selected_indices(17)


def test_original_full_population_and_source_identity_unchanged():
    assert len(c.original_items()) == 1319
    assert c.reap_eval._digest(c.original_items()) == c.SOURCE_SHA


def test_original_q8_q4_baselines_source_bound_and_no_replay():
    sample = c.selected_indices()
    baselines = c.original_baselines(sample)
    assert set(baselines) == {"reap_q8_original", "reap_q4_original"}
    assert set(baselines["reap_q8_original"]) == {f"gsm8k-{i}" for i in sample}
    assert sum(baselines["reap_q8_original"].values()) == 15
    assert sum(baselines["reap_q4_original"].values()) == 15


def test_request_contains_only_matched_math_and_no_generated_code():
    source = c.request_body(c.MODELS["qwen38_27b"], fixture_item()["prompt"])
    assert source["temperature"] == 0
    assert source["max_tokens"] == 1024
    assert source["messages"] == [
        {"role": "system", "content": c.SYSTEM},
        {"role": "user", "content": fixture_item()["prompt"]}]
    assert source["reasoning"] == {"effort": "medium"}
    with pytest.raises(ValueError):
        c.request_body("anthropic/not-opus", "test")


def test_valid_hosted_scoring_follows_original_final_number_match():
    row = c.parse_response(fixture_response(), c.MODELS["qwen38_27b"], fixture_item(), 2.0)
    assert row["correct"] is True
    assert row["terminated"] is True
    assert row["response_sha256"] == hashlib.sha256(b"The answer is 12").hexdigest()


@pytest.mark.parametrize("change", [
    lambda x: x.update(model="qwen/smaller-model"),
    lambda x: x.update(choices=[]),
    lambda x: x["choices"][0].update(finish_reason="made_up"),
    lambda x: x["choices"][0]["message"].update(content=None),
    lambda x: x.update(usage={}),
    lambda x: x["usage"].update(completion_tokens=True),
    lambda x: x["usage"].update(completion_tokens=2000),
    lambda x: x["usage"].update(cost=float("nan")),
])
def test_no_silent_model_substitution_or_invented_response_success(change):
    source = copy.deepcopy(fixture_response())
    change(source)
    with pytest.raises(ValueError):
        c.parse_response(source, c.MODELS["qwen38_27b"], fixture_item(), 1.0)


def test_max_token_length_termination_cannot_be_correct():
    response = fixture_response()
    response["choices"][0]["finish_reason"] = "length"
    row = c.parse_response(response, c.MODELS["qwen38_27b"], fixture_item(), 1.0)
    assert row["correct"] is False


def test_partial_hosted_comparison_cannot_claim_winner_or_release():
    indices = c.selected_indices()
    observed = c.report(indices, {"qwen38_27b": {}, "opus55": {}}, c.original_baselines(indices))
    assert observed["status"] == "INCOMPLETE_NOT_RANKABLE"
    assert observed["arms"]["opus55"]["accuracy"] is None
    assert observed["gen0_status"] == "BLOCKED"
    assert observed["release_approved"] is False


def test_no_key_or_generated_code_written_in_the_request():
    from inspect import getsource
    source = getsource(c)
    assert "_humaneval_pass(" not in source
    assert "OPENROUTER_API_KEY" in source
    assert '"api_key":' not in source


def test_private_endpoint_model_aliases_are_explicit_and_exact():
    assert c.LOCAL_GATEWAY_MODELS == {
        "qwen38_27b": "groq-qwen3.8-27b", "opus55": "claude-opus-5-5"}
    data = fixture_response()
    data["model"] = "groq/qwen3.8-27b"
    row = c.parse_response(data, "groq-qwen3.8-27b", fixture_item(), 2.0)
    assert row["resolved_model"] == "groq/qwen3.8-27b"


def test_full_paired_synthetic_completion_never_promotes_gen0():
    indices = c.selected_indices()
    original = c.original_baselines(indices)
    ids = {f"gsm8k-{i}" for i in indices}
    synthetic = {"qwen38_27b": {i: {"correct": True} for i in ids},
                 "opus55": {i: {"correct": False} for i in ids}}
    report = c.report(indices, synthetic, original)
    assert report["status"] == "COMPLETE_PAIRED_FIXED_SAMPLE_RESEARCH_ONLY"
    assert report["arms"]["qwen38_27b"]["accuracy"] == 1.0
    assert report["arms"]["opus55"]["accuracy"] == 0.0
    assert report["gen0_status"] == "BLOCKED"
    assert report["release_approved"] is False
