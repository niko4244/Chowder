"""Tests for context-budget admission.

The trajectory below is the real measured Q4 AutomationBench prompt sequence
(24 requests, 1,067 -> 32,639 tokens) that died at a 32,768-token ceiling.
"""

from __future__ import annotations

import pytest

from chowder_batch.reap_context_budget_20261026 import (
    DEFAULT_GENERATION_RESERVE,
    FALLBACK_TOKENS_PER_REQUEST,
    affordable_steps,
    assess,
    required_context,
    step_token_growth,
    trajectory_from_native_log,
)

MEASURED_PROMPTS = [
    1067, 3592, 5160, 5303, 6993, 8513, 8651, 10076, 11881, 13573,
    15411, 17471, 20514, 22256, 23734, 24835, 25402, 26754, 27448,
    28215, 28968, 30204, 31951, 32639,
]

DEAD_LOG = "\n".join(
    [
        f"2.0{i:02d}.000.000 I slot   operator(): id  0 | task {10000 + i} | new prompt, "
        f"n_ctx_slot = 32768, n_keep = 0, task.n_tokens = {p}"
        for i, p in enumerate(MEASURED_PROMPTS)
    ]
    + [
        "2.29.054.431 I slot      release: id  0 | task 11911 | stop processing: "
        "n_tokens = 32767, truncated = 1",
        '2.29.060.808 W srv    operator(): got exception: {"error":{"code":500,'
        '"message":"Failed to parse tool call arguments as JSON: [json.exception'
        '.parse_error.101] parse error at line 1, column 89"}}',
    ]
    # Retries of the unchanged overflowing conversation: no growth at all.
    + [
        f"2.{30 + i}.054.431 I slot   operator(): id  0 | task {12000 + i} | "
        f"new prompt, n_ctx_slot = 32768, n_keep = 0, task.n_tokens = 32639"
        for i in range(3)
    ]
)


def test_measured_trajectory_growth_matches_reported_rate():
    stats = step_token_growth(MEASURED_PROMPTS)
    assert stats["requests"] == 24
    assert stats["first_prompt"] == 1067
    assert stats["last_prompt"] == 32639
    # 31,572 net tokens over 23 positive steps.
    assert stats["net_growth"] == 31572
    assert stats["mean_growth"] == pytest.approx(1372.7, abs=0.1)


def test_reclaim_events_are_not_counted_as_growth():
    stats = step_token_growth([100, 500, 450, 900])
    assert stats["positive_steps"] == 2
    assert stats["reclaim_events"] == 1
    assert stats["mean_growth"] == pytest.approx(425.0)


def test_frozen_overflow_is_reported_not_raised():
    stats = step_token_growth([32639, 32639, 32639])
    assert stats["positive_steps"] == 0
    assert stats["mean_growth"] == 0.0
    assert stats["reclaim_events"] == 2


def test_frozen_overflow_is_refused():
    observed = {"prompt_tokens": [32639, 32639, 32639], "truncated_generations": 1}
    result = assess(observed=observed, context=32768, max_steps=50)
    assert result["status"] == "REFUSED"
    assert "OBSERVED_CONVERSATION_FROZEN_AT_CONTEXT_CEILING" in result["reasons"]
    assert result["affordable_steps"] < 50


def test_required_context_is_arithmetic_not_guesswork():
    need = required_context(
        first_prompt=1067, growth=1372.7, max_steps=50,
        generation_reserve=DEFAULT_GENERATION_RESERVE,
    )
    # 1067 + 1372.7*49 + 2048 = 70,377.3, rounded up to a whole token.
    assert need == 70378


def test_measured_trial_is_refused_at_its_own_context():
    observed = trajectory_from_native_log(DEAD_LOG)
    result = assess(observed=observed, context=32768, max_steps=50)
    assert result["status"] == "REFUSED"
    assert result["shortfall"] == 70378 - 32768
    # It died at request 24, so 50 steps was never reachable.
    assert result["affordable_steps"] < 50
    assert "OBSERVED_TRUNCATED_TOOL_CALL_REJECTED_AS_INVALID_JSON" in result["reasons"]
    assert "RETRY_CANNOT_RECOVER_UNCHANGED_OVERFLOW" in result["reasons"]


def test_doubling_context_still_cannot_reach_fifty_steps():
    observed = trajectory_from_native_log(DEAD_LOG)
    result = assess(observed=observed, context=65536, max_steps=50)
    # ~70.4k needed, so 64k is still short: this is why context alone is not a fix.
    assert result["status"] == "REFUSED"
    assert result["required_context"] > 65536


def test_sufficient_context_is_admitted():
    observed = trajectory_from_native_log(DEAD_LOG)
    result = assess(observed=observed, context=73728, max_steps=50)
    assert result["status"] == "ADMITTED_TO_ATTEMPT"
    assert result["affordable_steps"] >= 50
    # The observation still reports the historical failure, but it is not a
    # capacity reason to refuse the *next* launch.
    assert "EPISODE_CANNOT_REACH_MAX_STEPS_IN_CONTEXT" not in result["reasons"]


def test_reduced_step_budget_fits_in_measured_doubled_context():
    need = required_context(
        first_prompt=1067, growth=1372.7, max_steps=45,
        generation_reserve=DEFAULT_GENERATION_RESERVE,
    )
    assert 65536 >= need  # 45 turns do fit in 64k; 50 do not.


def test_fallback_is_used_without_observation():
    result = assess(observed=None, context=16384, max_steps=50)
    assert result["growth_source"] == "measured_fallback_20261026"
    assert result["tokens_per_request"] == FALLBACK_TOKENS_PER_REQUEST
    assert result["status"] == "REFUSED"


def test_single_request_observation_uses_fallback_growth():
    result = assess(observed={"prompt_tokens": [1067]}, context=16384, max_steps=50)
    assert result["first_prompt"] == 1067
    assert result["growth_source"] == "measured_fallback_20261026"


def test_affordable_steps_bounds():
    assert affordable_steps(
        first_prompt=0, growth=1000, context=5000, generation_reserve=0
    ) == 6
    assert affordable_steps(
        first_prompt=9000, growth=1000, context=5000, generation_reserve=0
    ) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_prompt": 0, "growth": 1, "max_steps": 0, "generation_reserve": 1},
        {"first_prompt": 0, "growth": -1, "max_steps": 1, "generation_reserve": 1},
        {"first_prompt": 0, "growth": 1, "max_steps": 1, "generation_reserve": -1},
    ],
)
def test_required_context_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        required_context(**kwargs)


def test_assess_rejects_impossible_reserve():
    with pytest.raises(ValueError):
        assess(observed=None, context=1024, max_steps=5, generation_reserve=1024)


def test_log_parser_counts_failures_without_trusting_harness_summary():
    observed = trajectory_from_native_log(DEAD_LOG)
    assert observed["truncated_generations"] == 1
    assert observed["tool_call_json_failures"] == 1
    assert observed["declared_contexts"] == [32768]
    assert len(observed["prompt_tokens"]) == 27
