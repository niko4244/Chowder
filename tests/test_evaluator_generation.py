from types import SimpleNamespace

import pytest

from chowder.evaluators.generation import observed_span, resolve_eos_token_ids

PAD = 0
STOP = 1


def test_a_single_row_that_stopped_early_keeps_its_terminator_and_reports_stopping():
    own, produced, stopped = observed_span(
        continuation=[7, 8, STOP], max_new_tokens=5, pad_token_id=None
    )
    assert (own, produced, stopped) == ([7, 8, STOP], 3, True)


def test_a_single_row_that_hit_the_cap_reports_not_stopping():
    own, produced, stopped = observed_span(
        continuation=[7, 8, 9, 10, 11], max_new_tokens=5, pad_token_id=None
    )
    assert (own, produced, stopped) == ([7, 8, 9, 10, 11], 5, False)


def test_a_batched_row_padded_after_it_finished_reports_exactly_its_own_span():
    """The batch ran on for another row; this row's facts may not absorb that."""
    own, produced, stopped = observed_span(
        continuation=[7, 8, STOP, PAD, PAD], max_new_tokens=5, pad_token_id=PAD
    )
    assert (own, produced, stopped) == ([7, 8, STOP], 3, True)


def test_the_last_row_to_finish_is_not_mistaken_for_a_capped_one():
    """Nothing was padded after it, but the batch still ended early."""
    own, produced, stopped = observed_span(
        continuation=[7, 8, STOP], max_new_tokens=5, pad_token_id=PAD
    )
    assert (own, produced, stopped) == ([7, 8, STOP], 3, True)


def test_a_batched_row_that_hit_the_cap_is_reported_as_capped():
    own, produced, stopped = observed_span(
        continuation=[7, 8, 9, 10, 11], max_new_tokens=5, pad_token_id=PAD
    )
    assert (own, produced, stopped) == ([7, 8, 9, 10, 11], 5, False)


def test_a_row_padded_to_the_cap_still_never_reports_the_cap():
    own, produced, stopped = observed_span(
        continuation=[10, STOP, PAD, PAD, PAD], max_new_tokens=5, pad_token_id=PAD
    )
    assert (own, produced, stopped) == ([10, STOP], 2, True)


def test_an_empty_continuation_reports_nothing_produced():
    assert observed_span(continuation=[], max_new_tokens=5, pad_token_id=PAD) == ([], 0, True)


@pytest.mark.parametrize("pad_token_id", [None, PAD])
@pytest.mark.parametrize("max_new_tokens", [1, 5])
def test_a_row_alone_and_a_row_in_a_batch_agree_whenever_nothing_followed_it(
    pad_token_id, max_new_tokens
):
    """Equality is the whole point: one rule decides both cases."""
    continuation = [7, 8, 9][:max_new_tokens]
    without = observed_span(
        continuation=continuation, max_new_tokens=max_new_tokens, pad_token_id=None
    )
    with_pad = observed_span(
        continuation=continuation, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id
    )
    assert without == with_pad

_NO_GENERATION_CONFIG = object()


def _tokenizer(eos_token_id):
    return SimpleNamespace(eos_token_id=eos_token_id)


def _model(generation_config_eos_token_id=_NO_GENERATION_CONFIG):
    if generation_config_eos_token_id is _NO_GENERATION_CONFIG:
        return SimpleNamespace()
    return SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=generation_config_eos_token_id)
    )


def test_prefers_the_models_multi_token_generation_config_over_the_tokenizer_scalar():
    tokenizer = _tokenizer(eos_token_id=151643)
    model = _model(generation_config_eos_token_id=[151643, 151645])
    assert resolve_eos_token_ids(tokenizer, model) == [151643, 151645]


def test_falls_back_to_tokenizer_eos_when_model_has_no_generation_config():
    tokenizer = _tokenizer(eos_token_id=2)
    model = SimpleNamespace()
    assert resolve_eos_token_ids(tokenizer, model) == 2


def test_falls_back_to_tokenizer_eos_when_generation_config_declares_none():
    tokenizer = _tokenizer(eos_token_id=2)
    model = _model(generation_config_eos_token_id=None)
    assert resolve_eos_token_ids(tokenizer, model) == 2


def test_a_single_scalar_generation_config_eos_is_still_honored():
    tokenizer = _tokenizer(eos_token_id=0)
    model = _model(generation_config_eos_token_id=7)
    assert resolve_eos_token_ids(tokenizer, model) == 7
