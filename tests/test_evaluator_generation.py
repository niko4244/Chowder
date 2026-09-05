from types import SimpleNamespace

from chowder.evaluators.generation import resolve_eos_token_ids

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
