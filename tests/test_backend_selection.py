from __future__ import annotations

import pytest

from chowder.backend_selection import (
    BackendSelectionError,
    create_training_executor,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
from chowder.backends.transformers_peft import TransformersPeftExecutor


def _config(backend):
    return {"backend": backend}


def test_legacy_transformers_peft_remains_transformers_without_engine_key():
    config = _config({"type": "transformers-peft"})
    assert resolve_training_engine(config) == "transformers"
    assert normalize_training_config_for_executor(config) == config


def test_legacy_transformers_peft_accepts_explicit_transformers_engine():
    config = _config({"type": "transformers-peft", "engine": "transformers"})
    assert resolve_training_engine(config) == "transformers"
    assert normalize_training_config_for_executor(config) == {
        "backend": {"type": "transformers-peft"}
    }


def test_canonical_peft_transformers_normalizes_to_existing_executor_contract():
    config = _config(
        {
            "type": "peft",
            "engine": "transformers",
            "base_model": "example/model",
        }
    )
    normalized = normalize_training_config_for_executor(config)
    assert resolve_training_engine(config) == "transformers"
    assert normalized == {
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
        }
    }
    assert config["backend"]["type"] == "peft"
    assert config["backend"]["engine"] == "transformers"


def test_canonical_peft_requires_explicit_engine():
    with pytest.raises(BackendSelectionError, match="backend.engine is required"):
        resolve_training_engine(_config({"type": "peft"}))


def test_unknown_peft_engine_fails_closed():
    with pytest.raises(BackendSelectionError, match="unsupported PEFT training engine"):
        resolve_training_engine(_config({"type": "peft", "engine": "mystery"}))


def test_legacy_alias_cannot_silently_select_unsloth():
    with pytest.raises(BackendSelectionError, match="legacy backend.type"):
        resolve_training_engine(
            _config({"type": "transformers-peft", "engine": "unsloth"})
        )


def test_unsloth_is_recognized_but_not_falsely_reported_available():
    config = _config({"type": "peft", "engine": "unsloth"})
    assert resolve_training_engine(config) == "unsloth"
    assert normalize_training_config_for_executor(config) == config
    with pytest.raises(BackendSelectionError, match="isolated executor"):
        create_training_executor(config)


def test_factory_constructs_existing_transformers_executor_for_canonical_config():
    executor = create_training_executor(
        _config({"type": "peft", "engine": "transformers"})
    )
    assert isinstance(executor, TransformersPeftExecutor)
    assert executor.name == "transformers-peft"
