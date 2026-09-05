from __future__ import annotations

from typing import Any


def resolve_eos_token_ids(tokenizer: Any, model: Any) -> int | list[int]:
    """Resolve the id(s) that should stop ``model.generate()``.

    ``generate()`` defaults to ``model.generation_config.eos_token_id`` when
    no explicit ``eos_token_id`` kwarg is passed. Many instruction-tuned
    checkpoints (Qwen2/Qwen3, Llama-3, etc.) ship a ``generation_config.json``
    whose ``eos_token_id`` is a *list* that includes the chat template's
    turn-end token (e.g. Qwen's ``<|im_end|>``) in addition to the
    tokenizer's own base ``eos_token``. Passing the tokenizer's scalar
    ``eos_token_id`` as an explicit override -- as both evaluator workers
    used to do unconditionally, regardless of ``use_chat_template`` --
    discards that list. The model then has no way to signal "the chat turn
    is over" and keeps generating until ``max_new_tokens`` is exhausted,
    which silently breaks every ``use_chat_template=True`` suite scored
    with ``exact_match`` / ``normalized_exact_match``: the correct short
    answer is still in the output, buried in trailing rambling that fails
    full-string comparison.

    Prefer the model's own resolved generation config; fall back to the
    tokenizer's eos id only when the model does not declare one at all
    (e.g. some base/non-instruct configs with no generation_config.json).
    """
    generation_config = getattr(model, "generation_config", None)
    configured = (
        getattr(generation_config, "eos_token_id", None)
        if generation_config is not None
        else None
    )
    if configured is not None:
        return configured
    return tokenizer.eos_token_id
