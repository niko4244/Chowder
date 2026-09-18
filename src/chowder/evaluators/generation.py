from __future__ import annotations

from typing import Any


def observed_generation(
    *,
    generated: Any,
    prompt_tokens: int,
    eos_token_id: int | list[int] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    """The two facts one generation leaves behind, as prediction-row fields.

    Both text workers record this beside every prediction, because neither fact
    can be recovered from the decoded text afterwards: ``skip_special_tokens``
    drops the EOS token, and a generation that hit ``max_new_tokens`` looks
    exactly like one that stopped early. In the Gen-0 freeze that distinction
    *was* the measured weakness (cap rate 1.000, EOS rate 0.000), so a reader
    that guessed it would be guessing about the finding itself.

    ``generated`` is the model's own output tensor -- one row, prompt plus
    continuation -- so this needs no batch bookkeeping and no torch import.

    The rule is the Gen-1 instrument's, kept in one place so the two workers
    cannot drift apart on it: stopping on EOS means stopping *before* the cap.
    """
    produced = int(generated.shape[1]) - int(prompt_tokens)
    stopped = False
    if produced > 0 and eos_token_id is not None:
        stops = (
            {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
        )
        stopped = produced < max_new_tokens and int(generated[0, -1]) in stops
    return {"generated_tokens": max(0, produced), "eos_terminated": stopped}


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
