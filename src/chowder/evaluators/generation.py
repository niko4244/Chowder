from __future__ import annotations

from typing import Any, Sequence


def observed_span(
    *, continuation: Sequence[int], max_new_tokens: int, pad_token_id: int | None = None
) -> tuple[list[int], int, bool]:
    """The one rule for what a generation did, batched or not.

    Returns the row's own tokens (as they must be decoded), how many tokens it
    produced, and whether it stopped before the cap.

    One generation at a time a row's tensor is the row, so the caller can read
    its last position. A *batched* generation keeps decoding for the rows still
    running and marks the rows that finished, so a row's own span can be shorter
    than the tensor it sits in. Two things can appear after a row is done, and
    which one does depends on the ``transformers`` version: the step that
    produced the terminator holds it (this checkout's version), or that step is
    rewritten to ``pad_token_id`` while later steps are pads. This rule finds
    the row's end under either -- a trailing run of pads delimits it, and a stop
    token inside that delimitation is the terminator -- so a batched row reports
    the same facts a single-row pass in the same environment would report.

    ``pad_token_id`` is what the batched call pads with; ``None`` means the row
    sits alone and nothing can have been padded after it. Note the rule needs no
    stop-token set: the padding itself says where the row ended.
    """
    tokens = [int(token) for token in continuation]
    trailing = 0
    if pad_token_id is not None:
        while trailing < len(tokens) and tokens[len(tokens) - 1 - trailing] == pad_token_id:
            trailing += 1
    produced = len(tokens) - trailing
    if trailing:
        # The batch kept decoding after this row was done. The row's span is
        # what precedes the padding, and it ended on its own terminator.
        stopped = True
    else:
        # Nothing was padded after it: either it ran into the cap, or the whole
        # batch finished and its last step was a terminator.
        stopped = produced < max_new_tokens
    return tokens[: max(0, produced)], max(0, produced), stopped


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
    It is :func:`observed_span` with no padding, so a batched row and a single
    row are decided by the same code.
    """
    _, produced, stopped = observed_span(
        continuation=generated[0, int(prompt_tokens):].tolist(),
        max_new_tokens=max_new_tokens,
    )
    return {
        "generated_tokens": max(0, produced),
        "eos_terminated": bool(stopped and eos_token_id is not None),
    }


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
