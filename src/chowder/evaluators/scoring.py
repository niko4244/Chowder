"""The one scoring rule, shared by every text evaluator worker.

It used to be two. `base_text_worker` and `transformers_text_worker` each carried
their own copy of `_normalize`, `_FINAL_NUMBER`, `_final_number` and `_score`, and
the copies had drifted on the case that matters most for reasoning models: the base
worker discarded an unclosed ``<think>`` block before extracting an answer, and the
transformers worker scored the raw generation.

That is not a cosmetic difference. Chowder runs the automatic baseline through one
worker and the candidate through the other, so the two sides of a comparison were
not scored by the same rule. In the 2026-09-11 GSM8K run **all 50** baseline
responses had an unclosed ``<think>``, which made the recorded baseline 0.00 *by
construction* and the arm uninterpretable
(`docs/PRUNED_9B_REAL_TRAINING_CORRECTION.md`).

The strict rule wins, and not because it is more conservative
---------------------------------------------------------------
An unclosed ``<think>`` means the generation budget ran out mid-reasoning: the model
never emitted an answer span at all. Reading a number out of that text scores the
*chain of thought*, which is the one region of the output explicitly not the answer.

The lenient rule does not merely flatter such a response, it can score it **correct**.
In the dense-vs-pruned control the pruned checkpoint degenerated on 8 of 8 prompts
and still posted 0.125, because `final_number_match` takes the last number and one
repetition loop happened to stop on the right one
(`docs/PRUNED_9B_REAL_TRAINING_RESULT.md`). A scorer that rewards that is measuring
luck. Scoring it as a miss is the honest reading: failing to finish thinking inside
the configured budget is a real capability limit of the protocol.

Consequence, stated plainly: thinking models under tight `max_new_tokens` score
*lower* here than under the old transformers-worker rule. That is the correction,
not a regression. Raise `max_new_tokens` if the budget is the binding constraint --
do not loosen the scorer to hide it.

A second kind of scoring lives here too: one defined over what the generation
*was observed to do* rather than over its text. `eos_termination` scores an item
1.0 when the model stopped on its own end-of-turn token and 0.0 when it ran into
the token cap, which makes the row's mean -- the score every consumer reads --
the EOS termination rate itself. That is the Gen-1 instrument's definition, kept
here so the two workers cannot implement it two ways.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

__all__ = [
    "OBSERVED_SCORINGS",
    "SAMPLE_SEPARATOR",
    "TOOL_CALL_REASONS",
    "explain_tool_call",
    "final_answer",
    "final_number",
    "majority_vote_final_number",
    "normalize",
    "observed_score",
    "reasoning_answer",
    "reasoning_final_number",
    "parse_tool_calls",
    "score",
    "tool_schema_for",
    "validate_arguments",
]

#: Scorings defined over the recorded observation instead of the decoded text.
OBSERVED_SCORINGS = ("eos_termination",)


def observed_score(observation: Mapping[str, Any], scoring: str) -> float | None:
    """The item score for an observation-defined scoring, else ``None``.

    ``None`` -- not 0.0 -- means "this is a text scoring": the caller then
    scores the prediction against the expected answer with :func:`score`. The
distinction matters because a silent 0.0 here would score every item of every
suite as a failure.
    """
    if scoring == "eos_termination":
        return 1.0 if observation.get("eos_terminated") else 0.0
    return None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def final_answer(prediction: str) -> str:
    """Extract a thinking model's final answer from its raw generation.

    Qwen3-style reasoning models emit chain-of-thought, then a ``</think>`` close
    marker, then the answer. The answer is everything after the LAST close marker;
    without any marker the whole prediction is the answer (non-thinking models, or
    thinking disabled). An *unclosed* ``<think>`` means the budget was exhausted
    mid-reasoning, so there is no answer yet and the extraction is empty.
    """
    if "</think>" in prediction:
        return prediction.rsplit("</think>", 1)[1]
    if "<think>" in prediction:
        return ""
    return prediction


#: Final-number extraction for arithmetic word problems (GSM8K and similar).
#: The regex is comma-aware and the LAST number wins, both learned the hard way:
#: the frontier repo's FINDINGS-GSM8K-EVAL-BUG.md records an extractor using
#: r"[-]?\d+(?:\.\d+)?" that split "$70,000" into ["70", "000"] and scored a CORRECT
#: answer wrong, so a self-improvement loop then "trained on a failure" that was not
#: one. Commas are stripped and a trailing ".0" normalised, so
#: "70,000" == "70000" == "70000.0".
_FINAL_NUMBER = re.compile(r"[-]?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    matches = _FINAL_NUMBER.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw.endswith("."):
        raw = raw[:-1]
    return raw or None


def reasoning_answer(prediction: str) -> str:
    """Extract the answer span for templates that open ``<think>`` themselves.

    Spark-X2.5-style chat templates end the generation prompt with ``<think>``
    already emitted, so the model's generation begins by CLOSING thinking:
    ``</think>answer`` (sometimes followed by a trailing ``</think>`` as its
    end-of-turn marker). For that shape the answer span is between the FIRST
    ``</think>`` and the NEXT one (or end of generation) -- taking everything
    after the LAST marker, as :func:`final_answer` does, reads the empty tail
    after the trailing marker and scores a correct answer as a miss.

    The strict budget rule is unchanged: a prediction with no ``</think>`` at
    all but an unclosed ``<think>`` means the generation was exhausted
    mid-reasoning, so there is no answer span and this extraction is empty.
    A prediction with neither marker is answered whole (non-thinking output).
    """
    if "</think>" in prediction:
        after = prediction.split("</think>", 1)[1]
        if "</think>" in after:
            return after.split("</think>", 1)[0]
        return after
    if "<think>" in prediction:
        return ""
    return prediction


def reasoning_final_number(prediction: str) -> str | None:
    """Final number within the reasoning answer span.

    Same answer-span extraction as :func:`reasoning_answer`, then the
    :func:`final_number` comparison rule. Exists because ``final_number_match``
    builds on :func:`final_answer`, which takes everything after the LAST
    ``</think>`` -- and on templates that emit a trailing ``</think>``
    end-of-turn marker after the answer (Spark-X2.5), that tail is empty, so a
    correct answer scores as a miss. Here the number is read from the span
    between the first and next ``</think>``, which is where the answer lives
    for that template shape.
    """
    return final_number(reasoning_answer(prediction))


#: Joins the K sampled chains of one self-consistency row in its prediction
#: text. It must not collide with real model output: a literal "\n<sample>\n"
#: is not a string a GSM8K-style answer emits, and the worker's
#: ``skip_special_tokens`` decode never produces it.
SAMPLE_SEPARATOR = "\n<sample>\n"


def majority_vote_final_number(prediction: str, expected: str) -> float:
    """Score one self-consistency row: majority vote over K sampled chains.

    The row's ``prediction`` is K chains joined by :data:`SAMPLE_SEPARATOR`;
    each chain's final number is extracted with the same reasoning-span rule
    as :func:`reasoning_final_number` (per chain, so a chain that overruns
    its own budget scores as its own miss rather than poisoning the others).
    The row is correct when a strict majority of chains -- ``> K / 2`` --
    extracted the expected number. A tie (only possible when fewer than half
    the chains produced the answer) is a miss, and a row with no majority is
    scored against the single-shot rule, not the vote.
    """
    chains = prediction.split(SAMPLE_SEPARATOR)
    votes: dict[str, int] = {}
    for chain in chains:
        got = final_number(reasoning_answer(chain))
        if got is not None:
            votes[got] = votes.get(got, 0) + 1
    want = final_number(expected)
    if want is None or not votes:
        return 0.0
    top, count = max(votes.items(), key=lambda kv: (kv[1], kv[0] == want))
    if count * 2 <= len(chains):
        return 0.0
    return float(top == want)


# --------------------------------------------------------------------------
# tool-use scoring: does the next turn pick the right tool with valid,
# correct arguments?  Pure parsing and comparison -- no model, no dependency.
# --------------------------------------------------------------------------

#: The reason codes ``tool_call_exact`` distinguishes, in decision order.  The
#: scorer returns only the float every other scorer returns; the *reason* is
#: exposed by :func:`explain_tool_call` so a caller that can only pass the two
#: strings (:func:`score`) is not the only way to see why an item missed.
TOOL_CALL_REASONS: tuple[str, ...] = (
    "ok",
    "no_call",
    "multiple_calls",
    "wrong_name",
    "schema_invalid",
    "args_mismatch",
)

#: Qwen's native tool-call block.  Qwen2.5/3 wrap the call in
#: ``<tool_call>...</tool_call>``; the JSON form (``{"name": ..., "arguments":
#: {...}}``) and the XML form the Qwen3.5/3.6 chat template emits
#: (``<function=name><parameter=k>v</parameter></function>``) both appear here.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _coerce_xml_value(raw: str) -> Any:
    """One ``<parameter=...>`` value: JSON when it decodes, else the raw text.

    The template emits a string argument raw and a non-string one through
    ``tojson`` (chat_template.jinja line 122), so ``5``/``true``/``[...]`` are
    reconstructed while ``2024-06-01`` stays a string.  The one ambiguity this
    format cannot resolve -- a *string* whose text looks like JSON -- is
    documented rather than hidden; the JSON form has no such case.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_json_tool_call(block: str) -> dict[str, Any] | None:
    try:
        document = json.loads(block)
    except ValueError:
        return None
    if not isinstance(document, Mapping) or "name" not in document:
        return None
    arguments = document.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, Mapping):
        return None
    return {"name": str(document["name"]), "arguments": dict(arguments)}


def _parse_xml_tool_call(block: str) -> dict[str, Any] | None:
    match = _XML_FUNCTION.search(block)
    if match is None:
        return None
    arguments = {
        key: _coerce_xml_value(raw)
        for key, raw in _XML_PARAMETER.findall(match.group(2))
    }
    return {"name": match.group(1), "arguments": arguments}


def parse_tool_calls(completion: str) -> list[dict[str, Any]]:
    """Every tool call in one completion, in the order it appears.

    A block that is neither the JSON form nor the XML form contributes no call:
    a malformed emission must read as "the model did not make a usable call",
    never as a call with guessed arguments.
    """
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_BLOCK.findall(completion or ""):
        parsed = _parse_json_tool_call(block.strip()) or _parse_xml_tool_call(block)
        if parsed is not None:
            calls.append(parsed)
    return calls


#: JSON-schema ``type`` predicates.  ``bool`` is excluded from the numeric
#: types on purpose: Python makes ``True == 1``, and a JSON boolean is not a
#: number in any schema that declares one.  An integral float (``3.0``) is an
#: integer -- the scorer already compares ``3`` and ``3.0`` as equal, and a
#: model that renders a count with a decimal point made the same call.
_TYPE_CHECKS: Mapping[str, Any] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: (
        (isinstance(value, int) and not isinstance(value, bool))
        or (isinstance(value, float) and value.is_integer())
    ),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
    "object": lambda value: isinstance(value, Mapping),
    "array": lambda value: isinstance(value, list),
}


def _schema_error(value: Any, schema: Any) -> str | None:
    """``None`` if ``value`` satisfies ``schema``, else a short reason.

    The subset the instrument's tool schemas use: ``type``, ``properties``,
    ``required``, ``items``, ``enum``.  A schema with ``properties``/``required``
    is treated as an object even when it omits ``type`` (the Nemotron
    ``place_order`` schema does), so a non-object there is a type error.
    """
    if not isinstance(schema, Mapping):
        return None
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return "enum"
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or [])
    if ("properties" in schema or "required" in schema) and "object" not in types:
        types.append("object")
    if "items" in schema and "array" not in types:
        types.append("array")
    if types and not any(
        _TYPE_CHECKS.get(name, lambda _value: False)(value) for name in types
    ):
        return "type"
    if isinstance(value, Mapping):
        for key in schema.get("required") or []:
            if key not in value:
                return "required"
        for key, sub_schema in (schema.get("properties") or {}).items():
            if key in value:
                error = _schema_error(value[key], sub_schema)
                if error is not None:
                    return error
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for element in value:
            error = _schema_error(element, schema["items"])
            if error is not None:
                return error
    return None


def validate_arguments(arguments: Any, schema: Any) -> bool:
    """Whether ``arguments`` satisfies ``schema`` (see :func:`_schema_error`)."""
    return _schema_error(arguments, schema) is None


def tool_schema_for(tools: Any, name: str) -> Mapping[str, Any] | None:
    """The ``parameters`` schema of the named tool, or ``None`` if not declared."""
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and function.get("name") == name:
            parameters = function.get("parameters")
            return parameters if isinstance(parameters, Mapping) else {}
    return None


def _canonical(value: Any) -> Any:
    """A comparison form: strings stripped, numbers numeric, key order ignored.

    ``"5"`` and ``5`` differ (a string is not a number); ``5`` and ``5.0`` do
    not; ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` do not.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value.strip())
    if value is None:
        return ("null",)
    if isinstance(value, (int, float)):
        return ("num", value)
    if isinstance(value, Mapping):
        return ("obj", tuple(sorted((str(key), _canonical(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("arr", tuple(_canonical(item) for item in value))
    return ("other", repr(value))


def _tool_call_reason(
    completion: str, name: str, arguments: Any, schema: Any
) -> str:
    """The reason code for one expected call against one completion."""
    calls = parse_tool_calls(completion)
    if not calls:
        return "no_call"
    if len(calls) != 1:
        return "multiple_calls"
    call = calls[0]
    if call["name"] != name:
        return "wrong_name"
    if not validate_arguments(call["arguments"], schema):
        return "schema_invalid"
    if _canonical(call["arguments"]) != _canonical(arguments):
        return "args_mismatch"
    return "ok"


def explain_tool_call(completion: str, item: Mapping[str, Any]) -> str:
    """Why one instrument item's completion did or did not score (see

    :data:`TOOL_CALL_REASONS`).  ``item`` is one item from the committed
    instrument -- it carries the tool schemas beside the expected call -- or one
    row from the slice, whose ``expected`` already carries the schema.
    """
    expected = item.get("expected", {}) if isinstance(item, Mapping) else {}
    if not isinstance(expected, Mapping):
        expected = {}
    name = str(expected.get("name", ""))
    schema = expected.get("parameters")
    if not isinstance(schema, Mapping):
        schema = tool_schema_for(item.get("tools", ()), name)
    return _tool_call_reason(
        final_answer(completion),
        name,
        expected.get("arguments", {}),
        schema if schema is not None else {},
    )


def _tool_call_exact(prediction: str, expected: str) -> float:
    """Score ``tool_call_exact``: 1.0 iff the one call matches exactly.

    ``expected`` is the item's expected call as JSON.  It carries the tool's
    ``parameters`` schema beside ``name``/``arguments`` because the two-argument
    :func:`score` signature has no other way to reach the schema; the committed
    item file keeps ``expected`` as the bare call and the slice adds the schema.
    """
    try:
        document = json.loads(expected)
    except ValueError:
        return 0.0
    if not isinstance(document, Mapping) or "name" not in document or "arguments" not in document:
        return 0.0
    schema = document.get("parameters")
    reason = _tool_call_reason(
        prediction,
        str(document["name"]),
        document["arguments"],
        schema if isinstance(schema, Mapping) else {},
    )
    return 1.0 if reason == "ok" else 0.0


def score(prediction: str, expected: str, scoring: str) -> float:
    """Score one prediction. Thinking-aware extraction applies to every mode."""
    if scoring == "reasoning_answer_match":
        return float(normalize(reasoning_answer(prediction)) == normalize(expected))
    if scoring == "reasoning_final_number_match":
        # Same reasoning-span rule as reasoning_answer_match, with the
        # final-number comparison for arithmetic tasks: a model that shows
        # its work cannot exact-match a bare number.
        got = reasoning_final_number(prediction)
        want = final_number(expected)
        if got is None or want is None:
            return 0.0
        return float(got == want)
    if scoring == "self_consistency_final_number_match":
        return majority_vote_final_number(prediction, expected)
    answer = final_answer(prediction)
    if scoring == "tool_call_exact":
        return _tool_call_exact(answer, expected)
    if scoring == "exact_match":
        return float(answer.strip() == expected.strip())
    if scoring == "normalized_exact_match":
        return float(normalize(answer) == normalize(expected))
    if scoring == "final_number_match":
        # Compare the last number on each side, not the whole string: a model that
        # shows its work cannot exact-match a bare answer, and scoring it wrong
        # would manufacture failures rather than measure them.
        got = final_number(answer)
        want = final_number(expected)
        if got is None or want is None:
            return 0.0
        return float(got == want)
    raise ValueError(f"unsupported scoring: {scoring}")
