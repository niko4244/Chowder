"""The ``tool-use@reap-moe-v1`` instrument: its scorer, its committed items, and
its wiring into campaign preparation.

Nothing here loads a model or touches a GPU. The scorer is pure parsing; the
instrument's items are read from the committed file; the builder is exercised on
tiny synthetic sources so the held-out rule, the leak check and the per-tool cap
are each proved to *fire* on inputs that violate them, not just to be absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))  # chowder_batch is a script dir

from build_tool_use_instrument import (  # noqa: E402
    ITEMS_NAME,
    MANIFEST_NAME,
    build_instrument,
)
from chowder.evaluators import scoring, transformers_text  # noqa: E402
from chowder.growth import tool_use_instrument as instrument  # noqa: E402
from chowder.growth.campaign_prepare import (  # noqa: E402
    CampaignPrepareRefusal,
    PREPARE_SLICE_UNAVAILABLE,
    _scoring_for,
    load_pinned_slices,
)

_WEATHER = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city"],
}


def _expected(name: str, arguments: object, parameters: object = _WEATHER) -> str:
    return json.dumps({"name": name, "arguments": arguments, "parameters": parameters})


def _item(name: str, arguments: object, parameters: object) -> dict:
    """One committed-instrument item: the expected call plus the tool schemas."""
    return {
        "expected": {"name": name, "arguments": arguments},
        "tools": [{"type": "function", "function": {"name": name, "parameters": parameters}}],
    }


# -- the scorer -------------------------------------------------------------


def test_json_tool_call_block_scores_one() -> None:
    completion = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"city": "Paris", "days": 3}}</tool_call>'
    )
    assert scoring.score(completion, _expected("get_weather", {"city": "Paris", "days": 3}), "tool_call_exact") == 1.0


def test_xml_tool_call_block_scores_one_and_reconstructs_types() -> None:
    """The Qwen3.5/3.6 template form, with ``days`` a number and a date a string."""
    completion = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter>"
        "<parameter=days>3</parameter>"
        "<parameter=on>2024-06-01</parameter>"
        "</function></tool_call>"
    )
    expected = _expected(
        "get_weather", {"city": "Paris", "days": 3, "on": "2024-06-01"}
    )
    assert scoring.score(completion, expected, "tool_call_exact") == 1.0


def test_bare_json_without_the_wrapper_is_no_call() -> None:
    completion = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    item = {"expected": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    assert scoring.score(completion, _expected("get_weather", {"city": "Paris"}), "tool_call_exact") == 0.0
    assert scoring.explain_tool_call(completion, item) == "no_call"


def test_two_blocks_are_multiple_calls() -> None:
    block = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    item = {"expected": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    assert scoring.explain_tool_call(block * 2, item) == "multiple_calls"
    assert scoring.score(block * 2, _expected("get_weather", {"city": "Paris"}), "tool_call_exact") == 0.0


def test_a_different_tool_is_wrong_name() -> None:
    completion = '<tool_call>{"name": "get_time", "arguments": {"city": "Paris"}}</tool_call>'
    item = {"expected": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    assert scoring.explain_tool_call(completion, item) == "wrong_name"


def test_missing_required_argument_is_schema_invalid() -> None:
    item = _item("get_weather", {"city": "Paris"}, _WEATHER)
    completion = '<tool_call>{"name": "get_weather", "arguments": {"days": 3}}</tool_call>'
    assert scoring.explain_tool_call(completion, item) == "schema_invalid"


def test_wrong_argument_type_is_schema_invalid_and_bool_is_not_integer() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    item = _item("f", {"n": 1}, schema)
    completion = '<tool_call>{"name": "f", "arguments": {"n": true}}</tool_call>'
    assert scoring.explain_tool_call(completion, item) == "schema_invalid"
    assert not scoring.validate_arguments({"n": True}, schema)
    assert scoring.validate_arguments({"n": 2}, schema)


def test_enum_violation_is_schema_invalid() -> None:
    schema = {"type": "object", "properties": {"reason": {"enum": ["late", "faulty"]}}}
    item = _item("cancel", {"reason": "late"}, schema)
    completion = '<tool_call>{"name": "cancel", "arguments": {"reason": "other"}}</tool_call>'
    assert scoring.explain_tool_call(completion, item) == "schema_invalid"


def test_a_valid_call_with_the_wrong_value_is_args_mismatch() -> None:
    item = {"expected": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    completion = '<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>'
    assert scoring.explain_tool_call(completion, item) == "args_mismatch"


def test_arguments_compare_order_insensitively_and_numerically() -> None:
    expected = _expected("get_weather", {"city": "Paris", "days": 3})
    completion = '<tool_call>{"name": "get_weather", "arguments": {"days": 3.0, "city": "  Paris  "}}</tool_call>'
    assert scoring.score(completion, expected, "tool_call_exact") == 1.0


def test_a_string_is_not_a_number() -> None:
    """Unconstrained schema: the only difference is the argument type."""
    open_object = {"type": "object", "properties": {}}
    expected = _expected("f", {"n": 5}, open_object)
    completion = '<tool_call>{"name": "f", "arguments": {"n": "5"}}</tool_call>'
    assert scoring.score(completion, expected, "tool_call_exact") == 0.0


def test_explain_reads_the_schema_from_the_items_tools() -> None:
    item = {
        "expected": {"name": "get_weather", "arguments": {"city": "Paris"}},
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": _WEATHER}}],
    }
    completion = '<tool_call>{"name": "get_weather", "arguments": {"days": 3}}</tool_call>'
    assert scoring.explain_tool_call(completion, item) == "schema_invalid"
    assert scoring.tool_schema_for(item["tools"], "absent") is None


def test_unparseable_expected_scores_zero() -> None:
    assert scoring.score("anything", "not json", "tool_call_exact") == 0.0
    assert scoring.score("anything", json.dumps({"name": "f"}), "tool_call_exact") == 0.0


def test_a_call_committed_after_thinking_scores_one() -> None:
    call = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    expected = _expected("get_weather", {"city": "Paris"})
    assert scoring.score(f"reasoning...</think>{call}", expected, "tool_call_exact") == 1.0


def test_a_call_only_inside_thinking_is_no_call() -> None:
    """Same extraction as every other mode: only the final answer is scored."""
    call = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    expected = _expected("get_weather", {"city": "Paris"})
    assert scoring.score(f"<tool_call>{call}", expected, "tool_call_exact") == 0.0


def test_other_scorings_are_unchanged() -> None:
    assert scoring.score("391", "391", "exact_match") == 1.0


def test_the_spec_accepts_the_scoring_the_workers_implement() -> None:
    assert "tool_call_exact" in transformers_text._ALLOWED_SCORING


# -- the committed instrument ----------------------------------------------


def test_the_held_out_rule_is_the_public_formula() -> None:
    for uuid in ("a", "b", "c", "ff6ab2b0-7551-468c-8606-a7c3b0c75b68"):
        expected = int(hashlib.sha256(uuid.encode()).hexdigest()[:8], 16) % 10 == 0
        assert instrument.is_reserved_for_evaluation(uuid) is expected


def test_the_committed_items_verify_against_the_manifest() -> None:
    manifest = instrument.load_manifest()
    items = instrument.load_items()
    assert len(items) == manifest["count"] == 240
    assert hashlib.sha256(instrument.items_path().read_bytes()).hexdigest() == manifest["items_sha256"]


def test_no_committed_item_is_held_out_or_ill_formed() -> None:
    for item in instrument.load_items():
        assert set(item) == {"uuid", "tools", "messages", "expected"}
        assert not instrument.is_reserved_for_evaluation(item["uuid"])
        assert isinstance(item["expected"]["name"], str) and item["expected"]["name"]
        assert isinstance(item["expected"]["arguments"], dict)
        assert any(m.get("role") == "user" for m in item["messages"])
        assert {m.get("role") for m in item["messages"]} <= {"system", "user", "tool", "assistant"}


def test_no_single_tool_dominates_the_instrument() -> None:
    manifest = instrument.load_manifest()
    counts = manifest["per_tool_counts"]
    assert counts and max(counts.values()) * 10 <= manifest["count"]


def test_the_slice_scoring_the_expected_call_back_scores_one() -> None:
    """Every committed item round-trips: its own call, re-rendered, is a 1.0."""
    for row in instrument.load_tool_use_slice():
        expected = json.loads(row["expected"])
        assert set(expected) == {"name", "arguments", "parameters"}
        completion = "<tool_call>" + json.dumps(
            {"name": expected["name"], "arguments": expected["arguments"]}
        ) + "</tool_call>"
        assert scoring.score(completion, row["expected"], "tool_call_exact") == 1.0


def test_the_slice_prompt_carries_the_frozen_context() -> None:
    items = instrument.load_items()
    rows = instrument.load_tool_use_slice()
    assert len(rows) == len(items) == 240
    prompt = json.loads(rows[0]["prompt"])
    assert set(prompt) == {"tools", "messages"}
    assert prompt["messages"] == items[0]["messages"]
    assert json.loads(rows[0]["expected"])["arguments"] == items[0]["expected"]["arguments"]


def test_a_digest_mismatch_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instrument, "INSTRUMENTS_DIR", tmp_path)
    (tmp_path / ITEMS_NAME).write_text('{"uuid": "x"}\n', encoding="utf-8")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"items_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(instrument.ToolUseInstrumentRefusal) as exc:
        instrument.load_items()
    assert "DIGEST_MISMATCH" in str(exc.value)


def test_a_missing_items_file_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instrument, "INSTRUMENTS_DIR", tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"items_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(instrument.ToolUseInstrumentRefusal):
        instrument.load_items()


# -- the campaign wiring ----------------------------------------------------


def test_preparation_wires_the_scoring_and_the_items() -> None:
    items = load_pinned_slices("tool-use@reap-moe-v1")
    assert len(items) == 240
    assert set(items[0]) == {"prompt", "expected"}
    assert _scoring_for("tool-use@reap-moe-v1") == "tool_call_exact"


def test_preparation_surfaces_an_instrument_refusal_as_a_slice_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(instrument, "INSTRUMENTS_DIR", tmp_path)
    (tmp_path / ITEMS_NAME).write_text('{"uuid": "x"}\n', encoding="utf-8")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"items_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(CampaignPrepareRefusal) as exc:
        load_pinned_slices("tool-use@reap-moe-v1")
    assert PREPARE_SLICE_UNAVAILABLE in str(exc.value)


# -- the builder ------------------------------------------------------------


def _uuid_of_kind(reserved: bool) -> str:
    index = 0
    while True:
        uuid = f"00000000-0000-0000-0000-{index:012d}"
        if instrument.is_reserved_for_evaluation(uuid) is reserved:
            return uuid
        index += 1


def _row(uuid: str, tool: str, arguments: object, user: str, calls: int = 1) -> dict:
    tool_calls = [
        {"type": "function", "function": {"name": tool, "arguments": json.dumps(arguments)}}
        for _ in range(calls)
    ]
    return {
        "uuid": uuid,
        "tools": [{"type": "function", "function": {"name": tool, "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "tool_calls": tool_calls, "content": ""},
        ],
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _inputs(tmp_path: Path, rows: list[dict], calibration: list[dict]) -> tuple[Path, Path]:
    return (
        _write(tmp_path / "source.jsonl", rows),
        _write(tmp_path / "calibration.jsonl", calibration),
    )


def _build(tmp_path: Path, rows: list[dict], calibration: list[dict], count: int = 4) -> dict:
    source, cal = _inputs(tmp_path, rows, calibration)
    return build_instrument(source=source, calibration=cal, out_dir=tmp_path / "out", count=count)


def test_the_builder_is_byte_identical_across_runs(tmp_path: Path) -> None:
    rows = [_row(_uuid_of_kind(False), f"tool{i % 2}", {"a": i}, f"context number {i} " * 6) for i in range(8)]
    source, cal = _inputs(tmp_path, rows, [{"text": "unrelated"}])
    first = build_instrument(source=source, calibration=cal, out_dir=tmp_path / "a", count=4)
    second = build_instrument(source=source, calibration=cal, out_dir=tmp_path / "b", count=4)
    assert (tmp_path / "a" / ITEMS_NAME).read_bytes() == (tmp_path / "b" / ITEMS_NAME).read_bytes()
    assert (tmp_path / "a" / MANIFEST_NAME).read_bytes() == (tmp_path / "b" / MANIFEST_NAME).read_bytes()
    assert "timestamp" not in json.dumps(first)
    assert first == second


def test_a_reserved_uuid_is_dropped(tmp_path: Path) -> None:
    rows = [
        _row(_uuid_of_kind(True), "tool", {"a": 1}, "held out " * 9),
        _row(_uuid_of_kind(False), "tool", {"a": 2}, "kept " * 16),
    ]
    manifest = _build(tmp_path, rows, [{"text": "unrelated"}], count=4)
    assert manifest["held_out"]["reserved_uuids"] == 1
    kept = [json.loads(line) for line in (tmp_path / "out" / ITEMS_NAME).read_text(encoding="utf-8").splitlines()]
    assert [item["uuid"] for item in kept] == [rows[1]["uuid"]]


def test_an_item_leaking_into_the_calibration_corpus_is_dropped(tmp_path: Path) -> None:
    leaked = "the exact same long passage the model was calibrated on, verbatim, all of it"
    assert len(leaked) >= 64
    rows = [
        _row(_uuid_of_kind(False), "tool", {"a": 1}, leaked),
        _row(_uuid_of_kind(False), "tool", {"a": 2}, "a different passage that is only in the instrument " * 2),
    ]
    manifest = _build(tmp_path, rows, [{"text": f"prefix {leaked} suffix"}], count=4)
    assert manifest["leak_check"]["dropped"] == 1
    kept = (tmp_path / "out" / ITEMS_NAME).read_text(encoding="utf-8")
    assert leaked not in kept


def test_no_single_tool_may_exceed_the_cap(tmp_path: Path) -> None:
    rows = [_row(_uuid_of_kind(False), "only_tool", {"a": i}, f"context {i} " * 9) for i in range(30)]
    manifest = _build(tmp_path, rows, [{"text": "unrelated"}], count=10)
    assert manifest["per_tool_counts"] == {"only_tool": 1}  # cap = 10 // 10


def test_round_robin_spreads_across_tools(tmp_path: Path) -> None:
    rows = [
        _row(_uuid_of_kind(False), name, {"a": i}, f"context {name} {i} " * 9)
        for i, name in enumerate(["alpha", "alpha", "alpha", "beta", "beta", "beta"])
    ]
    manifest = _build(tmp_path, rows, [{"text": "unrelated"}], count=20)
    assert manifest["per_tool_counts"] == {"alpha": 2, "beta": 2}


def test_rows_that_do_not_qualify_contribute_nothing(tmp_path: Path) -> None:
    two_calls = _row(_uuid_of_kind(False), "tool", {"a": 1}, "context " * 9, calls=2)
    string_args = _row(_uuid_of_kind(False), "tool", {"a": 1}, "context " * 9)
    string_args["messages"][1]["tool_calls"][0]["function"]["arguments"] = '"a bare string"'
    no_user = _row(_uuid_of_kind(False), "tool", {"a": 1}, "context " * 9)
    no_user["messages"] = no_user["messages"][1:]
    manifest = _build(tmp_path, [two_calls, string_args, no_user], [{"text": "unrelated"}], count=4)
    assert manifest["count"] == 0 and manifest["candidates"] == 0
