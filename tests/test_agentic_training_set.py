"""The agentic-v1 set builder: row shape, the reserved split, the two arms, the firewall.

Every test runs on tiny synthetic source files in ``tmp_path`` with an injected
protected set, so nothing here loads a benchmark cache, a model or a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_agentic_training_set", ROOT / "chowder_batch" / "build_agentic_training_set.py"
)
assert _spec and _spec.loader
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

from chowder.growth.agentic_providers import is_reserved_for_evaluation  # noqa: E402
from chowder.growth.tool_use_instrument import INSTRUMENT_ID  # noqa: E402

#: Real ids pinned from the instrument's own formula: the first hashes to a
#: multiple of ten (reserved), the second does not.
RESERVED_UUID = "ff6ab2b0-7551-468c-8606-a7c3b0c75b68"
KEPT_UUID = "0a1b2c3d-0000-4000-8000-000000000000"

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    },
}

TOKENIZER_DIR = Path(r"G:\llm-cold\bf16\Qwen3.6-35B-A3B")

FABLE_MARKER = "FABLE-ARM-MARKER-8f21"
NEMOTRON_MARKER = "NEMOTRON-ARM-MARKER-4c07"


def _assistant_call(arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "Read", "arguments": arguments}}],
    }


def _glint_row(messages: list[dict], *, target: dict | None = None, tools: list | None = None) -> dict:
    return {
        "messages": messages,
        "target": target if target is not None else {"content": "done"},
        "tools": tools,
        "session_key": "s1",
    }


def _swarm_row(messages: list[dict]) -> dict:
    return {"messages": messages, "meta": {"id": "swarm-row"}}


def _nemotron_row(uuid: str, messages: list[dict], *, tools: list | None = None) -> dict:
    return {"uuid": uuid, "messages": messages, "tools": tools}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    return path


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pytest.importorskip("pyarrow", reason="the lordx64 source is a parquet file")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({name: [row[name] for row in rows] for name in rows[0]}), path)
    return path


def _lordx64_text(assistant_body: str) -> str:
    return (
        "<|im_start|>user\nlist files<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_body}<|im_end|>\n"
    )


def _sources(tmp_path: Path, *, fable_marker: str = FABLE_MARKER, nemotron_marker: str = NEMOTRON_MARKER) -> dict:
    """One tiny synthetic copy of each of the four real sources."""
    glint = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row(
                [
                    {"role": "user", "content": f"read a.py {fable_marker}"},
                    _assistant_call({"file_path": "a.py"}),
                ],
                tools=[READ_TOOL],
            )
        ],
    )
    swarm = _write_jsonl(
        tmp_path / "swarm.jsonl",
        [_swarm_row([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])],
    )
    lordx64 = _write_parquet(
        tmp_path / "lordx64.parquet",
        [{"text": _lordx64_text('<tool_use name="Read" id="c1">\n{"file_path": "b.py"}\n</tool_use>')}],
    )
    nemotron = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            _nemotron_row(
                KEPT_UUID,
                [
                    {"role": "user", "content": f"read c.py {nemotron_marker}"},
                    _assistant_call({"file_path": "c.py"}),
                ],
                tools=[READ_TOOL],
            )
        ],
    )
    return {"glint": str(glint), "lordx64": str(lordx64), "swarm": str(swarm), "nemotron": str(nemotron)}


def _build(tmp_path: Path, sources: dict, **kwargs) -> dict:
    kwargs.setdefault("protected", {})
    kwargs.setdefault("validate_template", False)
    return builder.build(out_dir=tmp_path / "out", **sources, **kwargs)


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _digests(lines: list[str]) -> set[str]:
    return {hashlib.sha256(line.encode("utf-8")).hexdigest() for line in lines}


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #


def test_tool_arguments_are_json_strings_that_parse_back(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    builder.build(
        out_dir=tmp_path / "out",
        **sources,
        protected={},
        validate_template=False,
        nemotron_rows=1,
        fable_rows=5,
    )
    rows = [json.loads(line) for line in _lines(tmp_path / "out" / "train-fable.jsonl")]
    calls = [
        call for row in rows for turn in row["messages"] for call in turn.get("tool_calls") or []
    ]
    assert calls
    # Every call's arguments are a JSON string that parses back to the object
    # the source carried -- never a nested Arrow-unified dict.
    for call in calls:
        assert isinstance(call["function"]["arguments"], str)
        assert isinstance(json.loads(call["function"]["arguments"]), dict)
    parsed = {json.dumps(json.loads(call["function"]["arguments"]), sort_keys=True) for call in calls}
    assert json.dumps({"file_path": "a.py"}, sort_keys=True) in parsed
    # tools is a JSON string (or "" when a row has none), never a nested object.
    for row in rows:
        assert isinstance(row["tools"], str)
        if row["tools"]:
            assert isinstance(json.loads(row["tools"]), list)


def test_a_reserved_nemotron_row_never_reaches_either_output(tmp_path: Path) -> None:
    assert is_reserved_for_evaluation(RESERVED_UUID)
    assert not is_reserved_for_evaluation(KEPT_UUID)
    sources = _sources(tmp_path)
    reserved = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            _nemotron_row(
                RESERVED_UUID,
                [{"role": "user", "content": "RESERVED-ROW-9d13"}, {"role": "assistant", "content": "no"}],
            ),
            _nemotron_row(
                KEPT_UUID,
                [{"role": "user", "content": NEMOTRON_MARKER}, {"role": "assistant", "content": "yes"}],
            ),
        ],
    )
    manifest = builder.build(
        out_dir=tmp_path / "out",
        **{**sources, "nemotron": str(reserved)},
        protected={},
        validate_template=False,
        nemotron_rows=5,
        fable_rows=5,
    )
    for name in ("train-fable.jsonl", "train-control.jsonl"):
        text = (tmp_path / "out" / name).read_text(encoding="utf-8")
        assert "RESERVED-ROW-9d13" not in text
        assert NEMOTRON_MARKER in text
    assert manifest["sources"]["nemotron"]["drops"]["reserved_uuid"] == 1


def test_control_is_fable_free_and_the_fable_set_contains_its_nemotron_rows(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    builder.build(
        out_dir=tmp_path / "out",
        **sources,
        protected={},
        validate_template=False,
        nemotron_rows=5,
        fable_rows=5,
    )
    control = _lines(tmp_path / "out" / "train-control.jsonl")
    fable = _lines(tmp_path / "out" / "train-fable.jsonl")
    assert control and fable
    assert not any(FABLE_MARKER in line for line in control)
    assert any(FABLE_MARKER in line for line in fable)
    assert _digests(control) <= _digests(fable)


def test_a_row_duplicating_a_protected_item_is_dropped_and_counted(tmp_path: Path) -> None:
    protected = "Question 77: compute the seventy seventh Fibonacci number exactly."
    sources = _sources(tmp_path)
    planted = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            _nemotron_row(KEPT_UUID, [{"role": "user", "content": protected}]),
            _nemotron_row(
                KEPT_UUID,
                [{"role": "user", "content": NEMOTRON_MARKER}, {"role": "assistant", "content": "ok"}],
            ),
        ],
    )
    manifest = builder.build(
        out_dir=tmp_path / "out",
        **{**sources, "nemotron": str(planted)},
        protected={"tiny_bench@2026-01": [protected]},
        validate_template=False,
        nemotron_rows=5,
        fable_rows=5,
    )
    text = (tmp_path / "out" / "train-control.jsonl").read_text(encoding="utf-8")
    assert "Fibonacci" not in text
    assert NEMOTRON_MARKER in text
    assert manifest["drops_total"]["contaminated"] == 1
    assert manifest["sources"]["nemotron"]["drops"]["contaminated"] == 1
    assert manifest["protected"]["tiny_bench@2026-01"] == 1


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    builder.build(out_dir=tmp_path / "a", **sources, protected={}, validate_template=False)
    builder.build(out_dir=tmp_path / "b", **sources, protected={}, validate_template=False)
    for name in ("train-fable.jsonl", "train-control.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_manifest_records_licences_digests_and_the_declared_arm(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    manifest = builder.build(out_dir=tmp_path / "out", **sources, protected={}, validate_template=False)
    assert manifest["outputs"]["train-fable.jsonl"]["fable_declared"] is True
    assert manifest["outputs"]["train-control.jsonl"]["fable_declared"] is False
    for name, entry in manifest["sources"].items():
        assert Path(entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
        assert entry["licence"]
        assert name in manifest["sources"]
    for entry in manifest["outputs"].values():
        assert entry["sha256"] == hashlib.sha256(
            (tmp_path / "out" / ("train-fable.jsonl" if entry["fable_declared"] else "train-control.jsonl")).read_bytes()
        ).hexdigest()


# --------------------------------------------------------------------------- #
# Optional: the real Qwen tokenizer
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason="the cold Qwen tokenizer is not present on this machine",
)
def test_three_rows_render_through_the_real_template_with_tools(tmp_path: Path) -> None:
    from chowder.backends.training_data import (
        _build_chat_example,
        _validate_chat_messages,
        parse_tools,
    )
    from transformers import AutoTokenizer

    sources = _sources(tmp_path)
    manifest = builder.build(
        out_dir=tmp_path / "out",
        **sources,
        protected={},
        validate_template=True,
        tokenizer_dir=str(TOKENIZER_DIR),
        fable_rows=3,
        nemotron_rows=0,
    )
    assert manifest["template_validated"] is True
    assert manifest["outputs"]["train-fable.jsonl"]["token_count_percentiles"]
    rows = [json.loads(line) for line in _lines(tmp_path / "out" / "train-fable.jsonl")]
    assert len(rows) == 3
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=False)
    for index, row in enumerate(rows):
        # Exactly the worker's chat path: JSONL -> _validate_chat_messages (parses
        # the JSON-string arguments back to mappings) -> _build_chat_example.
        messages = _validate_chat_messages(row["messages"], row_index=index)
        example = _build_chat_example(
            tokenizer,
            messages,
            max_length=4096,
            row_index=index,
            tools=parse_tools(row["tools"], row_index=index),
        )
        assert any(label != -100 for label in example["labels"])


def test_the_default_protected_set_names_the_four_benchmarks_and_the_instrument() -> None:
    benchmarks = {qualified_id for qualified_id, *_ in builder._PROTECTED_SPECS}
    assert benchmarks == {"gsm8k@2021-09", "math500@2024-04", "mgsm@2022-11", "humaneval@2021-07"}
    assert builder.INSTRUMENT_ID == INSTRUMENT_ID == "tool_use@reap-moe-v1"
