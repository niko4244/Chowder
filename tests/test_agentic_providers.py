"""The agentic tool-use providers serve frozen, digest-pinned tool conversations.

Both providers are built on the same discipline as ``OrnithCurriculumProvider``:
a pinned sha256 that is re-checked on every ``produce``, rows served at most
once, per-example provenance, and no re-implementation of the protected-text or
contamination guards (those live in ``materialise``).  What is specific to
agentic data is tested here:

* the three copied normalisations -- JSON-string tool arguments becoming dicts,
  lordx64's pre-rendered ``<tool_use>`` XML becoming native tool calls, and the
  local Fable ``target`` turn being appended as the final assistant turn;
* the cross-source dedupe rule (a repackaged row is dropped, a row that shares
  one line with the seed source is kept);
* Fable's licence gate (``allow_fable``) and the ``fable_derived`` mark;
* Nemotron's reserved-evaluation exclusion, which must be exact.

The rows are emitted in the ``messages`` shape, with the tools block also folded
into the leading system turn so a messages-only consumer still sees the schema;
``test_worker_chat_path_rejects_tool_turns`` pins the fact that the current
``transformers_worker`` chat path is *not* that consumer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder.growth.agentic_providers import (
    FABLE_SCHEMA,
    NEMOTRON_LICENCE,
    SEED,
    FableAgenticProvider,
    NemotronAgenticProvider,
    _native_tool_calls,
    _with_tools_system,
    is_reserved_for_evaluation,
    lordx64_messages,
)
from chowder.growth.data_providers import (
    CLEAN,
    DEFAULT_PROVIDERS,
    TrainingDataRefusal,
    TrainingDataRequest,
    materialise,
)

#: Real ids, pinned from the evaluation instrument's own formula.  Reserved ids
#: hash to a multiple of ten, kept ids do not.
RESERVED_UUID = "ff6ab2b0-7551-468c-8606-a7c3b0c75b68"
RESERVED_UUID_2 = "ffffffff-ffff-4fff-8fff-ffffffffffff"
KEPT_UUID = "0a1b2c3d-0000-4000-8000-000000000000"
KEPT_UUID_2 = "22222222-2222-4222-8222-222222222222"

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
}

TOKENIZER_DIR = Path(r"G:\llm-cold\bf16\Qwen3.6-35B-A3B")


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


def _long(tag: str) -> str:
    """An assistant text long enough (>=80 normalised chars) to fingerprint."""
    return (tag + " ") * 20


def _clean(_text: str) -> str:
    return CLEAN


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    )
    return path


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {name: [row[name] for row in rows] for name in rows[0]}
    pq.write_table(pa.table(columns), path)
    return path


def _glint_row(
    session_key: str,
    messages: list[dict],
    *,
    target: dict | None = None,
    tools: list[dict] | None = None,
) -> dict:
    return {
        "messages": messages,
        "target": target if target is not None else {"content": "ack"},
        "tools": tools,
        "session_key": session_key,
        "source": "glint",
        "truncated_context": False,
    }


def _swarm_row(messages: list[dict]) -> dict:
    return {"messages": messages, "meta": {"id": "swarm-row"}}


def _lordx64_text(assistant_body: str, *, user: str = "list files") -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_body}<|im_end|>\n"
    )


def _nemotron_row(
    uuid: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    reasoning: str = "",
) -> dict:
    return {
        "uuid": uuid,
        "messages": messages,
        "license": "cc-by-4.0",
        "used_in": ["train"],
        "tools": tools,
        "reasoning": reasoning,
    }


def _fable(
    sources: dict[str, tuple[str, str]], **kwargs: object
) -> FableAgenticProvider:
    kwargs.setdefault("skill", "coding.agentic")
    kwargs.setdefault("allow_fable", True)
    return FableAgenticProvider(sources, **kwargs)  # type: ignore[arg-type]


def _request(
    *,
    skill: str = "coding.agentic",
    verification: str = "curated_trusted",
    training_type: str = "sft",
    example_count: int = 1,
    item_id: str = "agentic-item",
) -> TrainingDataRequest:
    return TrainingDataRequest(
        item_id=item_id,
        skill=skill,
        training_type=training_type,
        role="TARGET",
        verification_method=verification,
        generation="gen3",
        example_count=example_count,
    )


# --------------------------------------------------------------------------- #
# The copied normalisations
# --------------------------------------------------------------------------- #


def test_native_tool_calls_parse_json_string_arguments() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
            ],
        }
    ]
    parsed = _native_tool_calls(messages)[0]["tool_calls"][0]["function"]["arguments"]
    assert parsed == {"file_path": "a.py"}


def test_native_tool_calls_wrap_unparseable_arguments() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "Raw", "arguments": "not json"}}],
        }
    ]
    parsed = _native_tool_calls(messages)[0]["tool_calls"][0]["function"]["arguments"]
    assert parsed == {"input": "not json"}


def test_lordx64_xml_becomes_native_tool_calls() -> None:
    text = _lordx64_text('<tool_use name="Read" id="c1">\n{"file_path": "a.py"}\n</tool_use>')
    messages = lordx64_messages(text)
    assert messages[0] == {"role": "user", "content": "list files"}
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ""
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "Read"
    assert call["function"]["arguments"] == {"file_path": "a.py"}


def test_fable_target_turn_is_appended_with_reasoning_and_calls(tmp_path: Path) -> None:
    call = {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "a.py"}}}
    target = {"content": "here it is", "reasoning": "because", "tool_calls": [call]}
    row = _glint_row(
        "s1",
        [{"role": "user", "content": "read a.py"}],
        target=target,
        tools=[READ_TOOL],
    )
    path = _write_jsonl(tmp_path / "glint.jsonl", [row])
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    payload = json.loads(provider.produce(_request())[0])
    last = payload["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "here it is"
    assert last["reasoning_content"] == "because"
    assert last["tool_calls"] == [call]


def test_json_string_arguments_are_dicts_in_the_served_line(tmp_path: Path) -> None:
    row = _glint_row(
        "s1",
        [
            {"role": "user", "content": "read a.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
                ],
            },
            {"role": "tool", "content": "print(1)"},
        ],
        tools=[READ_TOOL],
    )
    path = _write_jsonl(tmp_path / "glint.jsonl", [row])
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    payload = json.loads(provider.produce(_request())[0])
    call = next(m for m in payload["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert call["function"]["arguments"] == {"file_path": "a.py"}


# --------------------------------------------------------------------------- #
# Fable licence gate and construction refusals
# --------------------------------------------------------------------------- #


def test_fable_refuses_without_the_licence_decision(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "glint.jsonl", [_glint_row("s1", [{"role": "user", "content": "hi"}])])
    with pytest.raises(TrainingDataRefusal, match="AGPL"):
        _fable({"glint_local": (str(path), _sha256(path))}, allow_fable=False)


def test_fable_refuses_unknown_source_kinds(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "glint.jsonl", [_glint_row("s1", [{"role": "user", "content": "hi"}])])
    with pytest.raises(TrainingDataRefusal, match=FABLE_SCHEMA):
        _fable({"mystery": (str(path), _sha256(path))})


def test_fable_refuses_with_no_sources() -> None:
    with pytest.raises(TrainingDataRefusal, match=FABLE_SCHEMA):
        _fable({})


def test_digest_mismatch_refuses_at_construction(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "glint.jsonl", [_glint_row("s1", [{"role": "user", "content": "hi"}])])
    with pytest.raises(TrainingDataRefusal, match="digest"):
        _fable({"glint_local": (str(path), "0" * 64)})


def test_file_mutated_after_construction_refuses_on_produce(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("s1", [{"role": "user", "content": "hi"}]),
            _glint_row("s2", [{"role": "user", "content": "there"}]),
        ],
    )
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataRefusal, match="digest"):
        provider.produce(_request())


def test_serves_only_declared_skill_and_verification(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("s1", [{"role": "user", "content": "hi"}]),
            _glint_row("s2", [{"role": "user", "content": "there"}]),
        ],
    )
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    assert provider.supports(_request())
    assert not provider.supports(_request(skill="coding.python"))
    assert not provider.supports(_request(verification="executable_tests"))
    assert not provider.supports(_request(training_type="preference"))


def test_rows_are_never_reused(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("s1", [{"role": "user", "content": "one"}]),
            _glint_row("s2", [{"role": "user", "content": "two"}]),
        ],
    )
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    first = provider.produce(_request(example_count=2))
    assert len(set(first)) == 2
    assert not provider.supports(_request(example_count=1))
    with pytest.raises(TrainingDataRefusal, match=FABLE_SCHEMA):
        provider.produce(_request(example_count=1))


def test_order_is_deterministic_across_instances(tmp_path: Path) -> None:
    rows = [_glint_row(f"s{i}", [{"role": "user", "content": f"turn {i}"}]) for i in range(6)]
    path = _write_jsonl(tmp_path / "glint.jsonl", rows)
    sha = _sha256(path)
    first = _fable({"glint_local": (str(path), sha)}).produce(_request(example_count=6))
    second = _fable({"glint_local": (str(path), sha)}).produce(_request(example_count=6))
    assert first == second
    assert SEED == 20261017


# --------------------------------------------------------------------------- #
# Cross-source dedupe
# --------------------------------------------------------------------------- #


def test_repackaged_row_is_dropped_and_a_one_line_share_is_kept(tmp_path: Path) -> None:
    a, c, d, e, f = _long("alpha"), _long("charlie"), _long("delta"), _long("echo"), _long("foxtrot")
    glint = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("g0", [{"role": "user", "content": "q0"}, {"role": "assistant", "content": a}]),
            _glint_row("g1", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": c}]),
        ],
    )
    # swarm row 0 repackages gl1's assistant line with different casing (the
    # fingerprint normalises case, so it is still a duplicate) -> dropped.
    # swarm row 1 shares only one line out of its four -> kept.
    swarm = _write_jsonl(
        tmp_path / "swarm.jsonl",
        [
            _swarm_row([{"role": "user", "content": "q2"}, {"role": "assistant", "content": c.upper()}]),
            _swarm_row(
                [
                    {"role": "user", "content": "q3"},
                    {"role": "assistant", "content": a},
                    {"role": "assistant", "content": d},
                    {"role": "assistant", "content": e},
                    {"role": "assistant", "content": f},
                ]
            ),
        ],
    )
    provider = _fable(
        {
            "glint_local": (str(glint), _sha256(glint)),
            "swarm": (str(swarm), _sha256(swarm)),
        }
    )
    assert provider.dropped_duplicates == {"glint_local": 0, "swarm": 1}
    lines = provider.produce(_request(example_count=3))
    assert len(lines) == 3
    served = sorted(
        provider.provenance[hashlib.sha256(line.encode()).hexdigest()]["source"] for line in lines
    )
    assert served == ["glint_local", "glint_local", "swarm"]


def test_lordx64_parquet_rows_load_and_convert_to_native_calls(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="the lordx64 source is a parquet file")
    text = _lordx64_text('<tool_use name="Read" id="c1">\n{"file_path": "a.py"}\n</tool_use>')
    path = _write_parquet(tmp_path / "lordx64.parquet", [{"text": text, "session": "l0"}])
    provider = _fable({"lordx64": (str(path), _sha256(path))})
    payload = json.loads(provider.produce(_request())[0])
    call = next(m for m in payload["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert call["function"]["name"] == "Read"
    assert call["function"]["arguments"] == {"file_path": "a.py"}
    record = next(iter(provider.provenance.values()))
    assert record["source"] == "lordx64"
    assert record["fable_derived"] is True


def test_local_seed_rows_are_never_dropped_against_themselves(tmp_path: Path) -> None:
    row = _glint_row("dup", [{"role": "user", "content": "same"}, {"role": "assistant", "content": _long("alpha")}])
    path = _write_jsonl(tmp_path / "glint.jsonl", [row, dict(row), dict(row)])
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    assert provider.dropped_duplicates == {"glint_local": 0}
    assert len(provider.produce(_request(example_count=3))) == 3


# --------------------------------------------------------------------------- #
# Line shape: tools folded into the system turn, and kept as a key
# --------------------------------------------------------------------------- #


def test_tools_are_folded_into_a_prepended_system_turn(tmp_path: Path) -> None:
    row = _glint_row("s1", [{"role": "user", "content": "read a.py"}], tools=[READ_TOOL])
    path = _write_jsonl(tmp_path / "glint.jsonl", [row])
    provider = _fable({"glint_local": (str(path), _sha256(path))})
    payload = json.loads(provider.produce(_request())[0])
    first = payload["messages"][0]
    assert first["role"] == "system"
    assert "# Tools" in first["content"]
    assert '"name": "Read"' in first["content"]
    assert payload["messages"][1]["content"] == "read a.py"
    # ``messages`` only: the worker cannot take a ``tools`` key, so none is emitted.
    assert "tools" not in payload


def test_tools_block_prepends_an_existing_system_turn() -> None:
    folded = _with_tools_system(
        [{"role": "system", "content": "you are terse"}, {"role": "user", "content": "hi"}],
        [READ_TOOL],
    )
    assert folded[0]["role"] == "system"
    assert "# Tools" in folded[0]["content"]
    assert folded[0]["content"].endswith("you are terse")
    assert folded[1] == {"role": "user", "content": "hi"}


def test_with_tools_system_is_a_noop_without_tools() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert _with_tools_system(messages, None) == messages


# --------------------------------------------------------------------------- #
# Fable provenance, through materialise
# --------------------------------------------------------------------------- #


def test_every_fable_row_is_marked_fable_derived(tmp_path: Path) -> None:
    glint = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("g0", [{"role": "user", "content": "q0"}]),
            _glint_row("g1", [{"role": "user", "content": "q1"}]),
        ],
    )
    provider = _fable({"glint_local": (str(glint), _sha256(glint))})
    provider.produce(_request(example_count=2))
    records = list(provider.provenance.values())
    assert len(records) == 2
    assert all(record["fable_derived"] is True for record in records)
    assert all(record["source"] == "glint_local" for record in records)
    assert all(record["source_sha256"] == _sha256(glint) for record in records)


def test_fable_serves_the_declared_count_through_materialise(tmp_path: Path) -> None:
    glint = _write_jsonl(
        tmp_path / "glint.jsonl",
        [
            _glint_row("g0", [{"role": "user", "content": "q0"}], tools=[READ_TOOL]),
            _glint_row("g1", [{"role": "user", "content": "q1"}], tools=[READ_TOOL]),
        ],
    )
    sha = _sha256(glint)
    provider = _fable({"glint_local": (str(glint), sha)})
    corpus = materialise([_request(example_count=2)], providers=(provider,), contamination_check=_clean)
    assert len(corpus.material["agentic-item"]) == 2
    example = corpus.examples["agentic-item"][0]
    assert example.provider_id == "fable-agentic"
    assert example.source_id == provider.source_id
    assert example.source_id.startswith("fable-agentic:")
    assert example.verification == "curated_trusted"


def test_materialise_refuses_a_protected_line_from_fable(tmp_path: Path) -> None:
    glint = _write_jsonl(tmp_path / "glint.jsonl", [_glint_row("g0", [{"role": "user", "content": "q0"}])])
    sha = _sha256(glint)
    protected = _fable({"glint_local": (str(glint), sha)}).produce(_request())[0]
    with pytest.raises(TrainingDataRefusal, match="protected evaluation"):
        materialise(
            [_request()],
            providers=(_fable({"glint_local": (str(glint), sha)}),),
            protected_texts=[protected],
            contamination_check=_clean,
        )


def test_materialise_refuses_unmeasured_contamination_from_fable(tmp_path: Path) -> None:
    glint = _write_jsonl(tmp_path / "glint.jsonl", [_glint_row("g0", [{"role": "user", "content": "q0"}])])
    provider = _fable({"glint_local": (str(glint), _sha256(glint))})
    with pytest.raises(TrainingDataRefusal, match="contamination"):
        materialise([_request()], providers=(provider,))


def test_fable_is_not_registered_in_default_providers() -> None:
    assert not any(isinstance(p, FableAgenticProvider) for p in DEFAULT_PROVIDERS)


# --------------------------------------------------------------------------- #
# Nemotron: the reserved-evaluation exclusion, exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("uuid", "reserved"),
    [
        (RESERVED_UUID, True),
        (RESERVED_UUID_2, True),
        (KEPT_UUID, False),
        (KEPT_UUID_2, False),
        ("deadbeef-0000-4000-8000-000000000001", False),
        ("01234567-89ab-4cde-8f01-23456789abcd", False),
    ],
)
def test_is_reserved_for_evaluation_vectors(uuid: str, reserved: bool) -> None:
    assert is_reserved_for_evaluation(uuid) is reserved


@pytest.mark.parametrize(
    ("uuid", "reserved"),
    [
        (RESERVED_UUID, True),
        (RESERVED_UUID_2, True),
        (KEPT_UUID, False),
        (KEPT_UUID_2, False),
    ],
)
def test_is_reserved_for_evaluation_matches_the_pinned_formula(uuid: str, reserved: bool) -> None:
    expected = int(hashlib.sha256(uuid.encode()).hexdigest()[:8], 16) % 10 == 0
    assert is_reserved_for_evaluation(uuid) is reserved is expected


def test_reserved_rows_are_excluded_and_counted(tmp_path: Path) -> None:
    rows = [
        _nemotron_row(KEPT_UUID, [{"role": "user", "content": "a"}], tools=[READ_TOOL]),
        _nemotron_row(RESERVED_UUID, [{"role": "user", "content": "b"}], tools=[READ_TOOL]),
        _nemotron_row(KEPT_UUID_2, [{"role": "user", "content": "c"}], tools=[READ_TOOL]),
        _nemotron_row(RESERVED_UUID_2, [{"role": "user", "content": "d"}], tools=[READ_TOOL]),
    ]
    path = _write_jsonl(tmp_path / "nemotron.jsonl", rows)
    provider = NemotronAgenticProvider(str(path), _sha256(path), skill="coding.agentic")
    assert provider.reserved_excluded == 2
    assert provider.supports(_request(example_count=2))
    assert not provider.supports(_request(example_count=3))


def test_a_row_without_a_uuid_refuses(tmp_path: Path) -> None:
    row = _nemotron_row(KEPT_UUID, [{"role": "user", "content": "a"}])
    del row["uuid"]
    path = _write_jsonl(tmp_path / "nemotron.jsonl", [row])
    with pytest.raises(TrainingDataRefusal, match="uuid"):
        NemotronAgenticProvider(str(path), _sha256(path), skill="coding.agentic")


def test_nemotron_digest_mismatch_and_mutation_refuse(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [_nemotron_row(KEPT_UUID, [{"role": "user", "content": "a"}])],
    )
    with pytest.raises(TrainingDataRefusal, match="digest"):
        NemotronAgenticProvider(str(path), "0" * 64, skill="coding.agentic")
    provider = NemotronAgenticProvider(str(path), _sha256(path), skill="coding.agentic")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataRefusal, match="digest"):
        provider.produce(_request())


def test_nemotron_serves_licence_and_uuid_provenance_through_materialise(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            _nemotron_row(
                KEPT_UUID,
                [
                    {"role": "system", "content": "you are an agent"},
                    {"role": "user", "content": "read a.py"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
                        ],
                    },
                    {"role": "tool", "content": "print(1)"},
                    {"role": "assistant", "content": "It prints 1."},
                ],
                tools=[READ_TOOL],
            )
        ],
    )
    sha = _sha256(path)
    provider = NemotronAgenticProvider(str(path), sha, skill="coding.agentic")
    corpus = materialise([_request()], providers=(provider,), contamination_check=_clean)
    assert len(corpus.material["agentic-item"]) == 1
    line = corpus.material["agentic-item"][0]
    payload = json.loads(line)
    # Tools fold into the *existing* system turn; no ``tools`` key is emitted.
    assert payload["messages"][0]["role"] == "system"
    assert "# Tools" in payload["messages"][0]["content"]
    assert payload["messages"][0]["content"].endswith("you are an agent")
    assert "tools" not in payload
    # JSON-string arguments became dicts.
    call = next(m for m in payload["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert call["function"]["arguments"] == {"file_path": "a.py"}
    example = corpus.examples["agentic-item"][0]
    assert example.provider_id == "nemotron-agentic"
    assert example.source_id == f"nemotron-agentic:{sha[:16]}"
    record = provider.provenance[example.digest]
    assert record["uuid"] == KEPT_UUID
    assert record["licence"] == NEMOTRON_LICENCE == "CC-BY-4.0"


def test_nemotron_is_not_registered_in_default_providers() -> None:
    assert not any(isinstance(p, NemotronAgenticProvider) for p in DEFAULT_PROVIDERS)


def test_unknown_verification_refuses_through_materialise(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [_nemotron_row(KEPT_UUID, [{"role": "user", "content": "a"}])],
    )
    provider = NemotronAgenticProvider(str(path), _sha256(path), skill="coding.agentic")
    assert not provider.supports(_request(verification="executable_tests"))


# --------------------------------------------------------------------------- #
# The integration boundary the report names: the worker's chat path is limited
# --------------------------------------------------------------------------- #


def test_worker_chat_path_rejects_tool_turns() -> None:
    """The ``dataset_format="chat"`` worker validator accepts only
    system/user/assistant and keeps only ``role``/``content``.  Agentic rows
    carry ``tool`` turns and native ``tool_calls``, so they are served as
    messages for a tools-aware renderer (or as text), not through that path."""
    from chowder.backends.training_data import _validate_chat_messages

    agentic = [
        {"role": "user", "content": "read a.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "Read", "arguments": {"file_path": "a.py"}}}],
        },
        {"role": "tool", "content": "print(1)"},
        {"role": "assistant", "content": "It prints 1."},
    ]
    with pytest.raises(RuntimeError, match="unsupported message role 'tool'"):
        _validate_chat_messages(agentic, row_index=0)
    without_tool_turn = [m for m in agentic if m["role"] != "tool"]
    normalized = _validate_chat_messages(without_tool_turn, row_index=0)
    assert all(set(turn) == {"role", "content"} for turn in normalized)


# --------------------------------------------------------------------------- #
# Optional: render the served rows with the real Qwen template
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason="the cold Qwen tokenizer is not present on this machine",
)
def test_served_rows_render_with_the_real_qwen_template(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    path = _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            _nemotron_row(
                KEPT_UUID,
                [
                    {"role": "system", "content": "you are an agent"},
                    {"role": "user", "content": "read a.py"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
                        ],
                    },
                    {"role": "tool", "content": "print(1)"},
                    {"role": "assistant", "content": "It prints 1."},
                ],
                tools=[READ_TOOL],
            )
        ],
    )
    provider = NemotronAgenticProvider(str(path), _sha256(path), skill="coding.agentic")
    payload = json.loads(provider.produce(_request())[0])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=False)
    folded = tokenizer.apply_chat_template(
        payload["messages"], tokenize=False, add_generation_prompt=False
    )
    assert "Read" in folded
    tools_aware = tokenizer.apply_chat_template(
        payload["messages"], tools=[READ_TOOL], tokenize=False, add_generation_prompt=False
    )
    assert "<tool_call>" in tools_aware
