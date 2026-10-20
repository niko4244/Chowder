"""The MoE Gen-1 mix builder: caps, shortfall, determinism, the two arms, the firewall.

Every test runs on tiny synthetic sources in ``tmp_path`` with an injected
protected set and an injected GSM8K split, so nothing here loads a benchmark
cache, a model or a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "chowder_batch" / "build_moe_gen1_mix.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_moe_gen1_mix", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


class FakeTokenizer:
    """A prefix-consistent stand-in: one role token plus short content ids."""

    _ROLE = {"system": 1000, "user": 1001, "assistant": 1002, "tool": 1003}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None):
        ids: list[int] = []
        for message in messages:
            ids.append(self._ROLE[message["role"]])
            ids.extend(ord(char) % 50 for char in str(message.get("content") or "")[:20])
        if add_generation_prompt:
            ids.append(self._ROLE["assistant"])
        return {"input_ids": ids}


PROTECTED = {
    "gsm8k@2021-09": ["a protected evaluation question about apples and oranges"],
    "tool_use@reap-moe-v1": ["a protected instrument conversation"],
}

FABLE_MARKER = "FABLE-ROW-MARKER"
NEMOTRON_MARKER = "NEMOTRON-ROW-MARKER"
SHARED_SYSTEM = "you are a customer service agent that helps the user with the policy below"

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    },
}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    return path


def _canonical(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _leak_solution(secret: str, index: int = 0) -> str:
    """A long row with the 13-word secret buried in it, so the 8-gram firewall stays clean."""
    filler = " ".join(f"fillerword{number:03d}" for number in range(200))
    return f"{filler} {secret} {filler} worked solution number {index}"


def _sources(tmp_path: Path, *, rows: int = 6, leak: str | None = None) -> dict:
    openr1 = []
    for index in range(rows):
        solution = f"worked solution number {index}"
        if leak and index == 0:
            solution = _leak_solution(leak, index)
        openr1.append({"problem": f"math problem number {index}", "solution": solution})
    ot3 = [
        {
            "domain": "code",
            "conversations": [
                {"from": "human", "value": f"write code number {index}"},
                {"from": "gpt", "value": f"```python\nprint({index})\n```"},
            ],
        }
        for index in range(rows)
    ]
    ornith = [
        {
            "messages": [
                {"role": "user", "content": f"ornith task {index}"},
                {"role": "assistant", "content": f"ornith worked answer {index}"},
            ],
            "source": "curriculum",
        }
        for index in range(3)
    ]
    nemotron = [
        {
            "messages": [
                {"role": "system", "content": SHARED_SYSTEM},
                {"role": "user", "content": f"{NEMOTRON_MARKER} {index}"},
                {"role": "assistant", "content": f"nemotron answer {index}"},
            ],
            "tools": "",
        }
        for index in range(3)
    ]
    fable = [
        {
            "messages": [
                {"role": "user", "content": f"{FABLE_MARKER} {index}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "Read", "arguments": json.dumps({"file_path": "a.py"})},
                        }
                    ],
                },
            ],
            "tools": json.dumps([READ_TOOL]),
        }
        for index in range(2)
    ]
    return {
        "openr1": _write_jsonl(tmp_path / "openr1.jsonl", openr1),
        "ot3": _write_jsonl(tmp_path / "ot3.jsonl", ot3),
        "ornith": _write_jsonl(tmp_path / "ornith.jsonl", ornith),
        "agentic_control": _write_jsonl(tmp_path / "agentic-control.jsonl", nemotron),
        "agentic_fable": _write_jsonl(tmp_path / "agentic-fable.jsonl", [*nemotron, *fable]),
        "gsm8k_rows": [
            {"question": f"gsm8k question {index}", "answer": f"step one\nstep two\n#### {index}"}
            for index in range(rows)
        ],
    }


def _caps(**overrides: int) -> dict[str, int]:
    caps = {
        "openr1_math": 2,
        "gsm8k_train": 2,
        "ornith": 5,  # only three rows exist -> shortfall
        "ot3_code": 2,
        "nemotron": 2,
        "fable": 2,
    }
    caps.update(overrides)
    return caps


def _build(tmp_path: Path, sources: dict, *, out: str = "out", monkeypatch=None, **kwargs):
    if monkeypatch is not None:
        monkeypatch.setattr(
            builder, "ORNITH_SHA256", hashlib.sha256(sources["ornith"].read_bytes()).hexdigest()
        )
    options = {
        "out_dir": tmp_path / out,
        "caps": _caps(),
        "openr1": sources["openr1"],
        "ot3": sources["ot3"],
        "ornith": sources["ornith"],
        "agentic_fable": sources["agentic_fable"],
        "agentic_control": sources["agentic_control"],
        "gsm8k_rows": sources["gsm8k_rows"],
        "protected": PROTECTED,
        "validate_template": False,
    }
    options.update(kwargs)
    return builder.build(**options)


# --------------------------------------------------------------------------- #
# The builder itself
# --------------------------------------------------------------------------- #


def test_builder_exists_on_this_branch():
    """Fails on origin/main, where this builder does not exist."""
    assert BUILDER_PATH.is_file()
    assert callable(builder.build)
    assert callable(builder.main)


# --------------------------------------------------------------------------- #
# Caps, shortfall, determinism
# --------------------------------------------------------------------------- #


def test_caps_and_shortfall_are_recorded(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    manifest = _build(tmp_path, sources, monkeypatch=monkeypatch)

    assert manifest["sources"]["openr1_math"]["kept"] == 2
    assert manifest["sources"]["gsm8k_train"]["kept"] == 2
    assert manifest["sources"]["ornith"]["kept"] == 3
    assert manifest["sources"]["ornith"]["shortfall"] == 2
    assert manifest["sources"]["fable"]["kept"] == 2
    assert manifest["sources"]["nemotron"]["kept"] == 2
    assert manifest["sources"]["openr1_math"]["shortfall"] == 0


def test_same_seed_same_output_digest(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    first = _build(tmp_path, sources, out="a", monkeypatch=monkeypatch, seed=7)
    second = _build(tmp_path, sources, out="b", monkeypatch=monkeypatch, seed=7)
    third = _build(tmp_path, sources, out="c", monkeypatch=monkeypatch, seed=8)

    assert first["outputs"]["train-fable.jsonl"]["sha256"] == (
        second["outputs"]["train-fable.jsonl"]["sha256"]
    )
    assert first["outputs"]["train-control.jsonl"]["sha256"] == (
        second["outputs"]["train-control.jsonl"]["sha256"]
    )
    assert first["outputs"]["train-fable.jsonl"]["sha256"] != (
        third["outputs"]["train-fable.jsonl"]["sha256"]
    )


# --------------------------------------------------------------------------- #
# The two arms
# --------------------------------------------------------------------------- #


def test_control_is_the_fable_arm_minus_exactly_the_fable_rows(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    manifest = _build(tmp_path, sources, monkeypatch=monkeypatch)
    out = tmp_path / "out"

    fable = _read_jsonl(out / "train-fable.jsonl")
    control = _read_jsonl(out / "train-control.jsonl")
    fable_hashes = Counter(_canonical(row) for row in fable)
    control_hashes = Counter(_canonical(row) for row in control)

    removed = fable_hashes - control_hashes
    assert sum(removed.values()) == manifest["sources"]["fable"]["kept"]
    # The control is the arm file with the Fable rows dropped, in the same order.
    assert control == [row for row in fable if _canonical(row) in control_hashes]
    assert fable


def test_tool_call_arguments_stay_json_strings(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    _build(tmp_path, sources, monkeypatch=monkeypatch)
    rows = _read_jsonl(tmp_path / "out" / "train-fable.jsonl")
    fable_row = next(row for row in rows if FABLE_MARKER in json.dumps(row))

    arguments = fable_row["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"file_path": "a.py"}
    assert isinstance(fable_row["tools"], str)


def test_gsm8k_rendering_keeps_the_hash_answer_line(tmp_path, monkeypatch):
    sources = _sources(tmp_path, rows=1)
    _build(tmp_path, sources, monkeypatch=monkeypatch, caps=_caps(gsm8k_train=1, openr1_math=0))
    rows = _read_jsonl(tmp_path / "out" / "train-control.jsonl")
    gsm8k_row = next(row for row in rows if "gsm8k question" in row["messages"][0]["content"])

    assert gsm8k_row["messages"][1]["content"].endswith("#### 0")
    assert gsm8k_row["messages"][1]["content"].splitlines()[:2] == ["step one", "step two"]
    assert gsm8k_row["tools"] == ""


# --------------------------------------------------------------------------- #
# The firewall
# --------------------------------------------------------------------------- #


def test_a_13gram_leak_is_dropped_and_recorded(tmp_path, monkeypatch):
    secret = " ".join(f"leaktoken{index:02d}" for index in range(13))
    protected = {"gsm8k@2021-09": [secret]}
    sources = _sources(tmp_path, leak=secret)
    manifest = _build(
        tmp_path,
        sources,
        monkeypatch=monkeypatch,
        protected=protected,
        caps=_caps(openr1_math=10),
    )

    # The row is long enough that the 8-gram firewall alone stays clean; the
    # drop is the 13-gram rule, exactly as the plan requires.
    firewall = builder.build_firewall(protected)
    leak_text = builder._row_text(
        [{"role": "assistant", "content": _leak_solution(secret)}], None
    )
    assert firewall.check_text(leak_text).clean

    entry = manifest["sources"]["openr1_math"]
    assert entry["drops"]["contaminated"] >= 1
    assert entry["contamination_detectors"].get("ngram13") == 1
    assert manifest["protected_13grams"] >= 1
    rows = _read_jsonl(tmp_path / "out" / "train-fable.jsonl")
    assert not any("leaktoken" in json.dumps(row) for row in rows)


def test_ornith_sha_mismatch_refuses(tmp_path):
    sources = _sources(tmp_path)
    with pytest.raises(SystemExit):
        builder.build(
            out_dir=tmp_path / "out",
            ornith=sources["ornith"],
            openr1=sources["openr1"],
            ot3=sources["ot3"],
            agentic_fable=sources["agentic_fable"],
            agentic_control=sources["agentic_control"],
            gsm8k_rows=sources["gsm8k_rows"],
            protected=PROTECTED,
            validate_template=False,
        )


def test_shared_system_preamble_alone_is_not_contamination(tmp_path, monkeypatch):
    """The environment contract is not item content: same system turn, different task."""
    filler = " ".join(f"taskword{number:03d}" for number in range(150))
    item = {
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM},
            {"role": "user", "content": f"{filler} an order about a missing drink"},
            {"role": "assistant", "content": "I can help with that order"},
        ],
        "expected": {"name": "verify_guest_order", "arguments": {"confirmation_number": "1"}},
    }
    protected = {"tool_use@reap-moe-v1": [json.dumps(item, sort_keys=True)]}
    sources = _sources(tmp_path)
    manifest = _build(
        tmp_path, sources, monkeypatch=monkeypatch, protected=protected, caps=_caps()
    )

    assert manifest["sources"]["nemotron"]["kept"] == 2
    assert manifest["sources"]["nemotron"]["drops"]["contaminated"] == 0


# --------------------------------------------------------------------------- #
# Template rendering
# --------------------------------------------------------------------------- #


def test_fake_tokenizer_renders_every_kept_row(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    manifest = _build(
        tmp_path,
        sources,
        monkeypatch=monkeypatch,
        tokenizer=FakeTokenizer(),
        validate_template=True,
    )

    assert manifest["template_validated"] is True
    assert manifest["tokenizer"]["loaded"] is True
    assert "token_count_percentiles" in manifest["outputs"]["train-fable.jsonl"]
    assert manifest["outputs"]["train-fable.jsonl"]["rows"] > 0
