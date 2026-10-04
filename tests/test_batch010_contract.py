"""Producer/consumer contract for the batch-010 trajectory dataset.

``build_batch010_dataset`` (producer, local lane) and
``kaggle/run_qat_distill_lane.load_teacher_rows`` (consumer, Kaggle lane) agree
on a dataset schema that no single place in the repo declares. The consumer was
previously exercised only against hand-filled rows, so a producer-side rename or
a consumer-side relaxation would have surfaced on a paid GPU notebook instead of
in CI.

This module builds a **real** batch-010 file by running the runtime harness
(``RuntimeTask`` -> ``run_live_benchmark`` -> ``repair_trajectory_row`` ->
``build_batch010_dataset``) and feeds that exact file to the consumer, so both
directions of drift are caught locally:

* every field the consumer requires is emitted by the producer, and
* every one of those fields is *load-bearing* in the consumer -- mutating it on
  real producer output must raise -- so a consumer that quietly stops
  enforcing a field fails here as well.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

from chowder.runtime_eval import RuntimeTask, _SPAN, _is_green, run_live_benchmark  # noqa: E402
from exp_e_pipeline import build_batch010_dataset, repair_trajectory_row  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "kaggle_qat_lane_contract", ROOT / "kaggle" / "run_qat_distill_lane.py"
)
lane = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lane)

#: Fields ``load_teacher_rows`` reads. Each one is proven load-bearing below;
#: if the producer stops emitting one, the Kaggle lane would reject a dataset
#: this test suite certified as good.
CONSUMER_REQUIRED_FIELDS = ("split", "green_verified", "harness", "messages")


# The literal tag halves, split off the capturing groups in ``_SPAN.pattern``.
_TOOL_OPEN = _SPAN.pattern.split("(", 1)[0]
_TOOL_CLOSE = _SPAN.pattern.rsplit(")", 1)[1]


def _tool(name: str, **arguments: str) -> str:
    """Render a tool call in the harness's exact wire format.

    The open/close tags contain an invisible zero-width space, so the literals
    are reconstructed from ``runtime_eval._SPAN`` instead of being retyped.
    """
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>" for key, value in arguments.items()
    )
    return f"{_TOOL_OPEN}{name}{body}{_TOOL_CLOSE}"


def _repair_task(name: str, *, returns: int = 2) -> dict:
    return {
        "name": name,
        "question": f"Repair app.py so it returns {returns}.",
        "initial": {"app.py": f"def f(): return {returns - 1}"},
        "target": "app.py",
        "expected_fix": f"return {returns}",
        "test_count": 1,
        "test_success": "1 passed",
    }


def _solving_generator(returns: int):
    """A minimal honest agent: write the fix, run the tests, then report.

    The harness evaluates the written file against the task's own test, so the
    fix is parameterized by what that task actually requires.
    """
    def generate(messages):
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if tool_messages and _is_green(str(tool_messages[-1].get("content", ""))):
            return "Verified: app.py returns the expected value with a passing suite."
        if any(
            message.get("role") == "tool" and str(message.get("content", "")).startswith("OK")
            for message in messages
        ):
            return _tool("run_tests")
        return _tool("write_file", path="app.py", content=f"def f(): return {returns}")

    return generate


def _build_real_batch010(directory: Path, *, task_count: int = 2) -> Path:
    """Run the real producer end to end and return the written JSONL path."""
    evolve = [_repair_task(f"repair_evolve_{index}", returns=2 + index) for index in range(task_count)]
    heldout = [_repair_task(f"repair_heldout_{index}", returns=9 + index) for index in range(task_count)]
    # A fresh generator per task so each trajectory is a distinct run rather
    # than one trace replayed under two names.
    records = []
    for task in evolve:
        runtime_task = RuntimeTask(
            task["name"], task["question"], task["initial"], task["target"],
            task["expected_fix"], task["test_count"], task["test_success"],
        )
        benchmark = run_live_benchmark(
            _solving_generator(int(task["expected_fix"].split()[-1])),
            tasks=(runtime_task,),
            harness="state_aware",
            max_turns=4,
        )
        trace = benchmark["tasks"][0]["trace"]
        records.append(repair_trajectory_row(task, trace, split="evolve", harness="state_aware"))
    output = directory / "batch010.jsonl"
    assert build_batch010_dataset(
        records, evolve_tasks=evolve, heldout_tasks=heldout, output_path=output
    ) == task_count
    return output


@pytest.fixture(scope="module")
def real_batch010(tmp_path_factory) -> Path:
    return _build_real_batch010(tmp_path_factory.mktemp("batch010_contract"))


def test_real_batch010_output_is_accepted_by_the_kaggle_lane(real_batch010: Path):
    """The happy path: a genuinely produced dataset survives the consumer intact."""
    rows = lane.load_teacher_rows(real_batch010)

    lines = real_batch010.read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(lines) == 2, "one JSON object per line, order preserved"
    assert [row["id"] for row in rows] == [json.loads(line)["id"] for line in lines]

    # Pin the producer-side properties the consumer's strictness relies on, so
    # a producer that starts emitting tool messages (or non-evolve rows) fails
    # here rather than as a confusing rejection inside a Kaggle notebook.
    for row in rows:
        assert row["split"] == "evolve"
        assert row["green_verified"] is True
        assert row["harness"] in lane.ALLOWED_TRAJECTORY_HARNESSES
        assert {message["role"] for message in row["messages"]} <= {"system", "user", "assistant"}
        assert row["messages"][-1]["role"] == "assistant"
        assert row["messages"][-1]["content"].strip()
        assert all(str(message["content"]).strip() for message in row["messages"])


def test_tool_observations_survive_the_tool_free_rendering(real_batch010: Path):
    """`messages` is tool-free for portable templating; `trace` keeps the truth.

    The producer folds harness `tool` observations into user turns so the
    Kaggle lane's system/user/assistant rule holds. That fold is only honest if
    nothing is lost: every observation in the trace must still appear verbatim
    in the supervised transcript, and the trace must still mark it as a tool
    result.
    """
    row = json.loads(real_batch010.read_text(encoding="utf-8").splitlines()[0])
    observations = [
        entry["observation"] for entry in row["trace"] if entry.get("role") == "tool"
    ]
    assert observations, "the fixture must contain at least one tool observation"
    rendered = [str(message["content"]) for message in row["messages"]]
    for observation in observations:
        assert observation in rendered, f"observation lost in the tool-free rendering: {observation!r}"
        assert all(entry.get("role") == "tool" for entry in row["trace"] if entry.get("observation") == observation)


def test_producer_emits_every_field_the_consumer_requires(real_batch010: Path):
    row = json.loads(real_batch010.read_text(encoding="utf-8").splitlines()[0])
    missing = [field for field in CONSUMER_REQUIRED_FIELDS if field not in row]
    assert not missing, (
        f"batch-010 producer stopped emitting {missing}; kaggle/run_qat_distill_lane."
        "load_teacher_rows requires them and would reject the dataset on a GPU session"
    )


def _drift_cases():
    """(field, mutate, expected consumer error) over real producer output."""
    return [
        ("split", lambda row: {**row, "split": "heldout"}, "green-verified evolve trajectory"),
        ("green_verified", lambda row: {**row, "green_verified": False}, "green-verified evolve trajectory"),
        ("harness", lambda row: {**row, "harness": "plain"}, "state-aware harness"),
        ("messages", lambda row: {**row, "messages": []}, "missing its transcript"),
        (
            "messages",
            lambda row: {**row, "messages": row["messages"][:-1]},
            "end in a nonempty assistant report",
        ),
        (
            "messages",
            lambda row: {**row, "messages": [*row["messages"][:-1], {"role": "assistant", "content": "  "}]},
            "end in a nonempty assistant report",
        ),
        (
            "messages",
            lambda row: {**row, "messages": [{"role": "tool", "content": "1 passed"}, *row["messages"]]},
            "malformed message",
        ),
    ]


@pytest.mark.parametrize(
    "field,mutate,message",
    _drift_cases(),
    ids=[f"{name}-{index}" for index, (name, _mutate, _msg) in enumerate(_drift_cases())],
)
def test_contract_fields_are_load_bearing_in_the_consumer(
    real_batch010: Path, tmp_path: Path, field: str, mutate, message: str
):
    """Each required field must actually change the consumer's verdict.

    Applied to real producer output, so this is the drift assertion: a producer
    rename breaks the happy path, and a consumer that stops checking a field
    breaks this test.
    """
    row = json.loads(real_batch010.read_text(encoding="utf-8").splitlines()[0])
    drifted = mutate(row)
    assert drifted != row, f"mutation for {field} did not change the row"

    path = tmp_path / f"drift_{field}_{abs(hash(message))}.jsonl"
    path.write_text(json.dumps(drifted) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        lane.load_teacher_rows(path)


def test_consumer_still_rejects_an_empty_or_unparseable_dataset(real_batch010: Path, tmp_path: Path):
    """File-level contract: the lane reads one JSON object per nonempty line."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty dataset"):
        lane.load_teacher_rows(empty)

    not_objects = tmp_path / "not_objects.jsonl"
    not_objects.write_text(json.dumps(["a", "list"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a JSON object"):
        lane.load_teacher_rows(not_objects)
