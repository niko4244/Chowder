"""CPU-only quality and contamination gates for the isolated pilot."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill" / "prepare.py"
spec = importlib.util.spec_from_file_location("teacher_free_prepare", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf8")


def catalog(tmp_path, *, approved=True, kind="chat"):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"sources": {"test": {
        "approved": approved, "license": "test-only",
        "review_reference": "synthetic fixture",
        "revision": "synthetic-rev", "kind": kind
    }}}))
    return path


def test_license_gate_fails_closed(tmp_path):
    inp = tmp_path / "input.jsonl"
    write_jsonl(inp, [{"messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}
    ]}])
    with pytest.raises(PermissionError):
        mod.prepare(catalog(tmp_path, approved=False), {"test": inp}, tmp_path / "out")


def test_chat_mapping_and_dedup(tmp_path):
    inp = tmp_path / "input.jsonl"
    row = {"conversations": [
        {"from": "human", "value": "Compute 2 + 2"}, {"from": "gpt", "value": "4"}
    ]}
    write_jsonl(inp, [row, row, {"conversations": [
        {"from": "human", "value": "Compute 3 + 3"}, {"from": "gpt", "value": "6"}
    ]}])
    result = mod.prepare(catalog(tmp_path), {"test": inp}, tmp_path / "out", dev_percent=1)
    assert result["train_rows"] + result["dev_rows"] == 2
    assert result["counts"]["test:duplicate_or_conflict"] == 1
    rows = [json.loads(x) for file in ("train.jsonl", "dev.jsonl")
            for x in (tmp_path / "out" / file).read_text().splitlines()]
    assert all(set(row) == {"messages"} for row in rows)
    assert all(row["messages"][0]["role"] == "user" for row in rows)


def test_external_holdout_blocks_example(tmp_path):
    inp = tmp_path / "input.jsonl"
    write_jsonl(inp, [{"messages": [
        {"role": "user", "content": q}, {"role": "assistant", "content": "answer"}
    ]} for q in ("holdout question", "different question")])
    protected = tmp_path / "holdout.jsonl"
    write_jsonl(protected, [{"prompt": " HOLDOUT   QUESTION "}])
    result = mod.prepare(catalog(tmp_path), {"test": inp}, tmp_path / "out",
                         heldout=protected, dev_percent=1)
    assert result["counts"]["test:holdout_collision"] == 1
    assert result["train_rows"] + result["dev_rows"] == 1
    assert "HOLDOUT" not in (tmp_path / "out" / "train.jsonl").read_text()


def test_repair_requires_matching_local_evidence(tmp_path):
    events = [
        {"kind": "tool", "action": {"tool": "read_file", "path": "x.py"},
         "observation": "read x.py", "verdict": "verified_good"},
        {"kind": "test", "action": {"tool": "run_tests"}, "observation": "1 passed",
         "returncode": 0, "tests_executed": 1, "verdict": "verified_good"}
    ]
    record = {"task": "Fix bug", "repository": "synthetic/repo", "task_id": "r1",
              "events": events, "verification": {
                  "method": "sandbox_replay", "returncode": 0, "tests_executed": 1,
                  "trace_sha256": mod.digest(events)
              }}
    assert len(mod.repair_examples(record)) == 2
    record["verification"]["returncode"] = 1
    with pytest.raises(ValueError):
        mod.repair_examples(record)


def test_repair_claimed_success_without_observed_test_rejected():
    events = [{"kind": "tool", "action": {"tool": "write_file"},
               "observation": "done", "verdict": "verified_good"}]
    row = {"task": "Fix", "repository": "synthetic", "task_id": "a", "events": events,
           "verification": {"method": "sandbox_replay", "returncode": 0,
                            "tests_executed": 1, "trace_sha256": mod.digest(events)}}
    with pytest.raises(ValueError, match="observed green"):
        mod.repair_examples(row)


def test_repair_holdout_task_id(tmp_path):
    path = tmp_path / "holdout.jsonl"
    write_jsonl(path, [{"repository": "synthetic/repo", "task_id": "r1"}])
    assert "repair:synthetic/repo:r1" in mod.heldout_keys(path)
