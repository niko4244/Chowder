"""RED/GREEN source, task and fail-closed genuine WorldState receipt tests.

Synthetic fixtures are used ONLY as adversarial unit tests; never benchmark
scores, and no test here starts model inference or submits to a provider.
"""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_original600_live_independent_audit_20261027 as a


def fixture(tmp_path: Path, monkeypatch, *, completed: int = 0):
    source = {"source_git_commit": a.run.q4.BENCH_COMMIT,
              "benchmark_version": a.run.q4.BENCH_VERSION}
    monkeypatch.setattr(a.run.q4, "verify_official_source", lambda: source)
    tasks = [{"id": str(i // 2), "name": f"official.task_{i}"} for i in range(600)]
    owner = a.run.exact_owner(tasks, a.run.source_ids.PUBLIC_PROMPTS_SHA256,
                              a.run.q4.GGUF_SHA, a.run.q4.BENCH_COMMIT)
    tmp_path.joinpath("owner.json").write_bytes(a.run._canonical(owner))
    monkeypatch.setattr(a.run, "verify_item",
                        lambda *args: {"independent_score": {"passed_tasks": 0}})
    for i in range(completed):
        directory = tmp_path / f"task-{i:03d}"
        directory.mkdir()
        (directory / "receipt.json").write_text("{}")
        row = {"name": f"official.task_{i}", "passed": False,
               "assertions_total": 2, "assertions_passed": 1,
               "assertion_results": [
                   {"passed": True, "excluded": False},
                   {"passed": False, "excluded": False},
                   {"passed": True, "excluded": True},
               ],
               "num_model_calls": 3, "num_tool_calls": 5, "input_tokens": 11,
               "output_tokens": 7, "model_time_s": 1.2, "tool_time_s": 0.1}
        (directory / "public-automationbench.private.json").write_text(
            json.dumps({"tasks": [row],
                        "usage_by_task": [{"task_name": f"official.task_{i}",
                                           "input_tokens": 11, "output_tokens": 7}]}))
    return tmp_path


def test_zero_real_scored_rows_is_unknown_not_zero_result(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    data = a.inspect(root)
    assert data["completed_scored_tasks"] == 0
    assert data["strict_pass_rate_completed"] is None
    assert data["status"] == "PARTIAL_AUTHENTIC_OFFICIAL_TOOL_EVALUATOR_EXECUTION"
    assert data["frozen_gen0_status"] == "BLOCKED"
    assert data["official_complete_600_claim_allowed"] is False


def test_valid_partial_source_scored_rows_have_original_positive_assertions(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=3)
    result = a.inspect(root)
    assert result["completed_scored_tasks"] == 3
    assert result["totals"]["strict_passes"] == 0
    assert result["totals"]["positive_assertions"] == 6
    assert result["totals"]["positive_assertions_passed"] == 3
    assert result["totals"]["excluded_assertions_passed"] == 3
    assert result["totals"]["model_calls"] == 9
    assert result["totals"]["tool_calls"] == 15
    assert result["totals"]["input_tokens"] == 33
    assert result["rows"][0]["original_scorer_json_sha256"] == a.digest(
        root / "task-000" / "public-automationbench.private.json")


@pytest.mark.parametrize("name,value", [
    ("model_sha256", "0" * 64),
    ("original_task_prompt_sha256", "0" * 64),
    ("official_benchmark_commit", "a" * 40),
    ("native_flags", ["--fit", "off"]),
    ("context_guard_policy", "different"),
    ("release_approved", True),
    ("chosen", "Q4"),
    ("original_task_population", 599),
])
def test_original_owner_drift_or_gate_pass_fails_closed(tmp_path, monkeypatch, name, value):
    root = fixture(tmp_path, monkeypatch)
    file = root / "owner.json"
    data = a.run._json(file)
    data[name] = value
    file.write_bytes(a.run._canonical(data))
    with pytest.raises(ValueError, match="owner drift"):
        a.inspect(root)


def test_source_git_checkout_mismatch_refuses(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(a.run.q4, "verify_official_source",
                        lambda: {"source_git_commit": "0" * 40,
                                 "benchmark_version": a.run.q4.BENCH_VERSION})
    with pytest.raises(ValueError, match="evaluator source drift"):
        a.inspect(root)


def test_bogus_original_item_gap_refuses(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=2)
    (root / "task-000").rename(root / "task-003")
    with pytest.raises(ValueError, match="skipped|corrupted"):
        a.inspect(root)


def test_active_next_row_never_claimed_as_scored(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=2)
    (root / "task-002").mkdir()
    (root / "task-002" / "official-evaluator.private.log").write_text("partial model output")
    result = a.inspect(root)
    assert result["completed_scored_tasks"] == 2
    assert result["active_unscored_indices"] == [2]
    assert result["official_complete_600_claim_allowed"] is False


def test_wrong_active_index_or_multiple_unscored_refuses(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=2)
    (root / "task-005").mkdir()
    with pytest.raises(ValueError, match="immediate next"):
        a.inspect(root)
    (root / "task-002").mkdir()
    with pytest.raises(ValueError, match="immediate next"):
        a.inspect(root)


@pytest.mark.parametrize("bad", [
    {"model_time_s": -1}, {"model_time_s": float("nan")},
    {"tool_time_s": None}, {"assertions_total": True},
    {"input_tokens": -1}, {"assertion_results": [{"passed": "yes", "excluded": False}]},
])
def test_dangerous_original_metric_value_rejected(bad):
    row = {"model_time_s": 1.0, "tool_time_s": 0.01,
           "assertions_total": 1, "assertions_passed": 0,
           "assertion_results": [{"passed": False, "excluded": False}],
           "input_tokens": 2, "output_tokens": 0}
    row.update(bad)
    with pytest.raises(ValueError):
        a.record_type_is(row)


def test_forged_positive_assertion_total_refuses(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=1)
    item = root / "task-000" / "public-automationbench.private.json"
    d = json.loads(item.read_text())
    d["tasks"][0]["assertions_passed"] = 0
    item.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="positive assertion counts"):
        a.inspect(root)


def test_mismatched_raw_usage_refuses(tmp_path, monkeypatch):
    root = fixture(tmp_path, monkeypatch, completed=1)
    item = root / "task-000" / "public-automationbench.private.json"
    d = json.loads(item.read_text())
    d["usage_by_task"][0]["input_tokens"] = 99
    item.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="official usage"):
        a.inspect(root)
