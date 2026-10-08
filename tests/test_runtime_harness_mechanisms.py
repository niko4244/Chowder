"""Batch-009: controlled harness experiment.

These tests exercise the expanded runtime benchmark and the two generic harness
mechanisms (state-aware file discovery and post-failure recovery) without a GPU.
A scripted policy stands in for the frozen model so the mechanisms are judged by
the traces and named metrics they produce.
"""
from __future__ import annotations

import re

from chowder.runtime_eval import (
    HELDOUT_TASKS,
    RUNTIME_METRIC_KEYS,
    TASKS,
    RuntimeTask,
    _evaluate_task,
    _is_green,
    run_live_benchmark,
    _state_message,
)

FAMILIES = {"imports", "stateful", "multi_file", "failed_first_fix", "misleading_output", "wrong_second_fix"}


def _call(name: str, **args: str) -> str:
    body = "".join(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>" for key, value in args.items())
    return f"<tool_call>{name}{body}</tool_call>"


def _one(name: str, fix: str, *, target: str = "app.py") -> RuntimeTask:
    return RuntimeTask(name, f"Repair {target}.", {target: "def f():\n    return 1\n"}, target, fix, 2, "2 passed")


def test_runtime_task_families_are_expanded_disjoint_and_initially_red():
    dev_names = {task.name for task in TASKS}
    held_names = {task.name for task in HELDOUT_TASKS}
    assert len(TASKS) >= 20
    assert len(HELDOUT_TASKS) >= 8
    assert dev_names.isdisjoint(held_names)
    assert FAMILIES <= {task.family for task in TASKS}
    assert FAMILIES <= {task.family for task in HELDOUT_TASKS}

    # Every task must start red: a benchmark task that is already green would
    # credit any policy, including one that does nothing.
    for task in (*TASKS, *HELDOUT_TASKS):
        assert not _is_green(_evaluate_task(task, dict(task.initial))), task.name

    # A whitespace/duplicate-free dedup guard: no two tasks share a name or a
    # primary target path within a split.
    assert len(dev_names) == len(TASKS)
    assert len(held_names) == len(HELDOUT_TASKS)


def test_is_green_rejects_misleading_test_output():
    assert _is_green("2 passed")
    assert _is_green("3 passed in 0.10s")
    assert not _is_green("0 passed, 2 failed")
    assert not _is_green("1 failed, 1 passed")
    assert not _is_green("2 passed\nERROR collecting extra_tests.py\nTraceback (most recent call last):")
    assert not _is_green("1 failed, 1 passed\n(report generated from a stale cache)")


def test_state_aware_harness_exposes_real_workspace_and_cuts_invalid_reads():
    task = RuntimeTask(
        "t_state", "Repair app.py.",
        {"app.py": "def f():\n    return 1\n", "helper.py": "X = 1\n"},
        "app.py", "", 2, "2 passed",
        check=lambda files: "2 passed" if "return 2" in files.get("app.py", "") else "FAILED 2 - impl",
    )
    state = {"turn": 0}

    def generate(messages):
        turn = state["turn"]
        state["turn"] += 1
        if turn == 0:
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            if "Files:" in system:
                listed = re.search(r"Files: (.+?)\. Read", system)
                assert listed, "compact state-aware system message must list real files"
                first = listed.group(1).split(", ")[0]
                return _call("read_file", path=first)
            return _call("read_file", path="test_version.py")  # hallucinated path
        if turn == 1:
            return _call("write_file", path="app.py", content="def f():\n    return 2\n")
        if turn == 2:
            return _call("run_tests")
        return "The observed suite is green."

    plain = run_live_benchmark(generate, max_turns=4, harness="plain", tasks=(task,))
    state["turn"] = 0
    aware = run_live_benchmark(generate, max_turns=4, harness="state_aware", tasks=(task,))
    assert plain["metrics"]["runtime_nonexistent_read_rate"] == 1.0
    assert aware["metrics"]["runtime_nonexistent_read_rate"] == 0.0
    assert aware["tasks"][0]["green_seen"] is True
    assert all(action["event"] != "nonexistent_read" for row in aware["tasks"] for action in row["action_rewards"])


def test_recovery_harness_blocks_redundant_reruns_after_red():
    task = _one("t_recovery", "return 2")

    def make():
        state = {"turn": 0}

        def generate(messages):
            turn = state["turn"]
            state["turn"] += 1
            if turn == 0:
                return _call("read_file", path="app.py")
            if turn == 1:
                return _call("write_file", path="app.py", content="def f():\n    return 1\n")  # no progress
            if turn <= 4:
                return _call("run_tests")
            return "Done."

        return generate

    plain = run_live_benchmark(make(), max_turns=8, harness="plain", tasks=(task,))
    recovery = run_live_benchmark(make(), max_turns=8, harness="recovery", tasks=(task,))
    assert plain["metrics"]["runtime_green_rate"] == 0.0
    assert plain["metrics"]["runtime_repeated_action_rate"] > recovery["metrics"]["runtime_repeated_action_rate"]
    assert recovery["metrics"]["runtime_execution_cost"] < plain["metrics"]["runtime_execution_cost"]
    assert any(
        action["event"] == "blocked_run"
        for row in recovery["tasks"]
        for action in row["action_rewards"]
    )
    # Blocked runs are recorded but never executed.
    blocked = [
        entry for row in recovery["tasks"] for entry in row["trace"] if entry.get("synthetic")
    ]
    assert blocked and all(entry["cost"] == 0 for entry in blocked)


def test_live_benchmark_publishes_every_named_metric_and_detects_premature_completion():
    task = _one("t_metrics", "return 2")

    def premature(messages):
        return "All fixed!"  # reports without any work

    result = run_live_benchmark(premature, max_turns=3, harness="plain", tasks=(task,))
    assert set(result["metrics"]) == set(RUNTIME_METRIC_KEYS)
    assert result["metrics"]["runtime_premature_completion_rate"] == 1.0
    assert result["metrics"]["runtime_green_rate"] == 0.0
    assert result["harness"] == "plain"
    assert "single_file" in result["families"]
    assert result["metrics"]["prompt_message_chars"] > 0
    assert result["metrics"]["policy_message_chars"] > 0


def test_recovery_leaves_the_success_path_untouched():
    task = _one("t_success", "return 2")
    state = {"turn": 0}

    def generate(messages):
        turn = state["turn"]
        state["turn"] += 1
        if turn == 0:
            return _call("read_file", path="app.py")
        if turn == 1:
            return _call("write_file", path="app.py", content="def f():\n    return 2\n")
        if turn == 2:
            return _call("run_tests")
        return "Green; done."

    result = run_live_benchmark(generate, max_turns=5, harness="recovery", tasks=(task,))
    assert result["metrics"]["runtime_green_rate"] == 1.0
    assert result["metrics"]["runtime_premature_completion_rate"] == 0.0
    assert result["metrics"]["runtime_repeated_action_rate"] == 0.0


def test_wrong_second_fix_tasks_start_red_and_distinguish_partial_from_corrected_fix():
    evolve = [task for task in TASKS if task.family == "wrong_second_fix"]
    heldout = [task for task in HELDOUT_TASKS if task.family == "wrong_second_fix"]
    assert len(evolve) >= 2
    assert len(heldout) >= 2
    assert {task.name for task in evolve}.isdisjoint({task.name for task in heldout})
    for task in (*evolve, *heldout):
        assert not _is_green(_evaluate_task(task, task.initial))
        if "retry.py" in task.initial:
            partial = {**task.initial, task.target: "def should_retry(attempt, limit): return attempt <= limit"}
            fixed = {**task.initial, task.target: "def should_retry(attempt, limit): return attempt < limit"}
        elif "window.py" in task.initial:
            partial = {**task.initial, task.target: "def window(values, start, end): return values[:min(end, len(values))]"}
            fixed = {**task.initial, task.target: "def window(values, start, end): return values[:end + 1]"}
        elif "retry_policy2.py" in task.initial:
            partial = {**task.initial, task.target: "def permitted(retries, max_retries): return retries <= max_retries"}
            fixed = {**task.initial, task.target: "def permitted(retries, max_retries): return retries < max_retries"}
        else:
            partial = {**task.initial, task.target: "def prefix_through(items, index): return items[:min(index, len(items))]"}
            fixed = {**task.initial, task.target: "def prefix_through(items, index): return items[:index + 1]"}
        assert "FAILED 1" in _evaluate_task(task, partial), task.name
        assert _is_green(_evaluate_task(task, fixed)), task.name


def test_compact_state_aware_prompt_is_shorter_than_legacy_and_metrics_record_it():
    files = {"app.py": "def f(): return 1", "helper.py": "X = 1"}
    assert len(_state_message(files, compact=True)["content"]) < len(_state_message(files, compact=False)["content"])
    task = RuntimeTask(
        "t_compact_cost", "Fix app.py.", files, "app.py", "return 2", 1, "1 passed",
        check=lambda workspace: "1 passed" if "return 2" in workspace["app.py"] else "FAILED 1",
    )

    def make_generate():
        turn = {"value": 0}

        def generate(messages):
            index = turn["value"]
            turn["value"] += 1
            if index == 0:
                return _call("read_file", path="app.py")
            if index == 1:
                return _call("write_file", path="app.py", content="def f(): return 2")
            if index == 2:
                return _call("run_tests")
            return "Verified."

        return generate

    compact = run_live_benchmark(make_generate(), max_turns=4, harness="state_aware", tasks=(task,))
    legacy = run_live_benchmark(make_generate(), max_turns=4, harness="state_aware_legacy", tasks=(task,))
    assert compact["metrics"]["runtime_green_rate"] == legacy["metrics"]["runtime_green_rate"] == 1.0
    assert compact["metrics"]["prompt_message_chars"] < legacy["metrics"]["prompt_message_chars"]


def test_recovery_requires_a_changed_workspace_before_retesting():
    task = _one("t_unchanged_recovery", "return 2")
    actions = [
        _call("write_file", path="app.py", content="def f(): return 1"),
        _call("run_tests"),
        _call("run_tests"),  # blocked until the file changes
        _call("write_file", path="app.py", content="def f(): return 2"),
        _call("run_tests"),
        "Verified.",
    ]
    index = {"value": 0}

    def generate(_messages):
        result = actions[index["value"]]
        index["value"] += 1
        return result

    result = run_live_benchmark(generate, max_turns=6, harness="recovery", tasks=(task,))
    tool_rows = [row for row in result["tasks"][0]["trace"] if row.get("role") == "tool"]
    assert tool_rows[1]["tool"] == "run_tests" and not tool_rows[1].get("synthetic")
    assert tool_rows[2]["tool"] == "run_tests" and tool_rows[2].get("synthetic")
    assert tool_rows[2]["cost"] == 0
    assert tool_rows[4]["tool"] == "run_tests" and not tool_rows[4].get("synthetic")
    assert result["metrics"]["runtime_green_rate"] == 1.0


def test_wrong_second_fix_family_requires_a_real_partial_red_then_changed_fix():
    task = next(task for task in TASKS if task.name == "retry_boundary_second_fix")
    partial = "def should_retry(attempt, limit): return attempt <= limit"
    fixed = "def should_retry(attempt, limit): return attempt < limit"
    calls = [
        _call("write_file", path="retry.py", content=partial),
        _call("run_tests"),
        _call("write_file", path="retry.py", content=fixed),
        _call("run_tests"),
        "Verified after the corrected fix.",
    ]
    index = {"value": 0}

    def generate(_messages):
        value = calls[index["value"]]
        index["value"] += 1
        return value

    result = run_live_benchmark(generate, max_turns=5, harness="recovery", tasks=(task,))
    tool_rows = [row for row in result["tasks"][0]["trace"] if row.get("role") == "tool"]
    assert tool_rows[1]["observation"].startswith("FAILED 1, 2 passed")
    assert tool_rows[3]["observation"] == "3 passed"
    assert result["metrics"]["runtime_green_rate"] == 1.0
    assert result["tasks"][0]["premature_completion"] is False


def test_a_new_red_test_revokes_prior_green_until_another_green_run():
    task = _one("t_green_then_red", "return 2")
    calls = [
        _call("write_file", path="app.py", content="def f(): return 2"),
        _call("run_tests"),
        _call("write_file", path="app.py", content="def f(): return 1"),
        _call("run_tests"),
        # Green was revoked by the regressing write + red test; the model must
        # re-verify green before any report is legitimate.
        _call("run_tests"),
        "I am done.",
    ]
    index = {"value": 0}

    def generate(_messages):
        value = calls[index["value"]]
        index["value"] += 1
        return value

    # The script ends with a gated report, which never breaks the loop, so
    # max_turns must match the script length exactly.
    result = run_live_benchmark(generate, max_turns=6, harness="state_aware", tasks=(task,))
    assert result["tasks"][0]["green_seen"] is False
    assert result["tasks"][0]["premature_completion"] is True


def test_harness_metrics_bridge_feeds_regularized_selection():
    from chowder.harness_evolution import HarnessProposal, metrics_from_benchmark, select_first_round

    def bench(reward, green, nonexistent, cost, tokens):
        return {
            "metrics": {
                "runtime_reward": reward,
                "runtime_green_rate": green,
                "runtime_nonexistent_read_rate": nonexistent,
                "runtime_execution_cost": cost,
                "policy_tokens": tokens,
            }
        }

    incumbent = metrics_from_benchmark(bench(-6.0, 0.30, 0.50, 5.0, 400))
    candidate = metrics_from_benchmark(bench(-1.0, 0.80, 0.00, 4.0, 300))
    assert incumbent.cost == 5.0
    proposal = HarnessProposal("state-aware", "tooling", ("expose the live workspace file list",), candidate)
    accepted = select_first_round(
        incumbent, candidate,
        evolve_incumbent=incumbent, evolve_candidate=candidate,
        heldout_incumbent=incumbent, heldout_candidate=candidate,
        proposal=proposal, forbidden_terms=("app.py", "version.py"),
    )
    assert accepted["accepted"] is True

    leaked = HarnessProposal("leaky", "prompt", ("hard-code app.py",), candidate)
    rejected = select_first_round(
        incumbent, candidate,
        evolve_incumbent=incumbent, evolve_candidate=candidate,
        heldout_incumbent=incumbent, heldout_candidate=candidate,
        proposal=leaked, forbidden_terms=("app.py",),
    )
    assert rejected["accepted"] is False
    assert "leakage" in rejected["reason"]
