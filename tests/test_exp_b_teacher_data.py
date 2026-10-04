"""Regression tests for Experiment B's fail-closed data curation."""
from __future__ import annotations

import copy
import json

import pytest

from chowder_batch.exp_b_teacher_data import (
    DEV_REPAIR_TASKS,
    _assert_new_output_paths,
    _experiment_output_paths,
    _grade_code_candidate,
    _safe_code_namespace,
    _text_row_verdict,
    _tool_decision_matches,
    _trace_review_reasons,
    dedupe_and_review,
    grade_numeric,
    load_tool_decision_prompts,
)


def _row(category: str, prompt_id: str, content: str, *, answer=None, expected_action=None):
    return {
        "id": prompt_id,
        "category": category,
        "messages": [
            {"role": "user", "content": f"Prompt for {prompt_id}"},
            {"role": "assistant", "content": content},
        ],
        "provenance": {
            "source_prompt_id": prompt_id,
            "grade_verdict": "kept",
            "finish_reason": "stop",
            "expected_answer": answer,
            **({"expected_action": expected_action} if expected_action else {}),
        },
    }


def _tool_turn(tool: str, args: dict, observation: str) -> dict:
    return {"role": "tool", "tool": tool, "args": args, "observation": observation}


def _assistant_tool_call(tool: str, args: dict) -> str:
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in args.items()
    )
    return f"<tool_call>{tool}{body}</tool_call>"


def _verified_python_trace() -> dict:
    from chowder_batch.exp_b_teacher_data import _verify_python_execution_trace

    initial = copy.deepcopy(DEV_REPAIR_TASKS[0]["initial"])
    patched_helper = "def double(n):\n    return n * 2\n"
    row = {
        "id": "trace_two_file_fix",
        "task_name": "two_file_fix",
        "source_split": "development",
        "harness_validation_kind": "python_execution",
        "target": "api.py",
        "green": True,
        "python_execution_verified_green": True,
        "initial_workspace": initial,
        "trace": [
            {"role": "assistant", "text": _assistant_tool_call("read_file", {"path": "api.py"})},
            _tool_turn("read_file", {"path": "api.py"}, initial["api.py"]),
            {"role": "assistant", "text": _assistant_tool_call("write_file", {"path": "mathx.py", "content": patched_helper})},
            _tool_turn("write_file", {"path": "mathx.py", "content": patched_helper}, "OK mathx.py written"),
            {"role": "assistant", "text": _assistant_tool_call("run_tests", {})},
            _tool_turn("run_tests", {}, "2 passed"),
            {"role": "final_report", "text": "The observed test suite is green."},
        ],
        "provenance": {
            "observations_verified": True,
            "actual_code_executed": True,
            "python_execution_observations_match": True,
        },
    }
    return _verify_python_execution_trace(row)


def test_experiment_output_paths_refuse_existing_artifacts_without_overwriting(tmp_path):
    out, condition_a, condition_b, conditions = _experiment_output_paths(
        tmp_path / "teacher_demos_v2.json"
    )
    assert not any(path.exists() for path in (out, condition_a, condition_b, conditions))
    _assert_new_output_paths((out, condition_a, condition_b, conditions))

    condition_a.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _assert_new_output_paths((out, condition_a, condition_b, conditions))
    assert condition_a.read_text(encoding="utf-8") == "preserve me"


def test_numeric_grading_requires_exact_final_line_and_boundaries():
    assert grade_numeric("80", "Work shown.\nAnswer: 80 km/h")
    assert not grade_numeric("80", "Answer: 180")
    assert not grade_numeric("80", "Answer: 80\nBut I am uncertain")
    assert not grade_numeric("80", "Answer: 80 km/h and rising")


def test_text_verdict_keeps_only_exact_answer_and_nontruncated_output():
    prompt = {"pid": "m", "answer": "44"}
    assert _text_row_verdict(prompt, "Answer: 44", {"finish_reason": "stop"})[0] == "kept"
    assert _text_row_verdict(prompt, "Answer: 144", {"finish_reason": "stop"})[0] == "dropped_wrong_answer"
    assert _text_row_verdict(prompt, "Answer: 44", {"finish_reason": "length"})[0] == "review_required_truncated"


def test_tool_decisions_require_exact_schema_and_expected_choice():
    expected = load_tool_decision_prompts()[0]["expected"]
    assert _tool_decision_matches(expected, json.dumps(expected))
    assert _tool_decision_matches(expected, f"```json\n{json.dumps(expected)}\n```")
    assert not _tool_decision_matches(expected, json.dumps({**expected, "claim": "tests pass"}))
    assert not _tool_decision_matches(expected, '{"next_action":"write_file","path":"parse.py"}')
    assert not _tool_decision_matches(expected, "not json")


def test_code_grader_executes_only_small_restricted_contracts():
    good = "```python\ndef is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]\n```"
    assert _grade_code_candidate("code_001", good)[0] is True
    assert _grade_code_candidate("code_001", "```python\ndef is_palindrome(s):\n    return True\n```")[0] is False
    assert _grade_code_candidate("code_001", "No fenced code")[0] is None


def test_code_grader_selects_task_function_and_quarantines_interpreter_limits():
    answer = """```python
def average(values):
    return sum(values) / len(values)
```

```python
>>> average([1, 2, 3, 4, 5])
3.0
```"""
    assert _grade_code_candidate("code_003", answer)[0] is True

    imports = """```python
from collections import Counter
from typing import Dict

def word_frequencies(text: str) -> Dict[str, int]:
    words = text.lower().split()
    return dict(Counter(words))
```"""
    assert _grade_code_candidate("code_002", imports)[0] is True

    annotated = "```python\ndef count_to_ten() -> None:\n    for i in range(10):\n        print(i)\n```"
    assert _grade_code_candidate("code_004", annotated)[0] is True

    unsupported = "```python\ndef f(value: object):\n    return value\n```"
    assert _grade_code_candidate("code_001", unsupported)[0] is None


def test_restricted_runtime_executes_real_dev_task_assertions():
    from chowder_batch.exp_b_teacher_data import _execute_restricted_workspace

    task = DEV_REPAIR_TASKS[0]
    assert _execute_restricted_workspace("two_file_fix", task["initial"]).startswith("FAILED")
    fixed = dict(task["initial"])
    fixed["mathx.py"] = "def double(n):\n    return n * 2\n"
    assert _execute_restricted_workspace("two_file_fix", fixed) == "2 passed"

    task = DEV_REPAIR_TASKS[1]
    assert _execute_restricted_workspace("class_counter", task["initial"]).startswith("FAILED")
    fixed = dict(task["initial"])
    fixed["counter.py"] = (
        "class Counter:\n"
        "    def __init__(self):\n        self.total = 0\n\n"
        "    def add(self, n):\n        self.total += n\n        return self.total\n"
    )
    assert _execute_restricted_workspace("class_counter", fixed) == "2 passed"


def test_code_grader_rejects_imports_and_unbounded_loops():
    with pytest.raises(ValueError):
        _safe_code_namespace("import os\ndef f():\n    return 1\n")
    with pytest.raises(ValueError):
        _safe_code_namespace("def f():\n    while True:\n        pass\n")


def test_bounded_ast_interpreter_covers_loop_and_word_frequency_contracts():
    code = """```python
    def word_frequencies(text):
        counts = {}
        for word in text.lower().split():
            counts[word] = counts.get(word, 0) + 1
        return counts
    ```"""
    assert _grade_code_candidate("code_002", code)[0] is True

    loop = """```python
    def count_to_ten():
        for value in range(10):
            print(value)
    ```"""
    assert _grade_code_candidate("code_004", loop)[0] is True


def test_ast_interpreter_fails_closed_on_indirect_calls_dunders_and_large_loops():
    open_call = _safe_code_namespace("def f():\n    return open('should-not-exist')\n")["f"]
    with pytest.raises(ValueError, match="unknown name"):
        open_call()

    with pytest.raises(ValueError, match="dunder"):
        _safe_code_namespace("def f(value):\n    return value.__class__\n")

    indirect_recursion = _safe_code_namespace(
        "def f(n):\n    call_again = f\n    return call_again(n + 1)\n"
    )["f"]
    with pytest.raises(ValueError, match="depth"):
        indirect_recursion(0)

    with pytest.raises(ValueError, match="integer literal"):
        _safe_code_namespace("def f():\n    return [value for value in range(100000000)]\n")
    huge_range = _safe_code_namespace(
        "def f():\n    return [value for value in range(513)]\n"
    )["f"]
    with pytest.raises(ValueError, match="range"):
        huge_range()

    untrusted_call_target = _safe_code_namespace("def f(candidate):\n    return candidate()\n")["f"]
    with pytest.raises(ValueError, match="call target"):
        untrusted_call_target("not-callable")


def test_ast_interpreter_rejects_unsupported_indirect_import_and_eval_paths():
    with pytest.raises(ValueError, match="dunder"):
        _safe_code_namespace("def f():\n    return __import__('os')\n")
    candidate = _safe_code_namespace("def f():\n    return eval('1 + 1')\n")["f"]
    with pytest.raises(ValueError, match="unknown name"):
        candidate()

    with pytest.raises(ValueError, match="dunder"):
        _safe_code_namespace("def f(fn):\n    return fn.__globals__\n")

    candidate = _safe_code_namespace("def f(fn):\n    return fn()\n")["f"]
    with pytest.raises(ValueError, match="plain data"):
        candidate(lambda: None)


def test_ast_interpreter_copies_external_aliases_and_blocks_mutation_leaks():
    shared = [1]
    aliased = [shared, shared]
    candidate = _safe_code_namespace(
        "def mutate(data):\n    data[0].append(2)\n    return data[1]\n"
    )["mutate"]

    assert candidate(aliased) == [1, 2]
    assert shared == [1]
    assert aliased == [[1], [1]]


def test_ast_interpreter_bounds_external_and_computed_strings_and_collections():
    identity = _safe_code_namespace("def identity(value):\n    return value\n")["identity"]
    with pytest.raises(ValueError, match="external text"):
        identity("x" * 4_097)
    with pytest.raises(ValueError, match="external collection"):
        identity([0] * 513)

    duplicate_text = _safe_code_namespace(
        "def duplicate(text):\n    return text + text\n"
    )["duplicate"]
    with pytest.raises(ValueError, match="text result"):
        duplicate_text("x" * 2_049)

    duplicate_list = _safe_code_namespace(
        "def duplicate(values):\n    return values + values\n"
    )["duplicate"]
    assert len(duplicate_list([0] * 256)) == 512
    with pytest.raises(ValueError, match="collection"):
        duplicate_list([0] * 257)

    append = _safe_code_namespace(
        "def append(values):\n    values.append(1)\n    return values\n"
    )["append"]
    with pytest.raises(ValueError, match="collection"):
        append([0] * 512)


def test_trace_replay_rejects_a_write_after_the_last_green_test():
    row = _verified_python_trace()
    write_args = {"path": "mathx.py", "content": "def double(n):\n    return n * 2\n"}
    write_call = _assistant_tool_call("write_file", write_args)
    final_report_index = next(
        index for index, turn in enumerate(row["trace"])
        if turn.get("role") == "final_report"
    )
    row["trace"][final_report_index:final_report_index] = [
        {"role": "assistant", "text": write_call},
        _tool_turn("write_file", write_args, "OK mathx.py written"),
    ]

    reasons = _trace_review_reasons(row)
    assert "python_execution_did_not_finish_green" in reasons
    assert "python_execution_report_disagrees_with_replay" in reasons


def test_cached_training_approval_regrades_answer_code_and_action():
    answer_row = _row("math_reasoning", "math_001", "Answer: 44", answer="45")
    action = load_tool_decision_prompts()[0]["expected"]
    action_row = _row("tool_selection", "toolsel_001", "{\"next_action\":\"write_file\",\"path\":\"parse.py\"}", expected_action=action)
    code_row = _row("code_generation", "code_001", "```python\ndef is_palindrome(s):\n    return True\n```")
    approved, traces, review, stats = dedupe_and_review([answer_row, action_row, code_row])
    assert not approved and not traces
    assert len(review) == 3
    assert stats["quarantined"] == 3
    assert "expected_answer_check_failed_on_cached_content" in review[0]["review_reasons"]
    assert any("expected_tool_action_check_failed_on_cached_content" in row["review_reasons"] for row in review)
    assert any("restricted_code_check_failed_on_cached_content" in row["review_reasons"] for row in review)


def test_trace_replay_uses_python_execution_as_the_actual_test_oracle():
    from chowder.runtime_eval import TASKS, RuntimeTask, run_live_benchmark
    from chowder_batch.exp_b_teacher_data import _execute_restricted_workspace

    source = next(task for task in TASKS if task.name == "two_file_fix")
    task = next(item for item in DEV_REPAIR_TASKS if item["name"] == "two_file_fix")
    patched_helper = "def double(n):\n    return n * 2\n"
    script_call_count = 0

    def scripted_teacher(_messages):
        nonlocal script_call_count
        script_call_count += 1
        return [
            _assistant_tool_call("read_file", {"path": "api.py"}),
            _assistant_tool_call("write_file", {"path": "mathx.py", "content": patched_helper}),
            _assistant_tool_call("run_tests", {}),
            "The observed suite is green.",
        ][script_call_count - 1]

    runtime_task = RuntimeTask(
        name=source.name,
        goal="repair",
        initial=copy.deepcopy(task["initial"]),
        target=source.target,
        expected_fix=source.expected_fix,
        test_count=source.test_count,
        test_success=source.test_success,
        family=source.family,
        check=lambda files: _execute_restricted_workspace("two_file_fix", dict(files)),
    )
    result = run_live_benchmark(scripted_teacher, max_turns=4, tasks=(runtime_task,))
    observed = [turn["observation"] for turn in result["tasks"][0]["trace"] if turn.get("role") == "tool"]
    assert observed == [task["initial"]["api.py"], "OK mathx.py written", "2 passed"]
    assert result["tasks"][0]["green_seen"] is True


def test_trace_schema_requires_content_observed_write_and_post_write_green():
    row = _verified_python_trace()
    assert _trace_review_reasons(row) == []

    missing_content = copy.deepcopy(row)
    write = next(turn for turn in missing_content["trace"] if turn.get("tool") == "write_file")
    write["args"] = {"path": "api.py"}
    assert "tool_arguments_do_not_match_protocol" in _trace_review_reasons(missing_content)

    bad_read = copy.deepcopy(row)
    next(turn for turn in bad_read["trace"] if turn.get("tool") == "read_file")["observation"] = "fabricated"
    assert "read_observation_not_grounded_in_workspace" in _trace_review_reasons(bad_read)

    false_green = copy.deepcopy(row)
    next(turn for turn in false_green["trace"] if turn.get("tool") == "run_tests")["observation"] = "2 passed, 1 failed"
    assert "green_label_disagrees_with_test_observation" in _trace_review_reasons(false_green)

    forged_action = copy.deepcopy(row)
    forged_action["trace"][0]["text"] = _assistant_tool_call("read_file", {"path": "mathx.py"})
    assert "tool_arguments_disagree_with_assistant_action" in _trace_review_reasons(forged_action)


def test_harness_predicate_claim_is_quarantined_as_not_execution():
    row = _verified_python_trace()
    row["harness_validation_kind"] = "source_predicate_not_python_execution"
    row["provenance"]["actual_code_executed"] = False
    row["provenance"]["python_execution_observations_match"] = False
    reasons = _trace_review_reasons(row)
    assert "python_code_execution_not_verified" in reasons
    assert "python_execution_did_not_match_model_visible_observations" in reasons
    assert "validation_is_not_python_execution" in reasons


def test_dev_task_allowlist_is_explicit_and_rejects_heldout_name():
    names = {task["name"] for task in DEV_REPAIR_TASKS}
    assert names == {"two_file_fix", "class_counter"}
    row = _verified_python_trace()
    row["task_name"] = "config_defaults"
    assert "task_name_not_in_experiment_b_dev_allowlist" in _trace_review_reasons(row)


def test_negative_and_ungraded_examples_do_not_enter_sft():
    positive = _row("general_knowledge", "a", "Canberra", answer="Canberra")
    negative = _row("general_knowledge", "b", "Sydney", answer="Canberra")
    negative["provenance"]["grade_verdict"] = "dropped_wrong_answer"
    ungraded = _row("abstention", "c", "Please share the file.")
    ungraded["provenance"]["grade_verdict"] = "review_required_abstention"
    approved, _traces, review, _stats = dedupe_and_review([positive, negative, ungraded])
    assert [row["id"] for row in approved] == ["a"]
    assert {row["id"] for row in review} == {"b", "c"}
