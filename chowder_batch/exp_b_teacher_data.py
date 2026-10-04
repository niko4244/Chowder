"""Experiment B Phase 2: generate auditable teacher data and execution traces.

Only answer-checked final assistant content enters ``rows`` (the SFT-ready
split). Ungraded code and abstention examples stay in ``review_queue`` until
reviewed. Executed tool traces are a separate artifact; every file write must
carry both parsed arguments and every claimed success must have a green test
observation. No teacher statement is treated as execution ground truth.

The teacher is served by the existing local llama.cpp endpoint. Repair traces
use only the development tasks ``two_file_fix`` and ``class_counter``; held-out
runtime tasks and Experiment E's final repair set are never used here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chowder_batch"))

TEACHER_PORT = 18081
TEACHER_ID = "Qwen3.8-9B-abliterated-25 (local llama.cpp)"

# The content key is essential: the harness pairs each arg_key with exactly
# one following arg_value. Without it, a write is executed with empty content.
TOOL_PROTOCOL = (
    " Every action must be exactly one tool call. For example: "
    "<tool_call>read_file<arg_key>path</arg_key><arg_value>api.py</arg_value></tool_call>. "
    "A write_file call MUST include both keyed arguments, in this form: "
    "<tool_call>write_file<arg_key>path</arg_key><arg_value>api.py</arg_value>"
    "<arg_key>content</arg_key><arg_value>complete file contents</arg_value></tool_call>. "
    "The other available tool is run_tests, called as "
    "<tool_call>run_tests</tool_call>. After an observed green run_tests result, "
    "reply with one plain sentence and no tool call."
)

_GREEN = re.compile(r"\b[1-9]\d*\s+passed\b", re.I)
_RED = re.compile(r"\b(failed|error|traceback|exception)\b", re.I)
_SUCCESS_CLAIM = re.compile(
    r"\b(?:tests? (?:all )?(?:pass|passed|are green)|all checks passed|"
    r"verified green|successfully verified)\b",
    re.I,
)
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.I | re.S)


# The two development repairs do not overlap the four held-out Experiment E
# repair tasks. Each task's exact tests come from chowder.runtime_eval.TASKS.
DEV_REPAIR_TASKS = (
    {
        "name": "two_file_fix",
        "target": "api.py",
        "initial": {
            "api.py": "from mathx import double\n\ndef quadruple(n):\n    return double(double(n))\n",
            "mathx.py": "def double(n):\n    return n + 2\n",
        },
    },
    {
        "name": "class_counter",
        "target": "counter.py",
        "initial": {
            "counter.py": "class Counter:\n    total = 0\n\n    def add(self, n):\n        Counter.total += n\n        return Counter.total\n",
        },
    },
)


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def chat(
    port: int,
    messages: list[dict],
    *,
    max_tokens: int,
    include_reasoning: bool = False,
) -> tuple[str, dict]:
    """Call the local teacher and keep final content separate from reasoning.

    Ordinary SFT uses the assistant's final ``content`` only. Tool execution
    may inspect ``reasoning_content`` to locate its requested action, but that
    channel is not mistaken for the final answer or silently graded as one.
    """
    started = time.perf_counter()
    data = _post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {"messages": messages, "max_tokens": max_tokens, "temperature": 0},
    )
    wall = time.perf_counter() - started
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if include_reasoning:
        visible = "\n".join(part for part in (reasoning, content) if part)
    else:
        visible = content
    usage = data.get("usage", {})
    return visible, {
        "wall_s": round(wall, 2),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
        "final_content_present": bool(content.strip()),
        "reasoning_channel_present": bool(reasoning.strip()),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _experiment_output_paths(out: Path) -> tuple[Path, Path, Path, Path]:
    return (
        out,
        out.with_name(out.stem + "_condition_a.jsonl"),
        out.with_name(out.stem + "_condition_b.jsonl"),
        out.with_name(out.stem + "_training_conditions.json"),
    )


def _assert_new_output_paths(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite Experiment B artifacts: {existing}")


def _write_new_artifact(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as artifact:
        artifact.write(content)


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def _last_nonempty_line(response: str) -> str:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].replace("**", "").replace("`", "")


def grade_numeric(answer: str, response: str) -> bool:
    """Accept only an exact numeric answer on the final, unambiguous line."""
    expected = _as_decimal(str(answer))
    line = _last_nonempty_line(response)
    if expected is None or not line:
        return False
    number = r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    unit = r"(?:\s*(?:km\s*/\s*h|miles?\s*/\s*hour|hours?|minutes?|seconds?|cups?|continents?|degrees?|dollars?|usd|%|percent))?"
    currency = "(?:" + re.escape(chr(36)) + "|€|£)?"
    currency_after = "(?:" + re.escape(chr(36)) + "|€|£)?"
    patterns = (
        rf"(?:final\s+)?answer\s*[:=]\s*{currency}{number}{currency_after}{unit}[.!]?",
        rf"(?:the\s+)?larger\s+integer\s+is\s*{number}[.!]?",
        rf"{currency}{number}{currency_after}{unit}[.!]?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, line, re.I)
        if match:
            return _as_decimal(match.group(1)) == expected
    return False


def _normalized_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def _phrase_present(phrase: str, response: str) -> bool:
    expected = _normalized_phrase(phrase)
    final_line = _normalized_phrase(_last_nonempty_line(response))
    if not expected:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", final_line))


def grade_expected_answer(answer: str, response: str) -> bool:
    """Grade numeric targets by final-answer position and phrases by boundaries."""
    if _as_decimal(str(answer)) is not None:
        return grade_numeric(str(answer), response)
    return _phrase_present(str(answer), response)


def load_knowledge_prompts() -> list[dict]:
    """Prompts with objectively checkable answers; other rows need review."""
    return [
        {"pid": "kg_001", "category": "general_knowledge", "question": "What is the capital of Australia? Answer with the city name only.", "answer": "Canberra"},
        {"pid": "kg_002", "category": "general_knowledge", "question": "How many continents are there on Earth? Answer with a number on the final line in the format Answer: <number>.", "answer": "7"},
        {"pid": "kg_003", "category": "general_knowledge", "question": "What gas do plants absorb from the atmosphere for photosynthesis? Answer in one short phrase.", "answer": "carbon dioxide"},
        {"pid": "kg_004", "category": "general_knowledge", "question": "Who wrote the play 'Romeo and Juliet'? Answer with the name only.", "answer": "Shakespeare"},
        {"pid": "kg_005", "category": "general_knowledge", "question": "What is the chemical symbol for gold? Answer with just the symbol.", "answer": "Au"},
        {"pid": "math_001", "category": "math_reasoning", "question": "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the steps, then end with Answer: <number> on its own final line.", "answer": "80"},
        {"pid": "math_002", "category": "math_reasoning", "question": "If a shirt costs $40 after a 20% discount, what was the original price? Show the steps, then end with Answer: <number> on its own final line.", "answer": "50"},
        {"pid": "math_003", "category": "math_reasoning", "question": "What is 15% of 240? Show the steps, then end with Answer: <number> on its own final line.", "answer": "36"},
        {"pid": "math_004", "category": "math_reasoning", "question": "A recipe needs 3/4 cup of sugar per batch. How much sugar is needed for 5 batches? Show the steps, then end with Answer: <number> on its own final line.", "answer": "3.75"},
        {"pid": "math_005", "category": "math_reasoning", "question": "The sum of two consecutive integers is 87. What is the larger integer? Show the steps, then end with Answer: <number> on its own final line.", "answer": "44"},
        {"pid": "code_001", "category": "code_generation", "question": "Write a Python function is_palindrome(s) that returns True if string s reads the same forwards and backwards, ignoring case. Include a brief explanation.", "answer": None},
        {"pid": "code_002", "category": "code_generation", "question": "Write a Python function word_frequencies(text) that returns a dict mapping each lowercase whitespace-separated word to its count. Include a brief explanation.", "answer": None},
        {"pid": "code_003", "category": "code_debugging", "question": "This code raises an exception. Explain the bug and give corrected code for nonempty input lists:\n\ndef average(values):\n    return sum(values) / 0", "answer": None},
        {"pid": "code_004", "category": "code_debugging", "question": "This loop never terminates. Explain the bug and give a corrected function count_to_ten() in a Python code block that prints 0 through 9 exactly once, using a bounded for-loop:\n\ni = 0\nwhile i < 10:\n    print(i)", "answer": None},
    ]


def _extract_code(content: str, *, function_name: str) -> str | None:
    blocks = _CODE_BLOCK.findall(content)
    matching = [
        block.strip()
        for block in blocks
        if re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", block)
    ]
    return matching[0] if matching else None


def _safe_code_namespace(source: str) -> dict[str, Any]:
    """Return interpreted functions; candidate source is never compiled or executed."""
    try:
        from .exp_b_restricted_python import candidate_namespace
    except ImportError:  # direct script invocation
        from exp_b_restricted_python import candidate_namespace
    return candidate_namespace(source)


def _grade_code_candidate(prompt_id: str, content: str) -> tuple[bool | None, str]:
    function_names = {
        "code_001": "is_palindrome",
        "code_002": "word_frequencies",
        "code_003": "average",
        "code_004": "count_to_ten",
    }
    function_name = function_names.get(prompt_id)
    if function_name is None:
        return None, "no restricted interpreter contract is defined for this code prompt"
    source = _extract_code(content, function_name=function_name)
    if source is None:
        return None, f"no fenced Python definition for {function_name} to test"
    try:
        namespace = _safe_code_namespace(source)
        if prompt_id == "code_001":
            fn = namespace["is_palindrome"]
            passed = fn("Racecar") is True and fn("A man") is False and fn("level") is True
        elif prompt_id == "code_002":
            fn = namespace["word_frequencies"]
            passed = fn("The cat cat") == {"the": 1, "cat": 2}
        elif prompt_id == "code_003":
            fn = namespace["average"]
            passed = abs(float(fn([1, 2, 3])) - 2.0) < 1e-9 and abs(float(fn([2, 3])) - 2.5) < 1e-9
        else:
            fn = namespace["count_to_ten"]
            fn()
            lines = fn.last_stdout.splitlines()
            passed = lines == [str(value) for value in range(10)]
    except Exception as error:
        try:
            from .exp_b_restricted_python import RestrictedPythonError
        except ImportError:  # direct script invocation
            from exp_b_restricted_python import RestrictedPythonError
        if isinstance(error, RestrictedPythonError):
            return None, f"not graded: restricted interpreter limitation: {str(error)[:160]}"
        return False, f"restricted code check failed: {type(error).__name__}: {str(error)[:160]}"
    return bool(passed), (
        "bounded AST interpretation passed the task contract; prose explanation not graded"
        if passed
        else "bounded AST interpretation failed the task contract"
    )


def _text_row_verdict(prompt: dict, content: str, meta: dict) -> tuple[str, str]:
    if not content.strip():
        return "review_required_no_final_content", "teacher returned no final content"
    if meta.get("finish_reason") in {"length", "max_tokens"}:
        return "review_required_truncated", "completion ended at the token limit"
    answer = prompt.get("answer")
    if answer is None:
        if prompt.get("category") in {"code_generation", "code_debugging"}:
            passed, note = _grade_code_candidate(prompt["pid"], content)
            if passed is True:
                return "kept", note
            if passed is False:
                return "dropped_wrong_answer", note
            return "review_required_ungraded", note
        return "review_required_ungraded", "no automatic correctness test is defined"
    if grade_expected_answer(str(answer), content):
        return "kept", "exact expected-answer check passed on final content"
    return "dropped_wrong_answer", "exact expected-answer check failed on final content"


def generate_text_rows(out_rows: list[dict]) -> None:
    """Generate final-answer-only knowledge/math/code candidates."""
    for prompt in load_knowledge_prompts():
        content, meta = chat(
            TEACHER_PORT,
            [{"role": "user", "content": prompt["question"]}],
            max_tokens=1024,
        )
        verdict, grade_note = _text_row_verdict(prompt, content, meta)
        out_rows.append({
            "id": prompt["pid"],
            "category": prompt["category"],
            "messages": [
                {"role": "user", "content": prompt["question"]},
                {"role": "assistant", "content": content},
            ],
            "provenance": {
                "source_prompt_id": prompt["pid"],
                "generator": TEACHER_ID,
                "grade_verdict": verdict,
                "grade_note": grade_note,
                "expected_answer": prompt["answer"],
                "completion_tokens": meta["completion_tokens"],
                "prompt_tokens": meta["prompt_tokens"],
                "finish_reason": meta["finish_reason"],
                "reasoning_channel_present_not_used": meta["reasoning_channel_present"],
                "generated_at": _timestamp(),
            },
        })
        print(f"{prompt['pid']}: {verdict}", flush=True)


def load_tool_decision_prompts() -> list[dict]:
    """Small, objectively graded next-action choices for condition B."""
    common = (
        "Choose exactly one next action. Return exactly one JSON object on the final line "
        "with keys next_action and path; do not use a Markdown code fence. Valid "
        "next_action values are read_file, write_file, run_tests, and report_success. "
        "Use null for path unless the action is read_file. Do not claim tests passed "
        "unless the supplied observation says so.\n\n"
    )
    return [
        {
            "pid": "toolsel_001", "category": "tool_selection",
            "question": common + "Task: repair parse.py. Workspace listing: parse.py, helpers.py. "
            "No file contents have been inspected yet.",
            "expected": {"next_action": "read_file", "path": "parse.py"},
        },
        {
            "pid": "toolsel_002", "category": "tool_selection",
            "question": common + "Task: repair parse.py. You read parse.py and observed it imports "
            "normalize, but the workspace listing contains only parse.py and helpers.py; "
            "helpers.py defines normalise. No edits have been made.",
            "expected": {"next_action": "read_file", "path": "helpers.py"},
        },
        {
            "pid": "toolsel_003", "category": "tool_selection",
            "question": common + "Task: repair parse.py. You have inspected the file, prepared a "
            "corrected implementation, and must apply it before verification. No write has "
            "occurred yet.",
            "expected": {"next_action": "write_file", "path": "parse.py"},
        },
        {
            "pid": "toolsel_004", "category": "failed_patch_recovery",
            "question": common + "Task: repair parse.py. You wrote a first patch, then the test "
            "observation was 'FAILED 2 - ImportError: cannot import name normalise'. The "
            "workspace listing contains parse.py and helpers.py. Inspect the current target "
            "before choosing a corrected patch.",
            "expected": {"next_action": "read_file", "path": "parse.py"},
        },
        {
            "pid": "toolsel_005", "category": "tool_selection",
            "question": common + "Task: repair parse.py. The latest tool observation is 'OK parse.py written'. "
            "No test has been run since this write.",
            "expected": {"next_action": "run_tests", "path": None},
        },
        {
            "pid": "toolsel_006", "category": "tool_selection",
            "question": common + "Task: repair parse.py. The tool observation after your write was "
            "'2 passed'. This is the latest observation.",
            "expected": {"next_action": "report_success", "path": None},
        },
        {
            "pid": "toolsel_007", "category": "tool_selection",
            "question": common + "Task: repair parse.py. You attempted to read missing.py and got "
            "'ERROR: no such file: missing.py. Available files: parse.py, helpers.py'. "
            "The target has not been inspected.",
            "expected": {"next_action": "read_file", "path": "parse.py"},
        },
    ]


def _tool_decision_matches(expected: dict, response: str) -> bool:
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        actual = json.loads(text)
    except json.JSONDecodeError:
        return False

    return (
        isinstance(actual, dict)
        and set(actual) == {"next_action", "path"}
        and actual.get("next_action") == expected["next_action"]
        and actual.get("path") == expected["path"]
    )


def generate_tool_decision_rows(out_rows: list[dict]) -> None:
    """Cache teacher next-action choices and grade them against fixed scenarios."""
    for prompt in load_tool_decision_prompts():
        content, meta = chat(
            TEACHER_PORT,
            [{"role": "user", "content": prompt["question"]}],
            max_tokens=160,
        )
        matches = (
            meta["finish_reason"] not in {"length", "max_tokens"}
            and _tool_decision_matches(prompt["expected"], content)
        )
        verdict = "kept" if matches else "dropped_wrong_action"
        out_rows.append({
            "id": prompt["pid"],
            "category": prompt["category"],
            "messages": [
                {"role": "user", "content": prompt["question"]},
                {"role": "assistant", "content": content},
            ],
            "provenance": {
                "source_prompt_id": prompt["pid"],
                "generator": TEACHER_ID,
                "grade_verdict": verdict,
                "grade_note": "exact JSON next-action/path match" if matches else "teacher action did not match scenario oracle",
                "expected_action": prompt["expected"],
                "completion_tokens": meta["completion_tokens"],
                "prompt_tokens": meta["prompt_tokens"],
                "finish_reason": meta["finish_reason"],
                "generated_at": _timestamp(),
            },
        })
        print(f"{prompt['pid']}: {verdict}", flush=True)


def generate_abstention_rows(out_rows: list[dict]) -> None:
    """Generate abstention candidates; they remain quarantined for review."""
    prompts = [
        {"pid": "abst_001", "question": "What is my current account balance?", "reason": "personal data is not present in the conversation"},
        {"pid": "abst_002", "question": "As of 2026-09-24, who won the 2027 World Series?", "reason": "the event is in the future as of the recorded generation date"},
        {"pid": "abst_003", "question": "What does the config file in my project set for the timeout value?", "reason": "requires inspecting a file that was not provided"},
    ]
    for prompt in prompts:
        content, meta = chat(
            TEACHER_PORT,
            [{
                "role": "user",
                "content": prompt["question"] + " If the answer is not available from the information given, say so and state what evidence you need.",
            }],
            max_tokens=768,
        )
        out_rows.append({
            "id": prompt["pid"],
            "category": "abstention",
            "messages": [
                {"role": "user", "content": prompt["question"]},
                {"role": "assistant", "content": content},
            ],
            "provenance": {
                "source_prompt_id": prompt["pid"],
                "generator": TEACHER_ID,
                "grade_verdict": "review_required_abstention",
                "expected_behaviour": "explicit abstention or evidence request",
                "completion_truncated": meta["finish_reason"] in {"length", "max_tokens"},
                "abstention_reason": prompt["reason"],
                "completion_tokens": meta["completion_tokens"],
                "prompt_tokens": meta["prompt_tokens"],
                "finish_reason": meta["finish_reason"],
                "generated_at": _timestamp(),
            },
        })
        print(f"{prompt['pid']}: review required", flush=True)


def _execute_restricted_workspace(task_name: str, files: dict[str, str]) -> str:
    """Run the two development task checks through the bounded AST interpreter."""
    try:
        from .exp_b_restricted_python import execute_workspace
    except ImportError:  # direct script invocation
        from exp_b_restricted_python import execute_workspace
    return execute_workspace(task_name, files)


def _verify_python_execution_trace(row: dict) -> dict:
    """Replay all actions and run task assertions against actual in-memory code."""
    files = dict(row.get("initial_workspace", {}))
    mismatches = []
    run_tests_count = 0
    green_count = 0
    latest_test_green = False
    for index, turn in enumerate(row.get("trace", [])):
        if not isinstance(turn, dict) or turn.get("role") != "tool":
            continue
        tool = turn.get("tool")
        args = turn.get("args", {})
        observation = str(turn.get("observation", ""))
        if not isinstance(args, dict):
            mismatches.append(f"turn_{index}:malformed_args")
        elif tool == "read_file":
            path = args.get("path")
            actual = files.get(path, f"ERROR: no such file: {path}") if isinstance(path, str) else "ERROR: invalid read path"
            if actual != observation:
                mismatches.append(f"turn_{index}:read_observation_mismatch")
        elif tool == "write_file":
            latest_test_green = False
            path, content = args.get("path"), args.get("content")
            if not isinstance(path, str) or not path or not isinstance(content, str):
                mismatches.append(f"turn_{index}:invalid_write_args")
            else:
                files[path] = content
                if observation != f"OK {path} written":
                    mismatches.append(f"turn_{index}:write_observation_mismatch")
        elif tool == "run_tests":
            run_tests_count += 1
            actual = _execute_restricted_workspace(str(row.get("task_name", "")), files)
            if observation != actual:
                mismatches.append(f"turn_{index}:test_observation_mismatch")
            latest_test_green = _green_observation(actual)
            if latest_test_green:
                green_count += 1
        else:
            mismatches.append(f"turn_{index}:unknown_tool")

    actual_green = run_tests_count > 0 and latest_test_green
    row["python_execution"] = {
        "actual_code_executed": run_tests_count > 0,
        "test_runs": run_tests_count,
        "green_test_runs": green_count,
        "green": actual_green,
        "matched_harness_observations": not mismatches,
        "mismatches": mismatches,
    }
    provenance = row.setdefault("provenance", {})
    provenance["actual_code_executed"] = run_tests_count > 0
    provenance["python_execution_observations_match"] = not mismatches
    provenance["python_test_runs"] = run_tests_count
    provenance["python_green_test_runs"] = green_count
    row["harness_validation_kind"] = "python_execution"
    row["python_execution_verified_green"] = actual_green
    return row


def generate_tooluse_rows() -> list[dict]:
    """Execute teacher actions against development-only runtime workspaces."""
    from chowder.runtime_eval import HELDOUT_TASKS, TASKS, RuntimeTask, run_live_benchmark

    development = {task.name: task for task in TASKS}
    heldout_names = {task.name for task in HELDOUT_TASKS}
    exp_e_tasks_path = Path(__file__).resolve().parents[1] / ".chowder-spark-calib" / "exp-e" / "tasks.json"
    exp_e_eval_repairs: set[str] = set()
    if exp_e_tasks_path.is_file():
        exp_e_tasks = json.loads(exp_e_tasks_path.read_text(encoding="utf-8"))
        exp_e_eval_repairs = {
            str(item["name"])
            for item in exp_e_tasks.get("tasks", [])
            if item.get("split") == "eval" and item.get("kind") == "repair"
        }
    traces = []
    generated_at = _timestamp()

    for task in DEV_REPAIR_TASKS:
        if (
            task["name"] not in development
            or task["name"] in heldout_names
            or task["name"] in exp_e_eval_repairs
        ):
            raise RuntimeError(f"Experiment B task is not development-only: {task['name']}")
        source = development[task["name"]]
        if task["initial"] != source.initial:
            raise RuntimeError(f"cached initial workspace diverges from runtime task {task['name']}")
        goal = f"Repair {task['target']} so its tests pass." + TOOL_PROTOCOL
        runtime_task = RuntimeTask(
            name=source.name,
            goal=goal,
            initial=dict(source.initial),
            target=source.target,
            expected_fix=source.expected_fix,
            test_count=source.test_count,
            test_success=source.test_success,
            family=source.family,
            check=lambda files, task_name=task["name"]: _execute_restricted_workspace(
                task_name, dict(files)
            ),
        )
        teacher_requests: list[dict[str, Any]] = []

        def generate(messages: list[dict[str, str]]) -> str:
            teacher_requests.append(copy.deepcopy(messages))
            content, _meta = chat(
                TEACHER_PORT, messages, max_tokens=512, include_reasoning=True
            )
            return content

        result = run_live_benchmark(
            generate,
            max_turns=8,
            harness="state_aware",
            tasks=(runtime_task,),
            split="exp_b_dev",
        )
        outcome = result["tasks"][0]
        traces.append({
            "id": f"tool_{task['name']}",
            "generated_at": generated_at,
            "category": "tool_use" if outcome["green_seen"] else "failed_patch_recovery",
            "task_name": task["name"],
            "source_split": "development",
            "goal": goal,
            "source_prompt": f"Target file: {source.target}. {goal}",
            "teacher_requests": teacher_requests,
            "target": source.target,
            "initial_workspace": dict(source.initial),
            "harness": "state_aware",
            "harness_validation_kind": "pending_python_execution",
            "green": outcome["green_seen"],
            "trace": outcome["trace"],
            "action_rewards": outcome["action_rewards"],
            "provenance": {
                "source_prompt_id": f"runtime:{task['name']}",
                "generator": f"{TEACHER_ID} via Chowder runtime harness",
                "grade_verdict": "candidate_actual_execution_not_sft",
                "observations_verified": True,
                "actual_code_executed": False,
                "python_execution_observations_match": False,
                "nonexistent_reads": outcome["nonexistent_reads"],
                "premature_completion": outcome["premature_completion"],
                "repeated_actions": outcome["repeated_actions"],
                "execution_cost": outcome["execution_cost"],
                "generated_at": generated_at,
            },
        })
        _verify_python_execution_trace(traces[-1])
        traces[-1]["category"] = (
            "tool_use"
            if traces[-1]["python_execution_verified_green"]
            and traces[-1]["python_execution"]["matched_harness_observations"]
            else "failed_patch_recovery"
        )
        print(
            f"tool_{task['name']}: harness_green={outcome['green_seen']} "
            f"python_green={traces[-1]['python_execution_verified_green']} "
            f"cost={outcome['execution_cost']}",
            flush=True,
        )
    return traces


def _green_observation(observation: str) -> bool:
    return bool(_GREEN.search(observation)) and not bool(_RED.search(observation))


def _trace_review_reasons(row: dict) -> list[str]:
    """Validate trace schema and source-predicate observations without overclaiming execution."""
    reasons = []
    provenance = row.get("provenance", {})
    trace = row.get("trace", [])
    if row.get("source_split") != "development":
        reasons.append("task_split_not_proven_development")
    if not provenance.get("observations_verified"):
        reasons.append("harness_observations_not_verified")
    dev_names = {task["name"] for task in DEV_REPAIR_TASKS}
    if row.get("task_name") not in dev_names:
        reasons.append("task_name_not_in_experiment_b_dev_allowlist")
    else:
        expected_task = next(task for task in DEV_REPAIR_TASKS if task["name"] == row["task_name"])
        if row.get("target") != expected_task["target"]:
            reasons.append("target_disagrees_with_development_task")
        if row.get("initial_workspace") != expected_task["initial"]:
            reasons.append("initial_workspace_disagrees_with_development_task")
    if row.get("harness_validation_kind") != "python_execution":
        reasons.append("validation_is_not_python_execution")

    replayed = _verify_python_execution_trace(copy.deepcopy(row))
    python_report = replayed["python_execution"]
    if not python_report["actual_code_executed"] or provenance.get("actual_code_executed") is not True:
        reasons.append("python_code_execution_not_verified")
    if provenance.get("actual_code_executed") is not python_report["actual_code_executed"]:
        reasons.append("python_execution_claim_disagrees_with_replay")
    if provenance.get("python_execution_observations_match") is not True:
        reasons.append("python_execution_did_not_match_model_visible_observations")
    if provenance.get("python_execution_observations_match") is not python_report["matched_harness_observations"]:
        reasons.append("python_observation_claim_disagrees_with_replay")
    if not python_report["matched_harness_observations"]:
        reasons.extend(
            f"python_execution_replay_{mismatch}"
            for mismatch in python_report["mismatches"]
        )
    if provenance.get("python_test_runs") != python_report["test_runs"]:
        reasons.append("python_test_run_count_disagrees_with_replay")
    if provenance.get("python_green_test_runs") != python_report["green_test_runs"]:
        reasons.append("python_green_test_count_disagrees_with_replay")
    if not python_report["green"]:
        reasons.append("python_execution_did_not_finish_green")
    if row.get("python_execution") != python_report:
        reasons.append("python_execution_report_disagrees_with_replay")
    if row.get("python_execution_verified_green") is not python_report["green"]:
        reasons.append("python_green_label_disagrees_with_replayed_code")

    files = dict(row.get("initial_workspace", {}))
    observed_green = False
    current_test_green = False
    write_seen = False
    allowed_args = {"read_file": {"path"}, "write_file": {"path", "content"}, "run_tests": set()}
    from chowder.runtime_eval import parse_tool_call

    for turn_index, turn in enumerate(trace):
        if not isinstance(turn, dict):
            reasons.append("trace_turn_is_not_an_object")
            continue
        if turn.get("role") == "tool":
            tool = turn.get("tool")
            args = turn.get("args", {})
            assistant_turn = (
                trace[turn_index - 1]
                if turn_index > 0 and trace[turn_index - 1].get("role") == "assistant"
                else None
            )
            parsed_call = parse_tool_call(str(assistant_turn.get("text", ""))) if assistant_turn else None
            if parsed_call != (tool, args):
                reasons.append("tool_arguments_disagree_with_assistant_action")
            observation = str(turn.get("observation", ""))
            if tool not in allowed_args:
                reasons.append("unknown_tool_in_trace")
            elif not isinstance(args, dict) or set(args) != allowed_args[tool]:
                reasons.append("tool_arguments_do_not_match_protocol")
            elif tool == "read_file":
                path = args.get("path")
                if not isinstance(path, str) or not path:
                    reasons.append("read_file_path_missing")
                elif path in files:
                    if observation != files[path]:
                        reasons.append("read_observation_not_grounded_in_workspace")
                elif not observation.startswith(f"ERROR: no such file: {path}"):
                    reasons.append("missing_read_lacks_nonexistent_file_observation")
            elif tool == "write_file":
                path, content = args.get("path"), args.get("content")
                if not isinstance(path, str) or not path or not isinstance(content, str):
                    reasons.append("write_file_missing_parsed_path_or_content")
                elif observation != f"OK {path} written":
                    reasons.append("write_observation_does_not_confirm_write")
                else:
                    files[path] = content
                    write_seen = True
                    current_test_green = False
            elif tool == "run_tests":
                current_test_green = _green_observation(observation)
                if current_test_green:
                    if write_seen:
                        observed_green = True
                    else:
                        reasons.append("green_observation_before_any_write")
        if turn.get("role") in {"early_report", "final_report"}:
            if _SUCCESS_CLAIM.search(str(turn.get("text", ""))) and not current_test_green:
                reasons.append("success_claim_without_prior_green_observation")
    if bool(row.get("green")) != observed_green:
        reasons.append("green_label_disagrees_with_test_observation")
    return sorted(set(reasons))


def dedupe_and_review(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Return SFT-approved rows, verified traces, review queue, and honest stats."""
    seen: set[str] = set()
    approved_sft = []
    execution_traces = []
    review_queue = []
    stats = {
        "seen": len(rows),
        "dropped_duplicate": 0,
        "dropped_wrong_answer": 0,
        "approved_sft": 0,
        "verified_execution_traces": 0,
        "quarantined": 0,
        "quarantine_reasons": {},
    }

    for row in rows:
        if "trace" in row:
            key = f"trace::{row.get('task_name', row.get('id', 'unknown'))}"
            reasons = _trace_review_reasons(row)
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)
            if reasons:
                row["review_status"] = "quarantined"
                row["review_reasons"] = reasons
                review_queue.append(row)
                for reason in reasons:
                    stats["quarantine_reasons"][reason] = stats["quarantine_reasons"].get(reason, 0) + 1
            else:
                row["review_status"] = "verified_execution_not_sft"
                execution_traces.append(row)
            continue

        messages = row.get("messages", [])
        prompt = messages[0].get("content", "") if messages else ""
        key = "prompt::" + re.sub(r"\s+", " ", prompt.casefold()).strip()
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        provenance = row.get("provenance", {})
        verdict = provenance.get("grade_verdict")
        reasons = []
        if verdict == "dropped_wrong_answer":
            stats["dropped_wrong_answer"] += 1
            reasons.append("answer_check_failed")
        elif verdict != "kept":
            reasons.append(str(verdict or "missing_grade_verdict"))
        if len(messages) < 2 or not messages[-1].get("content", "").strip():
            reasons.append("missing_assistant_final_content")
        if provenance.get("finish_reason") in {"length", "max_tokens"}:
            reasons.append("completion_truncated")
        expected = provenance.get("expected_answer")
        if expected is not None and messages and messages[-1].get("content"):
            if not grade_expected_answer(str(expected), messages[-1]["content"]):
                reasons.append("expected_answer_check_failed_on_cached_content")
        if row.get("category") in {"code_generation", "code_debugging"} and verdict == "kept":
            prompt_id = provenance.get("source_prompt_id", "")
            passed, _note = _grade_code_candidate(prompt_id, messages[-1].get("content", "") if messages else "")
            if passed is not True:
                reasons.append("restricted_code_check_failed_on_cached_content")
        expected_action = provenance.get("expected_action")
        if expected_action is not None and messages and messages[-1].get("content"):
            if not _tool_decision_matches(expected_action, messages[-1]["content"]):
                reasons.append("expected_tool_action_check_failed_on_cached_content")
        if reasons:
            row["review_status"] = "quarantined"
            row["review_reasons"] = sorted(set(reasons))
            review_queue.append(row)
            for reason in set(reasons):
                stats["quarantine_reasons"][reason] = stats["quarantine_reasons"].get(reason, 0) + 1
        else:
            row["review_status"] = "approved_sft"
            approved_sft.append(row)

    stats["approved_sft"] = len(approved_sft)
    stats["verified_execution_traces"] = len(execution_traces)
    stats["quarantined"] = len(review_queue)
    return approved_sft, execution_traces, review_queue, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out, condition_a_path, condition_b_path, conditions_path = _experiment_output_paths(Path(args.out))
    _assert_new_output_paths((out, condition_a_path, condition_b_path, conditions_path))

    candidates: list[dict] = []
    generate_text_rows(candidates)
    generate_tool_decision_rows(candidates)
    generate_abstention_rows(candidates)
    candidates.extend(generate_tooluse_rows())

    approved, traces, review_queue, review_stats = dedupe_and_review(candidates)
    condition_a = [
        row for row in approved
        if row["category"] not in {"tool_selection", "failed_patch_recovery"}
    ]
    condition_b = list(approved)
    out.parent.mkdir(parents=True, exist_ok=True)
    for dataset_path, selected_rows in (
        (condition_a_path, condition_a),
        (condition_b_path, condition_b),
    ):
        _write_new_artifact(
            dataset_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
        )
    condition_a_sha = hashlib.sha256(condition_a_path.read_bytes()).hexdigest()
    condition_b_sha = hashlib.sha256(condition_b_path.read_bytes()).hexdigest()
    incremental_action_rows = [
        row for row in condition_b
        if row["category"] in {"tool_selection", "failed_patch_recovery"}
    ]
    training_conditions = {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "status": "pilot_recipe_only; full checkpoint load and training not yet verified",
        "shared_recipe": {
            "backend": "transformers-peft",
            "dataset_format": "chat",
            "messages_field": "messages",
            "base_model": "ibm-granite/granite-4.0-h-tiny",
            "quantization": "4bit",
            "precision": "auto",
            "max_length": 512,
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "learning_rate": 0.00005,
                "epochs": 1,
                "max_steps": 20,
                "logging_steps": 1,
                "save_strategy": "steps",
                "save_steps": 10,
            },
            "lora": {
                "r": 4,
                "alpha": 8,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
                    "input_linear", "output_linear",
                ],
            },
        },
        "condition_a": {
            "dataset": str(condition_a_path),
            "dataset_sha256": condition_a_sha,
            "rows": len(condition_a),
            "ready": bool(condition_a),
        },
        "condition_b": {
            "dataset": str(condition_b_path),
            "dataset_sha256": condition_b_sha,
            "rows": len(condition_b),
            "incremental_action_rows": len(incremental_action_rows),
            "ready": bool(condition_a) and bool(incremental_action_rows),
            "requires_distinct_training_run": True,
        },
        "condition_c": {
            "status": "rejected_tokenizer_mismatch",
            "reason": "Qwen and Granite vocabulary indices do not align; token-level KL is invalid",
        },
    }
    _write_new_artifact(
        conditions_path,
        json.dumps(training_conditions, indent=2, ensure_ascii=False) + "\n",
    )
    trace_candidates = [row for row in review_queue if "trace" in row]
    payload = {
        "dataset": Path(args.out).stem,
        "teacher": TEACHER_ID,
        "teacher_endpoint": f"http://127.0.0.1:{TEACHER_PORT}/v1/chat/completions",
        "training_rows": approved,
        "verified_execution_traces": traces,
        "condition_a_training_rows": condition_a,
        "condition_b_training_rows": condition_b,
        "unverified_harness_trace_candidates": trace_candidates,
        "review_queue": review_queue,
        "condition_c": training_conditions["condition_c"],
        "training_conditions_file": str(conditions_path),
        "categories": sorted({row["category"] for row in approved + review_queue}),
        "review": review_stats,
        "training_ready": {
            "condition_a": bool(condition_a),
            "condition_b": bool(condition_a) and bool(incremental_action_rows),
        },
    }
    _write_new_artifact(out, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {out}: {len(approved)} approved SFT rows, "
        f"{len(trace_candidates)} quarantined harness traces, "
        f"{len(review_queue)} total quarantined; "
        f"A={len(condition_a)} rows / B adds {len(incremental_action_rows)} action rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
