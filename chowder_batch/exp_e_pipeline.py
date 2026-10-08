"""Injectable combined Experiment E pipeline and evolve-only repair data.

Factual tasks use small-model answers with dense retrieval. Repair tasks use a
teacher generation callback configured externally with n-gram speculation, and
only green observations from the actual runtime harness count as completed.
A red small-model repair escalates to one clean teacher attempt. The callback
interface makes all routing testable offline; this module does not start servers
or claim a speculative speedup by itself.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_e_corpus import RetrievalSubsystem
from exp_e_run import grade_answer, grade_citation, build_messages


def _call_generator(generator: Callable[..., Any], messages: list[dict[str, str]], task: Mapping[str, Any]) -> Any:
    try:
        parameters = inspect.signature(generator).parameters.values()
        accepts_task = any(
            parameter.name == "task" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_task = False
    return generator(messages, task=task) if accepts_task else generator(messages)


def _chat_call(generator: Callable[..., Any], task: Mapping[str, Any], context: str = "") -> tuple[str, dict]:
    messages = build_messages(dict(task), context)
    result = _call_generator(generator, messages, task)
    if isinstance(result, tuple) and len(result) == 2:
        text, metadata = result
    else:
        text, metadata = result, {}
    if not isinstance(text, str):
        raise TypeError("pipeline generator must return text or (text, metadata)")
    if not isinstance(metadata, Mapping):
        raise TypeError("pipeline generator metadata must be a mapping")
    return text, dict(metadata)


def _task_runtime_task(task: Mapping[str, Any]):
    from chowder.runtime_eval import RuntimeTask

    check = task.get("check")
    return RuntimeTask(
        name=str(task["name"]), goal=str(task["question"]),
        initial=dict(task["initial"]), target=str(task["target"]),
        expected_fix=str(task.get("expected_fix", "")),
        test_count=int(task.get("test_count", 1)),
        test_success=str(task.get("test_success", "1 passed")),
        family=str(task.get("family", "exp_e_repair")),
        check=check if callable(check) else None,
    )


def run_repair_pipeline(
    task: Mapping[str, Any], *,
    small_generate: Callable[..., Any],
    teacher_ngram_generate: Callable[..., Any],
    max_turns: int = 8,
    harness: str = "state_aware+recovery",
) -> dict[str, Any]:
    """Run Spark first, then optionally one teacher speculative repair pass."""
    from chowder.runtime_eval import run_live_benchmark

    protocol_goal = str(task["question"]) + (
        " Use exactly one action per turn in the format "
        "<tool_call>tool_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>. "
        "Available actions are read_file(path), write_file(path, content), run_tests(). "
        "Report success only after run_tests returned a green result."
    )
    rt = _task_runtime_task({**task, "question": protocol_goal})
    small_call = _make_runtime_generator(small_generate, task)
    small_result = run_live_benchmark(
        small_call,
        max_turns=max_turns, harness=harness, tasks=(rt,), split="eval_repair_small",
    )
    small_row = small_result["tasks"][0]
    route = "small_verified_green"
    teacher_result = None
    teacher_call = None
    final_row = small_row
    if not small_row["green_seen"]:
        route = "teacher_ngram_escalation"
        # Runtime tasks are pure dict-backed and a new benchmark run gives the
        # escalation an isolated, reset workspace rather than trusting red state.
        teacher_call = _make_runtime_generator(teacher_ngram_generate, task)
        teacher_result = run_live_benchmark(
            teacher_call,
            max_turns=max_turns, harness=harness, tasks=(rt,), split="eval_repair_teacher",
        )
        final_row = teacher_result["tasks"][0]
    return {
        "task": task["name"], "kind": "repair", "route": route,
        "green": bool(final_row["green_seen"]),
        "small_green": bool(small_row["green_seen"]),
        "teacher_escalated": teacher_result is not None,
        "harness": harness,
        "teacher_generation_used": teacher_result is not None,
        "teacher_spec_type": teacher_call.metadata.get("spec_type") if teacher_call else None,
        "teacher_ngram_configured": bool(teacher_call and teacher_call.metadata.get("spec_type") == "ngram-simple"),
        "teacher_speculation_speedup_measured": False,
        "teacher_backend_metadata": teacher_call.metadata if teacher_call else None,
        "small_metrics": small_result["metrics"],
        "teacher_metrics": teacher_result["metrics"] if teacher_result else None,
        "trace": final_row["trace"],
    }


def _make_runtime_generator(generator: Callable[..., Any], task: Mapping[str, Any]):
    """Wrap an injected backend while forwarding token counters and metadata."""
    initial_metadata = getattr(generator, "metadata", {}) or {}
    metadata = dict(initial_metadata) if isinstance(initial_metadata, Mapping) else {}

    def invoke(messages: list[dict[str, str]]) -> str:
        value = _call_generator(generator, messages, task)
        if isinstance(value, tuple) and len(value) == 2:
            text, call_metadata = value
            if isinstance(call_metadata, Mapping):
                invoke.policy_tokens += int(call_metadata.get("completion_tokens", 0) or 0)
                invoke.prompt_tokens += int(call_metadata.get("prompt_tokens", 0) or 0)
                metadata.update(call_metadata.get("backend", {}) if isinstance(call_metadata.get("backend"), Mapping) else {})
                if call_metadata.get("spec_type"):
                    metadata["spec_type"] = call_metadata["spec_type"]
        else:
            text = value
        if not isinstance(text, str):
            raise TypeError("runtime generator must return tool-call text")
        return text

    invoke.policy_tokens = 0
    invoke.prompt_tokens = 0
    invoke.metadata = metadata
    return invoke


def run_mixed_pipeline(
    tasks: Sequence[Mapping[str, Any]], *,
    corpus: RetrievalSubsystem,
    small_generate: Callable[..., Any],
    teacher_ngram_generate: Callable[..., Any],
    factual_retrieval_method: str = "dense",
    k: int = 2,
    max_turns: int = 8,
) -> dict[str, Any]:
    """Run factual retrieval and harness-verified repair orchestration only."""
    rows = []
    for task in tasks:
        kind = task.get("kind")
        if kind == "factual":
            picked, lookup_ms = corpus.retrieve(task["question"], method=factual_retrieval_method, k=k)
            context = corpus.context_block(picked)
            response, metadata = _chat_call(small_generate, task, context)
            ids = [str(doc["doc_id"]) for doc in picked]
            rows.append({
                "task": task["name"], "kind": kind, "route": "small_dense_rag",
                "correct": grade_answer(dict(task), response),
                "citation_ok": grade_citation(dict(task), response, ids),
                "retrieved": ids, "retrieval_latency_ms": lookup_ms,
                "small_latency_s": metadata.get("wall_seconds"),
                "small_tokens": metadata.get("completion_tokens"),
            })
        elif kind == "repair":
            rows.append(run_repair_pipeline(
                task, small_generate=small_generate,
                teacher_ngram_generate=teacher_ngram_generate,
                max_turns=max_turns,
            ))
        else:
            raise ValueError(f"unsupported mixed-pipeline task kind: {kind!r}")
    factual = [row for row in rows if row["kind"] == "factual"]
    repairs = [row for row in rows if row["kind"] == "repair"]
    return {
        "distribution": {
            "tasks": len(rows), "factual": len(factual), "repair": len(repairs),
        },
        "summary": {
            "factual_accuracy": sum(row["correct"] for row in factual) / max(len(factual), 1),
            "factual_citation_rate": sum(row["citation_ok"] for row in factual) / max(len(factual), 1),
            "repair_green_rate": sum(bool(row["green"]) for row in repairs) / max(len(repairs), 1),
            "teacher_escalation_rate": sum(bool(row["teacher_escalated"]) for row in repairs) / max(len(repairs), 1),
            "n_eval": len(rows),
        },
        "tasks": rows,
    }


def _task_value(task: Any, name: str, *fallbacks: str, default: Any = "") -> Any:
    if isinstance(task, Mapping):
        for key in (name, *fallbacks):
            if key in task:
                return task[key]
    else:
        for key in (name, *fallbacks):
            if hasattr(task, key):
                return getattr(task, key)
    return default


def _task_identity(task: Any, *, include_name: bool = True) -> dict[str, Any]:
    identity = {
        "question": str(_task_value(task, "question", "goal")),
        "initial": dict(_task_value(task, "initial", default={}) or {}),
        "target": str(_task_value(task, "target")),
        "expected_fix": str(_task_value(task, "expected_fix")),
        "test_count": int(_task_value(task, "test_count", default=0)),
        "test_success": str(_task_value(task, "test_success")),
        "family": str(_task_value(task, "family")),
    }
    if include_name:
        identity["name"] = str(_task_value(task, "name"))
    return identity


def _task_digest(task: Any, *, include_name: bool = True) -> str:
    canonical = json.dumps(_task_identity(task, include_name=include_name), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verified_final_green(trace: Sequence[Mapping[str, Any]]) -> bool:
    from chowder.runtime_eval import _is_green

    green = False
    final_after_green = False
    for row in trace:
        role = row.get("role")
        if role == "tool":
            if row.get("tool") == "write_file" and row.get("workspace_changed"):
                green = False
            elif row.get("tool") == "run_tests":
                green = _is_green(str(row.get("observation", "")))
        elif role == "final_report":
            if green:
                final_after_green = True
            else:
                final_after_green = False
        elif role == "budget_exhausted":
            final_after_green = False
    return green and final_after_green


def _messages_from_trace(trace: Sequence[Mapping[str, Any]], expected_question: str) -> list[dict[str, str]]:
    final_reports = [row for row in trace if row.get("role") == "final_report"]
    if not final_reports:
        raise ValueError("repair trajectory is missing its final report")
    # The final report is recorded as its own `final_report` row without
    # prompt_messages; the assistant row that actually received the final-turn
    # prompt is the last assistant row carrying a list of prompt_messages.
    turns = [row for row in trace if row.get("role") == "assistant" and isinstance(row.get("prompt_messages"), list)]
    if not turns or not isinstance(turns[-1].get("prompt_messages"), list):
        raise ValueError("repair trajectory is missing the actual final prompt history")
    messages = [dict(message) for message in turns[-1]["prompt_messages"]]
    from chowder.runtime_eval import parse_tool_call
    if not any(
        message.get("role") == "user" and expected_question in str(message.get("content", ""))
        for message in messages
    ):
        raise ValueError("runtime trace prompt does not match the declared evolve task")
    if any(
        message.get("role") == "assistant" and parse_tool_call(str(message.get("content", ""))) is None
        for message in messages
    ):
        raise ValueError("runtime repair transcript contains an ungrounded early report")
    # The supervised transcript is rendered tool-free. Chat templates do not
    # agree on a `tool` branch (Qwen2.5 wraps it in <tool_response>, several
    # small models have no branch at all), so a transcript carrying tool roles
    # cannot be tokenized portably -- and the Kaggle QAT lane refuses any role
    # outside system/user/assistant. Observations become user turns; the
    # faithful tool roles and the exact observation bytes stay in `trace`.
    for message in messages:
        if message.get("role") == "tool":
            message["role"] = "user"
    messages.append({"role": "assistant", "content": str(final_reports[-1].get("text", ""))})
    return messages


def repair_trajectory_row(task: Any, trace: Sequence[Mapping[str, Any]], split: str, *, harness: str = "state_aware") -> dict[str, Any]:
    if split != "evolve":
        raise ValueError("repair trajectories may only be exported from the evolve split")
    if harness not in {"state_aware", "state_aware+recovery"}:
        raise ValueError("repair trajectories require a selected state-aware harness")
    if not _verified_final_green(trace):
        raise ValueError("refusing trajectory without green tests followed by a final report")
    identity = _task_identity(task)
    task_name = identity["name"]
    task_digest = _task_digest(task)
    return {
        "id": f"batch010-{task_name}-{task_digest[:12]}",
        "goal": identity["question"],
        "initial": identity["initial"],
        "target": identity["target"],
        "split": "evolve", "task_name": task_name,
        "harness": harness,
        "task_sha256": task_digest,
        "task_content_sha256": _task_digest(task, include_name=False),
        "trace": [dict(row) for row in trace],
        "messages": _messages_from_trace(trace, identity["question"]),
        "green_verified": True,
    }


def build_batch010_dataset(
    trajectory_records: Sequence[Mapping[str, Any]],
    *,
    evolve_tasks: Sequence[Mapping[str, Any]],
    heldout_tasks: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> int:
    """Write only green-verified evolve trajectories; reject any held-out identity."""
    held_names = {str(_task_value(task, "name")) for task in heldout_tasks}
    evolve_by_name = {str(_task_value(task, "name")): task for task in evolve_tasks}
    if len(evolve_by_name) != len(evolve_tasks) or len(held_names) != len(heldout_tasks):
        raise ValueError("evolve and held-out task names must be unique")
    if held_names & set(evolve_by_name):
        raise ValueError("evolve and held-out task names overlap")
    held_content_hashes = {_task_digest(task, include_name=False) for task in heldout_tasks}
    rows = []
    seen: set[str] = set()
    for record in trajectory_records:
        if record.get("split") != "evolve":
            raise ValueError("held-out or unlabeled trajectory rejected")
        name = str(record.get("task_name", ""))
        if name in held_names or name not in evolve_by_name:
            raise ValueError(f"trajectory task is not on evolve split: {name}")
        task = evolve_by_name[name]
        digest = str(record.get("task_sha256", ""))
        expected = _task_digest(task)
        content_digest = _task_digest(task, include_name=False)
        if record.get("harness") not in {"state_aware", "state_aware+recovery"}:
            raise ValueError(f"trajectory was not generated under a selected state-aware harness: {name}")
        if content_digest in held_content_hashes:
            raise ValueError(f"evolve trajectory duplicates held-out task content: {name}")
        if record.get("heldout_task_names") or record.get("heldout_task_sha256"):
            raise ValueError(f"trajectory contains held-out identity metadata: {name}")
        if not digest or digest != expected or digest in seen:
            raise ValueError(f"missing, stale, or duplicate trajectory/task hash: {name}")
        if record.get("task_content_sha256") != content_digest:
            raise ValueError(f"trajectory content hash mismatch: {name}")
        trace = record.get("trace")
        if not isinstance(trace, list):
            raise ValueError(f"trajectory is missing the verified runtime trace: {name}")
        rebuilt = repair_trajectory_row(task, trace, split="evolve", harness=str(record["harness"]))
        if not record.get("green_verified") or record.get("messages") != rebuilt["messages"]:
            raise ValueError(f"trajectory is not a faithful green-verified trace: {name}")
        seen.add(digest)
        rows.append(rebuilt)
    if not rows:
        raise ValueError("batch-010 training set requires at least one verified evolve trajectory")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return len(rows)
