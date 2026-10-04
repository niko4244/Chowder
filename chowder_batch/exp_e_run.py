"""Experiment E driver: baselines, retrieval variants, routing, and evaluation.

All model calls go through llama.cpp servers (uniform protocol, greedy
temperature 0). Phases:

* Phase 1 -- baselines on the eval split: A) teacher alone, B) small alone,
  C) small + retrieval (each retrieval variant from exp_e_corpus).
* Phase 4 -- router calibration on dev tasks, then adaptive routing on eval
  tasks, with an always-large control and harness-verified repair escalation.
* Phase 5 -- the comparative metrics roll up here: accuracy, invalid tool
  actions, unsupported claims, retrieval correctness, cost, latency, and
  large-model invocation frequency.

Repair tasks always run through the chowder runtime harness; a claimed fix is
accepted only after a green test observation (never on model self-report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chowder_batch"))

from exp_e_confidence import (  # noqa: E402
    ARM_ADMISSION_VERSION,
    DEFAULT_MARGIN_SHIFT_TOLERANCE,
    DEFAULT_MAX_GREEN_LOSS_FRACTION,
    aggregate_margin,
    calibrate_margin_threshold,
    green_loss_fails_closed,
    heldout_transfer_gate,
    margin_shift_fails_closed,
    quant_route_allowed,
    quantized_arm_admission,
    small_route_allowed,
    validate_quantized_margin_shift,
)
from exp_e_corpus import RETRIEVAL_METHODS, RetrievalSubsystem, build_corpus, load_verified_corpus  # noqa: E402
from exp_f_ptq_margin import load_quantized_evidence  # noqa: E402

TEACHER_PORT = 18081
SPARK_PORT = 18082


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def chat(
    port: int, messages: list[dict], *, max_tokens: int,
    request_logprobs: bool = False,
) -> tuple[str, dict]:
    """Chat completion; returns (visible_output, meta).

    Both local models emit long chain-of-thought in ``reasoning_content`` and
    the final answer in ``content``. Grading uses the *full visible output*
    (reasoning + content) so an answer buried in reasoning still counts;
    empty-content truncations are recorded as truncation, not as correctness.
    """
    t0 = time.perf_counter()
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    logprob_error = None
    if request_logprobs and max_tokens < 1:
        raise ValueError("max_tokens must be positive when requesting confidence logprobs")
    try:
        data = _post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {**payload, **({"logprobs": True, "top_logprobs": 5} if request_logprobs else {})},
        )
    except urllib.error.HTTPError as exc:
        if not request_logprobs:
            raise
        logprob_error = f"HTTP {exc.code}: {exc.reason}"
        data = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    wall = time.perf_counter() - t0
    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    visible = (content or reasoning) if not content.strip() else reasoning + "\n" + content
    margin = aggregate_margin(data) if request_logprobs and logprob_error is None else None
    if request_logprobs and margin is None and logprob_error is None:
        logprob_error = "response omitted usable selected-token logprobs"
    predicted_ms = timings.get("predicted_ms") or 0
    token_rate = usage.get("completion_tokens", 0) / (predicted_ms / 1000) if predicted_ms > 0 else None
    return (
        visible,
        {
            "wall_seconds": round(wall, 3),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "content_empty": not content.strip(),
            "finish_reason": choice.get("finish_reason"),
            "tokens_per_second": round(token_rate, 2) if token_rate is not None else None,
            "logprobs_supported": margin is not None if request_logprobs else None,
            "logprob_margin": margin,
            "logprob_error": logprob_error,
        },
    )


def run_repair_task(task: dict, *, spark_port: int, teacher_port: int, max_turns: int = 8) -> dict:
    """Drive the runtime repair harness with Spark; escalate to the teacher on failure.

    Returns harness-verified metrics; the model's claim is never trusted.
    """
    from chowder.runtime_eval import RuntimeTask, run_live_benchmark

    def make_generate(port: int):
        def generate(messages: list[dict[str, str]]) -> str:
            # 512: the models reason (reasoning_content) before emitting the
            # tool call, so a small cap produces empty actions and premature
            # reports -- a budget artifact, not a capability result.
            content, _meta = chat(port, messages, max_tokens=512)
            return content

        return generate

    # The llama.cpp chat endpoint does not inject a tool protocol, so the
    # exact tool-call syntax is stated in the goal (protocol only -- no fix,
    # no filename beyond the task's own target).
    goal = task["question"] + (
        " Every action must be exactly one tool call in this format: "
        "<tool_call>tool_name<arg_key>argument</arg_key><arg_value>value</arg_value></tool_call>. "
        "Available tools: read_file (path), write_file (path, content), run_tests (). "
        "After a green run_tests observation, reply with a plain sentence and no tool call."
    )
    rt = RuntimeTask(
        name=task["name"], goal=goal, initial=dict(task["initial"]),
        target=task["target"], expected_fix=task["expected_fix"], test_count=task["test_count"],
        test_success=task["test_success"],
    )
    spark_result = run_live_benchmark(
        make_generate(spark_port), max_turns=max_turns, harness="state_aware", tasks=(rt,), split="exp_e"
    )
    row = spark_result["tasks"][0]
    if row["green_seen"]:
        return {
            "green": True, "escalated": False, "invalid_reads": row["nonexistent_reads"],
            "premature": row["premature_completion"], "repeated": row["repeated_actions"],
            "cost": row["execution_cost"], "tokens_spark": spark_result["metrics"]["policy_tokens"],
            "tokens_teacher": 0,
        }
    # Escalation: one teacher attempt on a fresh workspace copy.
    teacher_result = run_live_benchmark(
        make_generate(teacher_port), max_turns=max_turns, harness="state_aware", tasks=(rt,), split="exp_e"
    )
    trow = teacher_result["tasks"][0]
    return {
        "green": trow["green_seen"], "escalated": True,
        "invalid_reads": row["nonexistent_reads"] + trow["nonexistent_reads"],
        "premature": row["premature_completion"] or trow["premature_completion"],
        "repeated": row["repeated_actions"] + trow["repeated_actions"],
        "cost": row["execution_cost"] + trow["execution_cost"],
        "tokens_spark": spark_result["metrics"]["policy_tokens"],
        "tokens_teacher": teacher_result["metrics"]["policy_tokens"],
    }


def grade_answer(task: dict, response: str) -> bool:
    """Ground-truth grading (exact-match on normalized answer)."""
    normalized = response.lower().replace(",", "").strip()
    answer = str(task["answer"]).lower().replace(",", "").strip()
    return answer in normalized


def grade_citation(task: dict, response: str, retrieved_doc_ids: list[str]) -> bool:
    """Require the answer to cite the gold source and that source to be retrieved."""
    gold_doc_id = str(task["doc_id"])
    cited = f"[source: {gold_doc_id}]" in response
    return cited and gold_doc_id in retrieved_doc_ids


def build_messages(task: dict, context_block: str) -> list[dict]:
    if context_block:
        return [{"role": "user", "content": context_block + "\n\nQuestion: " + task["question"]}]
    return [{"role": "user", "content": task["question"]}]


def quantized_guard_state(
    *,
    quantized_margin_shift: float | None,
    max_quantized_margin_shift: float,
    quantized_green_loss_fraction: float | None,
    max_quantized_green_loss_fraction: float,
) -> dict[str, Any]:
    """Build the Phase-4 quantized-lane guard fields for a calibration record.

    The lane is active when *either* measurement is supplied, so providing one
    and omitting the other leaves the active lane blocked (``True`` in the
    corresponding ``*_fails_closed`` field) instead of unguarded; supplying
    neither leaves the pure-BF16 lane untouched (both fields inactive).
    """
    active = quantized_margin_shift is not None or quantized_green_loss_fraction is not None
    return {
        "quantized_guard_active": active,
        "quantized_margin_shift": quantized_margin_shift,
        "max_quantized_margin_shift": max_quantized_margin_shift if active else None,
        "quantized_margin_shift_fails_closed": margin_shift_fails_closed(
            quantized_margin_shift, max_quantized_margin_shift if active else None
        ),
        "quantized_green_loss_fraction": quantized_green_loss_fraction,
        "max_quantized_green_loss_fraction": max_quantized_green_loss_fraction if active else None,
        "quantized_green_loss_fails_closed": green_loss_fails_closed(
            quantized_green_loss_fraction, max_quantized_green_loss_fraction if active else None
        ),
    }


def quantized_route_decision(
    margin: float | None,
    calibration: Mapping[str, Any],
    *,
    guard_state: Mapping[str, Any],
    arm_admission: Mapping[str, Any] | None = None,
    heldout_gate: Mapping[str, Any] | None = None,
) -> bool:
    """The Phase-4 small-route decision for one query on a serving arm.

    An inactive lane (the BF16 arm) keeps the original behaviour: only the
    calibrated threshold governs. An active quantized lane must clear the
    per-precision tag, the held-out transfer gate measured for its threshold,
    the margin-shift bound and the green-retention bound through
    ``quant_route_allowed`` -- and, when supplied, the aggregate admission
    artifact, so routing can never diverge from the recorded verdicts.
    """
    route_small = small_route_allowed(
        margin,
        calibration,
        quantized_margin_shift=guard_state["quantized_margin_shift"],
        max_quantized_margin_shift=calibration.get("max_quantized_margin_shift"),
    )
    if not guard_state["quantized_guard_active"]:
        return route_small
    if arm_admission is not None and not bool(arm_admission.get("admitted")):
        return False
    return route_small and quant_route_allowed(
        margin,
        calibration,
        quantized_margin_shift=guard_state["quantized_margin_shift"],
        max_quantized_margin_shift=calibration.get("max_quantized_margin_shift"),
        green_loss_fraction=guard_state["quantized_green_loss_fraction"],
        max_green_loss_fraction=calibration.get("max_quantized_green_loss_fraction"),
        heldout_gate=heldout_gate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int,default=256)
    parser.add_argument("--limit", type=int, default=0, help="cap tasks per split for smoke runs")
    parser.add_argument("--corpus", help="provenance-verified corpus JSON; without it, use the historical 7-doc seed")
    parser.add_argument(
        "--quantized-margin-shift", type=float, default=None,
        help="measured mean quantized-margin minus BF16-margin shift (from Experiment F); "
             "omitting it while serving a quantized arm fails routing closed",
    )
    parser.add_argument(
        "--max-quantized-margin-shift", type=float, default=DEFAULT_MARGIN_SHIFT_TOLERANCE,
        help=f"tolerance for the quantized margin shift (default {DEFAULT_MARGIN_SHIFT_TOLERANCE})",
    )
    parser.add_argument(
        "--quantized-green-loss-fraction", type=float, default=None,
        help="measured fraction of the reference (BF16) arm's green tasks the quantized arm "
             "loses (from Experiment F); omitting it while serving a quantized arm fails routing closed",
    )
    parser.add_argument(
        "--max-quantized-green-loss-fraction", type=float, default=DEFAULT_MAX_GREEN_LOSS_FRACTION,
        help="tolerance for lost reference greens (default "
             f"{DEFAULT_MAX_GREEN_LOSS_FRACTION}: no reference green may be lost)",
    )
    parser.add_argument(
        "--quantized-heldout-rows",
        help="JSONL of held-out per-task margin rows (task, margin, correct) for the "
             "serving precision; without them a quantized arm cannot serve the small route",
    )
    parser.add_argument(
        "--quantized-precision-arm",
        help="name of the quantized serving arm the held-out rows were measured on "
             "(default: int8_smoothquant, or the arm named by --quantized-evidence)",
    )
    parser.add_argument(
        "--quantized-evidence",
        help="measured Experiment F report JSON; the serving precision, margin shift, green "
             "retention, and held-out rows are derived from its per-task measurements instead "
             "of being passed as scalars",
    )
    args = parser.parse_args()

    # A measured Experiment F report is the sanctioned source for a quantized
    # arm's shift and green retention: derive them from its per-task data
    # instead of accepting hand-typed scalars, and refuse to mix sources.
    if args.quantized_evidence:
        conflicting = [
            flag for flag, value in (
                ("--quantized-margin-shift", args.quantized_margin_shift),
                ("--quantized-green-loss-fraction", args.quantized_green_loss_fraction),
                ("--quantized-heldout-rows", args.quantized_heldout_rows),
                ("--quantized-precision-arm", args.quantized_precision_arm),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(
                "--quantized-evidence derives those measurements from the report; remove: "
                + ", ".join(conflicting)
            )
        evidence = load_quantized_evidence(
            args.quantized_evidence,
            max_quantized_margin_shift=args.max_quantized_margin_shift,
            max_green_loss_fraction=args.max_quantized_green_loss_fraction,
        )
        quantized_margin_shift = evidence["margin_shift"]
        quantized_green_loss_fraction = evidence["green_loss_fraction"]
        quantized_precision_arm = evidence["precision_arm"]
        quantized_heldout_rows = evidence["heldout_rows"]
    else:
        evidence = None
        quantized_margin_shift = args.quantized_margin_shift
        quantized_green_loss_fraction = args.quantized_green_loss_fraction
        quantized_precision_arm = args.quantized_precision_arm or "int8_smoothquant"
        quantized_heldout_rows = (
            [
                json.loads(line)
                for line in Path(args.quantized_heldout_rows).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if args.quantized_heldout_rows
            else []
        )

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))["tasks"]
    corpus_docs = load_verified_corpus(args.corpus) if args.corpus else build_corpus()
    corpus = RetrievalSubsystem(corpus_docs)
    eval_tasks = [t for t in tasks if t["split"] == "eval"]
    dev_tasks = [t for t in tasks if t["split"] == "dev"]
    if args.limit:
        eval_tasks = eval_tasks[: args.limit]
        dev_tasks = dev_tasks[: args.limit]
    print(f"{len(eval_tasks)} eval / {len(dev_tasks)} dev tasks", flush=True)

    results: dict[str, dict] = {
        "phase1": {}, "phase4": {},
        "corpus_manifest": {
            "n_documents": len(corpus_docs),
            "provenance_verified": bool(args.corpus),
            "source": "verified_corpus_json" if args.corpus else "historical_seed_unverified",
        },
    }

    # ---- Phase 1C variants: small alone and small + each retrieval variant.
    for method in RETRIEVAL_METHODS:
        rows = []
        for task in eval_tasks:
            if task["kind"] == "repair":
                continue  # repairs measured in phase 4 (harness-verified)
            if method == "none":
                context = ""
                picked, lookup_ms = [], 0.0
            else:
                picked, lookup_ms = corpus.retrieve(task["question"], method=method, k=2)
                context = corpus.context_block(picked)
            content, meta = chat(SPARK_PORT, build_messages(task, context), max_tokens=args.max_tokens)
            correct = grade_answer(task, content)
            row = {
                "task": task["name"], "kind": task["kind"], "difficulty": task["difficulty"],
                "correct": correct, "latency_s": meta["wall_seconds"], "tokens": meta["completion_tokens"],
                "truncated": meta["finish_reason"] == "length",
                "retrieval_latency_ms": round(lookup_ms, 2), "retrieved": [d["doc_id"] for d in picked],
                "response": content[-200:],
            }
            if task["kind"] == "factual":
                row["citation_ok"] = grade_citation(task, content, [d["doc_id"] for d in picked])
            rows.append(row)
        by_kind = {}
        for kind in ("gsm8k", "factual"):
            subset = [r for r in rows if r["kind"] == kind]
            if subset:
                by_kind[kind] = {
                    "accuracy": sum(r["correct"] for r in subset) / len(subset),
                    "latency_s": round(sum(r["latency_s"] for r in subset) / len(subset), 2),
                    "tokens": round(sum(r["tokens"] for r in subset) / len(subset), 1),
                    "retrieval_latency_ms": round(sum(r["retrieval_latency_ms"] for r in subset) / len(subset), 2),
                    "n": len(subset),
                }
        factual_rows = [r for r in rows if r["kind"] == "factual"]
        if factual_rows and method != "none":
            by_kind["factual"]["citation_rate"] = sum(r["citation_ok"] for r in factual_rows) / len(factual_rows)
        results["phase1"][method] = {"tasks": rows, "summary": by_kind}
        print(f"phase1[{method}]: " + json.dumps(by_kind), flush=True)

    # ---- Phase 1A: teacher alone on eval tasks (the quality reference).
    rows = []
    for task in eval_tasks:
        if task["kind"] == "repair":
            continue
        content, meta = chat(TEACHER_PORT, build_messages(task, ""), max_tokens=args.max_tokens)
        rows.append({
            "task": task["name"], "kind": task["kind"], "difficulty": task["difficulty"],
            "correct": grade_answer(task, content), "latency_s": meta["wall_seconds"],
            "tokens": meta["completion_tokens"], "response": content[:200],
        })
    by_kind = {}
    for kind in ("gsm8k", "factual"):
        subset = [r for r in rows if r["kind"] == kind]
        if subset:
            by_kind[kind] = {
                "accuracy": sum(r["correct"] for r in subset) / len(subset),
                "latency_s": round(sum(r["latency_s"] for r in subset) / len(subset), 2),
                "tokens": round(sum(r["tokens"] for r in subset) / len(subset), 1),
                "n": len(subset),
            }
    results["phase1"]["teacher_alone"] = {"tasks": rows, "summary": by_kind}
    print("phase1[teacher_alone]: " + json.dumps(by_kind), flush=True)

    # ---- Phase 4: chosen-token logprob margins; calibration sees dev only.
    # Missing or malformed response logprobs remain None and fail closed to the teacher.
    dev_rows = []
    for task in dev_tasks:
        if task["kind"] == "repair":
            continue
        content, meta = chat(
            SPARK_PORT, build_messages(task, ""), max_tokens=args.max_tokens,
            request_logprobs=True,
        )
        dev_rows.append({
            "task": task["name"], "kind": task["kind"], "difficulty": task["difficulty"],
            "correct": grade_answer(task, content), "latency_s": meta["wall_seconds"],
            "tokens": meta["completion_tokens"], "response": content[:200],
            "margin": meta["logprob_margin"], "logprob_error": meta["logprob_error"],
        })
    calibration = calibrate_margin_threshold(
        [{"task": row["task"], "margin": row["margin"], "correct": row["correct"]} for row in dev_rows],
        min_precision=0.80,
        min_samples=4,
    )
    calibration["dev_accuracy"] = sum(row["correct"] for row in dev_rows) / max(len(dev_rows), 1)
    calibration["dev_tasks"] = [row["task"] for row in dev_rows]
    calibration["dev_logprobs_available"] = sum(row["margin"] is not None for row in dev_rows)
    # Quantized-arm routing guards. The guards activate only when a quantized
    # arm is actually served (either measurement provided); the pure-BF16 lane
    # is unaffected. Supplying one measurement and omitting the other fails
    # closed, because an active quantized lane with an unmeasured guard input
    # must never route.
    guard_state = quantized_guard_state(
        quantized_margin_shift=quantized_margin_shift,
        max_quantized_margin_shift=args.max_quantized_margin_shift,
        quantized_green_loss_fraction=quantized_green_loss_fraction,
        max_quantized_green_loss_fraction=args.max_quantized_green_loss_fraction,
    )
    calibration.update(guard_state)
    # Held-out transfer gate. A quantized arm must clear the dev-set threshold
    # on tasks the calibration never saw, measured on that same arm; otherwise
    # it escalates every query to the teacher. The BF16 lane is unaffected.
    calibration["precision_arm"] = (
        quantized_precision_arm if calibration["quantized_guard_active"] else "bf16"
    )
    calibration["heldout_transfer_gate"] = heldout_transfer_gate(
        calibration,
        quantized_heldout_rows if calibration["quantized_guard_active"] else [],
        min_precision=0.80,
        min_samples=4,
    ) if calibration["quantized_guard_active"] else {
        "status": "not_applicable_bf16_lane",
        "precision_arm": "bf16",
        "n_heldout": 0,
    }
    # One artifact aggregating every guard: admission is granted only when all
    # of them pass, so no single passing bound can license the arm.
    if calibration["quantized_guard_active"]:
        arm_admission = quantized_arm_admission(
            calibration,
            quantized_margin_shift=quantized_margin_shift,
            max_quantized_margin_shift=calibration["max_quantized_margin_shift"],
            green_loss_fraction=quantized_green_loss_fraction,
            max_green_loss_fraction=calibration["max_quantized_green_loss_fraction"],
            heldout_gate=calibration["heldout_transfer_gate"],
        )
    else:
        arm_admission = {
            "artifact": ARM_ADMISSION_VERSION,
            "status": "not_applicable_bf16_lane",
            "precision_arm": "bf16",
        }
    results["phase4"]["arm_admission"] = arm_admission
    if evidence is not None:
        # The report-side admission was measured on the report's own suite (its
        # calibration and gate), so it is recorded alongside the serving-time
        # artifact, never merged with it.
        results["phase4"]["quantized_evidence"] = {
            "source": evidence["source"],
            "precision_arm": evidence["precision_arm"],
            "margin_shift": evidence["margin_shift"],
            "green_retention": evidence["green_retention"],
            "n_heldout_rows": len(evidence["heldout_rows"]),
            "report_heldout_gate": evidence["report_heldout_gate"],
            "admission": evidence["admission"],
            "admission_verification": evidence["admission_verification"],
        }
    results["phase4"]["calibration"] = calibration
    results["phase4"]["dev_rows"] = dev_rows
    print("phase4 calibration:", json.dumps(calibration), flush=True)

    # ---- Phase 4: adaptive routing on eval tasks.
    for task in eval_tasks:
        if task["kind"] == "repair":
            outcome = run_repair_task(task, spark_port=SPARK_PORT, teacher_port=TEACHER_PORT)
            results["phase4"].setdefault("repair", []).append({"task": task["name"], **outcome})
            continue
        content, spark_meta = chat(
            SPARK_PORT, build_messages(task, ""), max_tokens=args.max_tokens,
            request_logprobs=True,
        )
        margin = spark_meta["logprob_margin"]
        route_small = quantized_route_decision(
            margin,
            calibration,
            guard_state=guard_state,
            arm_admission=arm_admission,
            heldout_gate=calibration["heldout_transfer_gate"],
        )
        spark_row = {
            "task": task["name"], "kind": task["kind"], "difficulty": task["difficulty"],
            "spark_correct": grade_answer(task, content), "spark_margin": margin,
            "logprobs_supported": spark_meta["logprobs_supported"],
            "used_teacher": not route_small, "spark_latency_s": spark_meta["wall_seconds"],
            "spark_tokens": spark_meta["completion_tokens"],
            "logprob_error": spark_meta["logprob_error"],
            "quantized_margin_shift": quantized_margin_shift,
            "quantized_green_loss_fraction": quantized_green_loss_fraction,
            "heldout_transfer_gate_status": calibration["heldout_transfer_gate"]["status"],
        }
        if route_small:
            results["phase4"].setdefault("routed_small", []).append({
                **spark_row, "correct": spark_row["spark_correct"], "teacher_latency_s": 0.0,
                "teacher_tokens": 0,
            })
        else:
            teacher_content, teacher_meta = chat(
                TEACHER_PORT, build_messages(task, ""), max_tokens=args.max_tokens
            )
            results["phase4"].setdefault("routed_large", []).append({
                **spark_row, "correct": grade_answer(task, teacher_content),
                "teacher_latency_s": teacher_meta["wall_seconds"],
                "teacher_tokens": teacher_meta["completion_tokens"],
            })
    phase4_rows = [*results["phase4"].get("routed_small", []), *results["phase4"].get("routed_large", [])]
    results["phase4"]["routing_summary"] = {
        "n_eval": len(phase4_rows),
        "accuracy": sum(row["correct"] for row in phase4_rows) / max(len(phase4_rows), 1),
        "nonrepair_eval_tasks": len(phase4_rows),
        "repair_eval_tasks": len(results["phase4"].get("repair", [])),
        "teacher_invocations": sum(bool(row["used_teacher"]) for row in phase4_rows),
        "teacher_invocation_rate": sum(bool(row["used_teacher"]) for row in phase4_rows) / max(len(phase4_rows), 1),
        "small_route_count": len(results["phase4"].get("routed_small", [])),
        "logprob_unavailable_count": sum(row["spark_margin"] is None for row in phase4_rows),
        "teacher_control_accuracy": results["phase1"]["teacher_alone"]["summary"],
        "accuracy_by_kind": {
            kind: sum(row["correct"] for row in phase4_rows if row["kind"] == kind)
            / max(sum(row["kind"] == kind for row in phase4_rows), 1)
            for kind in sorted({row["kind"] for row in phase4_rows})
        },
        "teacher_accuracy_by_kind": {
            kind: summary["accuracy"]
            for kind, summary in results["phase1"]["teacher_alone"]["summary"].items()
        },
        "always_small_accuracy_by_kind": {
            kind: sum(row["correct"] for row in results["phase1"]["none"]["tasks"] if row["kind"] == kind)
            / max(sum(row["kind"] == kind for row in results["phase1"]["none"]["tasks"]), 1)
            for kind in sorted({row["kind"] for row in results["phase1"]["none"]["tasks"]})
        },
    }
    # Always-large control is measured independently above in phase1.teacher_alone.

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
