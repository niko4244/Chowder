"""Genuine hosted Qwen3.8-27B / Opus5.5 original-source GSM8K comparison.

Only the private official GSM8K public math items are sent externally. No
HumanEval-generated Python is run, no REAP model is loaded, and no frozen
Gen-0 selector or scoring source is modified. Record per-call exact model,
provider, finish, timings, reported usage and original source hash. The
historical REAP Q8/Q4 scored rows are immutable ORIGINAL Linux cohorts, not
new concurrent runs or a GPU throughput comparison.

Default is offline plan; --execute explicitly permits up to 16 paired hosted
requests under a small predicted cost cap. Raw texts are PRIVATE outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

from chowder_batch import reap_eval
from chowder_batch.reap_external_full_audit import _strict_json
from chowder.evaluators.scoring import score

MODELS = {"qwen38_27b": "qwen/qwen3.8-27b",
          "opus55": "anthropic/claude-opus-5.5"}
LOCAL_GATEWAY_MODELS = {"qwen38_27b": "groq-qwen3.8-27b",
                        "opus55": "claude-opus-5-5"}
SYSTEM = reap_eval.SYSTEM
MAX_ITEMS = 16
MAX_OUTPUT_TOKENS = 1024
SEED = 20261026
MAX_BUDGET_USD = 0.75
# Conservative per-request upper bounds at the currently observed OpenRouter
# rate for 1024 output tokens and the bounded original GSM8K prompt.
MAX_PROJECTED_CALL_USD = {"qwen38_27b": 0.003, "opus55": 0.025}
API_URI = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_GATEWAY_URI = "http://127.0.0.1:4000/v1/chat/completions"
SOURCE_N, SOURCE_SHA = reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"]
SOURCE_Q8 = Path(r"F:\reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4\reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-gsm8k.jsonl")
SOURCE_Q4 = Path(r"F:\reap-v2-kaggle-q4-full-complete-private-evidence-20261026\reap-v2-a2a6b83e345145839cc333561b2f3baf-gsm8k.jsonl")
ORIGINAL_SHA = {
    "reap_q8_original": "9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749",
    "reap_q4_original": "3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db",
}
ORIGINAL_SESSIONS = {"reap_q8_original": "95876063255e45159f57e29306e0e885",
                     "reap_q4_original": "a2a6b83e345145839cc333561b2f3baf"}


def selected_indices(n: int = MAX_ITEMS) -> tuple[int, ...]:
    if type(n) is not int or not 1 <= n <= MAX_ITEMS:
        raise ValueError("test sample must contain 1..16 original GSM8K problems")
    # No adaptation to historical success/failure or future API answers.
    return tuple(sorted(random.Random(SEED).sample(range(SOURCE_N), n)))


def original_items() -> list[dict]:
    items = reap_eval._load("gsm8k")
    if len(items) != SOURCE_N or reap_eval._digest(items) != SOURCE_SHA:
        raise ValueError("registered original GSM8K prompts/answers source mismatch")
    return items


def original_baselines(indices: tuple[int, ...]) -> dict[str, dict]:
    result = {}
    for name, path in (("reap_q8_original", SOURCE_Q8), ("reap_q4_original", SOURCE_Q4)):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("original REAP baseline missing or unbounded")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA[name]:
            raise ValueError("immutable original REAP source-row hash mismatch")
        lines = raw.splitlines()
        if len(lines) != SOURCE_N:
            raise ValueError("original REAP GSM8K cohort incomplete")
        by_id = {}
        for line in lines:
            row = json.loads(line)
            identifier = row.get("id")
            if (not isinstance(identifier, str) or identifier in by_id
                    or row.get("eval_session_id") != ORIGINAL_SESSIONS[name]
                    or type(row.get("correct")) is not bool
                    or type(row.get("terminated")) is not bool):
                raise ValueError("original REAP row/session integrity mismatch")
            by_id[identifier] = row
        if set(by_id) != {f"gsm8k-{i}" for i in range(SOURCE_N)}:
            raise ValueError("original population IDs differ from frozen source")
        result[name] = {f"gsm8k-{i}": by_id[f"gsm8k-{i}"]["correct"] for i in indices}
    return result


def request_body(model: str, prompt: str) -> dict:
    if (model not in tuple(MODELS.values()) + tuple(LOCAL_GATEWAY_MODELS.values())
            or not isinstance(prompt, str) or len(prompt) > 10000):
        raise ValueError("unknown comparison model or unbounded official source")
    return {"model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            # OpenRouter's explicit medium reasoning applies to both hosted
            # models, but is NOT equivalent to original REAP thinking-disabled.
            "reasoning": {"effort": "medium"}}


def parse_response(data: dict, expected_model: str, item: dict, elapsed: float) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or len(data["choices"]) != 1:
        raise ValueError("hosted model returned no unambiguous response")
    choice = data["choices"][0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("hosted model returned missing text")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise ValueError("hosted model lacks resolved model identity")
    canonical = ("qwen3.8-27b" if "qwen3.8-27b" in expected_model.lower()
                 else "opus-5-5")
    if canonical not in data["model"].lower():
        raise ValueError("requested hosted model was silently substituted")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length", "content_filter"):
        raise ValueError("unrecognized final response completion state")
    usage = data.get("usage")
    if (not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or not 0 <= usage["prompt_tokens"] <= 10000
            or not 0 <= usage["completion_tokens"] <= MAX_OUTPUT_TOKENS + 16):
        raise ValueError("missing finite source-bound token usage")
    content = message["content"]
    if len(content) > 100_000:
        raise ValueError("unbounded provider response")
    correct = (finish == "stop" and score(content, item["expected"], "final_number_match") == 1.0)
    cost = usage.get("cost")
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        raise ValueError("provider returned invalid dollar usage")
    return {"id": item["id"], "requested_model": expected_model,
            "resolved_model": data["model"], "provider": data.get("provider"),
            "api_response_id": data.get("id"),
            "finish_reason": finish, "terminated": finish == "stop",
            "correct": bool(correct), "seconds": round(elapsed, 3),
            "usage": usage, "content": content,
            "response_sha256": hashlib.sha256(content.encode()).hexdigest()}


def post_one(key: str, model: str, item: dict, *, provider: str = "openrouter") -> dict:
    if not isinstance(key, str) or len(key.strip()) < 10:
        raise ValueError("OpenRouter API key not configured")
    raw = json.dumps(request_body(model, item["prompt"]), allow_nan=False).encode()
    if provider not in {"openrouter", "litellm"}:
        raise ValueError("external comparison provider must be explicitly supported")
    endpoint = API_URI if provider == "openrouter" else LOCAL_GATEWAY_URI
    req = urllib.request.Request(
        endpoint, raw, headers={"Authorization": "Bearer " + key,
                               "Content-Type": "application/json",
                               "HTTP-Referer": "https://github.com/niko4244/Chowder",
                               "X-Title": "Chowder research paired GSM8K evaluation"}, method="POST")
    started = time.monotonic()
    # No automatic retry or alternative provider/model fallback. Failures
    # mean INCOMPLETE, never incorrect answers or fabricated latency/usage.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    with urllib.request.build_opener(NoRedirect).open(req, timeout=120) as response:
        data_raw = response.read(256_001)
        if len(data_raw) > 256_000:
            raise ValueError("hosted API returned oversized response")
        data = _strict_json(data_raw)
    return parse_response(data, model, item, time.monotonic() - started)


def _save_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def report(indices: tuple[int, ...], results: dict, baselines: dict) -> dict:
    names = ("reap_q8_original", "reap_q4_original", "qwen38_27b", "opus55")
    result = {"status": "INCOMPLETE_NOT_RANKABLE", "source_sha256": SOURCE_SHA,
              "selection_seed": SEED, "indices": list(indices), "n_requested": len(indices),
              "arms": {}, "original_windows_gate_eligible": False,
              "gen0_status": "BLOCKED", "release_approved": False}
    for arm in names:
        rows = baselines.get(arm, {}) if arm.startswith("reap") else results.get(arm, {})
        all_ids = {f"gsm8k-{i}" for i in indices}
        complete = set(rows) == all_ids
        correct = sum((val if isinstance(val, bool) else val.get("correct") is True)
                      for val in rows.values())
        result["arms"][arm] = {"n": len(rows), "correct": correct,
                                "accuracy": (correct / len(rows) if complete else None),
                                "complete": complete,
                                "qualification": ("ORIGINAL_LINUX_RESEARCH" if arm.startswith("reap")
                                                  else "HOSTED_API_PROVISIONAL")}
    if all(row["complete"] for row in result["arms"].values()):
        result["status"] = "COMPLETE_PAIRED_FIXED_SAMPLE_RESEARCH_ONLY"
    return result


def run(output: Path, *, count: int = MAX_ITEMS,
        cost_limit: float = MAX_BUDGET_USD, provider: str = "openrouter") -> dict:
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("choose a fresh exclusive absolute private output directory")
    private_out = output.resolve(strict=False)
    protected = (Path(__file__).resolve().parents[1], SOURCE_Q8.parent, SOURCE_Q4.parent)
    if any(private_out == path.resolve() or private_out.is_relative_to(path.resolve())
           for path in protected):
        raise ValueError("comparison responses must not be stored in Git or protected original sources")
    if type(cost_limit) not in (int, float) or not 0 < cost_limit <= MAX_BUDGET_USD:
        raise ValueError("API comparison cost authorization limit out of bounds")
    indices = selected_indices(count)
    source = original_items()
    baselines = original_baselines(indices)
    if provider not in {"openrouter", "litellm"}:
        raise ValueError("unsupported provider; no arbitrary URL/key forwarding")
    key = os.environ.get("OPENROUTER_API_KEY" if provider == "openrouter" else "LITELLM_API_KEY", "")
    if len(key) < 10:
        raise ValueError("configured comparison endpoint credential unavailable")
    # Published model token prices were read immediately before evaluation.
    # Bound *maximum* projected total below user's $0.75 ceiling even if
    # every response hits 1024 output tokens; fail closed on pricing uncertainty.
    projected = count * sum(MAX_PROJECTED_CALL_USD.values())
    if projected > cost_limit:
        raise ValueError("planned paired run exceeds specified conservative cost cap")
    output.mkdir(mode=0o700)
    model_ids = MODELS if provider == "openrouter" else LOCAL_GATEWAY_MODELS
    _save_new(output / "manifest.json", {"schema": "reap-v2-qwen38-opus55-math.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": SOURCE_SHA, "source_n": SOURCE_N,
              "sample_seed": SEED, "indices": list(indices),
              "max_tokens": MAX_OUTPUT_TOKENS, "models": model_ids, "provider": provider,
              "sample_population": "registered source, preregistered random 16 of 1319",
              "temperature": 0, "reasoning_effort": "medium",
              "projected_cost_upper_bound_usd": projected,
              "budget_usd": cost_limit,
              "original_reap_row_sha256": ORIGINAL_SHA,
              "gen0_status": "BLOCKED", "release_approved": False})
    gathered: dict[str, dict] = {"qwen38_27b": {}, "opus55": {}}
    accrued_estimate = 0.0
    error = None
    try:
        for i in indices:
            item = source[i]
            for arm, model in model_ids.items():
                allowance = MAX_PROJECTED_CALL_USD[arm]
                if accrued_estimate + allowance > cost_limit + 1e-9:
                    raise ValueError("conservative hosted API budget exhausted")
                row = post_one(key, model, item, provider=provider)
                accrued_estimate += allowance
                _save_new(output / f"{arm}-{i:04d}.json", row)
                gathered[arm][item["id"]] = row
                print(f"{arm}/{item['id']}: correct={row['correct']} "
                      f"finish={row['finish_reason']} tokens={row['usage']['completion_tokens']}", flush=True)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            OSError, ValueError, KeyError, TypeError) as exc:
        error = {"error_class": type(exc).__name__}
        if isinstance(exc, urllib.error.HTTPError):
            error["http_status"] = exc.code
        # No provider body, request headers, token, original response or
        # exception string in the public/logged output.
    finally:
        result = report(indices, gathered, baselines)
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["api_request_upper_charge_usd"] = round(accrued_estimate, 4)
        if error is not None:
            result["endpoint_error"] = error
        _save_new(output / "research_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--count", type=int, default=MAX_ITEMS)
    parser.add_argument("--provider", choices=("openrouter", "litellm"), default="openrouter")
    parser.add_argument("--max-cost-usd", type=float, default=MAX_BUDGET_USD)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "arms": (
            MODELS if args.provider == "openrouter" else LOCAL_GATEWAY_MODELS),
                          "source_sha256": SOURCE_SHA,
                          "sample_indices": selected_indices(args.count),
                          "estimated_cost_ceiling_usd": round(args.count * sum(MAX_PROJECTED_CALL_USD.values()), 4),
                          "release_approved": False}, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--execute requires --out at an unused private directory")
    result = run(args.out, count=args.count, cost_limit=args.max_cost_usd,
                 provider=args.provider)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"].startswith("COMPLETE_PAIRED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
