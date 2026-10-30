"""Context-budget admission for agentic benchmarks on a fixed-context server.

Why this module exists
----------------------
A real local Q4_K_M AutomationBench trial (2026-10-26, expanded context
``-c 32768 -np 1``) died for a structural reason, not a memory reason:

* it served 24 requests, growing the prompt 1,067 -> 32,639 tokens
  (~1,373 tokens per request);
* the 25th request needed a tool call it could not finish, so llama-server
  logged ``stop processing: n_tokens = 32767, truncated = 1``;
* a truncated tool call is invalid JSON, so the server answered HTTP 500
  ``Failed to parse tool call arguments as JSON ... column 89``;
* the harness retried the same conversation, whose prompt was unchanged and
  therefore still at the ceiling, so every retry failed at the same byte.

The retries can never succeed: the request that overflows only gets larger.
This module turns that into an arithmetic admission check so a launch is
refused *before* it burns an hour of GPU producing zero valid episodes.

Pure functions only: no HTTP, no subprocess, no model, no filesystem writes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence

# Measured on the 2026-10-26 Q4 trajectory. Used only when no trajectory is
# supplied; a supplied trajectory always wins.
FALLBACK_TOKENS_PER_REQUEST = 1373

# A tool call must be allowed to finish. Cutting it produces invalid JSON that
# a server rejects as HTTP 500 rather than a scorable step.
DEFAULT_GENERATION_RESERVE = 2048

PROMPT_RE = re.compile(
    r"task (?P<task>-?\d+) \| new prompt, n_ctx_slot = (?P<ctx>\d+), "
    r"n_keep = \d+, task\.n_tokens = (?P<tokens>\d+)"
)
TRUNCATED_RE = re.compile(r"n_tokens = (?P<used>\d+), truncated = (?P<trunc>\d+)")
TOOL_JSON_500_RE = re.compile(r"Failed to parse tool call arguments as JSON")


def trajectory_from_native_log(text: str) -> dict[str, object]:
    """Extract prompt sizes and truncation evidence from a llama-server log.

    The log is the primary source of truth: it records what the server was
    actually asked to do, independent of any harness summary.
    """
    prompts = [int(m.group("tokens")) for m in PROMPT_RE.finditer(text)]
    contexts = sorted({int(m.group("ctx")) for m in PROMPT_RE.finditer(text)})
    truncated = [
        (int(m.group("used")), int(m.group("trunc")))
        for m in TRUNCATED_RE.finditer(text)
    ]
    return {
        "prompt_tokens": prompts,
        "declared_contexts": contexts,
        "generations": len(truncated),
        "truncated_generations": sum(1 for _, flag in truncated if flag),
        "max_observed_tokens": max((used for used, _ in truncated), default=0),
        "tool_call_json_failures": len(TOOL_JSON_500_RE.findall(text)),
    }


def step_token_growth(prompts: Sequence[int]) -> dict[str, float]:
    """Per-request prompt growth.

    Only positive deltas are growth. A non-positive delta means the server
    reused or reclaimed a prefix (checkpoint / slot reuse), which is not
    evidence about how much a step costs.
    """
    if len(prompts) < 2:
        raise ValueError("need at least two prompts to measure growth")
    if any(p < 0 for p in prompts):
        raise ValueError("prompt token counts must be non-negative")
    deltas = [b - a for a, b in zip(prompts, prompts[1:])]
    growth = [d for d in deltas if d > 0]
    if not growth:
        # A conversation frozen at the ceiling reports the same prompt every
        # retry. That is a real observation, so report it rather than raising.
        ordered = sorted(prompts)
        mid = len(ordered) // 2
        median = (ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)
        return {
            "requests": len(prompts),
            "first_prompt": float(prompts[0]),
            "last_prompt": float(prompts[-1]),
            "net_growth": float(prompts[-1] - prompts[0]),
            "positive_steps": 0,
            "mean_growth": 0.0,
            "median_growth": float(median),
            "max_growth": 0.0,
            "reclaim_events": len(deltas),
        }
    ordered = sorted(growth)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)
    return {
        "requests": len(prompts),
        "first_prompt": float(prompts[0]),
        "last_prompt": float(prompts[-1]),
        "net_growth": float(prompts[-1] - prompts[0]),
        "positive_steps": len(growth),
        "mean_growth": sum(growth) / len(growth),
        "median_growth": float(median),
        "max_growth": float(max(growth)),
        "reclaim_events": len(deltas) - len(growth),
    }


def required_context(
    *, first_prompt: int, growth: float, max_steps: int, generation_reserve: int
) -> int:
    """Context a full episode needs so no step is truncated.

    ``max_steps`` counts model turns; the first turn costs ``first_prompt``
    and each later turn appends ``growth`` tokens of assistant output plus
    tool result.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if growth < 0:
        raise ValueError("growth must be non-negative")
    if generation_reserve < 0:
        raise ValueError("generation_reserve must be non-negative")
    turns_after_first = max(0, max_steps - 1)
    return int(math.ceil(first_prompt + growth * turns_after_first + generation_reserve))


def affordable_steps(
    *, first_prompt: int, growth: float, context: int, generation_reserve: int
) -> int:
    """How many model turns fit in ``context`` without truncation."""
    if context <= 0:
        raise ValueError("context must be positive")
    budget = context - generation_reserve - first_prompt
    if growth <= 0:
        # Constant-size conversation: only the reserve bounds it.
        return 10**6 if budget >= 0 else 0
    if budget < 0:
        return 0
    return int(math.floor(budget / growth)) + 1


def assess(
    *,
    observed: Mapping[str, object] | None,
    context: int,
    max_steps: int,
    generation_reserve: int = DEFAULT_GENERATION_RESERVE,
) -> dict[str, object]:
    """Decide whether an episode of ``max_steps`` can complete in ``context``.

    ``REFUSED`` is returned when the arithmetic shows the episode cannot
    finish: launching would spend the whole budget and yield zero episodes.
    """
    if context <= 0:
        raise ValueError("context must be positive")
    if generation_reserve >= context:
        raise ValueError("generation reserve must leave room inside the context")

    reasons: list[str] = []
    stuck = False

    if observed is None:
        first_prompt = 0
        growth = float(FALLBACK_TOKENS_PER_REQUEST)
        growth_source = "measured_fallback_20261026"
        requests = 0
        reclaimed = 0
    else:
        prompts = list(observed.get("prompt_tokens") or [])
        if len(prompts) >= 2:
            stats = step_token_growth(prompts)
            requests = int(stats["requests"])
            reclaimed = int(stats["reclaim_events"])
            if int(stats["positive_steps"]) == 0:
                # Frozen overflow: the retried conversation never shrinks, so
                # its current size is the floor for any further attempt.
                stuck = True
                first_prompt = int(max(prompts))
                growth = float(FALLBACK_TOKENS_PER_REQUEST)
                growth_source = "measured_fallback_20261026_after_frozen_overflow"
            else:
                first_prompt = int(stats["first_prompt"])
                growth = float(stats["mean_growth"])
                growth_source = "observed_native_log"
        else:
            first_prompt = int(prompts[0]) if prompts else 0
            growth = float(FALLBACK_TOKENS_PER_REQUEST)
            growth_source = "measured_fallback_20261026"
            requests = len(prompts)
            reclaimed = 0

    need = required_context(
        first_prompt=first_prompt,
        growth=growth,
        max_steps=max_steps,
        generation_reserve=generation_reserve,
    )
    fits = affordable_steps(
        first_prompt=first_prompt,
        growth=growth,
        context=context,
        generation_reserve=generation_reserve,
    )

    reasons: list[str] = []
    if stuck:
        reasons.append("OBSERVED_CONVERSATION_FROZEN_AT_CONTEXT_CEILING")
    status = "ADMITTED_TO_ATTEMPT"
    if need > context:
        status = "REFUSED"
        reasons.append("EPISODE_CANNOT_REACH_MAX_STEPS_IN_CONTEXT")
    if fits < max_steps:
        if status != "REFUSED":
            status = "REFUSED"
        reasons.append("TRUNCATION_GUARANTEED_BEFORE_MAX_STEPS")

    truncated = int(observed.get("truncated_generations") or 0) if observed else 0
    json_failures = int(observed.get("tool_call_json_failures") or 0) if observed else 0
    if truncated and json_failures:
        reasons.append("OBSERVED_TRUNCATED_TOOL_CALL_REJECTED_AS_INVALID_JSON")
    if json_failures and truncated:
        # The retried conversation cannot shrink, so retries cannot recover.
        reasons.append("RETRY_CANNOT_RECOVER_UNCHANGED_OVERFLOW")

    return {
        "schema": "reap-v2-context-budget-admission.v1",
        "status": status,
        "context": context,
        "max_steps": max_steps,
        "generation_reserve": generation_reserve,
        "first_prompt": first_prompt,
        "tokens_per_request": round(growth, 1),
        "growth_source": growth_source,
        "requests_observed": requests,
        "reclaim_events": reclaimed,
        "required_context": need,
        "shortfall": max(0, need - context),
        "affordable_steps": fits,
        "observed_truncated_generations": truncated,
        "observed_tool_call_json_failures": json_failures,
        "reasons": sorted(set(reasons)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-log", required=True, help="llama-server log for the trial")
    parser.add_argument("--context", type=int, required=True, help="served -c value")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--generation-reserve", type=int, default=DEFAULT_GENERATION_RESERVE)
    args = parser.parse_args()

    with open(args.native_log, "r", encoding="utf-8", errors="replace") as handle:
        observed = trajectory_from_native_log(handle.read())

    result = assess(
        observed=observed,
        context=args.context,
        max_steps=args.max_steps,
        generation_reserve=args.generation_reserve,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ADMITTED_TO_ATTEMPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
