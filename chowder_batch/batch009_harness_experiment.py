"""Batch-009: controlled harness experiment on the frozen gen-2 model.

The model and the plain harness are the fixed reference. Each candidate harness
mechanism is evaluated on the *same* evolve tasks, decoding settings, and seeds,
then re-checked on the disjoint held-out split before it can be adopted. Full
tool traces are recorded so successful and failed runs can be inspected, not
just aggregate rewards.

Run with the plain incumbent plus one or more candidate harnesses:

    python chowder_batch/batch009_harness_experiment.py \
        --base F:/Huihui-Spark-X2.5-4B-abliterated \
        --parent .chowder-spark-calib/gsm8k/gen2/.chowder/runs/gsm8k-gen2-17e5020c6316/adapter \
        --out F:/chowder-campaign/batch009-harness/harness_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from chowder.harness_evolution import HarnessProposal, metrics_from_benchmark, select_first_round
from chowder.runtime_eval import HELDOUT_TASKS, TASKS, make_transformers_generate, run_live_benchmark

HARNESSES = (
    "plain", "state_aware", "state_aware_legacy", "recovery",
    "state_aware+recovery", "state_aware+recovery_legacy",
)

# Generic mechanism descriptions. Leakage screening rejects any wording that
# names a task, entity, file, or answer, so these stay mechanism-level.
PROPOSALS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "state_aware": (
        "state-aware-file-discovery",
        "tooling",
        ("expose the live workspace file list and report real available paths on a miss",),
    ),
    "recovery": (
        "failed-test-recovery-state",
        "control_flow",
        ("track an explicit post-red recovery state and block redundant run_tests until a write changes the workspace",),
    ),
    "state_aware_legacy": (
        "verbose-state-aware-file-discovery",
        "tooling",
        ("show the actual current workspace file list and identify real available paths",),
    ),
    "state_aware+recovery": (
        "combined-state-discovery-and-red-recovery",
        "interaction",
        (
            "expose the live workspace file list and report real available paths on a miss",
            "block post-red test reruns until a write changes the workspace",
        ),
    ),
    "state_aware+recovery_legacy": (
        "combined-legacy-state-discovery-and-red-recovery",
        "interaction",
        (
            "show the actual current workspace file list and identify real available paths",
            "block post-red test reruns until a write changes the workspace",
        ),
    ),
}


def forbidden_terms() -> tuple[str, ...]:
    names = {task.name for task in (*TASKS, *HELDOUT_TASKS)}
    paths = {path for task in (*TASKS, *HELDOUT_TASKS) for path in task.initial}
    return tuple(sorted(names | paths))


def trace_digest(result: Mapping[str, Any]) -> dict[str, str]:
    """One compact, human-readable action line per task for trace inspection."""
    digest: dict[str, str] = {}
    for row in result["tasks"]:
        steps = []
        for entry in row["trace"]:
            if entry.get("role") != "tool":
                continue
            name = entry["tool"]
            path = entry["args"].get("path", "")
            label = f"{name}({path})" if path else name
            if entry.get("synthetic"):
                label += "[blocked]"
            elif name == "run_tests":
                label += "[green]" if "pass" in str(entry["observation"]).lower() else "[red]"
            steps.append(label)
        if row["green_seen"]:
            steps.append("report")
        digest[row["task"]] = " -> ".join(steps) if steps else "(no tools)"
    return digest


def run_harness(generate, harness: str, max_turns: int, limit: int | None = None) -> dict[str, Any]:
    evolve = TASKS[:limit] if limit else TASKS
    heldout = HELDOUT_TASKS[:limit] if limit else HELDOUT_TASKS
    return {
        "evolve": run_live_benchmark(generate, max_turns=max_turns, harness=harness, tasks=evolve, split="evolve"),
        "heldout": run_live_benchmark(generate, max_turns=max_turns, harness=harness, tasks=heldout, split="heldout"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--harnesses", default=",".join(HARNESSES))
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="cap tasks per split for smoke runs; 0 means all")
    args = parser.parse_args()

    harnesses = tuple(name.strip() for name in args.harnesses.split(",") if name.strip())
    if not harnesses or len(harnesses) != len(set(harnesses)):
        raise SystemExit("harness names must be nonempty and unique")
    unknown_harnesses = set(harnesses) - set(HARNESSES)
    if unknown_harnesses:
        raise SystemExit(f"unknown harnesses: {sorted(unknown_harnesses)}")
    if "plain" not in harnesses:
        raise SystemExit("the plain harness must be included as the frozen reference")

    limit = args.limit or None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_run = {
        "task_names": {
            "evolve": [task.name for task in (TASKS[:limit] if limit else TASKS)],
            "heldout": [task.name for task in (HELDOUT_TASKS[:limit] if limit else HELDOUT_TASKS)],
        },
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "harnesses": list(harnesses),
        "base_model": str(Path(args.base).resolve()),
        "parent_adapter": str(Path(args.parent).resolve()),
        "harness_version": "batch009-v2-wrong-second-fix-compact-state-aware",
    }
    results: dict[str, Any] = {}
    if out.is_file():
        try:
            saved = json.loads(out.read_text(encoding="utf-8"))
            if saved.get("run_config") != expected_run:
                raise SystemExit(
                    f"checkpoint {out} is stale or belongs to another run; choose a new --out path"
                )
            for harness, arms in saved.get("harnesses", {}).items():
                if harness not in harnesses:
                    continue
                evolve = arms.get("evolve", {})
                heldout = arms.get("heldout", {})
                recorded = [row["task"] for row in evolve.get("tasks", [])]
                heldout_recorded = [row["task"] for row in heldout.get("tasks", [])]
                expected_evolve = expected_run["task_names"]["evolve"]
                expected_heldout = expected_run["task_names"]["heldout"]
                if (
                    recorded == expected_evolve
                    and heldout_recorded == expected_heldout
                    and "metrics" in evolve
                    and "metrics" in heldout
                ):
                    results[harness] = arms
                    print(f"[resume] reusing completed {harness} arm from checkpoint", flush=True)
                else:
                    print(f"[resume] discarding invalid {harness} arm", flush=True)
        except SystemExit:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"checkpoint {out} is unreadable; choose a new --out path ({exc})") from exc

    missing_harnesses = [harness for harness in harnesses if harness not in results]
    generate = None
    if missing_harnesses:
        from run_runtime_harness_compare import load_model

        tokenizer, model = load_model(args.base, args.parent)
        generate = make_transformers_generate(
            tokenizer, model, max_new_tokens=args.max_new_tokens, device=next(model.parameters()).device
        )
    # Resume only from a checkpoint whose explicit run configuration matches.
    for harness in harnesses:
        if harness in results:
            continue
        assert generate is not None
        results[harness] = run_harness(generate, harness, args.max_turns, limit)
        # Checkpoint after every harness so a restart never loses finished work.
        out.write_text(json.dumps({"run_config": expected_run, "harnesses": results, "complete": False}, indent=2) + "\n", encoding="utf-8")

    forbidden = forbidden_terms()
    incumbent_evolve = metrics_from_benchmark(results["plain"]["evolve"])
    incumbent_heldout = metrics_from_benchmark(results["plain"]["heldout"])
    selection: dict[str, Any] = {}
    for harness in harnesses:
        if harness == "plain" or harness not in PROPOSALS:
            continue
        name, component, changes = PROPOSALS[harness]
        candidate_evolve = metrics_from_benchmark(results[harness]["evolve"])
        candidate_heldout = metrics_from_benchmark(results[harness]["heldout"])
        proposal = HarnessProposal(name, component, changes, candidate_evolve)
        selection[harness] = select_first_round(
            incumbent_evolve,
            candidate_evolve,
            evolve_incumbent=incumbent_evolve,
            evolve_candidate=candidate_evolve,
            heldout_incumbent=incumbent_heldout,
            heldout_candidate=candidate_heldout,
            proposal=proposal,
            forbidden_terms=forbidden,
        )

    interaction: dict[str, Any] = {}
    if {"plain", "state_aware", "recovery", "state_aware+recovery"} <= set(results):
        for split in ("evolve", "heldout"):
            arms = {
                name: results[name][split]["metrics"]
                for name in ("plain", "state_aware", "recovery", "state_aware+recovery")
            }
            deltas = {}
            for metric in ("runtime_reward", "runtime_green_rate", "runtime_execution_cost", "policy_tokens", "prompt_tokens", "total_tokens"):
                deltas[metric] = {
                    "state_gain": arms["state_aware"][metric] - arms["plain"][metric],
                    "recovery_gain": arms["recovery"][metric] - arms["plain"][metric],
                    "combined_gain": arms["state_aware+recovery"][metric] - arms["plain"][metric],
                    "additive_interaction": (
                        arms["state_aware+recovery"][metric] - arms["state_aware"][metric]
                        - arms["recovery"][metric] + arms["plain"][metric]
                    ),
                }
            interaction[split] = {"arms": arms, "effects": deltas}

    payload = {
        "run_config": expected_run,
        "harnesses": results,
        "selection": selection,
        "factorial_interaction": interaction,
        "forbidden_terms": list(forbidden),
        "traces": {harness: {split: trace_digest(results[harness][split]) for split in ("evolve", "heldout")} for harness in harnesses},
    }
    out.write_text(json.dumps({**payload, "complete": True}, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        harness: {
            split: results[harness][split]["metrics"] for split in ("evolve", "heldout")
        }
        for harness in harnesses
    }, indent=2))
    print(json.dumps({"selection": {h: {"accepted": d.get("accepted"), "reason": d.get("reason")} for h, d in selection.items()}}, indent=2))
    if interaction:
        print("factorial interaction (combined - state - recovery + plain):", json.dumps(interaction, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
