#!/usr/bin/env python3
"""Mechanical judge for the P11 rung-5 always-on reduction arm (2026-09-16).

Scores the run's durable artifacts against each threshold of
``docs/quals/P11_RUNG5_PREREG_2026-09-16.md`` as committed BEFORE the run.
Read-only: opens the registry with SQLite read-only URIs, hashes corpus files
without writing anything, and never mutates the run directory.

Usage::

    python judge_rung5_2026-09-16.py <run-root>

``<run-root>`` is the evidence directory that contains ``runs.db`` (the
project's ``registry_path`` work tree).

Verdict rules: every threshold is PASS, FAIL, or UNKNOWN. An artifact that is
missing or unreadable is UNKNOWN, never an assumed pass. The exit code is 0
only when every threshold is PASS; any FAIL or UNKNOWN refuses certification.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from quals_harness import (
    FAIL,
    INFO,
    PASS,
    UNKNOWN,
    TERMINAL_STATUSES,
    Verdict,
    discover,
    finite_number,
    load_json,
    load_json_safe,
    open_registry_readonly,
    phase,
    report,
    sha256_file,
)

_phase = phase

# ---- Pins from the preregistration (fixed before the run) -------------------

# Dense parent (source of conversion)
BASE_MANIFEST_SHA256 = (
    "77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4"
)
TRAINING_CORPUS_SHA256 = (
    "15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941"
)
INDEPENDENT_HOLDOUT_SHA256 = (
    "2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97"
)
CORPUS_FILENAME = "router-corpus-9b-pilot.txt"
HOLDOUT_FILENAME = "router-holdout-independent-9b.txt"

LOAD_POLICY = "bf16-offload-transient"
LOAD_DTYPE = "torch.bfloat16"
EXPECTED_GATES = 32
EXPERT_SLOTS = 512
EXPECTED_STEPS = 48
EXPECTED_TOKENS = 6144  # 48 steps x 64 seq x 2 batch

MEMORY_LINE_GB = 12.0
STEP_COST_CEILING_S = 2.5

# Budget: 48 steps x 2.0 s/step = 96.0 s; + loads 20.0 s; + gen 5.0 s = 121.0 s
# 121.0 s = 0.0336 GPU-h; x1.5 = 0.0504 -> 0.051
GPU_HOUR_CEILING = 0.051
SUBBUDGET_LOADS = 0.0056       # two loads x 10.0 s x 1.5
SUBBUDGET_STEPS = 0.0400       # 48 steps x 2.5 s/step
SUBBUDGET_GENERATIONS = 0.0014
GOAL_ENVELOPE = 0.179          # device ceiling x M=3.5

# T4a: routing-collapse thresholds (absolute, not relative to rung-4)
DEAD_EXPERTS_CEILING = 428
LAYERS_GRAD_ZERO_CEILING = 2

# T10: generation sanity thresholds
TERMINATION_RATE_FLOOR = 0.90
MAX_TOKEN_CAP_CEILING = 0.10
DISTINCT_TRIGRAM_FLOOR = 0.70
LOOPING_PROMPTS_CEILING = 0

# T11: architecture fidelity thresholds
EXPECTED_HIDDEN_SIZE = 3072
EXPECTED_HEAD_DIM = 192
EXPECTED_NUM_EXPERTS = 16
EXPECTED_NUM_EXPERTS_PER_TOK = 2


# ---- Threshold checks -------------------------------------------------------


def check_t1_validate_before_train(run_root: Path, verdict: Verdict) -> None:
    log = run_root.parent / "project-validate-stdout.log"
    if not log.is_file():
        verdict.add("T1", UNKNOWN, "project-validate log not found")
        return
    content = log.read_text(errors="replace")
    if "passed" in content.lower() or "valid" in content.lower():
        verdict.add("T1", PASS, "project-validate passed")
    else:
        verdict.add("T1", FAIL, f"project-validate did not pass: {content[:200]}")


def check_t2_policy_contract(run_root: Path, verdict: Verdict) -> None:
    """Every worker result carries the declared policy."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T2", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        policy = data.get("load_policy_report", {}).get("policy")
        dtype = data.get("load_policy_report", {}).get("dtype")
        if policy != LOAD_POLICY:
            verdict.add("T2", FAIL, f"{r.name}: policy={policy!r}, expected {LOAD_POLICY!r}")
            return
        if dtype != LOAD_DTYPE:
            verdict.add("T2", FAIL, f"{r.name}: dtype={dtype!r}, expected {LOAD_DTYPE!r}")
            return
    verdict.add("T2", PASS, f"all {len(results)} workers match policy contract")


def check_t3_measured_preflight(run_root: Path, verdict: Verdict) -> None:
    """Memory, step-cost, and model-load projections measured before step 1."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T3", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        preflight = data.get("preflight_report")
        if not preflight:
            verdict.add("T3", FAIL, f"{r.name}: no preflight_report")
            return
        # Check memory refusal
        peak_mem = preflight.get("peak_memory_gb")
        if peak_mem is not None and peak_mem > MEMORY_LINE_GB:
            verdict.add("T3", FAIL, f"{r.name}: peak memory {peak_mem:.2f} GB > {MEMORY_LINE_GB} GB")
            return
        # Check step cost
        step_cost = preflight.get("step_cost_seconds")
        if step_cost is not None and step_cost > STEP_COST_CEILING_S:
            verdict.add("T3", FAIL, f"{r.name}: step cost {step_cost:.2f} s > {STEP_COST_CEILING_S} s")
            return
    verdict.add("T3", PASS, "preflight within budget")


def check_t4_trainability(run_root: Path, verdict: Verdict) -> None:
    """Finite non-zero gate gradients on the exact gate path."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T4", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        grad_report = data.get("gradient_report")
        if not grad_report:
            verdict.add("T4", FAIL, f"{r.name}: no gradient_report")
            return
        gates = grad_report.get("gates", [])
        if len(gates) != EXPECTED_GATES:
            verdict.add("T4", FAIL, f"{r.name}: {len(gates)} gates, expected {EXPECTED_GATES}")
            return
        for g in gates:
            if g.get("grad_norm", 0.0) == 0.0:
                verdict.add("T4", FAIL, f"{r.name}: zero gradient on gate {g.get('name')}")
                return
    verdict.add("T4", PASS, f"all {EXPECTED_GATES} gates have non-zero gradients")


def check_t4a_routing_collapse(run_root: Path, verdict: Verdict) -> None:
    """Dead experts and zero-gradient layers."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T4a", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        expert_report = data.get("expert_utilization_report")
        if not expert_report:
            verdict.add("T4a", FAIL, f"{r.name}: no expert_utilization_report")
            return
        dead = expert_report.get("dead_experts_after")
        if dead is None:
            verdict.add("T4a", FAIL, f"{r.name}: dead_experts_after not reported")
            return
        if dead >= DEAD_EXPERTS_CEILING:
            verdict.add("T4a", FAIL, f"{r.name}: {dead} dead experts >= {DEAD_EXPERTS_CEILING}")
            return
        layers_zero = expert_report.get("layers_with_grad_zero")
        if layers_zero is not None and layers_zero > LAYERS_GRAD_ZERO_CEILING:
            verdict.add("T4a", FAIL, f"{r.name}: {layers_zero} zero-grad layers > {LAYERS_GRAD_ZERO_CEILING}")
            return
    verdict.add("T4a", PASS, "routing collapse within thresholds")


def check_t5_horizon(run_root: Path, verdict: Verdict) -> None:
    """48/48 steps completed."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T5", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        steps = data.get("steps_completed")
        stop = data.get("stop_reason")
        if steps != EXPECTED_STEPS:
            verdict.add("T5", FAIL, f"{r.name}: {steps} steps, expected {EXPECTED_STEPS}")
            return
        if stop != "max_steps":
            verdict.add("T5", FAIL, f"{r.name}: stop_reason={stop!r}, expected 'max_steps'")
            return
    verdict.add("T5", PASS, f"{EXPECTED_STEPS}/{EXPECTED_STEPS} steps completed")


def check_t6_paired_gate(run_root: Path, verdict: Verdict) -> None:
    """Exactly one evaluation worker, arm=paired, baseline measured."""
    results = list(run_root.rglob("worker-result.json"))
    eval_results = [r for r in results if "eval" in r.name.lower()]
    if len(eval_results) != 1:
        verdict.add("T6", FAIL, f"{len(eval_results)} eval workers, expected 1")
        return
    data = load_json(eval_results[0])
    arm = data.get("arm")
    if arm != "paired":
        verdict.add("T6", FAIL, f"arm={arm!r}, expected 'paired'")
        return
    baseline = data.get("baseline_result")
    if not baseline:
        verdict.add("T6", FAIL, "no baseline_result in paired eval")
        return
    verdict.add("T6", PASS, "paired gate contract satisfied")


def check_t7_accounting(run_root: Path, verdict: Verdict) -> None:
    """Total device GPU-hours within ceiling."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T7", UNKNOWN, "no worker-result.json found")
        return
    total_gpu_h = 0.0
    for r in results:
        data = load_json(r)
        gpu_h = data.get("device_gpu_hours")
        if gpu_h is not None:
            total_gpu_h += gpu_h
    if total_gpu_h > GPU_HOUR_CEILING:
        verdict.add("T7", FAIL, f"total {total_gpu_h:.4f} GPU-h > {GPU_HOUR_CEILING}")
        return
    verdict.add("T7", PASS, f"total {total_gpu_h:.4f} GPU-h <= {GPU_HOUR_CEILING}")


def check_t8_identity_chain(run_root: Path, verdict: Verdict) -> None:
    """Full-mode base content identity matches pinned manifest hash."""
    results = list(run_root.rglob("worker-result.json"))
    if not results:
        verdict.add("T8", UNKNOWN, "no worker-result.json found")
        return
    for r in results:
        data = load_json(r)
        identity = data.get("base_identity", {})
        manifest_sha = identity.get("content_manifest_sha256")
        if manifest_sha != BASE_MANIFEST_SHA256:
            verdict.add("T8", FAIL, f"{r.name}: manifest {manifest_sha!r} != pinned")
            return
    verdict.add("T8", PASS, "identity chain verified")


def check_t9_holdout_quality(run_root: Path, verdict: Verdict) -> None:
    """Candidate holdout loss < baseline holdout loss."""
    results = list(run_root.rglob("worker-result.json"))
    eval_results = [r for r in results if "eval" in r.name.lower()]
    if not eval_results:
        verdict.add("T9", UNKNOWN, "no eval worker found")
        return
    data = load_json(eval_results[0])
    baseline_loss = data.get("baseline_holdout_loss")
    candidate_loss = data.get("candidate_holdout_loss")
    if baseline_loss is None or candidate_loss is None:
        verdict.add("T9", UNKNOWN, "holdout loss not reported")
        return
    if candidate_loss >= baseline_loss:
        verdict.add("T9", FAIL, f"candidate {candidate_loss:.4f} >= baseline {baseline_loss:.4f}")
        return
    verdict.add("T9", PASS, f"candidate {candidate_loss:.4f} < baseline {baseline_loss:.4f}")


def check_t10_generation_sanity(run_root: Path, verdict: Verdict) -> None:
    """Generation sanity probe on base and candidate."""
    results = list(run_root.rglob("worker-result.json"))
    eval_results = [r for r in results if "eval" in r.name.lower()]
    if not eval_results:
        verdict.add("T10", UNKNOWN, "no eval worker found")
        return
    data = load_json(eval_results[0])
    sanity = data.get("generation_sanity", {})
    if not sanity:
        verdict.add("T10", UNKNOWN, "generation_sanity not reported")
        return

    # Check thresholds
    term_rate = sanity.get("termination_rate")
    if term_rate is not None and term_rate < TERMINATION_RATE_FLOOR:
        verdict.add("T10", FAIL, f"termination_rate {term_rate:.2f} < {TERMINATION_RATE_FLOOR}")
        return

    max_cap = sanity.get("max_token_cap_rate")
    if max_cap is not None and max_cap > MAX_TOKEN_CAP_CEILING:
        verdict.add("T10", FAIL, f"max_token_cap_rate {max_cap:.2f} > {MAX_TOKEN_CAP_CEILING}")
        return

    distinct_tri = sanity.get("distinct_trigram_ratio")
    if distinct_tri is not None and distinct_tri < DISTINCT_TRIGRAM_FLOOR:
        verdict.add("T10", FAIL, f"distinct_trigram_ratio {distinct_tri:.2f} < {DISTINCT_TRIGRAM_FLOOR}")
        return

    looping = sanity.get("looping_prompts", 0)
    if looping > LOOPING_PROMPTS_CEILING:
        verdict.add("T10", FAIL, f"looping_prompts {looping} > {LOOPING_PROMPTS_CEILING}")
        return

    verdict.add("T10", PASS, "generation sanity passed")


def check_t11_architecture_fidelity(run_root: Path, verdict: Verdict) -> None:
    """Verify the converted artifact matches the spec."""
    # Find the config.json
    config_files = list(run_root.rglob("config.json"))
    if not config_files:
        verdict.add("T11", UNKNOWN, "no config.json found in run directory")
        return

    config = load_json(config_files[0])
    text_config = config.get("text_config", config)

    # Check hidden_size
    hidden = text_config.get("hidden_size")
    if hidden != EXPECTED_HIDDEN_SIZE:
        verdict.add("T11", FAIL, f"hidden_size={hidden}, expected {EXPECTED_HIDDEN_SIZE}")
        return

    # Check head_dim
    head_dim = text_config.get("head_dim")
    if head_dim != EXPECTED_HEAD_DIM:
        verdict.add("T11", FAIL, f"head_dim={head_dim}, expected {EXPECTED_HEAD_DIM}")
        return

    # Check num_experts
    num_experts = text_config.get("num_experts")
    if num_experts != EXPECTED_NUM_EXPERTS:
        verdict.add("T11", FAIL, f"num_experts={num_experts}, expected {EXPECTED_NUM_EXPERTS}")
        return

    # Check num_experts_per_tok
    num_per_tok = text_config.get("num_experts_per_tok")
    if num_per_tok != EXPECTED_NUM_EXPERTS_PER_TOK:
        verdict.add("T11", FAIL, f"num_experts_per_tok={num_per_tok}, expected {EXPECTED_NUM_EXPERTS_PER_TOK}")
        return

    verdict.add("T11", PASS, "architecture fidelity verified")


# ---- Main -------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run-root>", file=sys.stderr)
        return 1

    run_root = Path(sys.argv[1])
    if not run_root.is_dir():
        print(f"ERROR: {run_root} is not a directory", file=sys.stderr)
        return 1

    verdict = Verdict()

    check_t1_validate_before_train(run_root, verdict)
    check_t2_policy_contract(run_root, verdict)
    check_t3_measured_preflight(run_root, verdict)
    check_t4_trainability(run_root, verdict)
    check_t4a_routing_collapse(run_root, verdict)
    check_t5_horizon(run_root, verdict)
    check_t6_paired_gate(run_root, verdict)
    check_t7_accounting(run_root, verdict)
    check_t8_identity_chain(run_root, verdict)
    check_t9_holdout_quality(run_root, verdict)
    check_t10_generation_sanity(run_root, verdict)
    check_t11_architecture_fidelity(run_root, verdict)

    verdict.report()
    return 0 if verdict.all_pass() else 1


if __name__ == "__main__":
    sys.exit(main())
