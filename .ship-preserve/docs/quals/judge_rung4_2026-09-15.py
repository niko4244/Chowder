#!/usr/bin/env python3
"""Mechanical judge for the P11 rung-4 longer-horizon paired CUDA arm (2026-09-15).

Scores the run's durable artifacts against each threshold of
``docs/quals/P11_RUNG4_PREREG_2026-09-15.md`` as committed BEFORE the run.
Read-only: opens the registry with SQLite read-only URIs, hashes corpus files
without writing anything, and never mutates the run directory.

Usage::

    python judge_rung4_2026-09-15.py <run-root>

``<run-root>`` is the evidence directory that contains ``runs.db`` (the
project's ``registry_path`` work tree).

Verdict rules: every threshold is PASS, FAIL, or UNKNOWN. An artifact that is
missing or unreadable is UNKNOWN, never an assumed pass. The exit code is 0
only when every threshold is PASS; any FAIL or UNKNOWN refuses certification.
"""

from __future__ import annotations

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

MEMORY_LINE_GB = 14.5
STEP_COST_CEILING_S = 3.5

# Budget: 48 steps x 2.854 s/step = 137.0 s; + loads 27.5 s; + gen 7.5 s = 172.0 s
# 172.0 s = 0.0478 GPU-h; x1.5 = 0.0717 -> 0.072
GPU_HOUR_CEILING = 0.072
SUBBUDGET_LOADS = 0.0114       # two loads x 13.85 s x 1.5
SUBBUDGET_STEPS = 0.0574       # 48 steps x 3.5 s/step
SUBBUDGET_GENERATIONS = 0.0032
GOAL_ENVELOPE = 0.252          # device ceiling x M=3.5

# Rung-3c reproduction pins (info, not thresholds)
REPRO_BASE_LOSS = 2.9327
REPRO_CANDIDATE_LOSS = 2.8369
REPRO_DEAD_EXPERTS = 428.0
REPRO_TOLERANCE = 1e-3

# T4a: routing-collapse thresholds
DEAD_EXPERTS_CEILING = 428  # must be STRICTLY less than rung-3c's 428
LAYERS_GRAD_ZERO_CEILING = 2

# T10: generation sanity thresholds
TERMINATION_RATE_FLOOR = 0.90
MAX_TOKEN_CAP_CEILING = 0.10
DISTINCT_TRIGRAM_FLOOR = 0.70
LOOPING_PROMPTS_CEILING = 0


# ---- Threshold checks -------------------------------------------------------


def check_t1_validate_before_train(run_root: Path, verdict: Verdict) -> None:
    log = run_root.parent / "project-validate-stdout.log"
    if not log.is_file():
        log = run_root / "project-validate-stdout.log"
    if not log.is_file():
        verdict.add("T1", "validate before train", UNKNOWN,
                    "no project-validate stdout log found beside the run root")
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if "traceback" in lowered or "refused" in lowered or "error" in lowered:
        verdict.add("T1", "validate before train", FAIL,
                    "validation log contains a traceback/refusal/error")
        return
    verdict.add("T1", "validate before train", PASS,
                f"validation log present and clean ({log.name})")


def check_t2_policy_contract(train: list, evals: list, verdict: Verdict) -> None:
    problems: list[str] = []
    checked = 0
    for path, result in train + evals:
        report = result.get("load_policy_report")
        if not isinstance(report, dict):
            problems.append(f"{path.parent.name}: no load_policy_report")
            continue
        checked += 1
        if report.get("policy") != LOAD_POLICY:
            problems.append(f"{path.parent.name}: policy={report.get('policy')!r}")
        if report.get("dtype") != LOAD_DTYPE:
            problems.append(f"{path.parent.name}: dtype={report.get('dtype')!r}")
        census = report.get("placement_census")
        if not isinstance(census, dict) or census.get("verified") is not True:
            problems.append(f"{path.parent.name}: placement_census not verified")
            continue
        if census.get("expert_params_on_device") != 0:
            problems.append(
                f"{path.parent.name}: expert_params_on_device="
                f"{census.get('expert_params_on_device')!r}"
            )
        if census.get("gate_params_on_device") != census.get("gate_params_total"):
            problems.append(f"{path.parent.name}: gates not fully on device")
    if not checked:
        verdict.add("T2", "policy contract", UNKNOWN, "no worker result carried a load policy report")
        return
    if problems:
        verdict.add("T2", "policy contract", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T2", "policy contract", PASS,
                    f"{checked} worker results carry {LOAD_POLICY}, {LOAD_DTYPE}, verified CPU-expert placement")


def check_t3_measured_preflight(train: list, evals: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("T3", "measured preflight", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    preflight = train[0][1].get("device_preflight")
    if not isinstance(preflight, dict):
        verdict.add("T3", "measured preflight", FAIL, "training result carries no device_preflight")
        return
    problems: list[str] = []
    if preflight.get("projected_oom") is not False:
        problems.append(f"projected_oom={preflight.get('projected_oom')!r}")
    step = preflight.get("step_cost_probe")
    if not isinstance(step, dict) or step.get("measured") is not True:
        problems.append("step_cost_probe missing or unmeasured")
    else:
        seconds = step.get("step_seconds")
        if not finite_number(seconds) or float(seconds) > STEP_COST_CEILING_S:
            problems.append(f"step_seconds={seconds!r} against a {STEP_COST_CEILING_S}s ceiling")
        if step.get("would_exceed_budget") is not False:
            problems.append("step-cost projection would exceed the wall budget")
    load = train[0][1].get("load_budget")
    if not isinstance(load, dict) or load.get("measured") is not True:
        problems.append("load_budget missing or unmeasured")
    else:
        if load.get("would_exceed_load_budget") is not False:
            problems.append("load budget exceeded on the training worker")
        load_gpu = load.get("load_gpu_hours")
        if not finite_number(load_gpu) or float(load_gpu) <= 0:
            problems.append(f"load_gpu_hours={load_gpu!r} is not a measured positive cost")
    for path, result in evals:
        load = (result.get("load_budget") or {}) if isinstance(result.get("load_budget"), dict) else {}
        if not load or load.get("measured") is not True:
            problems.append(f"{path.parent.name}: eval load_budget missing or unmeasured")
        elif load.get("would_exceed_load_budget") is not False:
            problems.append(f"{path.parent.name}: eval load budget exceeded")
    if problems:
        verdict.add("T3", "measured preflight", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T3", "measured preflight", PASS,
                    "memory, step-cost, and model-load projections measured before step 1, none exceeding")


def check_t4_trainability(train: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("T4", "trainability", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    trainability = result.get("trainability")
    components = (trainability or {}).get("components") if isinstance(trainability, dict) else None
    if not isinstance(components, dict) or not components:
        verdict.add("T4", "trainability", FAIL, "no trainability components reported")
        return
    problems: list[str] = []
    grad_zero: list[str] = []
    steps = list(range(EXPECTED_STEPS))
    for name, component in components.items():
        if not isinstance(component, dict):
            problems.append(f"{name}: unreadable component")
            continue
        states = component.get("gradient_states") or []
        if component.get("trainable") is not True:
            problems.append(f"{name}: trainable={component.get('trainable')!r}")
        if "grad-nonzero" not in states:
            problems.append(f"{name}: gradient states {states!r} never nonzero")
        nonzero = component.get("nonzero_steps")
        if (
            not isinstance(nonzero, list)
            or not nonzero
            or sorted(nonzero) != nonzero
            or any(not isinstance(step, int) or step < 0 or step >= EXPECTED_STEPS for step in nonzero)
        ):
            problems.append(f"{name}: nonzero_steps {nonzero!r} is not a step list within 0..{EXPECTED_STEPS - 1}")
        if component.get("update_steps") != steps:
            problems.append(f"{name}: update_steps {component.get('update_steps')!r} != every step 0..{EXPECTED_STEPS - 1}")
        if "grad-zero" in states:
            grad_zero.append(name)
    if len(components) != EXPECTED_GATES:
        problems.append(f"component count {len(components)} != {EXPECTED_GATES}")
    frozen = result.get("frozen")
    if not isinstance(frozen, dict) or frozen.get("ok") is not True:
        problems.append("frozen-tensor evidence missing or not ok")
    else:
        if frozen.get("changed") != {}:
            problems.append(f"frozen digests report changed={frozen.get('changed')!r}")
        if "full" not in (frozen.get("digest_strategy") or []):
            problems.append("no full-strategy frozen digest")
    if problems:
        verdict.add("T4", "trainability", FAIL, "; ".join(problems[:6]))
        return
    verdict.add("T4", "trainability", PASS,
                f"{len(components)}/{EXPECTED_GATES} gates trainable with real updates on all {EXPECTED_STEPS} steps; "
                "frozen changed={}, ok")
    if grad_zero:
        verdict.add("INFO", "grad-zero on exact-zero gradient steps", UNKNOWN,
                    f"{len(grad_zero)} gate(s) recorded grad-zero on some steps (e.g. {grad_zero[0]}): exact-zero "
                    "gradients while updates continued --- honestly recorded by the worker; inspect before "
                    "certifying quality claims")


def check_t4a_routing_collapse(train: list, verdict: Verdict) -> None:
    """T4a: routing-collapse threshold (NEW in rung-4)."""
    if len(train) != 1:
        verdict.add("T4a", "routing-collapse", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    metrics = result.get("metrics") or {}
    trainability = result.get("trainability") or {}
    components = trainability.get("components") or {}

    problems: list[str] = []

    # dead_experts_after must be strictly less than 428 (rung-3c's value)
    dead = metrics.get("dead_experts_after") or metrics.get("dead_experts")
    if not finite_number(dead):
        problems.append(f"dead_experts_after={dead!r} unmeasured")
    elif float(dead) >= DEAD_EXPERTS_CEILING:
        problems.append(f"dead_experts_after={float(dead):.0f} >= {DEAD_EXPERTS_CEILING} (must be strictly less)")

    # layers_with_grad_zero <= 2
    grad_zero_layers: list[str] = []
    for name, component in components.items():
        if isinstance(component, dict) and "grad-zero" in (component.get("gradient_states") or []):
            grad_zero_layers.append(name)
    if len(grad_zero_layers) > LAYERS_GRAD_ZERO_CEILING:
        problems.append(
            f"layers_with_grad_zero={len(grad_zero_layers)} > {LAYERS_GRAD_ZERO_CEILING} "
            f"(e.g. {', '.join(grad_zero_layers[:3])})"
        )

    # Any layer with grad_zero_steps == EXPECTED_STEPS must have evidence
    saturated: list[str] = []
    for name, component in components.items():
        if isinstance(component, dict):
            gz_steps = component.get("grad_zero_steps")
            if gz_steps == EXPECTED_STEPS:
                saturated.append(name)
    if saturated:
        verdict.add("INFO", "saturated layers (grad_zero on every step)", UNKNOWN,
                    f"{len(saturated)} layer(s) saturated: {', '.join(saturated[:3])} --- "
                    "inspect topology-vs-optimizer evidence")

    if problems:
        verdict.add("T4a", "routing-collapse", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T4a", "routing-collapse", PASS,
                    f"dead_experts_after={float(dead):.0f}/{EXPERT_SLOTS}, "
                    f"layers_with_grad_zero={len(grad_zero_layers)}/{EXPECTED_GATES}")


def check_t5_horizon(train: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("T5", "horizon", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    limits = result.get("limits") or {}
    steps = result.get("steps_completed")
    stop = limits.get("stop_reason")
    tokens = limits.get("tokens_consumed")
    problems: list[str] = []
    if steps != EXPECTED_STEPS:
        problems.append(f"steps_completed={steps!r}")
    if stop != "max_steps":
        problems.append(f"stop_reason={stop!r}")
    if tokens != EXPECTED_TOKENS:
        problems.append(f"tokens_consumed={tokens!r} (expected the fixed {EXPECTED_TOKENS})")
    if problems:
        verdict.add("T5", "horizon", FAIL, "; ".join(problems))
    else:
        verdict.add("T5", "horizon", PASS,
                    f"{steps}/{EXPECTED_STEPS} steps, stop_reason=max_steps, {tokens} tokens exactly the fixed workload")


def check_t6_paired_gate_contract(
    run_root: Path, evals: list, registry, verdict: Verdict
) -> None:
    if len(evals) != 1:
        names = ", ".join(sorted(path.parent.name for path, _ in evals)) or "none"
        verdict.add("T6", "paired gate contract", FAIL,
                    f"expected exactly one evaluation worker (the resident pair), found {len(evals)}: {names}"
                    + (" --- a separate baseline spawn ran" if len(evals) > 1 else ""))
        return
    path, result = evals[0]
    problems: list[str] = []
    if result.get("arm") != "paired":
        problems.append(f"arm={result.get('arm')!r}, expected 'paired'")
    if result.get("pair_error") is not None:
        problems.append(f"pair_error={result.get('pair_error')!r}")
    base_loss = result.get("base_holdout_loss")
    cand_loss = result.get("candidate_holdout_loss")
    delta = result.get("holdout_loss_delta")
    if not finite_number(base_loss):
        problems.append(f"base_holdout_loss={base_loss!r} unmeasured")
    if not finite_number(cand_loss):
        problems.append(f"candidate_holdout_loss={cand_loss!r} unmeasured")
    if finite_number(base_loss) and finite_number(cand_loss):
        if not finite_number(delta) or abs(float(delta) - (float(cand_loss) - float(base_loss))) > 1e-9:
            problems.append("holdout_loss_delta inconsistent with the two arm scores")
    control = result.get("application_control") or {}
    if not control.get("applied_parameters"):
        problems.append("no applied parameters: no evidence the candidate leg ran after the base score")
    load_phase = _phase(result, "model_load")
    if load_phase.get("measured") is not True or not load_phase.get("seconds"):
        problems.append("the pair's single model_load phase is missing or unmeasured")
    for phase_name in ("baseline_generation", "candidate_generation"):
        phase = _phase(result, phase_name)
        if phase.get("measured") is not True or not phase.get("seconds"):
            problems.append(f"{phase_name} not measured inside the pair")
    if registry is not None:
        try:
            rows = list(registry.execute(
                "SELECT status FROM experiments WHERE experiment_id='baseline'"
            ))
            if not rows:
                problems.append("no baseline experiment row in the registry")
            elif rows[0][0] != "passed":
                problems.append(f"baseline row status={rows[0][0]!r}, expected passed")
            evidence_rows = list(registry.execute(
                "SELECT evidence_json FROM results WHERE experiment_id='baseline'"
            ))
            if not evidence_rows:
                problems.append("baseline has no recorded result")
            else:
                evidence = load_json_safe(evidence_rows[0][0])
                source = (evidence or {}).get("baseline_source")
                if source != "paired-candidate-evaluation":
                    problems.append(
                        f"baseline evidence baseline_source={source!r}, expected the resident pair"
                    )
        except sqlite3.Error as exc:
            problems.append(f"registry unreadable: {exc}")
    else:
        problems.append("registry not found; baseline completion unverifiable")
    if problems:
        verdict.add("T6", "paired gate contract", FAIL, "; ".join(problems[:8]))
        return
    verdict.add("T6", "paired gate contract", PASS,
                f"one resident pair at {path.parent.name}: arm=paired, base {float(base_loss):.4f} before apply, "
                "one measured load, both generations measured, baseline row completed from paired evidence")


def check_t7_accounting(
    train: list, evals: list, registry, verdict: Verdict
) -> None:
    if len(train) != 1 or len(evals) != 1:
        verdict.add("T7", "accounting", UNKNOWN,
                    "accounting needs exactly one train and one eval worker result")
        return
    train_total = (train[0][1].get("lifecycle") or {}).get("measured_gpu_hours")
    eval_total = (evals[0][1].get("lifecycle") or {}).get("measured_gpu_hours")
    if not finite_number(train_total) or not finite_number(eval_total):
        verdict.add("T7", "accounting", FAIL,
                    f"measured_gpu_hours missing (train={train_total!r}, eval={eval_total!r})")
        return
    total = float(train_total) + float(eval_total)
    problems: list[str] = []
    detail_parts = [f"total {total:.7f} GPU-h vs ceiling {GPU_HOUR_CEILING}"]
    if total > GPU_HOUR_CEILING:
        problems.append(f"total {total:.7f} exceeds the load-budgeted ceiling {GPU_HOUR_CEILING}")
    loads = 0.0
    for _, result in train + evals:
        phase = _phase(result, "model_load")
        gpu = phase.get("gpu_hours")
        if finite_number(gpu):
            loads += float(gpu)
        else:
            problems.append("a model_load phase lacks a measured gpu_hours cost")
    steps_gpu = _phase(train[0][1], "steady_state_steps").get("gpu_hours")
    if not finite_number(steps_gpu):
        problems.append("steady_state_steps lacks a measured gpu_hours cost")
        steps_gpu = 0.0
    rest = total - loads - float(steps_gpu)
    detail_parts.append(f"loads {loads:.7f}/{SUBBUDGET_LOADS}")
    if loads > SUBBUDGET_LOADS:
        problems.append(f"load sub-budget exceeded: {loads:.7f} > {SUBBUDGET_LOADS}")
    detail_parts.append(f"steps {float(steps_gpu):.7f}/{SUBBUDGET_STEPS}")
    if float(steps_gpu) > SUBBUDGET_STEPS:
        problems.append(f"step sub-budget exceeded: {float(steps_gpu):.7f} > {SUBBUDGET_STEPS}")
    detail_parts.append(f"generations+pub+closeout {rest:.7f}/{SUBBUDGET_GENERATIONS}")
    if rest > SUBBUDGET_GENERATIONS:
        problems.append(f"generation/publication sub-budget exceeded: {rest:.7f} > {SUBBUDGET_GENERATIONS}")
    incidents = 0
    stranded: list[str] = []
    if registry is not None:
        try:
            incidents = list(registry.execute("SELECT count(*) FROM execution_incidents"))[0][0]
            if incidents:
                problems.append(f"{incidents} execution incidents recorded")
            for experiment_id, status in registry.execute(
                "SELECT experiment_id, status FROM experiments"
            ):
                if status in TERMINAL_STATUSES:
                    continue
                has_result = list(registry.execute(
                    "SELECT count(*) FROM results WHERE experiment_id=?", (experiment_id,)
                ))[0][0]
                if has_result:
                    stranded.append(f"{experiment_id} carries a result while {status!r}")
            if stranded:
                problems.append("stranded results: " + ", ".join(stranded))
        except sqlite3.Error as exc:
            problems.append(f"registry unreadable: {exc}")
    wall_charges = 0.0
    if registry is not None:
        try:
            for (gpu_hours,) in registry.execute("SELECT gpu_hours FROM results"):
                if finite_number(gpu_hours):
                    wall_charges += float(gpu_hours)
        except sqlite3.Error:
            pass
    envelope_note = (
        f"wall charges {wall_charges:.4f} GPU-h inside the {GOAL_ENVELOPE} envelope"
        if wall_charges <= GOAL_ENVELOPE
        else f"wall charges {wall_charges:.4f} GPU-h EXCEED the {GOAL_ENVELOPE} envelope (recorded, per the prereg)"
    )
    if problems:
        verdict.add("T7", "accounting", FAIL, "; ".join(problems[:6]) + " | " + "; ".join(detail_parts))
        return
    if wall_charges > GOAL_ENVELOPE:
        verdict.add("T7", "accounting", UNKNOWN,
                    "measured device time is within every budget but the wall-charge exceedance needs human recording: "
                    + envelope_note)
        return
    verdict.add("T7", "accounting", PASS,
                "; ".join(detail_parts) + f"; incidents 0, no stranded results; {envelope_note}")


def check_t8_identity_chain(
    run_root: Path, train: list, evals: list, verdict: Verdict
) -> None:
    if len(train) != 1 or len(evals) != 1:
        verdict.add("T8", "identity chain", UNKNOWN,
                    "identity chain needs exactly one train and one eval worker result")
        return
    problems: list[str] = []
    content_hashes: set[str] = set()
    for path, result in train + evals:
        identity = result.get("base_identity") or {}
        if identity.get("manifest_sha256") != BASE_MANIFEST_SHA256:
            problems.append(f"{path.parent.name}: manifest hash {identity.get('manifest_sha256', '')[:16]}... != pinned")
        if identity.get("mode") != "full":
            problems.append(f"{path.parent.name}: identity mode={identity.get('mode')!r}")
        if identity.get("binding") != "local-content":
            problems.append(f"{path.parent.name}: binding={identity.get('binding')!r}")
        if identity.get("content_sha256"):
            content_hashes.add(identity["content_sha256"])
        else:
            problems.append(f"{path.parent.name}: no content hash")
    if len(content_hashes) > 1:
        problems.append("train and eval workers re-derived different base content hashes")
    # Check training corpus pin
    corpus_path = run_root.parent / CORPUS_FILENAME
    if not corpus_path.is_file():
        corpus_path = run_root / CORPUS_FILENAME
    if corpus_path.is_file():
        if sha256_file(corpus_path) != TRAINING_CORPUS_SHA256:
            problems.append("training corpus on disk hashes differently than pinned")
    else:
        problems.append("training corpus file not found beside the run root; pin unverified")
    # Check independent holdout corpus pin (NEW in rung-4)
    holdout_path = run_root.parent / HOLDOUT_FILENAME
    if not holdout_path.is_file():
        holdout_path = run_root / HOLDOUT_FILENAME
    if holdout_path.is_file():
        if sha256_file(holdout_path) != INDEPENDENT_HOLDOUT_SHA256:
            problems.append("independent holdout corpus on disk hashes differently than pinned")
    else:
        problems.append("independent holdout corpus file not found; pin unverified")
    # Check eval holdout hash from worker result
    holdout = (evals[0][1].get("holdout") or {})
    if holdout.get("sha256") and holdout.get("sha256") != INDEPENDENT_HOLDOUT_SHA256:
        problems.append("eval holdout corpus hash != pinned independent holdout hash")
    verification = evals[0][1].get("payload_verification")
    if not isinstance(verification, dict):
        problems.append("eval carries no payload_verification")
    else:
        if content_hashes and verification.get("base_content_sha256") not in content_hashes:
            problems.append("payload is not bound to the re-derived base content")
        if not verification.get("manifest_sha256"):
            problems.append("payload manifest hash missing")
    if problems:
        verdict.add("T8", "identity chain", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T8", "identity chain", PASS,
                    "full-mode local-content identity matches the pinned manifest in both workers; "
                    "payload bound to the re-derived base content; corpora hash to their pins")


def check_t9_independent_holdout(train: list, evals: list, verdict: Verdict) -> None:
    """T9: independent holdout quality (NEW in rung-4).

    Baseline must be re-measured on the independent holdout corpus.
    Candidate must improve on that holdout (direction only, no margin).
    """
    if len(evals) != 1:
        verdict.add("T9", "independent holdout", UNKNOWN,
                    f"expected exactly one eval worker, found {len(evals)}")
        return
    result = evals[0][1]

    # Check that the eval was scored on the independent holdout
    holdout = result.get("holdout") or {}
    holdout_sha = holdout.get("sha256")
    if holdout_sha != INDEPENDENT_HOLDOUT_SHA256:
        verdict.add("T9", "independent holdout", FAIL,
                    f"eval holdout sha256={holdout_sha!r}, expected the independent holdout pin")
        return

    # Check that baseline was measured on the independent holdout
    base_ind = result.get("baseline_independent_holdout_loss")
    cand_ind = result.get("candidate_independent_holdout_loss")

    # Fallback: check if the eval was specifically on the independent holdout
    # by looking at the holdout corpus reference
    if not finite_number(base_ind) or not finite_number(cand_ind):
        # Try alternative field names
        base_ind = result.get("base_independent_holdout_loss") or result.get("base_holdout_loss")
        cand_ind = result.get("candidate_independent_holdout_loss") or result.get("candidate_holdout_loss")

    if not finite_number(base_ind):
        verdict.add("T9", "independent holdout", FAIL,
                    "baseline independent holdout loss not measured")
        return
    if not finite_number(cand_ind):
        verdict.add("T9", "independent holdout", FAIL,
                    "candidate independent holdout loss not measured")
        return

    base_f = float(base_ind)
    cand_f = float(cand_ind)
    delta = cand_f - base_f

    if cand_f >= base_f:
        verdict.add("T9", "independent holdout", FAIL,
                    f"candidate {cand_f:.4f} >= baseline {base_f:.4f} (delta={delta:+.4f}, "
                    "must improve on independent holdout)")
    else:
        verdict.add("T9", "independent holdout", PASS,
                    f"baseline {base_f:.4f}, candidate {cand_f:.4f}, delta={delta:+.4f} (improved)")


def check_t10_generation_sanity(train: list, evals: list, verdict: Verdict) -> None:
    """T10: generation sanity probe (NEW in rung-4)."""
    if len(evals) != 1:
        verdict.add("T10", "generation sanity", UNKNOWN,
                    f"expected exactly one eval worker, found {len(evals)}")
        return
    result = evals[0][1]
    gen = result.get("generation_sanity") or {}
    if not gen:
        verdict.add("T10", "generation sanity", UNKNOWN,
                    "no generation_sanity data in eval worker result")
        return

    problems: list[str] = []

    # Termination rate
    term_rate = gen.get("termination_rate")
    if not finite_number(term_rate):
        problems.append(f"termination_rate={term_rate!r} unmeasured")
    elif float(term_rate) < TERMINATION_RATE_FLOOR:
        problems.append(f"termination_rate={float(term_rate):.3f} < {TERMINATION_RATE_FLOOR}")

    # Max-token-cap hit rate
    cap_rate = gen.get("max_token_cap_rate")
    if not finite_number(cap_rate):
        problems.append(f"max_token_cap_rate={cap_rate!r} unmeasured")
    elif float(cap_rate) > MAX_TOKEN_CAP_CEILING:
        problems.append(f"max_token_cap_rate={float(cap_rate):.3f} > {MAX_TOKEN_CAP_CEILING}")

    # Distinct trigram ratio
    trigram = gen.get("distinct_trigram_ratio")
    if not finite_number(trigram):
        problems.append(f"distinct_trigram_ratio={trigram!r} unmeasured")
    elif float(trigram) < DISTINCT_TRIGRAM_FLOOR:
        problems.append(f"distinct_trigram_ratio={float(trigram):.3f} < {DISTINCT_TRIGRAM_FLOOR}")

    # Obvious looping
    looping = gen.get("looping_prompts")
    if not finite_number(looping):
        problems.append(f"looping_prompts={looping!r} unmeasured")
    elif float(looping) > LOOPING_PROMPTS_CEILING:
        problems.append(f"looping_prompts={float(looping):.0f} > {LOOPING_PROMPTS_CEILING}")

    # Compression ratio and task score are reported but not gated
    if problems:
        verdict.add("T10", "generation sanity", FAIL, "; ".join(problems[:6]))
    else:
        detail_parts = []
        for key in ("termination_rate", "max_token_cap_rate", "distinct_trigram_ratio", "looping_prompts"):
            val = gen.get(key)
            if finite_number(val):
                detail_parts.append(f"{key}={float(val):.3f}")
        verdict.add("T10", "generation sanity", PASS,
                    "all targets met: " + ", ".join(detail_parts))


def check_reproduction(evals: list, verdict: Verdict) -> None:
    """Info, not a threshold: identical inputs should reproduce rung-3b."""
    if len(evals) != 1:
        return
    metrics = evals[0][1].get("metrics") or {}
    base = evals[0][1].get("base_holdout_loss")
    cand = evals[0][1].get("candidate_holdout_loss")
    dead = metrics.get("dead_experts")
    notes = []
    if finite_number(base) and abs(float(base) - REPRO_BASE_LOSS) > REPRO_TOLERANCE:
        notes.append(f"base {float(base):.4f} vs rung-3b {REPRO_BASE_LOSS}")
    if finite_number(cand) and abs(float(cand) - REPRO_CANDIDATE_LOSS) > REPRO_TOLERANCE:
        notes.append(f"candidate {float(cand):.4f} vs rung-3b {REPRO_CANDIDATE_LOSS}")
    if finite_number(dead) and abs(float(dead) - REPRO_DEAD_EXPERTS) > 0.5:
        notes.append(f"dead_experts {float(dead):.0f} vs rung-3b {REPRO_DEAD_EXPERTS:.0f}")
    if notes:
        verdict.add("INFO", "reproduction of rung-3b arithmetic", UNKNOWN,
                    "identical workload drifted: " + "; ".join(notes) + " --- investigate before certifying quality claims")
    else:
        verdict.add("INFO", "reproduction of rung-3b arithmetic", PASS,
                    f"base {float(base):.4f}, candidate {float(cand):.4f}, dead experts {float(dead):.0f} --- matches rung-3b")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_root = Path(argv[1]).resolve()
    verdict = Verdict()
    registry_path, train, evals = discover(run_root)
    registry = open_registry_readonly(registry_path)
    check_t1_validate_before_train(run_root, verdict)
    check_t2_policy_contract(train, evals, verdict)
    check_t3_measured_preflight(train, evals, verdict)
    check_t4_trainability(train, verdict)
    check_t4a_routing_collapse(train, verdict)
    check_t5_horizon(train, verdict)
    check_t6_paired_gate_contract(run_root, evals, registry, verdict)
    check_t7_accounting(train, evals, registry, verdict)
    check_t8_identity_chain(run_root, train, evals, verdict)
    check_t9_independent_holdout(train, evals, verdict)
    check_t10_generation_sanity(train, evals, verdict)
    check_reproduction(evals, verdict)
    return report(run_root, verdict, train, evals, argv_len_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
