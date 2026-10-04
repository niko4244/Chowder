#!/usr/bin/env python3
"""Mechanical judge for the P11 rung-4c evaluation-protocol qualification (2026-09-16).

Scores the run's durable artifacts against each threshold of
``docs/quals/P11_RUNG4C_PREREG_2026-09-16.md`` as committed BEFORE the run.
Read-only: opens the registry with SQLite read-only URIs, hashes corpus files
without writing anything, and never mutates the run directory.

Usage::

    python judge_rung4c_2026-09-16.py <run-root>

``<run-root>`` is the directory that contains ``runs.db`` and ``.chowder/`` (the
project's ``work`` tree); the prereg/pin files are looked for there and, failing
that, beside it.

Verdict rules: every threshold is PASS, FAIL, or UNKNOWN. An artifact that is
missing or unreadable is UNKNOWN, never an assumed pass. The exit code is 0 only
when every threshold is PASS; any FAIL or UNKNOWN refuses certification.

What makes this judge different from the rung-4/4b judges:

* T4a is a **reproduction** threshold, not a progress threshold: the census must
  reproduce rung 3c's 431 exactly, with the untrained control reproducing 480.
  No routing-progress claim is made or judged here.
* T11 is new: the trained tensor file must be bit-identical to rung 3c's.
  The manifest hash is deliberately NOT gated -- it embeds ``spec_digest`` and the
  declared budget, so it must differ by construction.
* The horizon is 12 steps / 1536 tokens.
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
GEN_PROBE_PROMPTS_SHA256 = (
    "9db01036879af115635d2aa5452a3a43915e47c580ebff90b6e79a5db1ed1219"
)
CORPUS_FILENAME = "router-corpus-9b-pilot.txt"
HOLDOUT_FILENAME = "router-holdout-independent-9b.txt"
GEN_PROBE_FILENAME = "router-gen-probe-prompts.txt"

LOAD_POLICY = "bf16-offload-transient"
LOAD_DTYPE = "torch.bfloat16"
EXPECTED_GATES = 32
EXPERT_SLOTS = 512
EXPECTED_STEPS = 12
EXPECTED_TOKENS = 1536  # 12 steps x 64 seq x 2 batch
EXPECTED_PARAMETER_COUNT = 2097152  # 32 gates x 16 experts x 4096 hidden

MEMORY_LINE_GB = 14.5
STEP_COST_CEILING_S = 3.5
MAX_LOAD_SECONDS = 20.0

# Budget derived from MEASUREMENT (rung 4, 2026-09-16), not estimated:
#   loads   2 x (12.820 + 12.945) s = 25.765 s = 0.007157 GPU-h -> 0.0110 (1.537x)
#   steps   12 x 2.8976779 s        = 34.772 s = 0.009659 GPU-h -> 0.0145 (1.501x)
#   gen     both arms (unchanged probe, byte-identical pins)
#                                   = 182.127 s = 0.050591 GPU-h -> 0.0759 (1.500x)
#   sum                             = 0.1014 exactly
#   unmargined total 0.067425 GPU-h; 0.1014 / 0.067425 = 1.504x
GPU_HOUR_CEILING = 0.1014
SUBBUDGET_LOADS = 0.0110
SUBBUDGET_STEPS = 0.0145
SUBBUDGET_GENERATIONS = 0.0759
GOAL_ENVELOPE = 0.2028  # device ceiling x M=2.0
MEASURED_TRAIN_WALL_MULTIPLIER = 1.3343

# T4a: census reproduction (measured in the rung-3c re-census, 2026-09-16)
RUNG4_DEAD_BEFORE = 480        # the untrained control this run must reproduce
RUNG3C_DEAD_RECENSUS = 431     # rung 3c's trained payload on this corpus
RUNG4_DEAD_AFTER = 436         # rung 4's trained payload, reported for context
RUNG3C_DEAD_PILOT_CORPUS = 428  # cross-corpus reference only, reported

# T11: stack reproduction
RUNG3C_TENSOR_FILE_SHA256 = (
    "cceaedd792ee8f0eae7f2f6bc46a2208f98ff5654b5b0cd665979550a4a83792"
)
RUNG3C_MANIFEST_SHA256 = (
    "a43b4d962ecc739f958ad677d7d49f2ce3376d49f1c36ffbc4d2737b3435aa01"
)

# T10: generation sanity thresholds (unchanged across rungs)
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
    lowered = log.read_text(encoding="utf-8", errors="replace").lower()
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
        policy_report = result.get("load_policy_report")
        if not isinstance(policy_report, dict):
            problems.append(f"{path.parent.name}: no load_policy_report")
            continue
        checked += 1
        if policy_report.get("policy") != LOAD_POLICY:
            problems.append(f"{path.parent.name}: policy={policy_report.get('policy')!r}")
        if policy_report.get("dtype") != LOAD_DTYPE:
            problems.append(f"{path.parent.name}: dtype={policy_report.get('dtype')!r}")
        census = policy_report.get("placement_census")
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
        verdict.add("T2", "policy contract", UNKNOWN,
                    "no worker result carried a load policy report")
        return
    if problems:
        verdict.add("T2", "policy contract", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T2", "policy contract", PASS,
                    f"{checked} worker results carry {LOAD_POLICY}, {LOAD_DTYPE}, "
                    "verified CPU-expert placement")


def check_t3_measured_preflight(train: list, evals: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("T3", "measured preflight", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    preflight = train[0][1].get("device_preflight")
    if not isinstance(preflight, dict):
        verdict.add("T3", "measured preflight", FAIL,
                    "training result carries no device_preflight")
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
        if not finite_number(load.get("load_gpu_hours")):
            problems.append(f"load_gpu_hours={load.get('load_gpu_hours')!r} unmeasured")
    for path, result in evals:
        eval_load = result.get("load_budget") if isinstance(result.get("load_budget"), dict) else {}
        if not eval_load or eval_load.get("measured") is not True:
            problems.append(f"{path.parent.name}: eval load_budget missing or unmeasured")
        elif eval_load.get("would_exceed_load_budget") is not False:
            problems.append(f"{path.parent.name}: eval load budget exceeded")
    if problems:
        verdict.add("T3", "measured preflight", FAIL, "; ".join(problems[:6]))
    else:
        verdict.add("T3", "measured preflight", PASS,
                    "memory, step-cost and model-load projections measured before step 1, "
                    f"none exceeding (ceiling {GPU_HOUR_CEILING} GPU-h)")


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
    incomplete: list[str] = []
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
            problems.append(
                f"{name}: {len(nonzero)} nonzero steps outside 0..{EXPECTED_STEPS - 1}"
            )
        updates = component.get("update_steps")
        if not isinstance(updates, list) or not updates:
            problems.append(f"{name}: no recorded optimizer update")
        elif len(updates) < EXPECTED_STEPS:
            incomplete.append(f"{name}({len(updates)}/{EXPECTED_STEPS})")
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
                f"{len(components)}/{EXPECTED_GATES} gates trainable with finite non-zero "
                "gradients and real updates; frozen changed={}, ok")
    if incomplete:
        verdict.add("INFO", "per-step update coverage (reported)", UNKNOWN,
                    f"{len(incomplete)} gate(s) did not change on every step: "
                    f"{', '.join(incomplete[:4])}")


def check_t4a_census_reproduction(train: list, verdict: Verdict) -> None:
    """T4a: census reproduction and saturation.

    A reproduction threshold, not a progress threshold. The progress bar
    (``< 431``) belongs to runs that claim routing improvement; a 12-step replay
    cannot be expected to beat the 12-step predecessor, so gating it here would
    gate a bar existing measurement says is unreachable.
    """
    if len(train) != 1:
        verdict.add("T4a", "census reproduction", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    metrics = result.get("metrics") or {}
    trainability = result.get("trainability") or {}
    components = trainability.get("components") or {}

    problems: list[str] = []

    # (census) the measurement basis must be declared, not inferred
    for field in (
        "census_basis",
        "census_corpus_sha256",
        "census_blocks_used",
        "census_blocks_available",
        "expert_slots",
    ):
        if metrics.get(field) in (None, ""):
            problems.append(f"census field {field!r} missing")
    corpus_sha = metrics.get("census_corpus_sha256")
    if corpus_sha is not None and corpus_sha != INDEPENDENT_HOLDOUT_SHA256:
        problems.append(
            f"census_corpus_sha256={str(corpus_sha)[:16]}... != the independent-holdout pin"
        )
    slots = metrics.get("expert_slots")
    if finite_number(slots) and int(slots) != EXPERT_SLOTS:
        problems.append(f"expert_slots={slots!r} != {EXPERT_SLOTS}")

    dead = metrics.get("dead_experts_after")
    dead_before = metrics.get("dead_experts_before")

    # (a) the control: the comparison basis must be live
    if not finite_number(dead_before):
        problems.append(f"dead_experts_before={dead_before!r} unmeasured (control missing)")
    elif int(dead_before) != RUNG4_DEAD_BEFORE:
        problems.append(
            f"the untrained control reads {int(dead_before)} instead of rung 4's measured "
            f"{RUNG4_DEAD_BEFORE}: the census instrument or corpus drifted, so the "
            f"reproduction against rung 3c's {RUNG3C_DEAD_RECENSUS} is invalid"
        )

    # (b) the reproduction itself
    if not finite_number(dead):
        problems.append(f"dead_experts_after={dead!r} unmeasured")
    elif int(dead) != RUNG3C_DEAD_RECENSUS:
        problems.append(
            f"dead_experts_after={int(dead)} did not reproduce rung 3c's "
            f"{RUNG3C_DEAD_RECENSUS} on this corpus with this instrument "
            "(reproduction requires equality)"
        )

    # (c) still strictly de-collapsed from the untrained state
    if finite_number(dead) and finite_number(dead_before) and not (float(dead) < float(dead_before)):
        problems.append(
            f"dead_experts_after={float(dead):.0f} is not strictly below "
            f"dead_experts_before={float(dead_before):.0f}: the router did not de-collapse"
        )

    # (d) full-horizon saturation needs named evidence
    saturated: list[str] = []
    grad_zero_layers: list[str] = []
    incomplete_layers: list[str] = []
    for name, component in components.items():
        if not isinstance(component, dict):
            continue
        if "grad-zero" in (component.get("gradient_states") or []):
            grad_zero_layers.append(name)
        updates = component.get("update_steps")
        if not isinstance(updates, list) or len(updates) < EXPECTED_STEPS:
            incomplete_layers.append(name)
        if component.get("grad_zero_steps") == EXPECTED_STEPS:
            saturated.append(name)

    if problems:
        verdict.add("T4a", "census reproduction", FAIL, "; ".join(problems[:6]))
        return

    detail = (
        f"untrained control {int(dead_before)} reproduced; rung 3c's "
        f"{RUNG3C_DEAD_RECENSUS} reproduced exactly (dead_experts_after={int(dead)} "
        f"of {EXPERT_SLOTS}, strictly de-collapsed); census basis "
        f"{metrics.get('census_basis')!r} on {str(corpus_sha)[:12]}..., blocks "
        f"{metrics.get('census_blocks_used')}/{metrics.get('census_blocks_available')}"
    )
    if saturated:
        verdict.add("INFO", "saturated gates (grad-zero on the full horizon)", UNKNOWN,
                    f"{len(saturated)} gate(s) saturated over all {EXPECTED_STEPS} steps: "
                    f"{', '.join(saturated[:3])} --- topology-vs-optimizer evidence must be "
                    "named in the result doc or this reports UNKNOWN")
    verdict.add("INFO", "saturation and update coverage (reported, not gated at 12 steps)",
                UNKNOWN,
                f"layers_with_grad_zero={len(grad_zero_layers)}/{EXPECTED_GATES}, "
                f"incomplete_update_coverage={len(incomplete_layers)}/{EXPECTED_GATES}; "
                "the <=2 bars are 48-step improvement bars and are not claimed by a "
                "12-step replay (rung 3c recorded 3/32, rung 4 measured 4/32)")
    verdict.add("INFO", "comparison basis (reported, not gated here)", UNKNOWN,
                "like-for-like on this corpus with this instrument: untrained "
                f"{RUNG4_DEAD_BEFORE} -> rung 3c(12 steps) {RUNG3C_DEAD_RECENSUS} -> "
                f"rung 4(48 steps) {RUNG4_DEAD_AFTER}. The progress bar (< "
                f"{RUNG3C_DEAD_RECENSUS}) is not gated here: this rung makes no "
                "routing-progress claim. Rung 3c's cross-corpus "
                f"{RUNG3C_DEAD_PILOT_CORPUS} is a reference only")
    verdict.add("T4a", "census reproduction", PASS, detail)


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
                    f"{steps}/{EXPECTED_STEPS} steps, stop_reason=max_steps, "
                    f"{tokens} tokens exactly the fixed workload")


def check_t6_paired_gate_contract(evals: list, registry, verdict: Verdict) -> None:
    if len(evals) != 1:
        names = ", ".join(sorted(path.parent.name for path, _ in evals)) or "none"
        verdict.add("T6", "paired gate contract", FAIL,
                    f"expected exactly one evaluation worker (the resident pair), found "
                    f"{len(evals)}: {names}"
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
        generation_phase = _phase(result, phase_name)
        if generation_phase.get("measured") is not True or not generation_phase.get("seconds"):
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
                f"one resident pair at {path.parent.name}: arm=paired, base "
                f"{float(base_loss):.4f} before apply, one measured load, both generations "
                "measured, baseline row completed from paired evidence")


def check_t7_accounting(train: list, evals: list, registry, verdict: Verdict) -> None:
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
        problems.append(f"total {total:.7f} exceeds the measured-basis ceiling {GPU_HOUR_CEILING}")
    loads = 0.0
    for _, result in train + evals:
        load_phase = _phase(result, "model_load")
        gpu = load_phase.get("gpu_hours")
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
    detail_parts.append(f"generations+publication {rest:.7f}/{SUBBUDGET_GENERATIONS}")
    if rest > SUBBUDGET_GENERATIONS:
        problems.append(
            f"generation/publication sub-budget exceeded: {rest:.7f} > {SUBBUDGET_GENERATIONS}"
        )
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
        f"wall charges {wall_charges:.4f} GPU-h inside the {GOAL_ENVELOPE} envelope "
        f"(M=2.0; measured training-leg M was {MEASURED_TRAIN_WALL_MULTIPLIER})"
        if wall_charges <= GOAL_ENVELOPE
        else f"wall charges {wall_charges:.4f} GPU-h EXCEED the {GOAL_ENVELOPE} envelope "
             "(recorded, per the prereg)"
    )
    if problems:
        verdict.add("T7", "accounting", FAIL,
                    "; ".join(problems[:6]) + " | " + "; ".join(detail_parts))
        return
    if wall_charges > GOAL_ENVELOPE:
        verdict.add("T7", "accounting", UNKNOWN,
                    "measured device time is within every budget but the wall-charge "
                    "exceedance needs human recording: " + envelope_note)
        return
    verdict.add("T7", "accounting", PASS,
                "; ".join(detail_parts)
                + f"; incidents 0, no stranded results; {envelope_note}")


def check_t8_identity_chain(run_root: Path, train: list, evals: list, verdict: Verdict) -> None:
    if len(train) != 1 or len(evals) != 1:
        verdict.add("T8", "identity chain", UNKNOWN,
                    "identity chain needs exactly one train and one eval worker result")
        return
    problems: list[str] = []
    content_hashes: set[str] = set()
    for path, result in train + evals:
        identity = result.get("base_identity") or {}
        if identity.get("manifest_sha256") != BASE_MANIFEST_SHA256:
            problems.append(
                f"{path.parent.name}: manifest hash "
                f"{str(identity.get('manifest_sha256', ''))[:16]}... != pinned"
            )
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
    for filename, pin, label in (
        (CORPUS_FILENAME, TRAINING_CORPUS_SHA256, "training corpus"),
        (HOLDOUT_FILENAME, INDEPENDENT_HOLDOUT_SHA256, "independent holdout corpus"),
        (GEN_PROBE_FILENAME, GEN_PROBE_PROMPTS_SHA256, "generation probe prompt set"),
    ):
        path = run_root.parent / filename
        if not path.is_file():
            path = run_root / filename
        if path.is_file():
            if sha256_file(path) != pin:
                problems.append(f"{label} on disk hashes differently than pinned")
        else:
            problems.append(f"{label} file not found beside the run root; pin unverified")
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
                    "full-mode local-content identity matches the pinned manifest in both "
                    "workers (the eval-side derivation is post-training); payload bound to the "
                    "re-derived base content; corpora and probe prompt set hash to their pins")


def check_t9_independent_holdout(evals: list, registry, verdict: Verdict) -> None:
    if len(evals) != 1:
        verdict.add("T9", "independent holdout", UNKNOWN,
                    f"expected exactly one eval worker, found {len(evals)}")
        return
    result = evals[0][1]
    holdout = result.get("holdout") or {}
    if holdout.get("sha256") != INDEPENDENT_HOLDOUT_SHA256:
        verdict.add("T9", "independent holdout", FAIL,
                    f"eval holdout sha256={str(holdout.get('sha256'))[:16]}..., expected the "
                    "independent holdout pin")
        return
    problems: list[str] = []
    if result.get("arm") != "paired":
        problems.append(f"arm={result.get('arm')!r}, so the baseline has no resident-pair source")
    control = result.get("application_control") or {}
    if not control.get("applied_parameters"):
        problems.append("no payload was applied after the base score, so the 'before' arm is unproven")
    if registry is not None:
        try:
            evidence_rows = list(registry.execute(
                "SELECT evidence_json FROM results WHERE experiment_id='baseline'"
            ))
            if not evidence_rows:
                problems.append("baseline has no recorded result: the baseline is not frozen")
            else:
                evidence = load_json_safe(evidence_rows[0][0]) or {}
                if evidence.get("baseline_source") != "paired-candidate-evaluation":
                    problems.append(
                        f"baseline_source={evidence.get('baseline_source')!r}, expected the "
                        "resident pair"
                    )
                if not finite_number(evidence.get("base_holdout_loss")):
                    problems.append("baseline evidence carries no numeric base_holdout_loss")
        except sqlite3.Error as exc:
            problems.append(f"registry unreadable: {exc}")
    base = result.get("base_holdout_loss")
    cand = result.get("candidate_holdout_loss")
    base_ind = result.get("baseline_independent_holdout_loss")
    cand_ind = result.get("candidate_independent_holdout_loss")
    if finite_number(base_ind):
        base = base_ind
    if finite_number(cand_ind):
        cand = cand_ind
    if not finite_number(base):
        problems.append(f"baseline holdout loss={base!r} unmeasured")
    if not finite_number(cand):
        problems.append(f"candidate holdout loss={cand!r} unmeasured")
    if problems:
        verdict.add("T9", "independent holdout", FAIL, "; ".join(problems[:6]))
        return
    base_f = float(base)
    cand_f = float(cand)
    delta = cand_f - base_f
    if cand_f >= base_f:
        verdict.add("T9", "independent holdout", FAIL,
                    f"candidate {cand_f:.4f} >= baseline {base_f:.4f} "
                    f"(delta={delta:+.4f}; equality and regression both fail)")
    else:
        verdict.add("T9", "independent holdout", PASS,
                    f"paired base arm {base_f:.4f} vs candidate {cand_f:.4f} "
                    f"(delta={delta:+.4f}, strictly improved); baseline frozen from the "
                    "resident pair before payload application")


def check_t10_generation_sanity(evals: list, verdict: Verdict) -> None:
    if len(evals) != 1:
        verdict.add("T10", "generation sanity", UNKNOWN,
                    f"expected exactly one eval worker, found {len(evals)}")
        return
    gen = evals[0][1].get("generation_sanity") or {}
    if not gen:
        verdict.add("T10", "generation sanity", UNKNOWN,
                    "no generation_sanity data in eval worker result")
        return
    problems: list[str] = []
    checks = (
        ("termination_rate", TERMINATION_RATE_FLOOR, "min"),
        ("max_token_cap_rate", MAX_TOKEN_CAP_CEILING, "max"),
        ("distinct_trigram_ratio", DISTINCT_TRIGRAM_FLOOR, "min"),
        ("looping_prompts", LOOPING_PROMPTS_CEILING, "max"),
    )
    for key, bound, direction in checks:
        value = gen.get(key)
        if not finite_number(value):
            problems.append(f"{key}={value!r} unmeasured")
        elif direction == "min" and float(value) < bound:
            problems.append(f"{key}={float(value):.3f} < {bound}")
        elif direction == "max" and float(value) > bound:
            problems.append(f"{key}={float(value):.3f} > {bound}")
    if problems:
        verdict.add("T10", "generation sanity", FAIL, "; ".join(problems[:6]))
        return
    detail_parts = []
    for key in ("termination_rate", "max_token_cap_rate", "distinct_trigram_ratio",
                "looping_prompts", "compression_ratio"):
        value = gen.get(key)
        if finite_number(value):
            detail_parts.append(f"{key}={float(value):.3f}")
    verdict.add("T10", "generation sanity", PASS,
                "all targets met: " + ", ".join(detail_parts)
                + " (loss is never substituted for generation sanity)")


def check_t11_stack_reproduction(train: list, verdict: Verdict) -> None:
    """T11: does the training stack reproduce rung 3c's tensors bit-for-bit?

    The tensor file is the test. The manifest hash embeds ``spec_digest`` and the
    declared budget, so it must differ by construction and is reported only.
    """
    if len(train) != 1:
        verdict.add("T11", "stack reproduction", UNKNOWN,
                    f"expected exactly one training worker result, found {len(train)}")
        return
    payload = train[0][1].get("payload")
    if not isinstance(payload, dict):
        verdict.add("T11", "stack reproduction", FAIL,
                    "the training worker published no payload")
        return
    problems: list[str] = []
    steps = payload.get("steps_completed")
    if steps != EXPECTED_STEPS:
        problems.append(f"payload steps_completed={steps!r}, expected {EXPECTED_STEPS}")
    if payload.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        problems.append(
            f"parameter_count={payload.get('parameter_count')!r}, expected "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    names = payload.get("parameter_names")
    if not isinstance(names, list) or len(names) != EXPECTED_GATES:
        problems.append(
            f"parameter_names count={len(names) if isinstance(names, list) else names!r}, "
            f"expected {EXPECTED_GATES}"
        )
    tensor_sha = payload.get("tensor_file_sha256")
    if not tensor_sha:
        problems.append("no tensor_file_sha256 recorded")
    elif tensor_sha != RUNG3C_TENSOR_FILE_SHA256:
        problems.append(
            f"tensor_file_sha256={str(tensor_sha)[:16]}... != rung 3c's "
            f"{RUNG3C_TENSOR_FILE_SHA256[:16]}...: the training stack is NOT bit-reproducible "
            "at this horizon"
        )
    manifest_sha = payload.get("manifest_sha256")
    verdict.add("INFO", "manifest hash (informational only)", UNKNOWN,
                f"manifest_sha256={str(manifest_sha)[:16]}... vs rung 3c's "
                f"{RUNG3C_MANIFEST_SHA256[:16]}... --- the manifest embeds spec_digest and the "
                "declared budget, so it must differ; only the tensors are gated")
    if problems:
        verdict.add("T11", "stack reproduction", FAIL, "; ".join(problems[:6]))
        return
    verdict.add("T11", "stack reproduction", PASS,
                f"the trained tensor file is bit-identical to rung 3c's "
                f"({RUNG3C_TENSOR_FILE_SHA256[:16]}...): {EXPECTED_GATES} gates, "
                f"{EXPECTED_PARAMETER_COUNT} parameters, {EXPECTED_STEPS} steps, replaying "
                "rung 3c's exact training inputs")


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
    check_t4a_census_reproduction(train, verdict)
    check_t5_horizon(train, verdict)
    check_t6_paired_gate_contract(evals, registry, verdict)
    check_t7_accounting(train, evals, registry, verdict)
    check_t8_identity_chain(run_root, train, evals, verdict)
    check_t9_independent_holdout(evals, registry, verdict)
    check_t10_generation_sanity(evals, verdict)
    check_t11_stack_reproduction(train, verdict)
    return report(run_root, verdict, train, evals, argv_len_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
