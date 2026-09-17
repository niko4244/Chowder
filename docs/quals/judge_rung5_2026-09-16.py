#!/usr/bin/env python3
"""Mechanical judge for the P11 rung-5 router-numerics fix (2026-09-16).

Scores the campaign's durable artifacts against every threshold of
``docs/quals/P11_RUNG5_PREREG_2026-09-16.md`` as committed BEFORE the
implementation and the run. Read-only: each arm's registry is opened with a
SQLite read-only URI, the corpus files are hashed without writing, and the
campaign directory is never mutated.

Usage::

    python judge_rung5_2026-09-16.py <campaign-root>

``<campaign-root>`` holds ``arm-A``..``arm-D``, each of which is a run root
containing ``runs.db`` and ``.chowder/``.

Verdict rules: every threshold is PASS, FAIL, or UNKNOWN. An artifact that is
missing or unreadable is UNKNOWN, never an assumed pass. The exit code is 0
only when every gating threshold on every arm is PASS; any FAIL or UNKNOWN
refuses certification.

This judge is the first whose contract requires the **saturation instrument**
(prereg section 4). A run carrying only the historical binary grad-nonzero
record cannot satisfy N1-N3, and must not be read as passing by absence.
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

# ---- Pins from the preregistration (fixed before the run) --------------------

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
EXPECTED_TOKENS = 6144  # 48 steps x 64 seq x 2 batch (rung 4's measured value)
EXPECTED_SAMPLES = 96
FROZEN_PARAMETERS = 523

MEMORY_LINE_GB = 14.5
STEP_COST_CEILING_S = 3.5

# Ceiling and decomposition: measured end to end, and the three parts sum to the
# ceiling exactly (0.0057 + 0.0610 + 0.0043 == 0.0710 in double precision).
GPU_HOUR_CEILING = 0.0710
SUBBUDGET_LOADS = 0.0057
SUBBUDGET_STEPS = 0.0610
SUBBUDGET_GENERATIONS = 0.0043
GOAL_ENVELOPE_PER_ARM = 0.1200

# The instrument's declared geometry and derived bars.
INSTRUMENT_VERSION = "saturation.v1"
GATE_INIT_VERSION = "gate-init.v1"
ROWS_PER_STEP = 128  # batch_size 2 x seq_len 64
ONE_HOT_GAP_THRESHOLD = 92.4  # bf16 exp-underflow gap; fp32 is ~103.3
INIT_SOFTMAX_INPUT_BOUND = 10.0
# Calibration of the liveness test against the measured counterfactual:
# gap 0 -> implied_update ~ 5.0e-02 vs half_ulp ~ 5.7e-04 (moves);
# gap 132 -> ~6.1e-10 vs the same half_ulp (does not move, ~9 orders below).
LIVENESS_MARGIN_DECADES = 1.0

# Control and comparison pins.
DEAD_EXPERTS_UNTRAINED_TIE_BREAK = 480  # reproduced exactly by two processes
DEAD_EXPERTS_RUNG4_AFTER = 436
CAPABILITY_GUARD_RATIO = 1.10

CONTROL_ARM = "arm-A"
FIX_ARM = "arm-D"

# The frozen arm assignment (prereg section 3.6). The judge reads each arm's
# run spec and requires it to match, so a 2x2 cannot silently become four
# copies of one configuration.
ARM_KNOBS: dict[str, dict[str, object]] = {
    "arm-A": {"gate_initialization": "artifact", "router_logit_scale": None, "router_logit_soft_cap": None},
    "arm-B": {"gate_initialization": "small_normal", "router_logit_scale": None, "router_logit_soft_cap": None},
    "arm-C": {"gate_initialization": "artifact", "router_logit_scale": 128.0, "router_logit_soft_cap": 30.0},
    "arm-D": {"gate_initialization": "small_normal", "router_logit_scale": 128.0, "router_logit_soft_cap": 30.0},
}
ARMS = tuple(sorted(ARM_KNOBS))


# ---- Helpers -----------------------------------------------------------------


def arm_roots(campaign_root: Path) -> dict[str, Path]:
    return {arm: campaign_root / arm for arm in ARMS}


def gates_of(train_result: dict) -> dict[str, dict]:
    """The 32 trainable gate records, keyed by parameter name."""
    components = (train_result.get("trainability") or {}).get("components") or {}
    return {
        name: entry
        for name, entry in components.items()
        if isinstance(entry, dict) and name.endswith(".mlp.gate.weight")
    }


def instrument_of(train_result: dict) -> dict:
    block = (train_result.get("trainability") or {}).get("instrument")
    return block if isinstance(block, dict) else {}


def series(entry: dict, key: str, expected_len: int) -> tuple[list | None, str | None]:
    """A per-step series of exactly ``expected_len`` entries, or a problem."""
    value = entry.get(key)
    if not isinstance(value, list):
        return None, f"{key} missing or not a list"
    if len(value) != expected_len:
        return None, f"{key} has {len(value)} entries, expected {expected_len}"
    return value, None


def numbers(values: list) -> bool:
    return all(finite_number(value) for value in values)


def booleans(values: list) -> bool:
    return all(isinstance(value, bool) for value in values)


def span(entries: list[str], limit: int = 6) -> str:
    return "; ".join(entries[:limit]) + ("" if len(entries) <= limit else f" (+{len(entries) - limit} more)")


def load_arm(campaign_root: Path, arm: str) -> tuple[Path, list, list, object]:
    root = campaign_root / arm
    registry_path, train, evals = discover(root)
    return root, train, evals, open_registry_readonly(registry_path)


# ---- T1: validate before train ----------------------------------------------


def check_t1(root: Path, arm: str, verdict: Verdict) -> None:
    label = f"T1/{arm}"
    log = root / "project-validate-stdout.log"
    project = root / "project.json"
    problems: list[str] = []
    if not project.is_file():
        problems.append("arm carries no frozen project.json")
    if not log.is_file():
        problems.append("no project-validate-stdout.log: validate-before-train unproven")
    else:
        text = log.read_text(encoding="utf-8", errors="replace")
        if "project-valid" not in text.lower() and "ok" not in text.lower():
            problems.append("validate log does not report a successful validation")
    if problems:
        verdict.add(label, "validate before train", FAIL if log.is_file() or project.is_file() else UNKNOWN,
                    "; ".join(problems))
        return
    verdict.add(label, "validate before train", PASS,
                "project.json present and project-validate reported success before training")


# ---- T2: policy contract and the frozen arm assignment -----------------------


def check_t2(train: list, evals: list, arm: str, verdict: Verdict) -> None:
    label = f"T2/{arm}"
    if len(train) != 1:
        verdict.add(label, "policy contract", UNKNOWN,
                    f"needs exactly one training worker result, found {len(train)}")
        return
    problems: list[str] = []
    for kind, results in (("train", train), ("eval", evals)):
        for path, result in results:
            report_block = result.get("load_policy_report") or {}
            if report_block.get("policy") != LOAD_POLICY:
                problems.append(f"{kind}: policy={report_block.get('policy')!r}")
            if report_block.get("dtype") != LOAD_DTYPE:
                problems.append(f"{kind}: dtype={report_block.get('dtype')!r}")
            if report_block.get("transient_forward_installed") is not True:
                problems.append(f"{kind}: transient forward not installed")
            census = report_block.get("placement_census") or {}
            if census.get("verified") is not True:
                problems.append(f"{kind}: placement census unverified")
            if census.get("expert_params_on_device") != 0:
                problems.append(f"{kind}: {census.get('expert_params_on_device')} expert params on device")
            if census.get("gate_params_on_device") != EXPECTED_GATES:
                problems.append(
                    f"{kind}: {census.get('gate_params_on_device')}/{EXPECTED_GATES} gate params on device"
                )
    # The arm assignment must be the configuration the worker actually ran.
    declarable = ARM_KNOBS[arm]
    spec_path = train[0][0].parent / "run-spec.json"
    spec = load_json(spec_path)
    if not isinstance(spec, dict):
        problems.append("no readable run-spec.json beside the training worker result")
    else:
        for knob, expected in declarable.items():
            actual = spec.get(knob, "<absent>")
            if isinstance(expected, float) and finite_number(actual):
                if abs(float(actual) - expected) > 1e-9:
                    problems.append(f"{knob}={actual!r}, arm declares {expected!r}")
            elif actual != expected:
                problems.append(f"{knob}={actual!r}, arm declares {expected!r}")
    if problems:
        verdict.add(label, "policy contract", FAIL, span(problems))
        return
    verdict.add(label, "policy contract", PASS,
                f"bf16-offload-transient, torch.bfloat16, experts CPU-resident (0/64), 32/32 gates on device, "
                f"run spec matches the declared arm assignment {declarable}")


# ---- T3: measured preflight -------------------------------------------------


def check_t3(train: list, evals: list, arm: str, verdict: Verdict) -> None:
    label = f"T3/{arm}"
    if len(train) != 1:
        verdict.add(label, "measured preflight", UNKNOWN,
                    f"needs exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    preflight = result.get("device_preflight") or {}
    problems: list[str] = []
    probe = preflight.get("step_cost_probe") or {}
    if not preflight:
        problems.append("device_preflight absent")
    if preflight.get("projected_oom") is not False:
        problems.append(f"projected_oom={preflight.get('projected_oom')!r}")
    if probe.get("measured") is not True:
        problems.append("step-cost probe unmeasured")
    step_seconds = probe.get("step_seconds")
    if not finite_number(step_seconds):
        problems.append("step probe reports no cost")
    elif float(step_seconds) > STEP_COST_CEILING_S:
        problems.append(f"step {float(step_seconds):.3f} s exceeds the {STEP_COST_CEILING_S} s ceiling")
    if probe.get("would_exceed_budget") is not False:
        problems.append("step probe would exceed its budget")
    load_budget = result.get("load_budget") or {}
    if load_budget.get("measured") is not True:
        problems.append("load budget unmeasured")
    if load_budget.get("would_exceed_load_budget") is not False:
        problems.append("load budget refused")
    for kind, results in (("train", train), ("eval", evals)):
        for path, payload in results:
            ceiling = payload.get("run_ceiling") or {}
            if ceiling.get("measured") is not True:
                problems.append(f"{kind}: run ceiling unmeasured")
                continue
            if ceiling.get("would_exceed_ceiling") is not False:
                problems.append(f"{kind}: run ceiling exceeded")
            for category, entry in (ceiling.get("sub_budgets") or {}).items():
                if entry.get("would_exceed") is not False:
                    problems.append(f"{kind}: {category} sub-budget exceeded")
            peak = (payload.get("resource_usage") or {}).get("peak_vram_gb_by_accelerator") or {}
            worst = max((float(v) for v in peak.values() if finite_number(v)), default=None)
            if worst is not None and worst > MEMORY_LINE_GB:
                problems.append(f"{kind}: peak VRAM {worst:.3f} GB over the {MEMORY_LINE_GB} GB line")
    # Gate-initialization preflight, arm-scoped per prereg section 4.4.
    init = result.get("gate_initialization") or {}
    declarable = ARM_KNOBS[arm]
    if init.get("version") != GATE_INIT_VERSION:
        problems.append(f"gate_initialization.version={init.get('version')!r}")
    if init.get("mode") != declarable["gate_initialization"]:
        problems.append(f"gate_initialization.mode={init.get('mode')!r}, arm declares {declarable['gate_initialization']!r}")
    init_gates = init.get("gates") if isinstance(init.get("gates"), dict) else {}
    if len(init_gates) != EXPECTED_GATES:
        problems.append(f"gate_initialization reports {len(init_gates)}/{EXPECTED_GATES} gates")
    equality_reads: list[int] = []
    for name, entry in init_gates.items():
        equality = entry.get("initial_rows_with_exactly_equal_top2")
        absmax = entry.get("initial_softmax_input_absmax")
        if not isinstance(equality, int) or isinstance(equality, bool):
            problems.append(f"{name.split('.')[2]}.gate: no initial tie count")
            continue
        equality_reads.append(equality)
        if not finite_number(absmax):
            problems.append(f"{name.split('.')[2]}.gate: no initial softmax-input absmax")
    if declarable["gate_initialization"] == "small_normal":
        tied = [value for value in equality_reads if value > 0]
        if tied:
            problems.append(
                f"initialization did not break the degeneracy: {len(tied)} gates still tie on the top-2 logits"
            )
        saturated = [
            entry.get("initial_softmax_input_absmax")
            for entry in init_gates.values()
            if finite_number(entry.get("initial_softmax_input_absmax"))
            and float(entry["initial_softmax_input_absmax"]) > INIT_SOFTMAX_INPUT_BOUND
        ]
        if saturated:
            problems.append(
                f"{len(saturated)} gates start above the {INIT_SOFTMAX_INPUT_BOUND} softmax-input bound"
            )
    if problems:
        verdict.add(label, "measured preflight", FAIL, span(problems))
        return
    verdict.add(label, "measured preflight", PASS,
                f"step probe {float(step_seconds):.3f} s/step, load budget measured, both legs within the "
                f"0.0710 ceiling and every sub-budget, peak VRAM under {MEMORY_LINE_GB} GB, "
                f"{len(init_gates)} gate initializations recorded under mode {declarable['gate_initialization']!r}")


# ---- T4: the instrument contract is present and coherent --------------------


def check_t4(train: list, arm: str, verdict: Verdict) -> None:
    label = f"T4/{arm}"
    if len(train) != 1:
        verdict.add(label, "trainability contract", UNKNOWN,
                    f"needs exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    gates = gates_of(result)
    problems: list[str] = []
    instrument = instrument_of(result)
    if instrument.get("version") != INSTRUMENT_VERSION:
        problems.append(f"trainability.instrument.version={instrument.get('version')!r}")
    if instrument.get("gates_observed") != EXPECTED_GATES:
        problems.append(f"instrument gates_observed={instrument.get('gates_observed')!r}")
    if instrument.get("rows_per_step") != ROWS_PER_STEP:
        problems.append(f"instrument rows_per_step={instrument.get('rows_per_step')!r}")
    if not finite_number(instrument.get("adam_eps")):
        problems.append("instrument records no Adam epsilon; the liveness derivation is unbound")
    if not finite_number(instrument.get("learning_rate")):
        problems.append("instrument records no learning rate; the implied-update derivation is unbound")
    if not instrument.get("dtype"):
        problems.append("instrument records no dtype; the half-ULP derivation is unbound")
    if len(gates) != EXPECTED_GATES:
        problems.append(f"{len(gates)}/{EXPECTED_GATES} gate records present")
        verdict.add(label, "trainability contract", FAIL, span(problems))
        return
    for name, entry in sorted(gates.items()):
        short = name.split(".")[2]
        observed = entry.get("observed_steps")
        if observed != EXPECTED_STEPS:
            problems.append(f"L{short}: observed_steps={observed!r}")
            continue
        zero = entry.get("grad_zero_steps")
        nonzero = entry.get("grad_nonzero_steps")
        if not finite_number(zero) or not finite_number(nonzero) or int(zero) + int(nonzero) != EXPECTED_STEPS:
            problems.append(f"L{short}: gradient step counts inconsistent")
        steps = entry.get("nonzero_steps")
        if not isinstance(steps, list) or any(
            (not isinstance(step, int)) or isinstance(step, bool) or not 0 <= step < EXPECTED_STEPS for step in steps
        ):
            problems.append(f"L{short}: nonzero_steps malformed")
        for key in ("grad_absmax_by_step", "softmax_input_absmax_by_step", "top2_gap_max_by_step",
                    "implied_update_by_step", "half_ulp_by_step"):
            values, problem = series(entry, key, EXPECTED_STEPS)
            if problem:
                problems.append(f"L{short}: {problem}")
            elif key != "half_ulp_by_step" and not numbers(values):
                problems.append(f"L{short}: {key} carries non-finite entries")
        for key in ("rows_exactly_one_hot_by_step",):
            values, problem = series(entry, key, EXPECTED_STEPS)
            if problem:
                problems.append(f"L{short}: {problem}")
            elif any((not isinstance(v, int)) or isinstance(v, bool) for v in values):
                problems.append(f"L{short}: {key} must be integer counts")
            elif any(v < 0 or v > ROWS_PER_STEP for v in values):
                problems.append(f"L{short}: {key} outside 0..{ROWS_PER_STEP}")
        values, problem = series(entry, "gradient_moves_parameter_by_step", EXPECTED_STEPS)
        if problem:
            problems.append(f"L{short}: {problem}")
        elif not booleans(values):
            problems.append(f"L{short}: gradient_moves_parameter_by_step must be booleans")
        if entry.get("rows_total_per_step") != ROWS_PER_STEP:
            problems.append(f"L{short}: rows_total_per_step={entry.get('rows_total_per_step')!r}")
        for key in ("any_exactly_one_hot_steps", "fully_one_hot_steps",
                    "steps_with_gradient_not_moving_parameter"):
            if not isinstance(entry.get(key), list):
                problems.append(f"L{short}: {key} missing")
        # The derived liveness test must agree with its own inputs, so a run
        # cannot report a healthy flag beside a dead gradient.
        implied, _ = series(entry, "implied_update_by_step", EXPECTED_STEPS)
        half, _ = series(entry, "half_ulp_by_step", EXPECTED_STEPS)
        moves, _ = series(entry, "gradient_moves_parameter_by_step", EXPECTED_STEPS)
        if implied is not None and half is not None and moves is not None:
            for index, (imp, ulp, flag) in enumerate(zip(implied, half, moves)):
                if not finite_number(imp) or not finite_number(ulp):
                    continue
                derived = float(imp) > float(ulp)
                if derived != bool(flag):
                    problems.append(f"L{short}: step {index} liveness flag disagrees with implied_update vs half_ulp")
                    break
        grad, _ = series(entry, "grad_absmax_by_step", EXPECTED_STEPS)
        implied, _ = series(entry, "implied_update_by_step", EXPECTED_STEPS)
        if grad is not None and implied is not None and finite_number(instrument.get("adam_eps")) \
                and finite_number(instrument.get("learning_rate")):
            eps = float(instrument["adam_eps"])
            lr = float(instrument["learning_rate"])
            for index, (value, imp) in enumerate(zip(grad, implied)):
                if not finite_number(value) or not finite_number(imp) or eps <= 0:
                    continue
                expected = lr * float(value) / (float(value) + eps)
                if expected > 0 and abs(float(imp) / expected - 1.0) > 1e-6:
                    problems.append(f"L{short}: step {index} implied_update inconsistent with grad_absmax")
                    break
    if problems:
        verdict.add(label, "trainability contract", FAIL, span(problems))
        return
    verdict.add(label, "trainability contract", PASS,
                f"all {EXPECTED_GATES} gates carry the {INSTRUMENT_VERSION} instrument over {EXPECTED_STEPS} steps "
                f"({ROWS_PER_STEP} rows/step), the legacy counts are consistent, and every liveness flag agrees "
                "with implied_update vs half_ulp")


# ---- T4a: routing collapse, arm-scoped --------------------------------------


def check_t4a(train: list, arm: str, verdict: Verdict) -> None:
    label = f"T4a/{arm}"
    if len(train) != 1:
        verdict.add(label, "routing collapse", UNKNOWN,
                    f"needs exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    metrics = result.get("metrics") or {}
    gates = gates_of(result)
    problems: list[str] = []
    before = metrics.get("dead_experts_before")
    after = metrics.get("dead_experts_after")
    if metrics.get("expert_slots") != EXPERT_SLOTS:
        problems.append(f"expert_slots={metrics.get('expert_slots')!r}")
    if not finite_number(before) or not finite_number(after):
        problems.append("dead_experts_before/after unmeasured")
    if metrics.get("census_blocks_used") != 2:
        problems.append(f"census_blocks_used={metrics.get('census_blocks_used')!r}, expected 2")
    if metrics.get("census_corpus_sha256") != INDEPENDENT_HOLDOUT_SHA256:
        problems.append("census was not taken on the pinned independent holdout")
    if len(gates) != EXPECTED_GATES:
        problems.append(f"{len(gates)}/{EXPECTED_GATES} gate records")
    dead_layers = sorted(
        name.split(".")[2] for name, entry in gates.items() if int(entry.get("grad_zero_steps") or 0) >= 1
    )
    if arm == FIX_ARM and dead_layers:
        problems.append(f"layers_with_grad_zero={len(dead_layers)} must be 0 on the fixed arm: {dead_layers}")
    if arm == CONTROL_ARM and not dead_layers:
        problems.append("the control arm recorded no zero-gradient step at all: the latch did not reproduce")
    if problems:
        verdict.add(label, "routing collapse", FAIL, span(problems))
        return
    verdict.add(label, "routing collapse", PASS,
                f"dead experts {float(before):.0f} -> {float(after):.0f} of {EXPERT_SLOTS} on the pinned census "
                f"corpus, layers_with_grad_zero={len(dead_layers)}")


# ---- T5: horizon ------------------------------------------------------------


def check_t5(train: list, arm: str, verdict: Verdict) -> None:
    label = f"T5/{arm}"
    if len(train) != 1:
        verdict.add(label, "horizon", UNKNOWN, f"needs exactly one training worker result, found {len(train)}")
        return
    result = train[0][1]
    limits = result.get("limits") or {}
    problems: list[str] = []
    if result.get("steps_completed") != EXPECTED_STEPS:
        problems.append(f"steps_completed={result.get('steps_completed')!r}")
    if result.get("global_step") != EXPECTED_STEPS:
        problems.append(f"global_step={result.get('global_step')!r}")
    losses = result.get("losses")
    if not isinstance(losses, list):
        problems.append("losses missing")
    elif result.get("losses_truncated"):
        limit = result.get("loss_log_limit")
        if not finite_number(limit) or len(losses) != int(limit):
            problems.append("losses truncated without a declared limit")
    elif len(losses) != EXPECTED_STEPS:
        problems.append(f"{len(losses)} losses for {EXPECTED_STEPS} steps")
    if not finite_number(result.get("loss_first")) or not finite_number(result.get("loss_last")):
        problems.append("loss_first/loss_last unmeasured")
    if limits.get("max_tokens") != EXPECTED_TOKENS:
        problems.append(f"limits.max_tokens={limits.get('max_tokens')!r}")
    if limits.get("tokens_consumed") != EXPECTED_TOKENS:
        problems.append(f"limits.tokens_consumed={limits.get('tokens_consumed')!r}")
    if limits.get("samples_consumed") != EXPECTED_SAMPLES:
        problems.append(f"limits.samples_consumed={limits.get('samples_consumed')!r}")
    if limits.get("stop_reason") != "max_steps":
        problems.append(f"limits.stop_reason={limits.get('stop_reason')!r}, expected max_steps")
    if problems:
        verdict.add(label, "horizon", FAIL, span(problems))
        return
    verdict.add(label, "horizon", PASS,
                f"{EXPECTED_STEPS}/{EXPECTED_STEPS} steps, {EXPECTED_TOKENS} tokens, {EXPECTED_SAMPLES} samples, "
                f"stop_reason=max_steps, loss {float(result['loss_first']):.4f} -> {float(result['loss_last']):.4f}")


# ---- T6: paired gate contract and the wall-charge ledger --------------------


def check_t6(root: Path, evals: list, registry, arm: str, verdict: Verdict) -> None:
    label = f"T6/{arm}"
    if len(evals) != 1:
        names = ", ".join(sorted(path.parent.name for path, _ in evals)) or "none"
        verdict.add(label, "paired gate contract", FAIL if evals else UNKNOWN,
                    f"expected exactly one resident evaluation worker, found {len(evals)}: {names}")
        return
    path, result = evals[0]
    problems: list[str] = []
    if result.get("arm") != "paired":
        problems.append(f"arm={result.get('arm')!r}, expected 'paired'")
    if result.get("pair_error") is not None:
        problems.append(f"pair_error={result.get('pair_error')!r}")
    base_loss = result.get("base_holdout_loss")
    cand_loss = result.get("candidate_holdout_loss")
    if not finite_number(base_loss) or not finite_number(cand_loss):
        problems.append(f"holdout scores unmeasured (base={base_loss!r}, candidate={cand_loss!r})")
    delta = result.get("holdout_loss_delta")
    if finite_number(base_loss) and finite_number(cand_loss):
        if not finite_number(delta) or abs(float(delta) - (float(cand_loss) - float(base_loss))) > 1e-9:
            problems.append("holdout_loss_delta inconsistent with the two arm scores")
    control = result.get("application_control") or {}
    if not control.get("applied_parameters"):
        problems.append("no applied parameters: no evidence the candidate leg ran after the base score")
    if "logits_before" not in control or "logits_after" not in control:
        problems.append("the base arm's pre-application reading is not recorded")
    phase_name = "model_load"
    if _phase(result, phase_name).get("measured") is not True:
        problems.append("the pair's single model_load phase is missing or unmeasured")
    for generation in ("baseline_generation", "candidate_generation"):
        if _phase(result, generation).get("measured") is not True:
            problems.append(f"{generation} not measured inside the pair")
    verification = result.get("payload_verification")
    if not isinstance(verification, dict):
        problems.append("eval carries no payload_verification")
    else:
        if not verification.get("manifest_sha256"):
            problems.append("payload manifest hash missing")
        if verification.get("base_content_sha256") != (result.get("base_identity") or {}).get("content_sha256"):
            problems.append("payload is not bound to the re-derived base content")
    if registry is not None:
        try:
            rows = list(registry.execute("SELECT status FROM experiments WHERE experiment_id='baseline'"))
            if not rows:
                problems.append("no baseline experiment row in the registry")
            elif rows[0][0] != "passed":
                problems.append(f"baseline row status={rows[0][0]!r}, expected passed")
            baseline = list(registry.execute(
                "SELECT gpu_hours, evidence_json FROM results WHERE experiment_id='baseline'"
            ))
            if not baseline:
                problems.append("baseline has no recorded result")
            else:
                charged, evidence_text = baseline[0]
                evidence = load_json_safe(evidence_text) or {}
                compute = evidence.get("compute") if isinstance(evidence.get("compute"), dict) else {}
                if evidence.get("baseline_source") != "paired-candidate-evaluation":
                    problems.append(f"baseline_source={evidence.get('baseline_source')!r}, expected the resident pair")
                if finite_number(charged) and float(charged) != 0.0:
                    if not finite_number(compute.get("shared_wall_gpu_hours")) or not compute.get("charged_to"):
                        problems.append(
                            f"the paired baseline row re-charges {float(charged):.7f} GPU-h with no "
                            "shared_wall_gpu_hours/charged_to disclosure (the double charge the ledger fix removes)"
                        )
        except sqlite3.Error as exc:
            problems.append(f"registry unreadable: {exc}")
    else:
        problems.append("registry not found; baseline completion unverifiable")
    if problems:
        verdict.add(label, "paired gate contract", FAIL, span(problems))
        return
    verdict.add(label, "paired gate contract", PASS,
                f"one resident pair at {path.parent.name}: base {float(base_loss):.4f} scored before payload "
                f"application, candidate {float(cand_loss):.4f}, one measured load, both generations measured, "
                "baseline row completed from paired evidence with no re-charge")


# ---- T7: accounting, device basis ------------------------------------------


def check_t7(train: list, evals: list, registry, arm: str, verdict: Verdict) -> None:
    label = f"T7/{arm}"
    if len(train) != 1 or len(evals) != 1:
        verdict.add(label, "accounting", UNKNOWN, "accounting needs exactly one train and one eval worker result")
        return
    problems: list[str] = []
    detail: list[str] = []
    for kind, (path, result) in (("train", train[0]), ("eval", evals[0])):
        lifecycle = result.get("lifecycle") or {}
        total = lifecycle.get("measured_gpu_hours")
        if not finite_number(total):
            problems.append(f"{kind}: lifecycle.measured_gpu_hours unmeasured")
            continue
        phases = lifecycle.get("phases") or {}
        derived = 0.0
        loads = 0.0
        steps = 0.0
        for name, entry in phases.items():
            if entry.get("measured") is not True:
                continue
            cost = entry.get("gpu_hours")
            if not finite_number(cost):
                problems.append(f"{kind}: phase {name} measured without a gpu_hours cost")
                continue
            if entry.get("sync_overhead_seconds") is None:
                problems.append(f"{kind}: phase {name} records no sync overhead")
            derived += float(cost)
            if name == "model_load":
                loads += float(cost)
            elif name == "steady_state_steps":
                steps += float(cost)
        if abs(derived - float(total)) > 1e-9:
            problems.append(
                f"{kind}: phase sum {derived:.12f} does not equal the reported {float(total):.12f}"
            )
        required = ["model_load"]
        if kind == "train":
            # A generation-bearing phase belongs to the evaluator; the training
            # leg's own required phases are its load, its steps and its payload
            # publication.
            required += ["steady_state_steps", "checkpoint_publication"]
        else:
            required += ["baseline_generation", "candidate_generation"]
        for name in required:
            entry = phases.get(name) or {}
            if entry.get("measured") is not True:
                problems.append(f"{kind}: {name} phase not measured")
        rest = float(total) - loads - steps
        if float(total) > GPU_HOUR_CEILING:
            problems.append(f"{kind}: {float(total):.7f} GPU-h exceeds the {GPU_HOUR_CEILING} ceiling")
        if loads > SUBBUDGET_LOADS:
            problems.append(f"{kind}: loads {loads:.7f} > {SUBBUDGET_LOADS}")
        if steps > SUBBUDGET_STEPS:
            problems.append(f"{kind}: steps {steps:.7f} > {SUBBUDGET_STEPS}")
        if rest > SUBBUDGET_GENERATIONS:
            problems.append(f"{kind}: generations+publication {rest:.7f} > {SUBBUDGET_GENERATIONS}")
        detail.append(
            f"{kind} {float(total):.7f} GPU-h (loads {loads:.7f}/{SUBBUDGET_LOADS}, "
            f"steps {steps:.7f}/{SUBBUDGET_STEPS}, rest {rest:.7f}/{SUBBUDGET_GENERATIONS})"
        )
    wall_charge = 0.0
    if registry is not None:
        try:
            for (charged,) in registry.execute("SELECT gpu_hours FROM results"):
                if finite_number(charged):
                    wall_charge += float(charged)
            incidents = list(registry.execute("SELECT count(*) FROM execution_incidents"))[0][0]
            if incidents:
                problems.append(f"{incidents} execution incidents recorded")
            for experiment_id, status in registry.execute("SELECT experiment_id, status FROM experiments"):
                if status in TERMINAL_STATUSES:
                    continue
                stranded = list(registry.execute(
                    "SELECT count(*) FROM results WHERE experiment_id=?", (experiment_id,)
                ))[0][0]
                if stranded:
                    problems.append(f"stranded result: {experiment_id} carries a result while {status!r}")
        except sqlite3.Error as exc:
            problems.append(f"registry unreadable: {exc}")
    else:
        problems.append("registry not found; wall charges unverifiable")
    envelope = (
        f"wall charge {wall_charge:.4f} GPU-h inside the {GOAL_ENVELOPE_PER_ARM} envelope"
        if wall_charge <= GOAL_ENVELOPE_PER_ARM
        else f"wall charge {wall_charge:.4f} GPU-h EXCEEDS the {GOAL_ENVELOPE_PER_ARM} envelope (recorded, per the prereg)"
    )
    if problems:
        verdict.add(label, "accounting", FAIL, span(problems) + " | " + "; ".join(detail))
        return
    if wall_charge > GOAL_ENVELOPE_PER_ARM:
        verdict.add(label, "accounting", UNKNOWN,
                    "measured device time is within every budget, but the wall-charge exceedance needs human "
                    "recording: " + envelope)
        return
    verdict.add(label, "accounting", PASS, "; ".join(detail) + f"; incidents 0, no stranded results; {envelope}")


# ---- T8: identity chain -----------------------------------------------------


def check_t8(root: Path, train: list, evals: list, arm: str, verdict: Verdict) -> None:
    label = f"T8/{arm}"
    if len(train) != 1 or len(evals) != 1:
        verdict.add(label, "identity chain", UNKNOWN, "needs exactly one train and one eval worker result")
        return
    problems: list[str] = []
    content_hashes: set[str] = set()
    source_hashes: set[str] = set()
    for path, result in train + evals:
        identity = result.get("base_identity") or {}
        if identity.get("manifest_sha256") != BASE_MANIFEST_SHA256:
            problems.append(f"{path.parent.name}: manifest hash != pinned")
        if identity.get("mode") != "full":
            problems.append(f"{path.parent.name}: identity mode={identity.get('mode')!r}")
        if identity.get("binding") != "local-content":
            problems.append(f"{path.parent.name}: binding={identity.get('binding')!r}")
        if identity.get("content_sha256"):
            content_hashes.add(identity["content_sha256"])
        else:
            problems.append(f"{path.parent.name}: no content hash")
        source = result.get("source_identity") or {}
        if source.get("source_sha256"):
            source_hashes.add(source["source_sha256"])
        else:
            problems.append(f"{path.parent.name}: no worker source hash")
        frozen = result.get("frozen") or {}
        if frozen.get("ok") is not True or frozen.get("changed"):
            problems.append(f"{path.parent.name}: frozen-weight digest reports drift")
        if frozen.get("frozen_parameters") != FROZEN_PARAMETERS:
            problems.append(
                f"{path.parent.name}: frozen_parameters={frozen.get('frozen_parameters')!r}, expected {FROZEN_PARAMETERS}"
            )
    if len(content_hashes) > 1:
        problems.append("train and eval workers re-derived different base content hashes")
    # Every identity file written beside a worker result must agree with it.
    identity_files = sorted(root.glob(".chowder/*/*/chowder-identity.json"))
    if not identity_files:
        problems.append("no chowder-identity.json under the arm root")
    for path in identity_files:
        payload = load_json(path)
        if not isinstance(payload, dict):
            problems.append(f"{path.parent.name}: chowder-identity.json unreadable")
            continue
        recorded = payload.get("source_sha256")
        if recorded not in source_hashes:
            problems.append(f"{path.parent.name}: identity file source hash disagrees with the worker result")
    for filename, pinned in ((CORPUS_FILENAME, TRAINING_CORPUS_SHA256), (HOLDOUT_FILENAME, INDEPENDENT_HOLDOUT_SHA256)):
        candidates = [root / filename, root.parent / filename]
        pinned_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if pinned_file is None:
            problems.append(f"{filename} not found beside the arm root; pin unverified")
        elif sha256_file(pinned_file) != pinned:
            problems.append(f"{filename} hashes differently than pinned")
    verification = evals[0][1].get("payload_verification")
    if not isinstance(verification, dict):
        problems.append("eval carries no payload_verification")
    elif content_hashes and verification.get("base_content_sha256") not in content_hashes:
        problems.append("payload is not bound to the re-derived base content")
    if problems:
        verdict.add(label, "identity chain", FAIL, span(problems))
        return
    verdict.add(label, "identity chain", PASS,
                f"full-mode local-content identity matches the pinned manifest in both workers, "
                f"{len(identity_files)} identity files agree on the worker source hash, corpora hash to their pins, "
                f"frozen digest clean over {FROZEN_PARAMETERS} parameters")


# ---- N1-N3, N5, N7: the saturation thresholds on the fixed arm --------------


def check_saturation(train: list, evals: list, verdict: Verdict) -> None:
    if len(train) != 1:
        for name in ("N1", "N2", "N3", "N5", "N7"):
            verdict.add(name, "saturation", UNKNOWN,
                        f"needs exactly one training worker result on {FIX_ARM}, found {len(train)}")
        return
    result = train[0][1]
    gates = gates_of(result)
    instrument = instrument_of(result)
    if not gates or instrument.get("version") != INSTRUMENT_VERSION:
        for name in ("N1", "N2", "N3", "N5"):
            verdict.add(name, "saturation", UNKNOWN,
                        f"the {INSTRUMENT_VERSION} instrument is absent from {FIX_ARM}: this threshold cannot be "
                        "decided, and absence is not a pass")
    else:
        fully = sorted(
            name.split(".")[2] for name, entry in gates.items() if entry.get("fully_one_hot_steps")
        )
        if fully:
            verdict.add("N1", "no fully-one-hot step", FAIL,
                        f"{len(fully)} gates report fully-one-hot rows on real steps: L{', L'.join(fully)}")
        else:
            verdict.add("N1", "no fully-one-hot step", PASS,
                        f"no gate records a fully-one-hot row across {EXPECTED_STEPS} steps x {ROWS_PER_STEP} rows")
        stalled = sorted(
            name.split(".")[2] for name, entry in gates.items()
            if entry.get("steps_with_gradient_not_moving_parameter")
        )
        if stalled:
            worst = max(
                len(entry.get("steps_with_gradient_not_moving_parameter") or [])
                for entry in gates.values()
            )
            verdict.add("N2", "every gradient can move its parameter", FAIL,
                        f"{len(stalled)} gates have steps whose gradient cannot move the parameter "
                        f"(worst {worst}/{EXPECTED_STEPS} steps): L{', L'.join(stalled)}")
        else:
            verdict.add("N2", "every gradient can move its parameter", PASS,
                        f"implied_update > half_ulp on all {EXPECTED_STEPS} steps for all {len(gates)} gates")
        gaps = [
            max(float(value) for value in entry["top2_gap_max_by_step"])
            for entry in gates.values()
            if isinstance(entry.get("top2_gap_max_by_step"), list) and entry["top2_gap_max_by_step"]
            and numbers(entry["top2_gap_max_by_step"])
        ]
        if len(gaps) != len(gates):
            verdict.add("N3", "the logit bound holds", UNKNOWN,
                        "top2_gap_max_by_step missing or non-finite on at least one gate")
        elif max(gaps) >= ONE_HOT_GAP_THRESHOLD:
            verdict.add("N3", "the logit bound holds", FAIL,
                        f"max top-2 gap {max(gaps):.3f} reaches the {ONE_HOT_GAP_THRESHOLD} underflow threshold")
        else:
            verdict.add("N3", "the logit bound holds", PASS,
                        f"max top-2 gap {max(gaps):.3f} across all gates, {ONE_HOT_GAP_THRESHOLD / max(gaps):.2f}x "
                        f"below the {ONE_HOT_GAP_THRESHOLD} underflow threshold")
        init = result.get("gate_initialization") or {}
        init_gates = init.get("gates") if isinstance(init.get("gates"), dict) else {}
        before = (result.get("metrics") or {}).get("dead_experts_before")
        problems: list[str] = []
        if init.get("mode") != "small_normal":
            problems.append(f"gate_initialization.mode={init.get('mode')!r}, expected small_normal")
        if len(init_gates) != EXPECTED_GATES:
            problems.append(f"{len(init_gates)}/{EXPECTED_GATES} gate initializations recorded")
        tied = [name for name, entry in init_gates.items() if entry.get("initial_rows_with_exactly_equal_top2")]
        if tied:
            problems.append(f"{len(tied)} gates still tie on the top-2 logits after initialization")
        if not finite_number(before):
            problems.append("dead_experts_before unmeasured")
        elif float(before) >= DEAD_EXPERTS_UNTRAINED_TIE_BREAK:
            problems.append(
                f"dead_experts_before={float(before):.0f} still at the all-rows-tie artifact "
                f"({DEAD_EXPERTS_UNTRAINED_TIE_BREAK})"
            )
        if problems:
            verdict.add("N5", "the degeneracy is gone", FAIL, span(problems))
        else:
            verdict.add("N5", "the degeneracy is gone", PASS,
                        f"0 tied rows on all {len(init_gates)} gates and dead_experts_before={float(before):.0f} "
                        f"below the {DEAD_EXPERTS_UNTRAINED_TIE_BREAK} tie-break artifact")
    if len(evals) != 1:
        verdict.add("N7", "capability guard", UNKNOWN,
                    f"needs exactly one evaluation worker result on {FIX_ARM}, found {len(evals)}")
        return
    eval_result = evals[0][1]
    base = eval_result.get("base_holdout_loss")
    candidate = eval_result.get("candidate_holdout_loss")
    if not finite_number(base) or not finite_number(candidate):
        verdict.add("N7", "capability guard", UNKNOWN,
                    f"the fixed arm's holdout pair is unmeasured (base={base!r}, candidate={candidate!r})")
    elif float(candidate) > float(base) * CAPABILITY_GUARD_RATIO:
        verdict.add("N7", "capability guard", FAIL,
                    f"candidate {float(candidate):.4f} / base {float(base):.4f} = "
                    f"{float(candidate) / float(base):.3f} exceeds the {CAPABILITY_GUARD_RATIO}x guard")
    else:
        verdict.add("N7", "capability guard", PASS,
                    f"candidate {float(candidate):.4f} / base {float(base):.4f} = "
                    f"{float(candidate) / float(base):.3f} within the {CAPABILITY_GUARD_RATIO}x guard")


# ---- N4 and N6: the non-inferiority bar and the control --------------------


def check_n4(train: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("N4", "concentration non-inferiority", UNKNOWN,
                    f"needs exactly one training worker result on {FIX_ARM}, found {len(train)}")
        return
    after = ((train[0][1].get("metrics") or {}).get("dead_experts_after"))
    if not finite_number(after):
        verdict.add("N4", "concentration non-inferiority", UNKNOWN,
                    "dead_experts_after unmeasured on the fixed arm")
        return
    if float(after) > DEAD_EXPERTS_RUNG4_AFTER:
        verdict.add("N4", "concentration non-inferiority", FAIL,
                    f"dead_experts_after={float(after):.0f} exceeds rung 4's measured "
                    f"{DEAD_EXPERTS_RUNG4_AFTER} on the same corpus")
        return
    verdict.add("N4", "concentration non-inferiority", PASS,
                f"dead_experts_after={float(after):.0f} <= rung 4's {DEAD_EXPERTS_RUNG4_AFTER} "
                "(non-inferiority only; this rung claims no concentration gain)")


def check_n6(train: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("N6", "control: the latch reproduces", UNKNOWN,
                    f"needs exactly one training worker result on {CONTROL_ARM}, found {len(train)}")
        return
    result = train[0][1]
    gates = gates_of(result)
    before = (result.get("metrics") or {}).get("dead_experts_before")
    problems: list[str] = []
    if not finite_number(before):
        problems.append("dead_experts_before unmeasured")
    elif float(before) != DEAD_EXPERTS_UNTRAINED_TIE_BREAK:
        problems.append(
            f"dead_experts_before={float(before):.0f}, expected the reproduced tie-break artifact "
            f"{DEAD_EXPERTS_UNTRAINED_TIE_BREAK} exactly"
        )
    latched = sorted(
        name.split(".")[2] for name, entry in gates.items() if int(entry.get("grad_zero_steps") or 0) >= 1
    )
    if not gates:
        problems.append("no gate records on the control arm")
    elif not latched:
        problems.append("no gate recorded a zero-gradient step: the latch did not reproduce")
    if problems:
        verdict.add("N6", "control: the latch reproduces", FAIL,
                    span(problems) + " --- the control is void, so the fix's effect cannot be attributed to the "
                    "knobs rather than to a changed corpus, instrument, or environment")
        return
    verdict.add("N6", "control: the latch reproduces", PASS,
                f"dead_experts_before={float(before):.0f} exactly as two prior processes measured, with "
                f"layers_with_grad_zero={len(latched)} (L{', L'.join(latched)})")


# ---- INFO rows: control corroboration and the declared predictions ----------


def check_control_corroboration(train: list, verdict: Verdict) -> None:
    if len(train) != 1:
        verdict.add("INFO", "control gate corroboration", UNKNOWN,
                    "no single control training worker result to read")
        return
    gates = gates_of(train[0][1])
    dead = {name: entry for name, entry in gates.items() if int(entry.get("grad_zero_steps") or 0) >= 1}
    if not dead:
        verdict.add("INFO", "control gate corroboration", UNKNOWN,
                    "no gate with a zero-gradient step; nothing to corroborate")
        return
    annotated: list[str] = []
    unexplained: list[str] = []
    for name, entry in sorted(dead.items()):
        short = name.split(".")[2]
        one_hot = entry.get("any_exactly_one_hot_steps")
        count = len(one_hot) if isinstance(one_hot, list) else None
        if count is None:
            annotated.append(f"L{short}: one-hot series missing")
            unexplained.append(short)
        elif count > 0:
            annotated.append(f"L{short}: {count}/{EXPECTED_STEPS} one-hot steps")
        else:
            annotated.append(f"L{short}: gradient died with 0 one-hot rows")
            unexplained.append(short)
    if unexplained:
        verdict.add("INFO", "control gate corroboration", UNKNOWN,
                    "gradient death without exactly-one-hot rows on its own training batches (the diagnosis's "
                    "mechanism does not explain these gates and must be re-examined): " + span(annotated))
        return
    verdict.add("INFO", "control gate corroboration", PASS,
                "every zero-gradient gate also records exactly-one-hot rows: " + span(annotated))


def check_predictions(arms_train: dict[str, list], verdict: Verdict) -> None:
    # P1: the bounded range alone (arm C) prevents the latch.
    if len(arms_train.get("arm-C", [])) == 1:
        gates = gates_of(arms_train["arm-C"][0][1])
        instrumented = gates and all(
            isinstance(entry.get("fully_one_hot_steps"), list)
            and isinstance(entry.get("steps_with_gradient_not_moving_parameter"), list)
            for entry in gates.values()
        )
        fully = [name for name, entry in gates.items() if entry.get("fully_one_hot_steps")]
        stalled = [name for name, entry in gates.items()
                   if entry.get("steps_with_gradient_not_moving_parameter")]
        if not gates:
            verdict.add("INFO", "P1 the bound is load-bearing", UNKNOWN, "arm C carries no gate records")
        elif not instrumented:
            # Absence of the instrument is not evidence that the bound worked.
            verdict.add("INFO", "P1 the bound is load-bearing", UNKNOWN,
                        "arm C carries no saturation instrument, so this prediction cannot be decided from its "
                        "evidence")
        elif fully or stalled:
            verdict.add("INFO", "P1 the bound is load-bearing", UNKNOWN,
                        f"REFUTED: arm C alone still shows {len(fully)} fully-one-hot and {len(stalled)} "
                        "parameter-immobile gates")
        else:
            verdict.add("INFO", "P1 the bound is load-bearing", PASS,
                        "arm C (bound, artifact init) shows neither a fully-one-hot step nor a parameter-immobile gradient")
    # P2: initialization alone (arm B) still latches.
    if len(arms_train.get("arm-B", [])) == 1:
        gates = gates_of(arms_train["arm-B"][0][1])
        counted = gates and all(finite_number(entry.get("grad_zero_steps")) for entry in gates.values())
        latched = [name for name, entry in gates.items() if int(entry.get("grad_zero_steps") or 0) >= 1]
        if not counted:
            verdict.add("INFO", "P2 init alone is not load-bearing", UNKNOWN,
                        "arm B carries no zero-gradient step counts, so this prediction cannot be decided")
        elif latched:
            verdict.add("INFO", "P2 init alone is not load-bearing", PASS,
                        f"arm B still latches on {len(latched)} layers (L{', L'.join(sorted(n.split('.')[2] for n in latched))})")
        else:
            verdict.add("INFO", "P2 init alone is not load-bearing", UNKNOWN,
                        "REFUTED: initialization alone prevents the latch, so the diagnosis's mechanism ranking is "
                        "wrong on that link and the diagnosis must be amended")
    # P3: only the non-degenerate arms show a genuine pre-training census.
    observed: list[str] = []
    for arm in ARMS:
        results = arms_train.get(arm, [])
        if len(results) != 1:
            continue
        before = (results[0][1].get("metrics") or {}).get("dead_experts_before")
        if finite_number(before):
            observed.append(f"{arm}={float(before):.0f}")
    if len(observed) == len(ARMS):
        verdict.add("INFO", "P3 pre-training census is genuine only under the fix", PASS,
                    "dead_experts_before " + ", ".join(observed))


# ---- main -------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    campaign_root = Path(argv[1]).resolve()
    if not campaign_root.is_dir():
        print(f"campaign root does not exist: {campaign_root}")
        return 2
    verdict = Verdict()
    all_train: list = []
    all_evals: list = []
    arms_train: dict[str, list] = {}
    arms_evals: dict[str, list] = {}
    missing: list[str] = []
    for arm in ARMS:
        root, train, evals, registry = load_arm(campaign_root, arm)
        arms_train[arm] = train
        arms_evals[arm] = evals
        all_train.extend(train)
        all_evals.extend(evals)
        if not root.is_dir():
            missing.append(arm)
            for threshold in ("T1", "T2", "T3", "T4", "T4a", "T5", "T6", "T7", "T8"):
                verdict.add(f"{threshold}/{arm}", "artifact present", UNKNOWN,
                            f"{arm} directory is missing: nothing to judge, and absence is not a pass")
            continue
        check_t1(root, arm, verdict)
        check_t2(train, evals, arm, verdict)
        check_t3(train, evals, arm, verdict)
        check_t4(train, arm, verdict)
        check_t4a(train, arm, verdict)
        check_t5(train, arm, verdict)
        check_t6(root, evals, registry, arm, verdict)
        check_t7(train, evals, registry, arm, verdict)
        check_t8(root, train, evals, arm, verdict)
    if not missing:
        check_saturation(arms_train.get(FIX_ARM, []), arms_evals.get(FIX_ARM, []), verdict)
        check_n4(arms_train.get(FIX_ARM, []), verdict)
        check_n6(arms_train.get(CONTROL_ARM, []), verdict)
        check_control_corroboration(arms_train.get(CONTROL_ARM, []), verdict)
        check_predictions(arms_train, verdict)
    else:
        for name, label in (
            ("N1", "no fully-one-hot step"),
            ("N2", "every gradient can move its parameter"),
            ("N3", "the logit bound holds"),
            ("N4", "concentration non-inferiority"),
            ("N5", "the degeneracy is gone"),
            ("N6", "control: the latch reproduces"),
            ("N7", "capability guard"),
        ):
            verdict.add(name, label, UNKNOWN, "an arm directory is missing: the campaign is incomplete")
    return report(campaign_root, verdict, all_train, all_evals, argv_len_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
