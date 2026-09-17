#!/usr/bin/env python3
"""Mutation proof for judge_rung5_2026-09-16.py.

The rung-5 judge's clauses are only worth their ink if each one can fail for
its own stated reason. This probe builds a synthetic campaign that satisfies
every threshold, verifies the judge certifies it, then mutates exactly one
field per case and requires the named clause -- and only that clause's family
-- to move.

Fully synthetic except for the two corpus files, which are copied from the
rung-4 run directory so their SHA-256 values really do match the pins; a
forged hash would make T8's corpus clause untestable.

Usage::

    python judge_mutation_probe.py <work-dir> <judge.py>
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE = Path(r"F:/chowder-worktrees/rung5/docs/quals/judge_rung5_2026-09-16.py")
REAL_RUN = Path(r"C:/Users/nikma/Chowder-Protected/runs/2026-09-16-router-healing-rung4")

CORPUS = "router-corpus-9b-pilot.txt"
HOLDOUT = "router-holdout-independent-9b.txt"
CONTENT_SHA = "a6b20aafa3789ccfcd2d197f0effb7954d643aee495a2354983f68ca4d0428a5"
MANIFEST_SHA = "77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4"
HOLDOUT_SHA = "2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97"
SOURCE_SHA = "69f408c9a9a35a2d83ff038dd7f7b546001a34f8cef3e749ed14d1f00b438297"

STEPS = 48
ROWS = 128
LR = 0.05
ADAM_EPS = 1e-8

ARMS = {
    "arm-A": {"mode": "artifact", "scale": None, "cap": None},
    "arm-B": {"mode": "small_normal", "scale": None, "cap": None},
    "arm-C": {"mode": "artifact", "scale": 128.0, "cap": 30.0},
    "arm-D": {"mode": "small_normal", "scale": 128.0, "cap": 30.0},
}
# A: latch reproduces (control). B: latch too (P2). C: bounded range alone (P1).
# D: the fix.
# Declared expectations, mirrored from the prereg's section 6.3: arms A and B
# latch, arms C and D do not.
LATCHED = {"arm-A", "arm-B"}
DEAD_BEFORE = {"arm-A": 480, "arm-B": 318, "arm-C": 480, "arm-D": 311}
DEAD_AFTER = {"arm-A": 436, "arm-B": 429, "arm-C": 430, "arm-D": 428}

TRAIN_PHASES = {
    "model_load": 12.820008400012739,
    "steady_state_steps": 139.08854140003677,
    "checkpoint_publication": 0.0661124000325799,
    "closeout": 4.499917849898338e-06,
}
EVAL_PHASES = {
    "model_load": 13.608863100002054,
    "baseline_generation": 3.7465822000522166,
    "candidate_generation": 3.777045400016314,
}


def phases_block(seconds: dict[str, float]) -> tuple[dict, float, float]:
    phases = {}
    total_gpu = 0.0
    total_seconds = 0.0
    for name, value in seconds.items():
        gpu = value / 3600.0
        total_gpu += gpu
        total_seconds += value
        phases[name] = {
            "phase": name,
            "measured": True,
            "seconds": value,
            "gpu_hours": gpu,
            "accelerator_count": 1,
            "sync_overhead_seconds": 0.0,
            "synchronized": False,
            "note": None,
        }
    for name in ("first_forward", "first_backward", "first_update", "reload"):
        phases.setdefault(name, {
            "phase": name, "measured": False, "seconds": None, "gpu_hours": None,
            "accelerator_count": 1, "sync_overhead_seconds": 0.0, "synchronized": None, "note": None,
        })
    return phases, total_gpu, total_seconds


def lifecycle(seconds: dict[str, float]) -> dict:
    phases, gpu, wall = phases_block(seconds)
    return {
        "accelerator_count": 1,
        "measured_gpu_hours": gpu,
        "measured_seconds": wall,
        "phases": phases,
    }


def gate_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.gate.weight"


def gate_record(layer: int, latched: bool) -> dict:
    grad, gap, one_hot, softmax_in = [], [], [], []
    for step in range(STEPS):
        if latched and layer == 31 and step >= 8:
            grad.append(0.0)
            gap.append(528.0)
            one_hot.append(ROWS)
            softmax_in.append(600.0)
        elif latched and layer == 22 and step >= 35:
            grad.append(1.214e-16)
            gap.append(132.0)
            one_hot.append(120)
            softmax_in.append(180.0)
        else:
            grad.append(4.7e-02)
            gap.append(4.3)
            one_hot.append(0)
            softmax_in.append(4.6)
    magnitude = 0.293 if latched else 0.81
    half_ulp = [0.5 * (2.0 ** -8) * magnitude for _ in range(STEPS)]
    implied = [LR * value / (value + ADAM_EPS) for value in grad]
    moves = [imp > ulp for imp, ulp in zip(implied, half_ulp)]
    zeros = [step for step, value in enumerate(grad) if value == 0.0]
    nonzero = [step for step in range(STEPS) if step not in zeros]
    fully = [step for step, value in enumerate(one_hot) if value == ROWS]
    any_one = [step for step, value in enumerate(one_hot) if value > 0]
    stalled = [step for step, flag in enumerate(moves) if not flag]
    return {
        "observed_steps": STEPS,
        "trainable": True,
        "gradient_states": sorted({"grad-zero"} if zeros else {"grad-nonzero"}),
        "grad_zero_steps": len(zeros),
        "grad_nonzero_steps": len(nonzero),
        "nonzero_steps": nonzero,
        "update_steps": list(range(STEPS - 16)),
        "grad_absmax_by_step": grad,
        "softmax_input_absmax_by_step": softmax_in,
        "top2_gap_max_by_step": gap,
        "rows_exactly_one_hot_by_step": one_hot,
        "implied_update_by_step": implied,
        "half_ulp_by_step": half_ulp,
        "gradient_moves_parameter_by_step": moves,
        "rows_total_per_step": ROWS,
        "fully_one_hot_steps": fully,
        "any_exactly_one_hot_steps": any_one,
        "steps_with_gradient_not_moving_parameter": stalled,
        "max_exactly_one_hot_fraction": (max(one_hot) / ROWS) if one_hot else 0.0,
        "max_top2_gap": max(gap),
        "min_softmax_input_absmax": min(softmax_in),
        "grad_absmax_min": min(grad),
        "min_implied_update": min(implied),
    }


def train_result(arm: str, mut: str) -> dict:
    latched = arm in LATCHED
    gates = {gate_name(layer): gate_record(layer, latched) for layer in range(32)}
    before = DEAD_BEFORE[arm]
    after = DEAD_AFTER[arm]
    tie = ROWS if arm in {"arm-A", "arm-C"} else 0
    init_absmax = 600.0 if arm in {"arm-A", "arm-C"} else 0.02
    if mut == "n1_fully_one_hot" and arm == "arm-D":
        gates[gate_name(12)]["fully_one_hot_steps"] = [5]
        gates[gate_name(12)]["any_exactly_one_hot_steps"] = [5]
    if mut == "n2_stalled" and arm == "arm-D":
        gates[gate_name(7)]["steps_with_gradient_not_moving_parameter"] = [10]
    if mut == "n3_gap" and arm == "arm-D":
        gates[gate_name(20)]["top2_gap_max_by_step"] = [4.3] * 47 + [528.0]
        gates[gate_name(20)]["max_top2_gap"] = 528.0
    if mut == "n4_dead_after" and arm == "arm-D":
        after = 437
    if mut == "n5_tie_remains" and arm == "arm-D":
        tie = 128
    if mut == "n5_census_artifact" and arm == "arm-D":
        before = 480
    if mut == "n6_census_drift" and arm == "arm-A":
        before = 479
    if mut == "n6_no_latch" and arm == "arm-A":
        for name in gates:
            gates[name] = gate_record(int(name.split(".")[2]), False)
    if mut == "t3_init_saturated" and arm == "arm-B":
        init_absmax = 11.0
    if mut == "t4_instrument_absent" and arm == "arm-D":
        pass  # handled by the caller
    if mut == "t4_missing_series" and arm == "arm-D":
        gates[gate_name(1)]["grad_absmax_by_step"] = gates[gate_name(1)]["grad_absmax_by_step"][:47]
    if mut == "t4_flag_lie" and arm == "arm-D":
        gates[gate_name(2)]["gradient_moves_parameter_by_step"][0] = False
    if mut == "t4_implied_inconsistent" and arm == "arm-D":
        # A lie that is self-consistent for the boolean flag but not for the
        # arithmetic: only T4's re-derivation can catch it, so this case must
        # move T4 and leave N2 alone.
        gates[gate_name(4)]["implied_update_by_step"][3] = 1.0e-09
        gates[gate_name(4)]["gradient_moves_parameter_by_step"][3] = False
    if mut == "t5_steps" and arm == "arm-D":
        pass  # handled by the caller
    if mut == "t7_steps_budget" and arm == "arm-D":
        pass  # handled by the caller

    instrument = {
        "version": "saturation.v1",
        "rows_per_step": ROWS,
        "gates_observed": 32,
        "declared_scale": ARMS[arm]["scale"],
        "declared_soft_cap": ARMS[arm]["cap"],
        "declared_init": ARMS[arm]["mode"],
        "adam_eps": ADAM_EPS,
        "learning_rate": LR,
        "dtype": "torch.bfloat16",
    }
    if mut == "t4_instrument_absent" and arm == "arm-D":
        instrument = {}

    init = {
        "version": "gate-init.v1",
        "mode": ARMS[arm]["mode"],
        "init_std": 1.0e-3 if ARMS[arm]["mode"] == "small_normal" else None,
        "init_seed": 1 if ARMS[arm]["mode"] == "small_normal" else None,
        "initialized_gate_digest": "b" * 64 if ARMS[arm]["mode"] == "small_normal" else None,
        "rows_total": ROWS,
        "refused": False,
        "refusal_reason": None,
        "gates": {
            name: {
                "initial_gate_output_absmax": init_absmax,
                "initial_softmax_input_absmax": init_absmax,
                "initial_logit_rms": init_absmax / 4.0,
                "initial_rows_with_exactly_equal_top2": tie,
            }
            for name in sorted(gates)
        },
    }

    steps_completed = 47 if (mut == "t5_steps" and arm == "arm-D") else STEPS
    stop_reason = "max_tokens" if (mut == "t5_stop_reason" and arm == "arm-D") else "max_steps"
    train_phases = dict(TRAIN_PHASES)
    if mut == "t7_steps_budget" and arm == "arm-D":
        train_phases["steady_state_steps"] = 0.07 * 3600.0

    losses = [2.65 - 0.01 * step for step in range(STEPS)]
    return {
        "kind": "router_healing_worker_result.v1",
        "steps_completed": steps_completed,
        "global_step": steps_completed,
        "losses": losses,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_log_limit": STEPS,
        "losses_truncated": False,
        "limits": {
            "max_seconds": None,
            "max_steps": STEPS,
            "max_tokens": 6144,
            "samples_consumed": 96,
            "stop_reason": stop_reason,
            "tokens_consumed": 6144,
        },
        "load_policy_report": {
            "policy": "bf16-offload-transient",
            "dtype": "torch.bfloat16",
            "transient_forward_installed": True,
            "patched_expert_modules": 32,
            "patched_expert_modules_after_freeze": 32,
            "meta_param_count": 555,
            "placement_census": {
                "expert_params_on_device": 0,
                "expert_params_total": 64,
                "gate_params_on_device": 32,
                "gate_params_total": 32,
                "verified": True,
            },
        },
        "device_preflight": {
            "device": "cuda",
            "free_memory_bytes": 5899288576,
            "projected_oom": False,
            "step_cost_probe": {
                "measured": True,
                "step_seconds": 3.9 if (mut == "t3_steep" and arm == "arm-D") else 3.049187000025995,
                "incremental_step_bytes": 2009024512,
                "peak_step_bytes": 11972909056,
                "resident_before_step_bytes": 9963884544,
                "projected_oom": False,
                "would_exceed_budget": False,
            },
        },
        "load_budget": {
            "measured": True,
            "load_seconds": TRAIN_PHASES["model_load"],
            "load_gpu_hours": TRAIN_PHASES["model_load"] / 3600.0,
            "max_load_seconds": 20.0,
            "max_load_gpu_hours": 20.0 / 3600.0,
            "would_exceed_load_budget": False,
        },
        "run_ceiling": {
            "measured": True,
            "would_exceed_ceiling": False,
            "projected_gpu_hours": sum(train_phases.values()) / 3600.0,
            "max_gpu_hours": 0.0710,
            "sub_budgets": {
                "loads": {"would_exceed": False},
                "steps": {"would_exceed": False},
                "generations": {"would_exceed": False},
            },
        },
        "resource_usage": {
            "peak_vram_gb_by_accelerator": {"0": 13.548635},
            "wall_seconds": sum(train_phases.values()),
            "sampling_device": "cuda:0",
        },
        "lifecycle": lifecycle(train_phases),
        "metrics": {
            "census_basis": "declared-census-corpus",
            "census_blocks_available": 12,
            "census_blocks_used": 2,
            "census_corpus_sha256": HOLDOUT_SHA,
            "dead_experts_before": before,
            "dead_experts_after": after,
            "dead_experts_per_layer": {},
            "dead_experts_per_layer_before": {},
            "expert_slots": 512,
        },
        "gate_initialization": init,
        "trainability": {
            "ok": True,
            "window_steps": 2,
            "intended_components": sorted(gates),
            "not_trainable": [],
            "components": gates,
            "instrument": instrument,
        },
        "base_identity": {
            "binding": "local-content",
            "content_sha256": CONTENT_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "mode": "full",
        },
        "source_identity": {"files": 164, "source_root": "F:/x/src/chowder", "source_sha256": SOURCE_SHA},
        "frozen": {
            "ok": True,
            "changed": {"model.layers.0.mlp.experts.0.gate_proj.weight": "drift"}
            if (mut == "t8_frozen_drift" and arm == "arm-A") else {},
            "frozen_parameters": 523,
            "digest_strategy": ["full", "sampled"],
        },
        "spec_digest": "c" * 64,
    }


def eval_result(arm: str, mut: str) -> dict:
    base = 2.9327452182769775
    candidate = 2.8369253873825073
    if mut == "n7_regression" and arm == "arm-D":
        candidate = 3.30  # 1.125x base, past the 1.10x guard
    eval_phases = dict(EVAL_PHASES)
    if mut == "t7_eval_total" and arm == "arm-D":
        eval_phases["model_load"] = 0.075 * 3600.0
    names = sorted(gate_name(layer) for layer in range(32))
    return {
        "kind": "router_healing_eval_result.v1",
        "arm": "paired",
        "pair_error": None,
        "base_holdout_loss": base,
        "candidate_holdout_loss": candidate,
        "holdout_loss_delta": candidate - base,
        "metrics": {"dead_experts": DEAD_AFTER[arm], "experts_per_token": 2.0, "holdout_loss": candidate},
        "application_control": {
            "applied_parameters": names,
            "logits_before": {"digest": "before"},
            "logits_after": {"digest": "after"},
        },
        "payload_verification": {"manifest_sha256": "a43b4d962ecc739f958ad677d7d49f2ce3376d49f1c36ffbc4d2737b3435aa01",
                                 "base_content_sha256": CONTENT_SHA},
        "lifecycle": lifecycle(eval_phases),
        "load_policy_report": {
            "policy": "bf16-offload-transient",
            "dtype": "torch.bfloat16",
            "transient_forward_installed": True,
            "placement_census": {
                "expert_params_on_device": 0,
                "expert_params_total": 64,
                "gate_params_on_device": 32,
                "gate_params_total": 32,
                "verified": True,
            },
        },
        "device_preflight": {"measured": True, "projected_oom": False},
        "load_budget": {"measured": True, "would_exceed_load_budget": False, "load_seconds": EVAL_PHASES["model_load"]},
        "run_ceiling": {
            "measured": True,
            "would_exceed_ceiling": False,
            "sub_budgets": {"loads": {"would_exceed": False}, "steps": {"would_exceed": False},
                            "generations": {"would_exceed": False}},
        },
        "resource_usage": {"peak_vram_gb_by_accelerator": {"0": 9.482414}},
        "base_identity": {
            "binding": "local-content",
            "content_sha256": CONTENT_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "mode": "full",
        },
        "source_identity": {"files": 164, "source_root": "F:/x/src/chowder", "source_sha256": SOURCE_SHA},
        "frozen": {"ok": True, "changed": {}, "frozen_parameters": 523},
        "spec_digest": "d" * 64,
    }


def write_registry(path: Path, arm: str, mut: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, parent_id TEXT,"
                 " estimated_gpu_hours REAL, hypothesis_json TEXT, config_json TEXT, status TEXT)")
    conn.execute("CREATE TABLE results (experiment_id TEXT, metrics_json TEXT, gpu_hours REAL,"
                 " artifact_ref TEXT, evidence_json TEXT)")
    conn.execute("CREATE TABLE execution_incidents (incident_id TEXT, experiment_id TEXT)")
    conn.execute("INSERT INTO experiments VALUES ('baseline', NULL, 0.01, '{}', '{}', 'passed')")
    conn.execute("INSERT INTO experiments VALUES ('router-pilot-9b', NULL, 0.0541, '{}', '{}', 'passed')")
    shared_wall = 0.01707477902776898
    if mut == "t6_double_charge" and arm == "arm-D":
        # The pre-2026-09-15 shape: the paired baseline re-charges the shared
        # resident wall with no disclosure at all.
        baseline_charge = shared_wall
        baseline_compute = {"model_loads": 1, "total_gpu_hours": baseline_charge}
    elif mut == "t6_disclosed_charge" and arm == "arm-D":
        # Charged, but disclosed: the post-fix contract, so T6 must pass.
        baseline_charge = shared_wall
        baseline_compute = {"model_loads": 1, "total_gpu_hours": baseline_charge,
                            "shared_wall_gpu_hours": shared_wall, "charged_to": "router-pilot-9b"}
    else:
        baseline_charge = 0.0
        baseline_compute = {"model_loads": 1, "shared_wall_gpu_hours": shared_wall,
                            "charged_to": "router-pilot-9b", "total_gpu_hours": 0.0}
    evidence = {"base_holdout_loss": 2.9327452182769775, "baseline_source": "paired-candidate-evaluation",
                "compute": baseline_compute}
    conn.execute("INSERT INTO results VALUES ('baseline', '{}', ?, NULL, ?)",
                 (baseline_charge, json.dumps(evidence)))
    conn.execute("INSERT INTO results VALUES ('router-pilot-9b', '{}', 0.073401, NULL, '{}')")
    conn.commit()
    conn.close()


def write_identity(path: Path) -> None:
    path.write_text(json.dumps({"files": 164, "source_root": "F:/x/src/chowder", "source_sha256": SOURCE_SHA}),
                    encoding="utf-8")


def build_arm(root: Path, arm: str, mut: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text("{}", encoding="utf-8")
    (root / "project-validate-stdout.log").write_text("project-validate: ok\n", encoding="utf-8")
    write_registry(root / "runs.db", arm, mut)
    run_dir = root / ".chowder" / "runs" / "router-pilot-9b-deadbeef0001"
    eval_dir = root / ".chowder" / "evals" / "router-pilot-9b-eval-deadbeef0001"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "gate_initialization": ARMS[arm]["mode"],
        "gate_init_std": 1.0e-3 if ARMS[arm]["mode"] == "small_normal" else None,
        "router_logit_scale": ARMS[arm]["scale"],
        "router_logit_soft_cap": ARMS[arm]["cap"],
    }
    if mut == "t2_knob_drift" and arm == "arm-C":
        spec["router_logit_scale"] = 64.0
    (run_dir / "run-spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run_dir / "worker-result.json").write_text(json.dumps(train_result(arm, mut)), encoding="utf-8")
    write_identity(run_dir / "chowder-identity.json")
    (eval_dir / "worker-result.json").write_text(json.dumps(eval_result(arm, mut)), encoding="utf-8")
    write_identity(eval_dir / "chowder-identity.json")


def build_campaign(root: Path, mut: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name in (CORPUS, HOLDOUT):
        source = REAL_RUN / name
        if not source.is_file():
            raise SystemExit(f"pinned corpus missing: {source}")
        shutil.copy2(source, root / name)
    for arm in ARMS:
        build_arm(root / arm, arm, mut)
    if mut == "t8_corpus_arm_local":
        shutil.copy2(REAL_RUN / HOLDOUT, root / "arm-D" / "router-corpus-9b-pilot.txt")


CASES: list[tuple[str, str | None, str]] = [
    ("clean", None, "QUALIFIED"),  # must certify: the mutations below are only
                                    # meaningful if the clean campaign passes.
    ("n1_fully_one_hot", "N1", "fully-one-hot"),
    ("n2_stalled", "N2", "cannot move the parameter"),
    ("n3_gap", "N3", "reaches the 92.4 underflow threshold"),
    ("n4_dead_after", "N4", "exceeds rung 4's measured"),
    ("n5_tie_remains", "N5", "still tie on the top-2 logits"),
    ("n5_census_artifact", "N5", "still at the all-rows-tie artifact"),
    ("n6_census_drift", "N6", "expected the reproduced tie-break artifact 480 exactly"),
    ("n6_no_latch", "N6", "the latch did not reproduce"),
    ("n7_regression", "N7", "exceeds the 1.1x guard"),
    ("t2_knob_drift", "T2/arm-C", "arm declares 128.0"),
    ("t3_steep", "T3/arm-D", "exceeds the 3.5 s ceiling"),
    ("t3_init_saturated", "T3/arm-B", "start above the 10.0 softmax-input bound"),
    ("t4_instrument_absent", "T4/arm-D", "instrument.version=None"),
    ("t4_missing_series", "T4/arm-D", "grad_absmax_by_step has 47 entries"),
    ("t4_flag_lie", "T4/arm-D", "liveness flag disagrees"),
    ("t4_implied_inconsistent", "T4/arm-D", "implied_update inconsistent with grad_absmax"),
    ("t5_steps", "T5/arm-D", "steps_completed=47"),
    ("t5_stop_reason", "T5/arm-D", "stop_reason='max_tokens'"),
    ("t6_double_charge", "T6/arm-D", "no shared_wall_gpu_hours/charged_to disclosure"),
    ("t6_disclosed_charge", None, "QUALIFIED"),
    ("t7_steps_budget", "T7/arm-D", "steps 0.0700000 > 0.061"),
    ("t7_eval_total", "T7/arm-D", "exceeds the 0.071 ceiling"),
    ("t8_frozen_drift", "T8/arm-A", "frozen-weight digest reports drift"),
    ("t8_corpus_arm_local", "T8/arm-D", "hashes differently than pinned"),
]

INFO_CASES = [
    ("p1_refuted", "P1 the bound is load-bearing", "REFUTED"),
    ("p2_refuted", "P2 init alone is not load-bearing", "REFUTED"),
]


def run_judge(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(JUDGE), str(root)],
        capture_output=True, text=True, cwd=str(JUDGE.parent),
    )
    return proc.returncode, proc.stdout + proc.stderr


def row_for(output: str, label: str | None) -> str:
    if label is None:
        return output
    for line in output.splitlines():
        if line.startswith(label + " "):
            return line
    return ""


def rows_matching(output: str, needle: str) -> str:
    """Every INFO row whose text mentions the prediction, joined."""
    return "\n".join(
        line for line in output.splitlines() if line.startswith("INFO ") and needle in line
    )


def main(argv: list[str]) -> int:
    global JUDGE
    if len(argv) > 2:
        JUDGE = Path(argv[2]).resolve()
    if not JUDGE.is_file():
        raise SystemExit(f"judge not found: {JUDGE}")
    work = Path(argv[1] if len(argv) > 1 else str(HERE / "_mut")).resolve()
    failures: list[str] = []
    for name, label, needle in CASES:
        root = work / name
        build_campaign(root, name)
        code, output = run_judge(root)
        verdict_line = next((line for line in output.splitlines() if line.startswith("VERDICT:")), "")
        if label is None or name == "clean":
            ok = code == 0 and "QUALIFIED" in verdict_line
            reason = f"exit={code} {verdict_line.strip()}"
        else:
            row = row_for(output, label)
            ok = code == 1 and needle in row
            reason = f"exit={code} row={row.strip()[:160]!r}"
        print(f"{'ok  ' if ok else 'MISS'} {name:26s} {reason}")
        if not ok:
            failures.append(name)
    for name, label, needle in INFO_CASES:
        # INFO rows must fire without changing the exit code.
        root = work / name
        build_campaign(root, "none")
        if name == "p1_refuted":
            # Arm C becomes latched: P1's refutation.
            path = root / "arm-C" / ".chowder" / "runs" / "router-pilot-9b-deadbeef0001" / "worker-result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["trainability"]["components"] = {
                gate_name(layer): gate_record(layer, True) for layer in range(32)
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        if name == "p2_refuted":
            path = root / "arm-B" / ".chowder" / "runs" / "router-pilot-9b-deadbeef0001" / "worker-result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["trainability"]["components"] = {
                gate_name(layer): gate_record(layer, False) for layer in range(32)
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        code, output = run_judge(root)
        rows = [line for line in output.splitlines() if line.startswith("INFO ")]
        row = next((line for line in rows if label in line), rows[0] if rows else "")
        ok = code == 0 and needle in row
        print(f"{'ok  ' if ok else 'MISS'} {name:26s} exit={code} row={row.strip()[:150]!r}")
        if not ok:
            failures.append(name)
    print()
    print(f"{len(CASES) + len(INFO_CASES) - len(failures)} ok, {len(failures)} missed"
          + (f": {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
