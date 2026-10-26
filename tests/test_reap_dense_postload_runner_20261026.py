"""Adversarial CPU-only admission tests for an unlaunched dense first-200 snapshot.

Extract only pure admission functions from the runner syntax tree. Importing the
actual notebook snapshot would incorrectly attempt live GPU/Kaggle execution.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reap_v2_kaggle_qualification_20261026"
    / "dense_q4_first200_material_postload.py"
)

EXPECTED_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]


def _functions():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    names = {"admit_material_dense_load", "admit_dense_postload"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    env = {"math": math, "GIB": 1024**3}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("PREVIOUS_INCOMPLETE_DENSE_CONSUMPTION_GIB",
                                      "MINIMUM_NEW_DENSE_CGROUP_GIB"):
                env[node.targets[0].id] = eval(
                    compile(ast.Expression(node.value), str(RUNNER), "eval"),
                    {"__builtins__": {}}, env,
                )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), env)
    return tree, env


def _start(**changes):
    sample = {"physical_free_gib": 55.0, "cgroup_unused_gib": 48.0,
              "gpu_free_mib": [15000, 15000]}
    sample.update(changes)
    return sample


def test_frozen_source_snapshot_keeps_exact_dense_model_identity_and_protocol():
    tree, _ = _functions()
    source = RUNNER.read_text(encoding="utf-8")
    assert "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049" in source
    assert "b2ef325bb9c3165439e669326746fdb9165a5a66975890c99cbfd56dda8c6c0a" in source
    assert "rows[:200]" in source
    assert "POSIX_FADV_DONTNEED" in source
    assert "R['material_new_host_preflight']=admit_material_dense_load(" in source
    assert "R['postload_measured_admission']=admit_dense_postload(" in source
    assert "not_official_windows_serving_attestation" in source
    assert "COMPLETE_DENSE_FIRST200_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE" in source
    assert "dense-q4_k_m.gguf" in source
    flags = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "FLAGS" for t in n.targets)]
    assert len(flags) == 1
    assert ast.literal_eval(flags[0]) == EXPECTED_FLAGS


def test_material_host_admission_happens_before_first_large_gguf_read():
    source = RUNNER.read_text(encoding="utf-8")
    first_hash = source.index("weights=source('dense-q4_k_m.gguf'")
    census = source.index("R['material_prehash_host_admission']=admit_material_dense_load(")
    after_hash = source.index("R['material_new_host_preflight']=admit_material_dense_load(")
    assert census < first_hash < after_hash
    # If hashing itself pressures the cgroup, sampling must interrupt the
    # 21.17-GB read rather than waiting until after the oversized input.
    assert "check_runtime(host())" in source[source.index("def hash_file"):source.index("def source")]
    assert "R['material_prehash_host']" in source


def test_terminal_complete_requires_stopped_supervisor_and_no_fatal():
    source = RUNNER.read_text(encoding="utf8")
    terminal = source.index("R['status']='COMPLETE_DENSE_FIRST200_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE'")
    quiesce = source.index("stop.set();guard.join(timeout=12)")
    final_abort = source.index("if guard.is_alive() or fatal:")
    assert quiesce < final_abort < terminal
    assert "check_runtime(host());check_owner()" in source[final_abort:terminal]


@pytest.mark.parametrize("invalid", [
    {"cgroup_unused_gib": 29.444},
    {"cgroup_unused_gib": 33.451},
    {"cgroup_unused_gib": float("nan")},
    {"cgroup_unused_gib": float("inf")},
    {"cgroup_unused_gib": True},
    {"cgroup_unused_gib": None},
    {"physical_free_gib": 6.99},
    {"physical_free_gib": float("nan")},
    {"gpu_free_mib": [15000, 10000]},
    {"gpu_free_mib": [15000]},
    {"gpu_free_mib": []},
    {"gpu_free_mib": [15000, True]},
    {"gpu_free_mib": [15000, 15000, 15000]},
])
def test_original_failed_capacity_and_corrupt_metrics_must_not_start_dense(invalid):
    _, env = _functions()
    with pytest.raises((ValueError, RuntimeError)):
        env["admit_material_dense_load"](_start(**invalid), 20 * 1024**3)


def test_provisioned_host_can_only_attempt_new_provisional_load():
    _, env = _functions()
    decision = env["admit_material_dense_load"](_start(), 20 * 1024**3)
    assert decision["status"] == "MAY_ATTEMPT_DENSE_FIRST200_PROVISIONAL_ONLY"
    assert decision["windows_gate_eligible"] is False
    assert decision["speed_certified"] is False
    assert decision["observed_incomplete_dense_load_gib"] == pytest.approx(19.452)
    assert decision["minimum_unconsumed_cgroup_gib"] == pytest.approx(33.452)


@pytest.mark.parametrize("before,after", [
    (10.0, 13.99), (11.0, 14.9), (20.0, 23.9),
    (9.0, 30.0), (10.0, float("nan")), (10.0, float("inf")),
    (True, 30.0), (10.0, None),
])
def test_postload_reclaim_must_be_actual_4_gib_and_14_gib_headroom(before, after):
    _, env = _functions()
    with pytest.raises((ValueError, RuntimeError)):
        env["admit_dense_postload"](before, after, 20.0, 15.0, 0, 0)


def test_good_postload_is_only_provisional_not_a_gate_pass():
    _, env = _functions()
    result = env["admit_dense_postload"](10.5, 26.0, 20.0, 4.0, 0, 0)
    assert result["measured_recovered_gib"] == pytest.approx(15.5)
    assert result["measured_file_cache_decrease_gib"] == pytest.approx(16.0)
    assert result["status"] == "POSTLOAD_RECLAIM_ADMITTED_TO_PROVISIONAL_SCORING_ONLY"
    assert result["windows_gate_eligible"] is False
    assert result["speed_certified"] is False


@pytest.mark.parametrize("pre,post,events_pre,events_post", [
    (20.0, 17.0, 0, 0),
    (20.0, 20.0, 0, 0),
    (20.0, 10.0, 0, 1),
    (20.0, 10.0, 1, 1),
    (20.0, float("nan"), 0, 0),
    (True, 1.0, 0, 0),
    (20.0, 10.0, False, 0),
    (20.0, 10.0, 0, None),
])
def test_dense_refuses_unproven_clean_cache_or_native_oom(pre, post, events_pre, events_post):
    _, env = _functions()
    with pytest.raises((RuntimeError, ValueError)):
        env["admit_dense_postload"](10.5, 25.0, pre, post, events_pre, events_post)
