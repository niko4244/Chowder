"""Adversarial fixed-fixture admission tests; no generated code or containers."""

from __future__ import annotations

import pytest

from chowder_batch import reap_humaneval_podman_semantics_20261027 as semantic


def test_actual_frozen_evaluator_function_pinned_and_unchanged():
    source = semantic.frozen_function()
    assert semantic.sha(source.encode()) == semantic.FUNCTION_SHA
    assert "with tempfile.TemporaryDirectory() as tmp:" in source
    assert 'Path(tmp) / "candidate.py"' in source
    assert 'subprocess.run([sys.executable, "-I", str(path)]' in source
    assert "cwd=tmp, capture_output=True, timeout=10" in source


@pytest.mark.parametrize("fixture", list(semantic.FIXTURES))
def test_fixed_sources_use_original_scorer_without_host_execution(fixture):
    source = semantic.fixed_source(fixture, semantic.frozen_function())
    assert semantic.sha(source.encode()) != semantic.FUNCTION_SHA
    assert "resource.setrlimit(resource.RLIMIT_AS" in source
    assert "resource.setrlimit(resource.RLIMIT_FSIZE" in source
    assert "passed = _humaneval_pass(item, completion)" in source
    assert "timeout=10" in source
    assert "exec(" not in source


@pytest.mark.parametrize("foreign", ["generated", "../pass", "", "PASS"])
def test_foreign_fixtures_refused(foreign):
    with pytest.raises(ValueError, match="fixture|cases"):
        semantic.fixed_source(foreign, semantic.frozen_function())


def test_changed_scorer_refused():
    with pytest.raises(ValueError, match="source-pinned"):
        semantic.fixed_source("pass", "def _humaneval_pass(*args): return True")


def vm_and_observed():
    vm = {"descendants": [
        {"pid": 110, "start_ticks": 55},
        {"pid": 111, "start_ticks": 56},
    ]}
    observed = {"processes": [
        {"pid": 110, "start_ticks": 55,
         "argv": ["python3", "-I", "-"], "cwd": "/"},
        {"pid": 111, "start_ticks": 56,
         "argv": ["/usr/local/bin/python3", "-I", "/tmp/tmpabc123/candidate.py"],
         "cwd": "/tmp/tmpabc123"},
    ]}
    return vm, observed


def test_vm_observer_accepts_only_exact_source_bound_candidate():
    vm, observed = vm_and_observed()
    assert semantic.verify_candidate_processes(vm, observed)["pid"] == 111


@pytest.mark.parametrize(("field", "bad"), [
    ("argv", ["/usr/bin/python", "candidate.py"]),
    ("argv", ["/usr/bin/python", "-I", "/host/candidate.py"]),
    ("argv", ["/usr/bin/python", "-I", "/tmp/tmpabc123/other.py"]),
    ("argv", ["/usr/bin/python", "-I", "/tmp/tmpabc123/../candidate.py"]),
    ("cwd", "/tmp"),
    ("start_ticks", 999),
    ("pid", 999),
])
def test_vm_observer_refuses_wrong_candidate_or_spoofed_identity(field, bad):
    vm, observed = vm_and_observed()
    observed["processes"][1][field] = bad
    with pytest.raises(ValueError, match="candidate|unbound"):
        semantic.verify_candidate_processes(vm, observed)


def test_vm_observer_refuses_duplicate_candidate():
    vm, observed = vm_and_observed()
    vm["descendants"].append({"pid": 112, "start_ticks": 57})
    observed["processes"].append({
        **observed["processes"][1], "pid": 112, "start_ticks": 57,
    })
    with pytest.raises(ValueError, match="exactly one"):
        semantic.verify_candidate_processes(vm, observed)


@pytest.mark.parametrize(("fixture", "passed", "elapsed"), [
    ("pass", True, 4.1), ("fail", False, 4.1), ("timeout", False, 10.02),
])
def test_original_benign_pass_fail_timeout_semantics(fixture, passed, elapsed):
    payload = {"passed": passed, "elapsed": elapsed, "python": [3, 13],
               "isolated_parent": True, "rlimit_as": 2 * 1024 ** 3,
               "rlimit_fsize": 1024 ** 2}
    candidate = {"pid": 111, "start_ticks": 56}
    post = {"same_container_init_alive": False,
            "same_descendants_alive": [], "cgroup_directory_extinct": True}
    semantic.verify_fixed_result(fixture, payload, candidate, post)


@pytest.mark.parametrize(("field", "bad"), [
    ("passed", True), ("python", [3, 11]), ("isolated_parent", False),
    ("rlimit_as", 0), ("rlimit_fsize", 0), ("elapsed", 8.0),
])
def test_timeout_semantics_fail_closed(field, bad):
    payload = {"passed": False, "elapsed": 10.01, "python": [3, 13],
               "isolated_parent": True, "rlimit_as": 2 * 1024 ** 3,
               "rlimit_fsize": 1024 ** 2}
    payload[field] = bad
    candidate = {"pid": 111, "start_ticks": 56}
    post = {"same_container_init_alive": False,
            "same_descendants_alive": [], "cgroup_directory_extinct": True}
    with pytest.raises(ValueError, match="semantics"):
        semantic.verify_fixed_result("timeout", payload, candidate, post)


def test_non_extinct_candidate_cgroup_refused():
    payload = {"passed": True, "elapsed": 4.01, "python": [3, 13],
               "isolated_parent": True, "rlimit_as": 2 * 1024 ** 3,
               "rlimit_fsize": 1024 ** 2}
    candidate = {"pid": 111, "start_ticks": 56}
    post = {"same_container_init_alive": False,
            "same_descendants_alive": [111], "cgroup_directory_extinct": False,
            "cgroup_pids_current_after": "1"}
    with pytest.raises(ValueError, match="teardown"):
        semantic.verify_fixed_result("pass", payload, candidate, post)
