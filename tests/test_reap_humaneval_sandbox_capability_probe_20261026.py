"""CPU-only source-level safety tests of a distinct harmless Kaggle probe."""

import ast
import hashlib
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "scripts"
          / "reap_v2_humaneval_sandbox_probe_20261026"
          / "sandbox_capability_probe.py")


def test_probe_is_trusted_fixed_code_and_never_admits_generated_code():
    raw = SOURCE.read_text("utf8")
    tree = ast.parse(raw)
    assert "reap-v2-private-human164-harmless-kaggle-sandbox-probe.v1" in raw
    assert "verified_complete_humaneval_execution_sandbox" in raw
    assert "result[\"verified_complete_humaneval_execution_sandbox\"] = False" in raw
    assert "no_generated_python_executed" in raw
    assert "no_network_request_issued" in raw
    assert "no_cgroup_modified" in raw
    assert "open('x'" not in raw
    assert "RESULT_PATH.open(\"x\"" in raw
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "chowder_batch"
                   for n in ast.walk(tree))


def test_capability_probe_requires_actual_distinct_kernel_namespaces():
    raw = SOURCE.read_text("utf8")
    for flag in ("--unshare-all", "--die-with-parent", "--new-session",
                 "--ro-bind", "--proc", "--dev", "--tmpfs",
                 "--map-root-user", "--mount", "--pid", "--net", "--fork"):
        assert flag in raw
    assert 'observed["ns"][n] != host[n]' in raw
    assert "cgroup_delegation_proven\": False" in raw
    assert "CAPABILITY_PROBE_ONLY_NOT_AN_EXECUTION_APPROVAL" in raw


def test_no_dynamic_candidate_source_or_private_credential_ingestion():
    raw = SOURCE.read_text("utf8")
    forbidden = ("kaggle/input", "model.gguf", "humaneval.jsonl",
                 "getpass", "import requests", "urllib.request", "socket.connect",
                 "candidate_source", "generated_source", "eval(")
    assert all(item not in raw for item in forbidden)


def test_actual_completed_private_cpu_probe_source_hash_is_immutable():
    original = SOURCE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "a78fb69d06ac15ffba481582f07eb1d2855807c66b53aabd05c1f383df678942"
    )
    tree = ast.parse(original)
    fixed = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "CHILD"
    )
    assert hashlib.sha256(fixed.encode()).hexdigest() == (
        "fdacc215f9aae207a476f4229b7e92414bc332d70d58b0f9f721262b2b97ea8d"
    )
