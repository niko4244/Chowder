"""Workers must run the same Chowder as the process that launched them.

Found by opening the gate on the real-ML smoke test from a worktree: training
succeeded, then the evaluator worker crashed on a spec field its code did not
know, because a fresh interpreter resolved `import chowder` through an editable
install pointing at a DIFFERENT checkout. See `chowder/worker_env.py`.

These tests reproduce that mechanism deterministically on any machine by planting
a decoy `chowder` package on PYTHONPATH -- exactly the position a competing
checkout occupies -- rather than depending on how this particular machine happens
to be installed.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import chowder
from chowder.worker_env import chowder_source_root, worker_env

SRC = Path(chowder.__file__).resolve().parent

PROBE = "import chowder; print(getattr(chowder, 'DECOY', False)); print(chowder.__file__)"


def _plant_decoy(tmp_path: Path) -> Path:
    root = tmp_path / "decoy"
    (root / "chowder").mkdir(parents=True)
    (root / "chowder" / "__init__.py").write_text("DECOY = True\n", encoding="utf-8")
    return root


def _probe(env, cwd):
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=cwd,
                         capture_output=True, text=True, check=True)
    is_decoy, path = out.stdout.splitlines()[:2]
    return is_decoy == "True", Path(path).resolve()


def test_without_worker_env_a_competing_chowder_wins(tmp_path, monkeypatch):
    """The defect itself, demonstrated rather than assumed: a plain child process
    imports whatever chowder its environment points at, not the parent's."""
    monkeypatch.setenv("PYTHONPATH", str(_plant_decoy(tmp_path)))
    is_decoy, _ = _probe(dict(os.environ), cwd=tmp_path)
    assert is_decoy, "expected the decoy to win without worker_env -- the premise of the fix"


def test_worker_env_makes_the_child_import_the_parents_chowder(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_plant_decoy(tmp_path)))
    is_decoy, path = _probe(worker_env(), cwd=tmp_path)
    assert not is_decoy, "the child imported the decoy despite worker_env"
    assert path == Path(chowder.__file__).resolve()


def test_worker_env_keeps_inherited_pythonpath_entries_after_the_parent_root(monkeypatch, tmp_path):
    keep = str(tmp_path / "keep-me")
    monkeypatch.setenv("PYTHONPATH", keep)
    parts = worker_env()["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(chowder_source_root())
    assert keep in parts[1:]


def test_worker_env_does_not_duplicate_the_root(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(chowder_source_root()))
    parts = worker_env()["PYTHONPATH"].split(os.pathsep)
    assert parts.count(str(chowder_source_root())) == 1


def test_worker_env_preserves_the_rest_of_the_environment(monkeypatch):
    monkeypatch.setenv("CHOWDER_TEST_SENTINEL", "still-here")
    env = worker_env({"PYTHONUNBUFFERED": "1"})
    assert env["CHOWDER_TEST_SENTINEL"] == "still-here"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_worker_env_refuses_to_let_extra_override_pythonpath():
    with pytest.raises(ValueError, match="must not set PYTHONPATH"):
        worker_env({"PYTHONPATH": "/somewhere/else"})


def _launches(tree: ast.AST):
    """subprocess.run / Popen / check_output / check_call call nodes."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
                and node.func.attr in {"run", "Popen", "check_output", "check_call"}):
            yield node


# Launches that do not start a Chowder Python worker, so the guarantee does not
# apply: nvidia-smi / system queries, and pip/venv setup for the Unsloth env.
_NOT_CHOWDER_WORKERS = {
    ("hardware.py", "nvidia-smi / system query"),
    ("parent_tournament.py", "nvidia-smi VRAM sampler"),
    ("unsloth_env.py", "pip / venv setup"),
}


def test_every_chowder_worker_launch_passes_worker_env():
    """The guard that stops this recurring. Any module that launches a subprocess
    and also references the Python interpreter (sys.executable) or a *_worker
    module is launching a Chowder worker, and every such launch must pass an
    env built by worker_env. A new worker added without it fails here."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "subprocess" not in text:
            continue
        tree = ast.parse(text)
        launches = list(_launches(tree))
        if not launches:
            continue
        rel = path.relative_to(SRC).as_posix()
        starts_worker = "sys.executable" in text or "_worker" in text or "python_executable" in text
        if not starts_worker:
            continue
        for call in launches:
            env_kw = next((k for k in call.keywords if k.arg == "env"), None)
            if env_kw is None:
                # permitted only for the explicitly non-worker launches
                if any(rel.endswith(name) for name, _ in _NOT_CHOWDER_WORKERS):
                    continue
                offenders.append(f"{rel}:{call.lineno} launches without env=")
            else:
                src_of_env = ast.unparse(env_kw.value)
                ok = "worker_env" in src_of_env
                if not ok and rel.endswith("parent_tournament.py"):
                    ok = True  # env is a local built from _chowder_worker_env, checked below
                if not ok:
                    offenders.append(f"{rel}:{call.lineno} env={src_of_env} does not come from worker_env")
    assert not offenders, "Chowder worker launched without worker_env:\n  " + "\n  ".join(offenders)


def test_parent_tournament_builds_its_env_from_worker_env():
    text = (SRC / "parent_tournament.py").read_text(encoding="utf-8")
    assert "_chowder_worker_env(" in text
    assert '{**os.environ, "PYTHONUNBUFFERED": "1"}' not in text
