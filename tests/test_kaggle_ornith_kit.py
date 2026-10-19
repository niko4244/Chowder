import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kaggle" / "ornith_curriculum"
sys.path.insert(0, str(KIT))

import make_dataset  # noqa: E402
import make_kernel  # noqa: E402
import run_ornith  # noqa: E402


def test_models_are_pinned():
    models = json.loads((KIT / "models.json").read_text())
    assert [m["alias"] for m in models] == ["ornith-1.5-9b", "qwythos-27b"]
    for m in models:
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"])


def test_server_args_match_local_run():
    assert run_ornith.PROPOSER_ARGS == ["--alias", "ornith-1.5-9b", "--jinja", "--ctx-size", "12288",
                                        "--n-gpu-layers", "999", "--flash-attn", "on"]
    assert run_ornith.CROSSCHECK_ARGS == ["--alias", "qwythos-27b", "--jinja", "--ctx-size", "8192", "--fit", "on",
                                          "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]


def test_kernel_inlines_models_and_is_private(tmp_path):
    make_kernel.build("me", tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["dataset_sources"] == ["me/ornith-hermes"]
    script = (tmp_path / "run_ornith.py").read_text()
    assert "__MODELS__" not in script
    ns: dict = {}
    exec(compile(script.split("WORK = Path")[0], "k", "exec"), ns)
    assert json.loads(ns["MODELS_JSON"])[1]["alias"] == "qwythos-27b"


def test_dataset_is_importable_offline(tmp_path):
    hermes = tmp_path / "repo" / "hermes"
    hermes.mkdir(parents=True)
    (hermes / "__init__.py").write_text("")
    (hermes / "admission.py").write_text("X = 1\n")
    (hermes / "junk.pyc").write_bytes(b"x")
    subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True)
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "-c", "user.email=a@b", "-c", "user.name=a",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    phase = tmp_path / "ornith_phase.py"
    phase.write_text("import hermes.admission\n")
    out = tmp_path / "ds"
    manifest = make_dataset.build("me", hermes, phase, out)
    assert not (out / "hermes" / "junk.pyc").exists()
    assert {"hermes/admission.py", "ornith_phase.py", "run_baseline.py"} <= set(manifest["files"])
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["hermes_commit"])
    done = subprocess.run([sys.executable, "-c", "import hermes.admission, ornith_phase, run_baseline"],
                          cwd=tmp_path, env={"PYTHONPATH": str(out), "SYSTEMROOT": "C:\\Windows"},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_propose_timeout_keeps_written_rows(tmp_path):
    out = tmp_path / "p.jsonl"
    out.write_text('{"a": 1}\n')
    slow = [sys.executable, "-c", "import time; time.sleep(5)"]
    assert run_ornith.run_budgeted(slow, 1, out) is False


def test_propose_empty_output_fails(tmp_path):
    out = tmp_path / "p.jsonl"
    with pytest.raises(RuntimeError, match="missing or empty"):
        run_ornith.run_budgeted([sys.executable, "-c", "pass"], 30, out)
