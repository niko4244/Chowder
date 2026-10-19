"""Tests for the Kaggle REAP baseline kit (``kaggle/reap_baseline/``).

What is testable without Kaggle, the network, a GPU or a model is tested here:
the frozen arm table, both generators' outputs (metadata, rewritten arm
constants, and the copied import closure -- asserted by importing that closure
in a subprocess with only the dataset directory on ``sys.path``), and the
streaming sha256 verifier's two outcomes. The kernel script needs a Kaggle GPU
image, so it is exercised through the parts that do not: it is imported and its
arm identity and verifier are called directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "kaggle" / "reap_baseline"
KIT_FILES = ("baselines.json", "run_baseline.py", "make_kernels.py", "make_dataset.py", "README.md")
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$", re.M)
# The closure the kernel imports at scoring time; the probe runs with -S so the
# only chowder on the path is the one the dataset generator copied.
PROBE = (
    "import json, reap_eval, chowder.evaluators.scoring as scoring;"
    "print(json.dumps({'eval': reap_eval.__file__, 'scoring': scoring.__file__,"
    " 'score': scoring.score('The answer is 12', '12', 'final_number_match')}))"
)
FROZEN_ARMS = {
    "qwen38_27b": (
        "unsloth/Qwen3.8-27B-GGUF",
        "fdd03b8bbd279c1694563650e79d85a2373d9934",
        "Qwen3.8-27B-Q4_K_M.gguf",
        "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
    ),
    "qwythos_27b": (
        "empero-ai/Qwythos-27B-v1-GGUF",
        "main",
        "Qwythos-27B-Q4_K_M.gguf",
        "817aa895cd3b6e0680c7aa8da62ec393f3125ff0ab676f8f06a8fc759d20e3d5",
    ),
    "ornith15_35b": (
        "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
        "fbbaed45c2f0e200276ffa51701a24d45dc7f57e",
        "Ornith-1.5-35B-Q4_K_M.gguf",
        "ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f",
    ),
}


def _load_kernel_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def _baselines() -> list[dict]:
    return json.loads((KIT / "baselines.json").read_text(encoding="utf-8"))


def test_kit_files_exist():
    """Fails on origin/main: the kit is what this branch adds."""
    missing = [name for name in KIT_FILES if not (KIT / name).is_file()]
    assert not missing, f"kaggle/reap_baseline is incomplete: missing {missing}"


def test_baselines_json_is_the_frozen_arm_table():
    entries = _baselines()
    assert [entry["arm"] for entry in entries] == list(FROZEN_ARMS)
    for entry in entries:
        assert set(entry) == {"arm", "hf_repo", "revision", "file", "sha256"}
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry
        assert entry["revision"] == "main" or re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry
        assert entry["file"].endswith(".gguf"), entry
        got = (entry["hf_repo"], entry["revision"], entry["file"], entry["sha256"])
        assert got == FROZEN_ARMS[entry["arm"]], entry["arm"]  # a typo here is a wrong download, not a typo


def test_make_kernels_writes_private_gpu_kernels_with_the_arm_rewritten(tmp_path):
    out = tmp_path / "kernels"
    done = _run(KIT / "make_kernels.py", "--owner", "someone", "--out", str(out))
    assert done.returncode == 0, done.stderr

    template = _load_kernel_module(KIT / "run_baseline.py", "kit_run_baseline_template")
    assert template.ARM == _baselines()[0]["arm"]  # the template ships arm 0's identity
    assert template.BASELINE == _baselines()[0]

    for entry in _baselines():
        arm = entry["arm"]
        arm_dir = out / arm
        assert (arm_dir / "run_baseline.py").is_file() and (arm_dir / "kernel-metadata.json").is_file()

        metadata = json.loads((arm_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert set(metadata) == {"id", "title", "code_file", "language", "kernel_type", "is_private",
                                 "enable_gpu", "enable_internet", "dataset_sources"}
        assert metadata["id"] == f"someone/reap-baseline-{arm.replace('_', '-')}"
        assert isinstance(metadata["title"], str) and 0 < len(metadata["title"]) <= 50
        assert metadata["code_file"] == "run_baseline.py"
        assert metadata["language"] == "python"
        assert metadata["kernel_type"] == "script"
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is True
        assert metadata["dataset_sources"] == ["someone/chowder-reap-eval"]

        source = (arm_dir / "run_baseline.py").read_text(encoding="utf-8")
        assert f'ARM = "{arm}"' in source
        generated = _load_kernel_module(arm_dir / "run_baseline.py", f"kit_kernel_{arm}")
        assert generated.ARM == arm
        assert generated.BASELINE == entry
        assert generated.OUT == Path("/kaggle/working/eval")  # arm results land in the kernel's own output


def test_make_dataset_copies_the_import_closure_that_imports_offline(tmp_path):
    out = tmp_path / "chowder-reap-eval"
    done = _run(KIT / "make_dataset.py", "--owner", "someone", "--out", str(out))
    assert done.returncode == 0, done.stderr

    required = ("reap_serve.py", "reap_eval.py", "chowder/evaluators/scoring.py", "dataset-metadata.json")
    for name in required:
        assert (out / name).is_file(), f"{name} missing from the dataset"

    metadata = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert set(metadata) == {"id", "title", "licenses", "isPrivate"}
    assert metadata["id"] == "someone/chowder-reap-eval"
    assert metadata["licenses"] == [{"name": "other"}]
    assert metadata["isPrivate"] is True

    manifest = {path: digest for digest, path in MANIFEST_LINE.findall(done.stdout)}
    assert set(required) <= set(manifest), f"manifest does not cover the closure: {sorted(manifest)}"
    for path, digest in manifest.items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == digest, path

    env = dict(os.environ)
    env["PYTHONPATH"] = str(out)
    env["HF_HUB_OFFLINE"] = env["HF_DATASETS_OFFLINE"] = "1"  # an import that needs the network fails here
    probe = subprocess.run([sys.executable, "-S", "-c", PROBE], capture_output=True, text=True,
                           timeout=180, env=env, cwd=str(tmp_path))
    assert probe.returncode == 0, probe.stderr
    payload = json.loads(probe.stdout.strip().splitlines()[-1])
    assert Path(payload["eval"]).parent == out
    assert Path(payload["scoring"]) == out / "chowder" / "evaluators" / "scoring.py"
    assert payload["score"] == 1.0  # the copied scorer runs, not just imports


def test_sha256_verifier_refuses_a_mismatch_and_accepts_a_match(tmp_path):
    module = _load_kernel_module(KIT / "run_baseline.py", "kit_run_baseline_verifier")
    payload = tmp_path / "fixture.bin"
    payload.write_bytes(b"chowder" * 1000)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    assert module.verify_sha256(payload, digest, chunk_size=7) == digest  # streaming, unaligned chunks
    with pytest.raises(ValueError) as raised:
        module.verify_sha256(payload, "0" * 64, chunk_size=7)
    assert "mismatch" in str(raised.value)
    with pytest.raises(FileNotFoundError):
        module.verify_sha256(tmp_path / "absent.bin", digest)
