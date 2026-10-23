"""Real CPU-only producer→evaluator→worker adapter binding contract.

The worker's own load must refuse replacement of an exported LoRA after the
controller's evidence was written, *before* importing torch/transformers or
reading the multi-GB REAP parent. Test bytes are synthetic, never GPU/model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.evaluators.adapter_identity import AdapterIdentityRefusal, verify_adapter_reload
from chowder.evaluators.base_text import BaseModelTextEvaluator, BaseTextEvalSpec
from chowder.evaluators.base_text_worker import evaluate as baseline_worker
from chowder.evaluators.transformers_text import EvalSuiteSpec, TransformersTextEvalSpec
from chowder.evaluators.transformers_text_worker import evaluate as candidate_worker
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.provenance import sha256_directory


def _adapter(tmp_path: Path) -> Path:
    directory = tmp_path / "exported-adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16}) + "\n",
        encoding="utf-8",
    )
    # The verifier reads bytes only to compute the SAME SHA-256 as production;
    # the live PEFT liveness guard separately checks saved tensor applicability.
    (directory / "adapter_model.safetensors").write_bytes(b"synthetic-adapter-bytes")
    return directory


def _suite(tmp_path: Path) -> EvalSuiteSpec:
    dataset = tmp_path / "slice.jsonl"
    dataset.write_text('{"prompt":"x","expected":"y"}\n', encoding="utf-8")
    return EvalSuiteSpec(name="tool-use", dataset=str(dataset))


def _baseline_spec(tmp_path: Path, adapter: Path, sha: str) -> BaseTextEvalSpec:
    return BaseTextEvalSpec(
        base_model="nonexistent/REAP160", output_dir=str(tmp_path / "result"),
        suites=(_suite(tmp_path),), adapter_dir=str(adapter), adapter_sha256=sha,
    )


def _candidate_spec(tmp_path: Path, adapter: Path, sha: str) -> TransformersTextEvalSpec:
    return TransformersTextEvalSpec(
        base_model="nonexistent/REAP160", output_dir=str(tmp_path / "result"),
        suites=(_suite(tmp_path),), adapter_dir=str(adapter), adapter_sha256=sha,
    )


def test_preflight_verifies_exact_exported_directory_using_production_digest(tmp_path):
    adapter = _adapter(tmp_path)
    sha = sha256_directory(adapter)
    assert verify_adapter_reload(adapter, sha, label="candidate") == sha
    assert verify_adapter_reload(None, None, label="bare REAP160") is None
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("problem", [
    "no_digest", "malformed_digest", "changed_weights", "changed_config",
    "extra_file", "missing_weights", "missing_config", "empty_weights", "missing_directory",
])
def test_adversarial_adapter_tamper_is_refused_before_model_load(tmp_path, problem):
    adapter = _adapter(tmp_path)
    sha = sha256_directory(adapter)
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if problem == "no_digest":
        sha = None
    elif problem == "malformed_digest":
        sha = "g" * 64
    elif problem == "changed_weights":
        weights.write_bytes(b"substituted-adapter-weights")
    elif problem == "changed_config":
        config.write_text('{"peft_type":"IA3"}\n', encoding="utf-8")
    elif problem == "extra_file":
        (adapter / "backdoored-extra.txt").write_text("new contents", encoding="utf-8")
    elif problem == "missing_weights":
        weights.unlink()
    elif problem == "missing_config":
        config.unlink()
    elif problem == "empty_weights":
        weights.write_bytes(b"")
    else:
        # No destructive filesystem mutation needed: an unmatched path is
        # enough to prove missing is not interpreted as zero/unknown evidence.
        adapter = tmp_path / "missing-adapter"
    with pytest.raises(AdapterIdentityRefusal):
        verify_adapter_reload(adapter, sha, label="candidate")


def test_both_workers_refuse_adapter_replacement_before_importing_model_stack(tmp_path):
    adapter = _adapter(tmp_path)
    sha = sha256_directory(adapter)
    baseline = _baseline_spec(tmp_path, adapter, sha)
    candidate = _candidate_spec(tmp_path, adapter, sha)
    # Model stack cannot load: the refusal happens on the very first line of
    # evaluate(), with no GPU, download, or parent checkpoint read.
    (adapter / "adapter_model.safetensors").write_bytes(b"replaced-after-controller-preflight")
    with pytest.raises(AdapterIdentityRefusal, match="parent baseline worker.*digest changed"):
        baseline_worker(baseline)
    with pytest.raises(AdapterIdentityRefusal, match="candidate worker.*digest changed"):
        candidate_worker(candidate)


def test_parent_baseline_project_spec_carries_exact_declared_parent_digest(tmp_path):
    adapter = _adapter(tmp_path)
    sha = sha256_directory(adapter)
    suite = _suite(tmp_path)
    config = {
        "backend": {
            "base_model": "nonexistent/REAP160",
            "parent_adapter": {"path": str(adapter), "sha256": sha},
        },
        "evaluation": {
            "suites": [{"name": suite.name, "dataset": suite.dataset}],
        },
    }
    spec = BaseTextEvalSpec.from_config(config, work_dir=tmp_path,
                                        output_dir=tmp_path / "result", seed=1)
    assert spec.adapter_dir == str(adapter.resolve())
    assert spec.adapter_sha256 == sha
    assert json.loads(spec.canonical_json())["adapter_sha256"] == sha


def test_production_parent_baseline_controller_refuses_before_subprocess(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    enrolled = sha256_directory(adapter)
    suite = _suite(tmp_path)
    config = {
        "backend": {"base_model": "nonexistent/REAP160",
                    "parent_adapter": {"path": str(adapter), "sha256": enrolled}},
        "evaluation": {"suites": [{"name": suite.name, "dataset": suite.dataset}]},
    }
    context = ExecutionContext(
        HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 1,
        resolved_config=config,
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation subprocess must not be started")
    monkeypatch.setattr("chowder.evaluators.base_text.subprocess.Popen", forbidden)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered parent")
    with pytest.raises(AdapterIdentityRefusal, match="parent baseline.*digest changed"):
        BaseModelTextEvaluator().evaluate(config=config, context=context)


def test_candidate_evaluator_spec_binds_training_artifact_directory_digest(tmp_path):
    adapter = _adapter(tmp_path)
    sha = sha256_directory(adapter)
    suite = _suite(tmp_path)
    artifact = TrainingArtifact(
        run_id="run-1", experiment_id="gen1-fable-lr5e-05",
        artifact_ref=str(adapter), gpu_hours=0.1,
        evidence={"artifact_sha256": sha, "model_provenance": {}},
    )
    config = {
        "backend": {"base_model": "nonexistent/REAP160"},
        "evaluation": {"suites": [{"name": suite.name, "dataset": suite.dataset}]},
    }
    spec = TransformersTextEvalSpec.from_context(
        config=config, artifact=artifact, work_dir=tmp_path,
        output_dir=tmp_path / "result", seed=1,
    )
    assert spec.adapter_sha256 == sha
    assert json.loads(spec.canonical_json())["adapter_sha256"] == sha


def test_base_only_eval_retains_legacy_spec_hash_and_cannot_claim_adapter_sha(tmp_path):
    bare = BaseTextEvalSpec(
        base_model="nonexistent/REAP160", output_dir=str(tmp_path / "result"),
        suites=(_suite(tmp_path),),
    )
    assert bare.adapter_dir is None
    assert "adapter_sha256" not in json.loads(bare.canonical_json())
    with pytest.raises(AdapterIdentityRefusal, match="base-only"):
        verify_adapter_reload(None, "a" * 64, label="base-only")


def test_peft_declared_local_parent_must_be_the_exact_local_base_reloaded(tmp_path):
    expected_base = tmp_path / "REAP160"
    wrong_base = tmp_path / "REAP128"
    expected_base.mkdir()
    wrong_base.mkdir()
    adapter = _adapter(tmp_path)
    config = adapter / "adapter_config.json"
    record = json.loads(config.read_text(encoding="utf-8"))
    record["base_model_name_or_path"] = str(wrong_base)
    config.write_text(json.dumps(record) + "\n", encoding="utf-8")
    # Even if a careless caller enrolls a NEW SHA matching this misattached
    # adapter, the metadata must not authorize a different REAP parent.
    new_sha = sha256_directory(adapter)
    with pytest.raises(AdapterIdentityRefusal, match="wrong-parent reload"):
        verify_adapter_reload(adapter, new_sha, label="Gen1 adapter", base_model=expected_base)
    record["base_model_name_or_path"] = str(expected_base)
    config.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert verify_adapter_reload(adapter, sha256_directory(adapter), label="Gen1 adapter",
                                 base_model=expected_base) == sha256_directory(adapter)


def test_symlinked_artifact_file_is_refused_even_if_hash_were_reenrolled(tmp_path):
    adapter = _adapter(tmp_path)
    target = tmp_path / "other.safetensors"
    target.write_bytes(b"untrusted-content")
    weights = adapter / "adapter_model.safetensors"
    weights.unlink()
    try:
        os.symlink(target, weights)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this test host")
    with pytest.raises(AdapterIdentityRefusal):
        verify_adapter_reload(adapter, "a" * 64, label="candidate")
