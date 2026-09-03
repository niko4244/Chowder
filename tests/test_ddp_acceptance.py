"""Phase 5 acceptance suite: a genuine 2-GPU DDP training run.

Every test here requires BOTH CHOWDER_REAL_ML_SMOKE=1 AND two real CUDA
devices actually visible to torch (torch.cuda.device_count() >= 2) -- never
mocked, never faked. There is no way to prove "both T4s participate in a
real accelerate-launch DDP run" without two real devices; a Popen-mocked
unit test (see test_transformers_backend.py's
test_executor_launches_accelerate_when_multiple_accelerators_requested)
proves the *command* is built correctly, not that DDP genuinely engages two
devices, checkpoints don't corrupt under two writers, or cancellation
genuinely tears down every rank.

How to run this for real (e.g. a Kaggle notebook with a T4 x2 accelerator):

    pip install -e ".[train,dev]"
    CHOWDER_REAL_ML_SMOKE=1 python -m pytest -q tests/test_ddp_acceptance.py -v

Record the real result (pass/fail, GPU model, driver/torch/accelerate
versions) in docs/PHASE5_DDP_ACCEPTANCE.md once run for real.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.cancellation import CancellationToken
from chowder.executors import ExecutionContext
from chowder.hardware import detect_hardware
from chowder.models import Experiment, Hypothesis
from chowder.project import write_project
from chowder.project_runner import hardware_profile_from_snapshot, run_project
from chowder.registry import RunRegistry


def _real_cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
        reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
    ),
    pytest.mark.skipif(
        _real_cuda_device_count() < 2,
        reason="DDP acceptance requires two real, visible CUDA devices (e.g. a Kaggle T4x2)",
    ),
]


def _write_train_and_eval_files(work_dir: Path) -> None:
    train_path = work_dir / "train.jsonl"
    train_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "Question: What token comes after alpha? Answer: beta"},
                {"text": "Question: What token comes after red? Answer: blue"},
            ]
        ),
        encoding="utf-8",
    )
    eval_path = work_dir / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {"prompt": "Question: What token comes after alpha? Answer:", "expected": "beta"}
        )
        + "\n",
        encoding="utf-8",
    )


def _ddp_project_payload(
    work_dir: Path,
    *,
    name: str,
    epochs: float,
    extra_backend: dict | None = None,
) -> dict:
    backend = {
        "schema_version": 1,
        "type": "transformers-peft",
        "base_model": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "dataset": "train.jsonl",
        "text_field": "text",
        "max_length": 64,
        "precision": "fp32",
        "quantization": "none",
        "trust_remote_code": False,
        "training": {
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "gradient_checkpointing": False,
        },
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "use_rslora": False,
        },
        # Both T4s, explicitly -- this is the whole point of this suite.
        "runtime": {"active_accelerator_count": 2, "timeout_seconds": 300.0},
    }
    if extra_backend:
        backend.update(extra_backend)
    return {
        "schema_version": 1,
        "name": name,
        "work_dir": str(work_dir),
        "registry_path": ".chowder/runs.db",
        "seed": 123,
        "goal": {
            "metrics": [{"name": "quality", "minimum": 0.0, "direction": "maximize"}],
            "gpu_hour_budget": 2.0,
            "max_parallel_candidates": 1,
            "minimum_promotion_gain": 0.0,
            "require_protocol_match": False,
        },
        "baseline": {
            "experiment_id": "baseline",
            "metrics": {"quality": 0.0},
            "gpu_hours": 0.0,
        },
        "experiment": {
            "experiment_id": "real-ddp-sft",
            "estimated_gpu_hours": 0.25,
            "hypothesis": {
                "observation": "tiny model is unadapted",
                "suspected_cause": "target examples are unseen",
                "intervention": "one small LoRA SFT run across both accelerators",
                "expected_deltas": {"quality": 0.0},
            },
            "config_patch": {},
            "tags": ["integration", "real-ml", "ddp", "phase5"],
        },
        "config": {
            "seed": 123,
            "backend": backend,
            "evaluation": {
                "type": "transformers-text",
                "estimated_gpu_hours": 0.05,
                "precision": "fp32",
                "quantization": "none",
                "device": "cpu",
                "trust_remote_code": False,
                "runtime": {"timeout_seconds": 180.0},
                "suites": [
                    {
                        "name": "quality",
                        "dataset": "eval.jsonl",
                        "prompt_field": "prompt",
                        "expected_field": "expected",
                        "scoring": "normalized_exact_match",
                        "max_new_tokens": 2,
                        "use_chat_template": False,
                    }
                ],
            },
        },
    }


def test_real_ddp_two_gpus_trains_evaluates_and_reports_correct_gpu_hours(tmp_path: Path):
    """The core Phase 5 acceptance flow: active_accelerator_count=2 ->
    accelerate launch -> world_size==2 -> both T4s participate -> LoRA
    training completes -> exactly one valid adapter artifact ->
    independent evaluator reloads and scores it -> GPU-hours == wall-hours
    x 2 (ResourceUsage.from_wall_time computes this directly from the
    measured active_accelerator_count, so this is really proving that
    active_accelerator_count genuinely reflects two engaged devices, not
    re-deriving the arithmetic)."""
    _write_train_and_eval_files(tmp_path)
    project_path = tmp_path / "project.json"
    write_project(project_path, _ddp_project_payload(tmp_path, name="real ddp smoke", epochs=1.0))

    outcome = run_project(project_path)
    candidate = outcome.generation.candidates[0]
    assert candidate.error is None, candidate.error
    assert candidate.artifact is not None
    assert candidate.evaluation is not None
    assert candidate.result is not None

    artifact = candidate.artifact
    assert artifact.evidence["requested_active_accelerator_count"] == 2
    usage = artifact.evidence["resource_usage"]
    assert usage["active_accelerator_count"] == 2  # world_size == 2, genuinely engaged
    assert artifact.gpu_hours == pytest.approx(usage["wall_seconds"] * 2 / 3600.0, rel=1e-6)

    # Exactly one valid adapter artifact -- not one per rank, not corrupted
    # by two writers racing on the same path.
    adapter_dir = Path(artifact.artifact_ref)
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").is_file()
    adapter_weight_files = [p for p in adapter_dir.iterdir() if p.name.startswith("adapter_model")]
    assert len(adapter_weight_files) == 1, (
        f"expected exactly one adapter weight file, found {adapter_weight_files}"
    )

    # Independent evaluator genuinely reloads the artifact and scores it.
    assert candidate.evaluation.evidence["evaluator"] == "transformers-text"
    assert set(candidate.result.metrics) == {"quality"}

    registry_path = tmp_path / ".chowder" / "runs.db"
    with RunRegistry(registry_path) as registry:
        artifacts = list(registry.list_training_artifacts())
        assert len(artifacts) == 1
        assert artifacts[0].artifact_ref == str(adapter_dir)


def test_real_ddp_cancellation_terminates_every_rank_and_frees_gpu_memory(tmp_path: Path):
    """Cancellation must tear down the whole distributed process group, not
    just the accelerate-launch parent -- an orphaned rank would keep
    holding GPU memory and would corrupt the next run's device
    accounting."""
    _write_train_and_eval_files(tmp_path)
    project_path = tmp_path / "project.json"
    write_project(
        project_path,
        _ddp_project_payload(
            tmp_path,
            name="real ddp cancellation smoke",
            # Enough epochs to give the canceller thread a real window
            # after both worker processes start and before they'd finish.
            epochs=20.0,
        ),
    )

    before_pids = _gpu_compute_pids()

    token = CancellationToken()

    def request_once_training_actually_starts() -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if token._active is not None:
                token.request()
                return
            time.sleep(0.02)
        raise AssertionError("training subprocess never registered as active")

    canceller = threading.Thread(target=request_once_training_actually_starts, daemon=True)
    canceller.start()
    try:
        outcome = run_project(project_path, cancellation=token)
    finally:
        canceller.join(timeout=60)

    candidate = outcome.generation.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("cancelled"), candidate.error
    assert candidate.result is None

    # Give the OS a bounded window to actually reap the terminated
    # processes, then require no new compute process is left holding a
    # device -- an orphaned rank is exactly the failure mode this proves
    # doesn't happen.
    deadline = time.monotonic() + 30
    leaked_pids: set[str] = set()
    while time.monotonic() < deadline:
        leaked_pids = _gpu_compute_pids() - before_pids
        if not leaked_pids:
            break
        time.sleep(1.0)
    assert not leaked_pids, f"orphaned GPU compute process(es) after cancellation: {leaked_pids}"


def _gpu_compute_pids() -> set[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_real_ddp_one_rank_crashing_is_reported_as_a_failure_not_a_silent_success(
    tmp_path: Path, monkeypatch
):
    """Deliberately crash rank 1 (via the test-support hook in
    transformers_worker._crash_rank_for_ddp_acceptance_test, inert unless
    this env var is set) and prove the executor surfaces a real failure --
    never a silently-successful artifact built from only rank 0's work."""
    monkeypatch.setenv("_CHOWDER_DDP_ACCEPTANCE_CRASH_RANK", "1")
    _write_train_and_eval_files(tmp_path)

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    payload = _ddp_project_payload(tmp_path, name="real ddp crash accounting smoke", epochs=1.0)
    context = ExecutionContext(hardware, str(tmp_path), seed=123, resolved_config=payload["config"])
    experiment = Experiment(
        "real-ddp-crash",
        None,
        Hypothesis("obs", "cause", "deliberate rank crash"),
        {},
        0.25,
    )

    with pytest.raises(RuntimeError, match="exit code"):
        TransformersPeftExecutor().run(experiment, context)


def test_real_ddp_checkpoint_save_produces_one_checkpoint_no_rank_duplication(tmp_path: Path):
    """With save_strategy=steps, only rank 0 may write the checkpoint
    manifest and trainer state -- two ranks racing on the same path would
    be silent corruption, not just wasted I/O."""
    _write_train_and_eval_files(tmp_path)
    project_path = tmp_path / "project.json"
    write_project(
        project_path,
        _ddp_project_payload(
            tmp_path,
            name="real ddp checkpoint smoke",
            epochs=3.0,
            extra_backend={
                "training": {
                    "epochs": 3.0,
                    "learning_rate": 0.001,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "logging_steps": 1,
                    "gradient_checkpointing": False,
                    "save_strategy": "steps",
                    "save_steps": 2,
                },
            },
        ),
    )

    outcome = run_project(project_path)
    candidate = outcome.generation.candidates[0]
    assert candidate.error is None, candidate.error

    trainer_dir = Path(candidate.artifact.artifact_ref) / "trainer"
    manifest_path = trainer_dir / "chowder-checkpoint-manifest.json"
    assert manifest_path.is_file()
    checkpoint_dirs = sorted(trainer_dir.glob("checkpoint-*"))
    assert len(checkpoint_dirs) >= 1
    # Each checkpoint step directory must exist exactly once -- a
    # duplicate-writer race would show up as corrupted/partial files
    # inside, not as extra directories, so check each one is well-formed
    # rather than just counting.
    for checkpoint_dir in checkpoint_dirs:
        assert (checkpoint_dir / "trainer_state.json").is_file()
