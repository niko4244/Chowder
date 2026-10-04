"""Parent cancellation must use the normal token and retain real CPU cost."""
import json
from types import SimpleNamespace

import pytest

from chowder.backends.router_healing import RouterHealingExecutor, RouterHealingRunSpec
from chowder.backends.router_healing import RouterHealingEvaluator, RouterHealingEvalSpec
from chowder.cancellation import CancellationToken
from chowder.cycle import ExperimentCycleRunner
from chowder.execution_failure import ExecutionFailure
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis


@pytest.mark.parametrize("terminal", ["cancel", "timeout", "invalid-result"])
def test_terminal_attempt_has_bound_identity_and_cpu_cost(tmp_path, monkeypatch, terminal):
    executor = RouterHealingExecutor()
    token = CancellationToken()
    ExperimentCycleRunner._bind_cancellation(None, executor, token)
    experiment = Experiment(
        experiment_id="parent-control", parent_id=None,
        hypothesis=Hypothesis(observation="o", suspected_cause="c", intervention="i"),
        config_patch={}, estimated_gpu_hours=0.1,
    )
    context = ExecutionContext(
        HardwareProfile(vram_gb=16, ram_gb=64, nvme_gb=1000, pcie_gbps=16,
                        ram_gbps=50, nvme_gbps=3), str(tmp_path), 0,
    )
    spec = RouterHealingRunSpec(
        base_model_dir="base", base_content_sha256="a" * 64,
        corpus_path="corpus", corpus_sha256="b" * 64, output_dir="output",
        max_steps=4, learning_rate=0.01, seq_len=16, batch_size=1, seed=0,
        probe_window=2, max_tokens=64,
    )
    monkeypatch.setattr(executor, "_spec_for", lambda *a, **k: spec)
    monkeypatch.setattr("chowder.backends.router_healing.chowder_source_identity", lambda: {})
    monkeypatch.setattr("chowder.backends.router_healing.uuid4",
                        lambda: SimpleNamespace(hex="0123456789ab"))
    ticks = iter(range(0, 100000, 1000))
    if terminal == "timeout":
        monkeypatch.setattr("chowder.backends.router_healing.time.perf_counter", lambda: next(ticks))

    class Process:
        returncode = None
        terminated = False

        def __init__(self, command, **kwargs):
            if terminal == "cancel":
                token.request()
            elif terminal == "invalid-result":
                from pathlib import Path
                Path(command[command.index("--result") + 1]).write_text("{}")
                self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -1

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("chowder.backends.router_healing.subprocess.Popen", Process)
    # Bound the old ignored-cancellation path too, so the RED control never hangs.
    if terminal == "cancel":
        monkeypatch.setattr("chowder.backends.router_healing._PROCESS_GRACE_SECONDS", -1)
    with pytest.raises(ExecutionFailure) as caught:
        executor.run(experiment, context)
    failure = caught.value
    assert failure.run_id == "parent-control-0123456789ab"
    assert failure.resource_usage.wall_seconds >= 0
    assert failure.gpu_hours_spent == 0
    assert failure.resource_usage.active_accelerator_count == 0
    path = tmp_path / ".chowder/runs" / failure.run_id / "run-failure.json"
    evidence = json.loads(path.read_text())
    assert evidence["run_id"] == failure.run_id
    assert evidence["wall_seconds"] == failure.resource_usage.wall_seconds
    assert evidence["terminal_state"] == ("cancelled" if terminal == "cancel" else "failed")
    assert not executor._processes
    assert not executor._cancelled


def _eval_spec(**changes):
    values = dict(
        base_model_dir="base", base_content_sha256="a" * 64, payload_dir=None,
        holdout_corpus_path="holdout", holdout_corpus_sha256="b" * 64,
        expected_parameter_paths=(), output_dir="output", seq_len=16, batches=2,
    )
    values.update(changes)
    return RouterHealingEvalSpec(**values)


def test_candidate_requires_receipt_and_protocol_is_shared_by_arms():
    base = _eval_spec()
    with pytest.raises(ValueError, match="recorded payload_manifest_sha256"):
        _eval_spec(payload_dir="payload", expected_parameter_paths=("gate",))
    candidate = _eval_spec(
        payload_dir="payload", expected_parameter_paths=("gate",),
        payload_manifest_sha256="c" * 64, payload_tensor_sha256="d" * 64,
    )
    assert candidate.protocol_digest("source") == base.protocol_digest("source")
    assert candidate.digest() != base.digest()
    assert _eval_spec(batches=3).protocol_digest("source") != base.protocol_digest("source")
    assert base.protocol_digest("different-source") != base.protocol_digest("source")


def test_evaluation_cancellation_is_not_lost_when_terminate_finishes_child(tmp_path, monkeypatch):
    evaluator = RouterHealingEvaluator()
    token = CancellationToken()
    evaluator.bind_cancellation(token)
    monkeypatch.setattr(evaluator, "_base_spec_for", lambda *a, **k: _eval_spec())
    monkeypatch.setattr("chowder.backends.router_healing.chowder_source_identity", lambda: {})

    class Process:
        returncode = None

        def __init__(self, *args, **kwargs):
            token.request()

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("chowder.backends.router_healing.subprocess.Popen", Process)
    context = ExecutionContext(
        HardwareProfile(vram_gb=16, ram_gb=64, nvme_gb=1000, pcie_gbps=16,
                        ram_gbps=50, nvme_gbps=3), str(tmp_path), 0,
    )
    with pytest.raises(ExecutionFailure) as caught:
        evaluator.evaluate_base(config={}, context=context)
    failure = caught.value
    assert failure.cause_type == "OperationCancelled"
    assert failure.gpu_hours_spent == 0
    assert failure.stage.value == "evaluate"
    path = tmp_path / ".chowder/evals" / failure.run_id / "run-failure.json"
    assert json.loads(path.read_text())["terminal_state"] == "cancelled"
    assert token._active is None
