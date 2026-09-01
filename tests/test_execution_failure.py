import pytest

from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.execution_failure import ExecutionStage, normalize_execution_exception
from chowder.executors import CostEstimate, ExecutionContext
from chowder.investigation import Investigation
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.incident import SignatureKind


def _hardware():
    return HardwareProfile(
        vram_gb=16,
        ram_gb=32,
        nvme_gb=100,
        pcie_gbps=12,
        ram_gbps=40,
        nvme_gbps=3,
    )


def _experiment():
    return Experiment(
        "crashy",
        None,
        Hypothesis("observe crash", "runtime mismatch", "investigate"),
        {},
        3.0,
    )


def _context(tmp_path):
    return ExecutionContext(
        hardware=_hardware(),
        work_dir=str(tmp_path),
        seed=1,
        resolved_config={
            "backend": {
                "runtime": {"active_accelerator_count": 2},
            }
        },
    )


def test_raw_kaggle_t4x2_exception_normalizes_to_two_gpu_hours(tmp_path):
    exc = RuntimeError(
        "CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"
    )
    failure = normalize_execution_exception(
        exc,
        experiment=_experiment(),
        executor_name="transformers-peft",
        context=_context(tmp_path),
        wall_seconds=3600,
    )
    assert failure.stage is ExecutionStage.TRAIN
    assert failure.resource_usage is not None
    assert failure.resource_usage.active_accelerator_count == 2
    assert failure.resource_usage.gpu_hours == pytest.approx(2.0)
    assert "CUBLAS_STATUS_EXECUTION_FAILED" in failure.cause_message


class _CuBlasCrashTrainer:
    name = "transformers-peft"

    def profile(self, experiment, context):
        return CostEstimate(gpu_hours=1.0)

    def run(self, experiment, context):
        raise RuntimeError(
            "qwen3_5 gated delta rule failed: CUBLAS_STATUS_EXECUTION_FAILED"
        )

    def cancel(self, run_id):
        return None


class _NeverEvaluator:
    name = "never"

    def evaluate(self, *, experiment, artifact, context):
        raise AssertionError("evaluation must not run after training crash")


def test_cycle_routes_live_cublas_crash_into_executor_investigator(tmp_path):
    experiment = _experiment()
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=1),), gpu_hour_budget=5),
        ExperimentResult("base", {"quality": 1}, 0),
    )
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_CuBlasCrashTrainer(),
        evaluator=_NeverEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config={
            "backend": {
                "runtime": {"active_accelerator_count": 2},
            }
        },
    )

    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.result is None
    assert candidate.execution_failure is not None
    assert candidate.executor_analysis is not None
    assert candidate.executor_analysis.fingerprint.signature_kind is SignatureKind.CUDA_EXECUTION_FAILED
    assert isinstance(candidate.executor_analysis.routed, Investigation)
    assert candidate.executor_analysis.capture.exception_type == "RuntimeError"
    assert "CUBLAS_STATUS_EXECUTION_FAILED" in candidate.executor_analysis.capture.exception_message
    assert candidate.executor_analysis.capture.environment.accelerator_count == 2
    assert engine.graph.nodes[experiment.experiment_id].status.value == "failed"
    assert engine.spent_gpu_hours > 0


def test_execution_failure_analysis_preserves_resolved_config_for_context_identity(tmp_path):
    experiment = _experiment()
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=1),), gpu_hour_budget=5),
        ExperimentResult("base", {"quality": 1}, 0),
    )
    engine.propose((experiment,))
    config = {
        "backend": {
            "base_model": "Qwen/Qwen3.5",
            "runtime": {"active_accelerator_count": 2},
        }
    }
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_CuBlasCrashTrainer(),
        evaluator=_NeverEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config=config,
    )
    candidate = runner.run_generation((experiment,)).candidates[0]
    assert candidate.executor_analysis is not None
    assert candidate.executor_analysis.capture.environment.config_patch == config
