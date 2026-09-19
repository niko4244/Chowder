"""One declared evaluation throughput, read by every path that measures a model.

Batched decoding is not a neutral detail of how an evaluation runs: measured on
the frozen Gen-0 base over the frozen 16-item Math500 slice at
``max_new_tokens=4``, a batch-16 pass agreed with a single-row pass on 14 of 16
rows, while two single-row passes agreed on 16 of 16. Batched and single-row
decoding therefore generate *different tokens*, so a batched arm and a
single-row arm are not comparable and the choice has to be declared, recorded
and shared -- never chosen per path.

These tests pin the sharing: the campaign declares ``evaluation_execution``,
the arms read that value into the spec they run, the candidate evaluator reads
the same value, and the value is recorded in the evidence each one writes. An
execution key nobody reads is refused rather than accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from chowder.growth.campaign import (
    CampaignManifest,
    CampaignManifestError,
    EvaluationExecution,
)

TARGET_ID = "generation-diagnostics@gen1-eval-protocol-v1"
PROTECTED_ID = "math500@2024-04"
BROAD_ID = "mgsm@2022-11"

PROTECTION: dict[str, Any] = {
    "trusted_ancestor_version": "gen0",
    "slice_regression_max": 0.0625,
    "n_samples": 16,
    "seed": 1234,
    "shuffle": False,
    "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 8},
    "prompt_policy": "chat_template",
}


def _manifest_document(tmp_path: Path) -> dict[str, Any]:
    return {
        "cycle_id": "gen2-execution-fixture",
        "parent_version": "gen1",
        "base_model_path": str(tmp_path / "base"),
        "base_model_digest": "a" * 64,
        "state_root": str(tmp_path / "state"),
        "target_benchmarks": [TARGET_ID],
        "protected_benchmarks": [PROTECTED_ID],
        "broad_benchmarks": [BROAD_ID],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": {
            "device_gpu_hours_ceiling_per_recipe": 0.30,
            "wall_gpu_hours_ceiling_per_recipe": 0.75,
            "device_gpu_hours_ceiling_campaign": 0.60,
            "wall_gpu_hours_ceiling_campaign": 1.50,
            "device_time_measured": False,
        },
        "recipes": ["recipe-a", "recipe-b"],
        "candidate_selection_policy": "first_successful",
        "stopping_rules": ["stop before compute on admission refusal"],
        "protection": dict(PROTECTION),
    }


def _manifest(tmp_path: Path, *, batch_size: int | None) -> CampaignManifest:
    document = _manifest_document(tmp_path)
    if batch_size is not None:
        document["evaluation_execution"] = {"batch_size": batch_size}
    return CampaignManifest.from_mapping(document, source="test")


def _slice_source(items: int = 16):
    def source(qualified_id: str):
        return tuple(
            {"prompt": f"{qualified_id} item {index}", "expected": str(index)}
            for index in range(items)
        )

    return source


class _RecordingEvalRunner:
    """Answers the arm worker without a model, keeping the spec it was given."""

    def __init__(self) -> None:
        self.specs: list[dict[str, Any]] = []

    def __call__(self, command, cwd, env, timeout):  # noqa: ANN001
        from chowder.growth.training_binding import SubprocessOutcome

        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.specs.append(spec)
        out_dir = Path(spec["output_dir"])
        suites: dict[str, Any] = {}
        for suite in spec["suites"]:
            dataset = Path(suite["dataset"])
            items = [
                json.loads(line)
                for line in dataset.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            predictions = out_dir / f"predictions-{suite['name']}.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": item["prompt"],
                            "expected": item["expected"],
                            "prediction": item["expected"],
                            "score": 1.0,
                        }
                    )
                    + "\n"
                    for item in items
                ),
                encoding="utf-8",
            )
            suites[suite["name"]] = {"predictions_file": str(predictions)}
        result_path.write_text(
            json.dumps(
                {
                    "metrics": {name: 1.0 for name in suites},
                    "suites": suites,
                    "runtime": {"gpu_count": 1},
                }
            ),
            encoding="utf-8",
        )
        return SubprocessOutcome(
            command=list(command),
            returncode=0,
            stdout="",
            stderr="",
            seconds=12.0,
            timed_out=False,
        )


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------


def test_an_absent_execution_block_means_one_row_per_call(tmp_path: Path) -> None:
    assert _manifest(tmp_path, batch_size=None).evaluation_execution.batch_size == 1


def test_an_unrecognised_execution_key_is_refused() -> None:
    with pytest.raises(CampaignManifestError) as error:
        EvaluationExecution.from_mapping({"batch_size": 4, "threads": 8}, source="test")
    assert "threads" in str(error.value)
    assert "nothing reads" in str(error.value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_an_execution_throughput_that_is_not_a_positive_integer_is_refused(
    value: Any,
) -> None:
    with pytest.raises(CampaignManifestError):
        EvaluationExecution.from_mapping({"batch_size": value}, source="test")


def test_the_manifest_refuses_an_execution_block_of_the_wrong_shape(tmp_path: Path) -> None:
    document = _manifest_document(tmp_path)
    document["evaluation_execution"] = 16
    with pytest.raises(CampaignManifestError):
        CampaignManifest.from_mapping(document, source="test")


# --------------------------------------------------------------------------
# the arms read it
# --------------------------------------------------------------------------


def test_both_arm_measurements_run_at_the_declared_throughput(tmp_path: Path) -> None:
    """The ancestor and the parent arm are measured the way the candidate will be."""
    from chowder.growth.campaign_prepare import measure_ancestor_arm

    manifest = _manifest(tmp_path, batch_size=8)
    runner = _RecordingEvalRunner()
    measure_ancestor_arm(
        manifest,
        out_path=tmp_path / "arm.json",
        slice_source=_slice_source(),
        runner=runner,
    )
    assert runner.specs, "the arm worker was never invoked"
    for spec in runner.specs:
        assert spec["suites"], "a spec with no suites measures nothing"
        assert {suite["batch_size"] for suite in spec["suites"]} == {8}


def test_the_arm_records_the_throughput_it_was_measured_at(tmp_path: Path) -> None:
    """A reader can tell a batched arm from a single-row one from the artifact."""
    from chowder.growth.campaign_prepare import measure_ancestor_arm

    manifest = _manifest(tmp_path, batch_size=8)
    arm = measure_ancestor_arm(
        manifest,
        out_path=tmp_path / "arm.json",
        slice_source=_slice_source(),
        runner=_RecordingEvalRunner(),
    )
    report = json.loads(Path(arm.report_path).read_text(encoding="utf-8"))
    for run in report["runs"]:
        assert run["metadata"]["decoding"]["batch_size"] == 8
        # The declared decoding is untouched: only the execution is recorded.
        assert run["metadata"]["decoding"]["max_new_tokens"] == 8
        assert run["metadata"]["decoding"]["do_sample"] is False


def test_the_arm_names_the_throughput_the_slice_fixture_expects(tmp_path: Path) -> None:
    from chowder.growth.campaign_prepare import measure_ancestor_arm

    manifest = _manifest(tmp_path, batch_size=None)
    runner = _RecordingEvalRunner()
    measure_ancestor_arm(
        manifest,
        out_path=tmp_path / "arm.json",
        slice_source=_slice_source(),
        runner=runner,
    )
    assert {suite["batch_size"] for suite in runner.specs[0]["suites"]} == {1}


# --------------------------------------------------------------------------
# the candidate reads the same value
# --------------------------------------------------------------------------


def test_the_candidate_evaluator_uses_the_same_declared_throughput(tmp_path: Path) -> None:
    from chowder.growth.campaign_runner import build_evaluator

    material = tmp_path / "evaluation-material.json"
    material.write_text(
        json.dumps(
            {
                "suites": [
                    {"benchmark_qualified_id": PROTECTED_ID, "dataset": str(tmp_path / "d.jsonl")}
                ]
            }
        ),
        encoding="utf-8",
    )
    for batch_size in (1, 8):
        document = _manifest_document(tmp_path)
        document["evaluation_material_path"] = str(material)
        document["evaluation_execution"] = {"batch_size": batch_size}
        manifest = CampaignManifest.from_mapping(document, source="test")
        evaluator = build_evaluator(manifest, state_root=tmp_path / "state")
        assert evaluator is not None
        assert evaluator.batch_size == batch_size


def test_the_declared_throughput_is_the_only_owner_of_the_batching_decision() -> None:
    """No path may take a batch size from anywhere else."""
    import inspect

    from chowder.growth import campaign_prepare, campaign_runner, evaluation_binding

    for module in (campaign_prepare, campaign_runner):
        source = inspect.getsource(module)
        assert "evaluation_execution.batch_size" in source, module.__name__
    binding = inspect.signature(evaluation_binding.SubprocessEvaluationFn.__init__)
    # It is a parameter, so the runner's single declared value is what reaches
    # it; nothing in the binding reads the manifest for itself.
    assert "batch_size" in binding.parameters
