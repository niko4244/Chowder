"""A declaration's required inputs are produced from durable evidence.

``chowder growth campaign prepare`` is the production path that emits the seven
inputs the run phase reads from disk (plus the contamination manifest) from
evidence the repository already holds: a real device probe, the pinned dataset
caches, the parent generation's own run root, and the production planner.

The tests fake only the two external seams -- the GPU probe and the dataset
loader -- so everything else is exercised through the real code.  The load-
bearing claim is that a prepared declaration's readiness stops reporting
``READINESS_DECLARED_INPUT`` because the inputs really exist, not because a
check was loosened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from chowder.cli import main as chowder_main
from chowder.growth.campaign import CampaignManifest
from chowder.growth.campaign_prepare import (
    CONTAMINATION_MANIFEST_FIELD,
    PREPARED_INPUT_FIELDS,
    PREPARE_PARENT_EVIDENCE_REQUIRED,
    PREPARE_SLICE_UNAVAILABLE,
    CampaignPrepareRefusal,
    prepare_campaign,
)
from chowder.growth.campaign_runner import check_campaign_readiness
from chowder.local_model_manifest import model_content_digest

TARGET_ID = "generation-diagnostics@gen1-eval-protocol-v1"
PROTECTED_ID = "math500@2024-04"
BROAD_ID = "mgsm@2022-11"

HARDWARE: Mapping[str, Any] = {
    "gpu_name": "probe-gpu",
    "vram_gb": 16.0,
    "measured_step_seconds_at_seq": {"512": 0.001, "1024": 0.002, "2048": 0.004},
    "measured_load_seconds": 1.0,
    "wall_multiplier": 3.5,
    "measurement_method": "fixture probe",
}

PROTECTION: dict[str, Any] = {
    "trusted_ancestor_version": "gen0",
    "slice_regression_max": 0.0625,
    "n_samples": 16,
    "seed": 1234,
    "shuffle": False,
    "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512},
    "prompt_policy": "chat_template",
}

STOPPING_RULES = [
    "stop before compute on admission refusal",
    "never enlarge a frozen threshold after candidate results are visible",
]


def _probe() -> Mapping[str, Any]:
    return dict(HARDWARE)


def _slice_source(items: int = 16):
    def source(qualified_id: str) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {"prompt": f"{qualified_id} prompt {index}", "expected": f"{index}"}
            for index in range(items)
        )

    return source


def _parent_evidence(root: Path, *, instrument: str = TARGET_ID) -> Path:
    """The parent generation's durable run root, as the prepare path reads it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate_evaluation.json").write_text(
        json.dumps(
            {
                "cycle_id": "gen1-protocol-compliance",
                "candidate": "gen1",
                "protocol": "gen1-eval-protocol-v1",
                "diagnostics": {
                    "instrument": instrument,
                    "metric": "eos_termination_rate",
                    "eos_termination_rate": 0.9375,
                    "n_prompts": 16,
                    "adapter_ref": "adapter-dir",
                },
                "protected": [
                    {
                        "benchmark_qualified_id": PROTECTED_ID,
                        "carried_from_parent": True,
                        "score": 0.0,
                    }
                ],
                "unmeasured": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _manifest_document(tmp_path: Path, *, declared_inputs: bool = False) -> dict[str, Any]:
    """A declaration that names none of the inputs prepare is asked to emit."""
    base = tmp_path / "base-model"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    document: dict[str, Any] = {
        "cycle_id": "gen2-prepare-fixture",
        "parent_version": "gen1",
        "base_model_path": str(base),
        "base_model_digest": model_content_digest(base).digest,
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
        "stopping_rules": list(STOPPING_RULES),
        "protection": dict(PROTECTION),
        "notes": "prepare fixture",
    }
    if declared_inputs:
        prepared = tmp_path / "prepared"
        document.update({field: str(prepared / field) for field in PREPARED_INPUT_FIELDS})
    return document


def _write_manifest(tmp_path: Path, document: Mapping[str, Any]) -> Path:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _load_manifest(tmp_path: Path, document: Mapping[str, Any]) -> CampaignManifest:
    return CampaignManifest.from_file(_write_manifest(tmp_path, document))


def test_prepare_emits_every_declared_input_from_evidence(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=_parent_evidence(tmp_path / "gen1"),
        probe=_probe,
        slice_source=_slice_source(),
    )
    for field_name in PREPARED_INPUT_FIELDS:
        path = Path(prepared.inputs[field_name])
        assert path.is_file(), field_name
        json.loads(path.read_text(encoding="utf-8"))
    assert Path(prepared.inputs[CONTAMINATION_MANIFEST_FIELD]).is_file()


def test_prepare_refuses_without_parent_evidence(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    with pytest.raises(CampaignPrepareRefusal) as error:
        prepare_campaign(
            manifest,
            out_dir=tmp_path / "prepared",
            parent_evidence=None,
            probe=_probe,
            slice_source=_slice_source(),
        )
    assert PREPARE_PARENT_EVIDENCE_REQUIRED in str(error.value)


def test_prepare_refuses_a_slice_shorter_than_the_protocol(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    with pytest.raises(CampaignPrepareRefusal) as error:
        prepare_campaign(
            manifest,
            out_dir=tmp_path / "prepared",
            parent_evidence=_parent_evidence(tmp_path / "gen1"),
            probe=_probe,
            slice_source=_slice_source(items=4),
        )
    assert PREPARE_SLICE_UNAVAILABLE in str(error.value)


def test_the_parent_arm_marks_what_the_parent_did_not_measure(tmp_path: Path) -> None:
    """A benchmark the parent never measured is UNMEASURED, never a copied score."""
    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=_parent_evidence(tmp_path / "gen1"),
        probe=_probe,
        slice_source=_slice_source(),
    )
    report = json.loads(
        Path(prepared.inputs["parent_eval_report_path"]).read_text(encoding="utf-8")
    )
    by_id = {run["benchmark_qualified_id"]: run for run in report["runs"]}
    # The parent's diagnostics instrument matches the declared target id, so it
    # is a real MEASURED_PARENT row.
    assert by_id[TARGET_ID]["measurement_origin"] == "MEASURED_PARENT"
    # The protected slice was carried, never measured, so it stays unmeasured.
    assert by_id[PROTECTED_ID]["measurement_origin"] == "UNMEASURED"
    assert by_id[BROAD_ID]["measurement_origin"] == "UNMEASURED"
    assert by_id[PROTECTED_ID]["score"] is None


def test_a_parent_measured_under_another_instrument_is_not_this_arm(tmp_path: Path) -> None:
    """A row for a different instrument version is not a measurement of this one."""
    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=_parent_evidence(
            tmp_path / "gen1", instrument="generation-diagnostics@gen1-other"
        ),
        probe=_probe,
        slice_source=_slice_source(),
    )
    report = json.loads(
        Path(prepared.inputs["parent_eval_report_path"]).read_text(encoding="utf-8")
    )
    assert all(run["measurement_origin"] == "UNMEASURED" for run in report["runs"])


def test_prepared_declaration_stops_reporting_declared_input(tmp_path: Path) -> None:
    """The whole point: readiness no longer refuses on the inputs prepare filled."""
    document = _manifest_document(tmp_path)
    manifest = _load_manifest(tmp_path, document)
    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=_parent_evidence(tmp_path / "gen1"),
        probe=_probe,
        slice_source=_slice_source(),
    )
    document.update(prepared.inputs)
    report = check_campaign_readiness(_load_manifest(tmp_path, document))
    checks = {check.check: check for check in report.checks}
    assert checks["declared_inputs"].status == "ok"
    assert "READINESS_DECLARED_INPUT" not in report.reason_codes
    # And the inputs the run reads were produced by production code, not by the
    # declaration pretending: the parent arm really parses.
    assert checks["parent_arm"].status == "ok"


def test_the_target_instrument_slice_comes_from_production() -> None:
    """The target benchmark's 16 prompts are a production-owned pinned dataset."""
    from chowder.growth.campaign_prepare import load_pinned_slices

    items = load_pinned_slices("generation-diagnostics@gen2-response-surface-v1")
    assert len(items) == 16
    assert items[0]["prompt"] == "Reply with exactly: ping"
    assert items[0]["expected"] == "ping"


def test_the_production_instrument_matches_the_frozen_judge() -> None:
    """The production target slice cannot drift from the frozen judge's copy."""
    import importlib.util

    from chowder.growth.generation_diagnostics import INSTRUMENT_PROMPTS

    judge_path = Path(__file__).resolve().parents[1] / "docs" / "gen2" / "judge_gen2.py"
    spec = importlib.util.spec_from_file_location("judge_gen2", judge_path)
    assert spec is not None and spec.loader is not None
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    assert tuple(INSTRUMENT_PROMPTS) == tuple(judge.INSTRUMENT_PROMPTS)


class _RecordingEvalRunner:
    """A process runner that answers the ancestor worker without a GPU.

    It reads the spec the real binding wrote, writes one scored prediction row
    per frozen-slice item into the suite's predictions file, and returns the
    result manifest the real worker returns -- so every part of the ancestor
    path except the model is production code.
    """

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, cwd, env, timeout):  # noqa: ANN001
        from chowder.growth.training_binding import SubprocessOutcome

        self.commands.append(list(command))
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        out_dir = Path(spec["output_dir"])
        suites: dict[str, Any] = {}
        for suite in spec["suites"]:
            dataset = Path(suite["dataset"])
            items = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
            predictions = out_dir / f"predictions-{suite['name']}.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps({"prompt": item["prompt"], "expected": item["expected"],
                                "prediction": item["expected"], "score": 1.0})
                    + "\n"
                    for item in items
                ),
                encoding="utf-8",
            )
            suites[suite["name"]] = {"predictions_file": str(predictions)}
        result_path.write_text(
            json.dumps({"metrics": {name: {} for name in suites}, "suites": suites,
                        "runtime": {"gpu_count": 1}}),
            encoding="utf-8",
        )
        return SubprocessOutcome(
            command=list(command), returncode=0, stdout="", stderr="",
            seconds=42.0, timed_out=False,
        )


def test_the_ancestor_arm_is_measured_and_bound_to_the_base(tmp_path: Path) -> None:
    """The Gen-0 arm is a real measurement: MEASURED_PARENT rows, base identity."""
    from chowder.growth.campaign_prepare import measure_ancestor_arm

    document = _manifest_document(tmp_path)
    document["protection"]["trusted_ancestor_version"] = "gen0"
    document["baseline_eval_report_path"] = str(tmp_path / "gen0" / "baseline.json")
    manifest = _load_manifest(tmp_path, document)
    runner = _RecordingEvalRunner()
    arm = measure_ancestor_arm(
        manifest,
        slice_source=_slice_source(),
        runner=runner,
        state_root=tmp_path / "state",
    )
    report = json.loads(Path(arm.report_path).read_text(encoding="utf-8"))
    assert report["generation_version"] == "gen0"
    assert report["model_identity"]["base_model_digest"] == document["base_model_digest"]
    assert {run["benchmark_qualified_id"] for run in report["runs"]} == {PROTECTED_ID, BROAD_ID}
    for run in report["runs"]:
        assert run["measurement_origin"] == "MEASURED_PARENT"
        assert run["generation_version"] == "gen0"
        assert run["n_samples"] == 16
        assert len(run["per_sample_scores"]) == 16
        assert run["metadata"]["artifact_sha256"]
        assert run["metadata"]["sample_indices"] == list(range(16))
    assert arm.gpu_count == 1
    # No adapter was loaded: the dense base is what was measured.
    spec = json.loads((arm.work_dir / "ancestor-eval-spec.json").read_text(encoding="utf-8"))
    assert spec["adapter_dir"] is None
    assert spec["base_model"] == document["base_model_path"]


def test_the_ancestor_arm_refuses_a_short_slice(tmp_path: Path) -> None:
    from chowder.growth.campaign_prepare import (
        ANCESTOR_ARM_SLICE_TOO_SHORT,
        measure_ancestor_arm,
    )

    document = _manifest_document(tmp_path)
    document["baseline_eval_report_path"] = str(tmp_path / "gen0" / "baseline.json")
    manifest = _load_manifest(tmp_path, document)
    with pytest.raises(CampaignPrepareRefusal) as error:
        measure_ancestor_arm(
            manifest,
            slice_source=_slice_source(items=3),
            runner=_RecordingEvalRunner(),
            state_root=tmp_path / "state",
        )
    assert ANCESTOR_ARM_SLICE_TOO_SHORT in str(error.value)


class _FailingEvalRunner:
    def __call__(self, command, cwd, env, timeout):  # noqa: ANN001
        from chowder.growth.training_binding import SubprocessOutcome

        return SubprocessOutcome(
            command=list(command), returncode=1, stdout="", stderr="boom",
            seconds=1.0, timed_out=False,
        )


def test_a_failed_ancestor_worker_refuses_rather_than_writing_an_arm(tmp_path: Path) -> None:
    from chowder.growth.campaign_prepare import ANCESTOR_ARM_PROCESS_FAILED, measure_ancestor_arm

    document = _manifest_document(tmp_path)
    document["baseline_eval_report_path"] = str(tmp_path / "gen0" / "baseline.json")
    manifest = _load_manifest(tmp_path, document)
    with pytest.raises(CampaignPrepareRefusal) as error:
        measure_ancestor_arm(
            manifest,
            slice_source=_slice_source(),
            runner=_FailingEvalRunner(),
            state_root=tmp_path / "state",
        )
    assert ANCESTOR_ARM_PROCESS_FAILED in str(error.value)
    assert not Path(document["baseline_eval_report_path"]).exists()


def test_cli_prepare_writes_a_declaration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chowder.growth import campaign_prepare

    monkeypatch.setattr(campaign_prepare, "probe_hardware", _probe)
    monkeypatch.setattr(campaign_prepare, "load_pinned_slices", _slice_source())
    evidence = _parent_evidence(tmp_path / "gen1")
    path = _write_manifest(tmp_path, _manifest_document(tmp_path))
    out_declaration = tmp_path / "prepared-campaign.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chowder",
            "growth",
            "campaign",
            "prepare",
            str(path),
            "--out-dir",
            str(tmp_path / "prepared"),
            "--parent-evidence",
            str(evidence),
            "--write-declaration",
            str(out_declaration),
        ],
    )
    assert chowder_main() == 0
    written = json.loads(out_declaration.read_text(encoding="utf-8"))
    for field_name in PREPARED_INPUT_FIELDS:
        assert Path(written[field_name]).is_file()
    assert Path(written[CONTAMINATION_MANIFEST_FIELD]).is_file()


def _with_parent_adapter(tmp_path: Path, document: dict[str, Any]) -> dict[str, Any]:
    adapter = tmp_path / "parent-adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_model.safetensors").write_text("gen1-parent", encoding="utf-8")
    from chowder.growth.training_binding import directory_digest

    digest, _entries = directory_digest(adapter)
    document["parent_adapter_path"] = str(adapter)
    document["parent_adapter_digest"] = digest
    document["parent_eval_report_path"] = str(tmp_path / "parent" / "parent-eval.json")
    return document


def test_the_parent_arm_measures_the_parent_under_the_declared_instrument(tmp_path: Path) -> None:
    """The parent arm gets a real target row under this campaign's instrument."""
    from chowder.growth.campaign_prepare import measure_parent_arm

    document = _with_parent_adapter(tmp_path, _manifest_document(tmp_path))
    manifest = _load_manifest(tmp_path, document)
    runner = _RecordingEvalRunner()
    arm = measure_parent_arm(
        manifest,
        slice_source=_slice_source(),
        runner=runner,
        state_root=tmp_path / "state",
    )
    report = json.loads(Path(arm.report_path).read_text(encoding="utf-8"))
    assert report["generation_version"] == "gen1"
    by_id = {run["benchmark_qualified_id"]: run for run in report["runs"]}
    # The declared target instrument now has a real parent row.
    assert by_id[TARGET_ID]["measurement_origin"] == "MEASURED_PARENT"
    assert by_id[TARGET_ID]["generation_version"] == "gen1"
    assert by_id[TARGET_ID]["n_samples"] == 16
    assert len(by_id[TARGET_ID]["per_sample_scores"]) == 16
    assert report["model_identity"]["adapter_digest"] == document["parent_adapter_digest"]
    parent_spec = json.loads(
        (arm.work_dir / "ancestor-eval-spec.json").read_text(encoding="utf-8")
    )
    assert parent_spec["adapter_dir"] == document["parent_adapter_path"]
    assert arm.which == "parent"


def test_the_parent_arm_refuses_when_no_adapter_is_declared(tmp_path: Path) -> None:
    from chowder.growth.campaign_prepare import CampaignPrepareRefusal, measure_parent_arm

    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    with pytest.raises(CampaignPrepareRefusal):
        measure_parent_arm(manifest, slice_source=_slice_source(), runner=_RecordingEvalRunner())


def test_the_predicted_input_paths_are_exactly_what_preparation_writes(
    tmp_path: Path,
) -> None:
    """The composer's names and the writer's names must be the same names.

    An automatic declaration is frozen *before* any compute, so the component
    that composes it has to name the inputs preparation will produce. If those
    two owners drift, a frozen declaration points at files nothing writes, and
    the campaign refuses later for a reason unrelated to the science.
    """
    from chowder.growth.campaign_prepare import prepared_input_paths

    manifest = _load_manifest(tmp_path, _manifest_document(tmp_path))
    out_dir = tmp_path / "prepared"

    prepared = prepare_campaign(
        manifest,
        out_dir=out_dir,
        parent_evidence=_parent_evidence(tmp_path / "gen1"),
        probe=_probe,
        slice_source=_slice_source(),
    )

    assert prepared_input_paths(out_dir) == dict(prepared.inputs)
