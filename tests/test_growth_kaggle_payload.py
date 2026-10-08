"""The declared kernel-side payload, exercised on a laptop.

A campaign declares what the kernel runs (:class:`CorpusTraining`), the kernel
writes the attempt context it verified, and the payload ports
``SubprocessTrainingFn``'s corpus materialization and production entry points
into the kernel. Every check -- the declaration, the ported gates, the corpus
bytes, the composed project, the production commands, the run summary, the
resume marker and the dispatcher-side merge -- runs here with an injected
command runner, so none of it needs a GPU or a Kaggle session.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chowder.growth.compute_backend import ComputeBackendRefusal
from chowder.growth.kaggle_payload import (
    CORPUS_NAME,
    CORPUS_TRAINING_KIND,
    KAGGLE_PAYLOAD_CONTAMINATION_DECLARED,
    KAGGLE_PAYLOAD_DECLARATION_MISSING,
    KAGGLE_PAYLOAD_INPUTS_MISSING,
    KAGGLE_PAYLOAD_ITEMS_EMPTY,
    KAGGLE_PAYLOAD_KIND_UNKNOWN,
    KAGGLE_PAYLOAD_MATERIAL_MISSING,
    KAGGLE_PAYLOAD_PROJECT_BUDGET,
    KAGGLE_PAYLOAD_SCHEMA,
    KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE,
    KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
    KAGGLE_PAYLOAD_VALIDATE_FAILED,
    PROJECT_NAME,
    RUN_SUMMARY_NAME,
    PRODUCTION_VERDICT_MISMATCH,
    CorpusTraining,
    merge_run_summary,
    run_payload,
)
from chowder.growth.recipe_planner import TrainingRecipe
from chowder.growth.training_binding import directory_digest

COMMIT = "a" * 40


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    item_id: str


def _recipe(
    recipe_id: str = "recipe-01", *, resume_from: str | None = None
) -> TrainingRecipe:
    return TrainingRecipe(
        recipe_id=recipe_id,
        curriculum_item_ids=("item-01", "item-02"),
        mixture={"TARGET": 1.0},
        learning_rate=2e-4,
        scheduler="cosine",
        warmup_steps=2,
        lora_rank=8,
        lora_alpha=16,
        target_modules=("q_proj",),
        seq_len=256,
        batch_size=1,
        gradient_accumulation=1,
        max_steps=20,
        objective="sft",
        replay_rate=0.0,
        dataset_manifest={},
        projected_device_gpu_hours=0.1,
        projected_wall_gpu_hours=0.2,
        resume_from_checkpoint=resume_from,
    )


def _template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "gen2",
        "goal": {
            "metrics": [{"name": "math500", "direction": "maximize"}],
            "gpu_hour_budget": 0.5,
        },
        "experiment": {"experiment_id": "gen2", "estimated_gpu_hours": 0.2},
        "config": {
            "backend": {
                "type": "transformers-peft",
                "base_model": "/models/base",
                "dataset": "{corpus}",
                "max_length": 512,
                "training": {"max_steps": 30, "learning_rate": 1e-4},
                "lora": {"r": 16, "alpha": 32},
            }
        },
    }


def _material() -> dict[str, Any]:
    return {
        "sources": {"item-01": "source-a", "item-02": "source-b"},
        "material": {
            "item-01": ["first   line", "", "second line"],
            "item-02": ["other item line"],
        },
    }


def _source_document(source_id: str, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "source_id": source_id,
        "dataset_name": source_id,
        "revision": "2026-09-18",
        "url": "file: tests/test_growth_kaggle_payload.py",
        "license": "Apache-2.0",
        "permitted_training_use": True,
        "domain": "synthetic-protocol",
        "language": "en",
        "source_type": "synthetic",
        "verification": "symbolic_numeric",
        "trust_class": "GOLD",
        "example_count": 2,
        "token_estimate": 120,
        "provenance": "test fixture",
        "acquisition_timestamp": "2026-09-18T00:00:00Z",
        "source_hash": "0" * 64,
        "contamination_relationship": "CLEAN",
        "quality_score": 1.0,
        "pii_reviewed": True,
        "secrets_reviewed": True,
        "inclusion_decision": "included",
    }
    document.update(overrides)
    return document


def _registry(
    *,
    source_a: dict[str, Any] | None = None,
    source_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sources": [
            _source_document("source-a", **(source_a or {})),
            _source_document("source-b", **(source_b or {})),
        ]
    }


def _contamination(**statuses: str) -> dict[str, Any]:
    return {
        "policy": {},
        "benchmarks": {},
        "training_sources": {
            source_id: {"status": status, "matches": []}
            for source_id, status in statuses.items()
        },
    }


def _write_inputs(
    tmp_path: Path,
    *,
    template: dict[str, Any] | None = None,
    material: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
    contamination: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bodies: dict[str, tuple[str, dict[str, Any]]] = {
        "project_template_path": ("project-template.json", template or _template()),
        "training_material_path": ("training-material.json", material or _material()),
        "data_registry_path": ("data-registry.json", registry or _registry()),
        "contamination_manifest_path": (
            "contamination.json",
            contamination if contamination is not None else _contamination(**{"source-a": "CLEAN", "source-b": "CLEAN"}),
        ),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, (filename, body) in bodies.items():
        path = tmp_path / filename
        path.write_text(json.dumps(body), encoding="utf-8")
        paths[name] = path
    return paths


def _context(
    locations: dict[str, Path],
    *,
    output_dir: Path,
    items: tuple[str, ...] = ("item-01", "item-02"),
    recipe: TrainingRecipe | None = None,
    budget: float = 0.5,
    resume_from: str | None = None,
    attempt_id: str = "attempt-01",
    declaration: Any = None,
) -> dict[str, Any]:
    if declaration is None:
        declaration = CorpusTraining(budget).payload(
            recipe or _recipe(), [_Item(item) for item in items]
        )
    return {
        "spec_id": f"gen2:{attempt_id}",
        "attempt_id": attempt_id,
        "source_commit_sha": COMMIT,
        "output_dir": str(output_dir),
        "input_locations": {
            name: str(path) for name, path in locations.items()
        },
        "declared_payload": declaration,
        "resume_from": resume_from,
    }


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeCli:
    """A stand-in for ``chowder.cli`` that records its calls by verb."""

    def __init__(
        self,
        *,
        validate_code: int = 0,
        train_code: int = 0,
        summary: dict[str, Any] | None = None,
    ) -> None:
        self.validate_code = validate_code
        self.train_code = train_code
        self.summary = summary
        self.calls: list[list[str]] = []

    def __call__(self, command, working_dir, environment):
        parts = [str(part) for part in command]
        self.calls.append(parts)
        if parts[3] == "project-validate":
            return _Completed(self.validate_code, stdout='{"ok": true}')
        # Production prints its summary pretty-printed (indent=2); the last
        # JSON object parser is line-oriented, so the fake prints the same way.
        return _Completed(
            self.train_code,
            stdout=(
                json.dumps(self.summary, indent=2, sort_keys=True)
                if self.summary is not None
                else ""
            ),
            stderr=("training exploded" if self.train_code else ""),
        )


def _train_summary(output_dir: Path, **overrides: Any) -> dict[str, Any]:
    artifact_ref = str(output_dir / "work" / "run" / "adapter")
    summary: dict[str, Any] = {
        "project": "gen2",
        "experiment_id": "gen2-a01",
        "succeeded": True,
        "promoted_experiment_id": "gen2-a01",
        "artifact_ref": artifact_ref,
        "metrics": {"math500": 0.25},
        "gpu_hours": 0.12,
        "error": None,
    }
    summary.update(overrides)
    return summary


def _run(
    tmp_path: Path,
    *,
    runner: _FakeCli,
    locations: dict[str, Path] | None = None,
    output_dir: Path | None = None,
    **context_overrides: Any,
) -> dict[str, Any]:
    context = _context(
        locations if locations is not None else _write_inputs(tmp_path),
        output_dir=output_dir or (tmp_path / "working"),
        **context_overrides,
    )
    return run_payload(context, runner=runner)


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------


def test_the_declaration_builds_the_command_and_the_payload() -> None:
    declaration = CorpusTraining(project_gpu_hour_budget=0.75)
    assert declaration.command() == ["python", "-m", "chowder.growth.kaggle_payload"]
    payload = declaration.payload(_recipe(), [_Item("item-01"), _Item("item-02")])
    assert payload["kind"] == CORPUS_TRAINING_KIND
    assert payload["item_ids"] == ["item-01", "item-02"]
    assert payload["project_gpu_hour_budget"] == 0.75
    assert payload["recipe"]["recipe_id"] == "recipe-01"
    assert payload["recipe"]["max_steps"] == 20


def test_a_declaration_without_items_or_with_a_bad_budget_refuses() -> None:
    declaration = CorpusTraining(project_gpu_hour_budget=0.75)
    with pytest.raises(ComputeBackendRefusal) as empty:
        declaration.payload(_recipe(), [])
    assert empty.value.code == KAGGLE_PAYLOAD_ITEMS_EMPTY
    with pytest.raises(ComputeBackendRefusal) as no_id:
        declaration.payload(_recipe(), [_Item("")])
    assert no_id.value.code == KAGGLE_PAYLOAD_SCHEMA
    for budget in (float("nan"), -1.0, True):
        with pytest.raises(ComputeBackendRefusal) as bad:
            CorpusTraining(project_gpu_hour_budget=budget)
        assert bad.value.code == KAGGLE_PAYLOAD_SCHEMA
    with pytest.raises(ComputeBackendRefusal):
        CorpusTraining(project_gpu_hour_budget=0.5, python="   ")


# --------------------------------------------------------------------------
# the ported training path
# --------------------------------------------------------------------------


def test_the_payload_materializes_trains_and_summarizes(tmp_path: Path) -> None:
    output_dir = tmp_path / "working"
    adapter = output_dir / "work" / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    runner = _FakeCli(summary=_train_summary(output_dir))

    summary = _run(tmp_path, runner=runner)

    assert summary["status"] == "succeeded", summary
    assert summary["succeeded"] is True
    # 1. the corpus is exactly the local executor's materialization.
    expected = "first line\nsecond line\nother item line\n"
    assert (output_dir / CORPUS_NAME).read_text(encoding="utf-8") == expected
    assert summary["corpus"]["examples"] == 3
    assert summary["corpus"]["item_ids"] == ["item-01", "item-02"]
    assert summary["corpus"]["source_ids"] == ["source-a", "source-b"]
    # 2. the production entry points ran in order, through the kernel python.
    assert [call[3] for call in runner.calls] == ["project-validate", "train"]
    assert all(call[0] == sys.executable for call in runner.calls)
    # 3. the project is composed from the declared template and the recipe.
    project = json.loads((output_dir / PROJECT_NAME).read_text(encoding="utf-8"))
    assert project["config"]["backend"]["dataset"] == str(output_dir / CORPUS_NAME)
    assert project["config"]["backend"]["max_length"] == 256
    assert project["config"]["backend"]["lora"]["r"] == 8
    assert project["config"]["backend"]["training"]["max_steps"] == 20
    assert project["experiment"]["experiment_id"] == "gen2-a01"
    assert project["work_dir"] == str(output_dir / "work")
    assert project["registry_path"] == str(output_dir / "runs.db")
    # 4. the run summary on disk is what was returned, and it names the artifact.
    written = json.loads((output_dir / RUN_SUMMARY_NAME).read_text(encoding="utf-8"))
    assert written == summary
    assert summary["training"]["artifact_path"] == "work/run/adapter"
    assert summary["training"]["artifact_exists"] is True
    assert summary["training"]["promoted_experiment_id"] == "gen2-a01"
    assert (output_dir / "validate.stdout.txt").is_file()
    assert (output_dir / "train.stdout.txt").is_file()
    # 5. a run without a declared resume writes no resume marker.
    assert not (output_dir / "resume-state.json").exists()


def test_a_failed_candidate_is_recorded_and_exits_unsuccessfully(tmp_path: Path) -> None:
    output_dir = tmp_path / "working"
    runner = _FakeCli(
        train_code=1,
        summary=_train_summary(
            output_dir, succeeded=False, promoted_experiment_id=None, error="no gain"
        ),
    )
    summary = _run(tmp_path, runner=runner)
    assert summary["status"] == "failed"
    assert summary["succeeded"] is False
    assert summary["failure_reason"] == "no gain"
    assert summary["training"]["succeeded"] is False


def test_a_project_refused_before_compute_never_trains(tmp_path: Path) -> None:
    output_dir = tmp_path / "working"
    runner = _FakeCli(validate_code=1)
    summary = _run(tmp_path, runner=runner)
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_VALIDATE_FAILED
    assert [call[3] for call in runner.calls] == ["project-validate"]


def test_the_declared_resume_reports_whether_the_continuation_took(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "working"
    adapter = output_dir / "work" / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    good = _FakeCli(summary=_train_summary(output_dir))
    summary = _run(tmp_path, runner=good, resume_from="ckpt-0007", recipe=_recipe(resume_from="ckpt-0007"))
    assert summary["status"] == "succeeded"
    marker = json.loads((output_dir / "resume-state.json").read_text(encoding="utf-8"))
    assert marker == {"resume_state": "resumed", "resume_from": "ckpt-0007"}

    failed_dir = tmp_path / "failed-working"
    bad = _FakeCli(
        train_code=1,
        summary=_train_summary(failed_dir, succeeded=False, error="loader died"),
    )
    summary = _run(
        tmp_path / "failed-inputs",
        runner=bad,
        output_dir=failed_dir,
        resume_from="ckpt-0007",
        recipe=_recipe(resume_from="ckpt-0007"),
    )
    assert summary["status"] == "failed"
    marker = json.loads((failed_dir / "resume-state.json").read_text(encoding="utf-8"))
    assert marker == {"resume_state": "not-a-resume", "resume_from": "ckpt-0007"}


# --------------------------------------------------------------------------
# the ported gates
# --------------------------------------------------------------------------


def test_a_declared_item_without_material_refuses_before_any_command(
    tmp_path: Path,
) -> None:
    runner = _FakeCli()
    summary = _run(tmp_path, runner=runner, items=("item-01", "item-99"))
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_MATERIAL_MISSING
    assert "item-99" in summary["refusal_reason"]
    assert runner.calls == []


def test_an_untrainable_source_refuses(tmp_path: Path) -> None:
    registry = _registry()
    registry["sources"][1]["contamination_relationship"] = "KNOWN_CONTAMINATION"
    locations = _write_inputs(tmp_path, registry=registry)
    summary = _run(tmp_path, runner=_FakeCli(), locations=locations)
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE
    assert "source-b" in summary["refusal_reason"]


def test_a_declared_contamination_verdict_refuses(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty"
    locations = _write_inputs(
        dirty,
        contamination=_contamination(
            **{"source-a": "CLEAN", "source-b": "KNOWN_CONTAMINATION"}
        ),
    )
    summary = _run(dirty, runner=_FakeCli(), locations=locations)
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_CONTAMINATION_DECLARED
    assert "source-b" in summary["refusal_reason"]

    # A source the manifest does not mention is recorded, not guessed at --
    # and the run proceeds; the gate refuses declared dirt, not silence.
    quiet = tmp_path / "quiet"
    locations = _write_inputs(quiet, contamination=_contamination())
    output_dir = quiet / "working"
    runner = _FakeCli(summary=_train_summary(output_dir))
    summary = _run(quiet, runner=runner, locations=locations)
    assert summary["status"] == "succeeded"
    assert summary["refused_by"] is None
    assert summary["contamination"] == {
        "checked": True,
        "training_sources": {"source-a": "not-declared", "source-b": "not-declared"},
    }


def test_a_loose_project_budget_refuses(tmp_path: Path) -> None:
    template = _template()
    template["goal"]["gpu_hour_budget"] = 2.0
    locations = _write_inputs(tmp_path, template=template)
    summary = _run(tmp_path, runner=_FakeCli(), locations=locations, budget=0.75)
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_PROJECT_BUDGET


def test_a_template_without_the_corpus_token_refuses(tmp_path: Path) -> None:
    template = _template()
    template["config"]["backend"].pop("dataset")
    locations = _write_inputs(tmp_path, template=template)
    summary = _run(tmp_path, runner=_FakeCli(), locations=locations)
    assert summary["status"] == "refused"
    assert summary["refused_by"] == KAGGLE_PAYLOAD_TEMPLATE_CONTRACT


def test_a_context_without_a_declaration_or_a_kind_refuses(tmp_path: Path) -> None:
    locations = _write_inputs(tmp_path)
    output_dir = tmp_path / "working"
    context = _context(locations, output_dir=output_dir)
    context.pop("declared_payload")
    summary = run_payload(context, runner=_FakeCli())
    assert summary["refused_by"] == KAGGLE_PAYLOAD_DECLARATION_MISSING

    context = _context(locations, output_dir=output_dir)
    context["declared_payload"]["kind"] = "something-else"
    summary = run_payload(context, runner=_FakeCli())
    assert summary["refused_by"] == KAGGLE_PAYLOAD_KIND_UNKNOWN


def test_a_missing_input_location_refuses(tmp_path: Path) -> None:
    output_dir = tmp_path / "working"
    locations = _write_inputs(tmp_path)
    locations.pop("training_material_path")
    context = _context(locations, output_dir=output_dir)
    summary = run_payload(context, runner=_FakeCli())
    assert summary["refused_by"] == KAGGLE_PAYLOAD_INPUTS_MISSING


# --------------------------------------------------------------------------
# the dispatcher-side merge
# --------------------------------------------------------------------------


def test_the_run_summary_merges_the_trained_artifact_into_evidence(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt-01"
    adapter = attempt / "work" / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (attempt / CORPUS_NAME).write_text("line\n", encoding="utf-8")
    (attempt / RUN_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "status": "succeeded",
                "corpus": {
                    "path": "/kaggle/working/corpus.txt",
                    "sha256": "d" * 64,
                    "examples": 1,
                    "item_ids": ["item-01"],
                    "source_ids": ["source-a"],
                },
                "training": {
                    "experiment_id": "gen2-a01",
                    "succeeded": True,
                    "promoted_experiment_id": "gen2-a01",
                    "artifact_ref": "/kaggle/working/work/run/adapter",
                    "artifact_path": "work/run/adapter",
                    "artifact_exists": True,
                    "metrics": {"math500": 0.25},
                    "gpu_hours": 0.12,
                    "error": None,
                    "exit_code": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "status": "SUCCEEDED",
        "candidate_succeeded": True,
        "artifact_ref": "attempt-context.json",
        "artifact_sha256": "e" * 64,
        "notes": [],
    }
    merged = merge_run_summary(evidence, attempt)
    assert merged["artifact_ref"] == str(adapter)
    digest, files = directory_digest(adapter)
    assert merged["artifact_sha256"] == digest
    assert merged["artifact_files"] == [dict(entry) for entry in files]
    assert merged["candidate_metrics"] == {"math500": 0.25}
    assert merged["promoted_experiment_id"] == "gen2-a01"
    assert merged["production_gpu_hours"] == 0.12
    assert merged["material"]["corpus_sha256"] == "d" * 64
    assert merged["material"]["corpus_path"] == str(attempt / CORPUS_NAME)


def test_a_production_verdict_that_contradicts_the_record_fails_the_evidence(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt-01"
    attempt.mkdir()
    (attempt / RUN_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "status": "failed",
                "training": {
                    "succeeded": False,
                    "error": "no measurable gain",
                    "artifact_path": None,
                },
            }
        ),
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "status": "SUCCEEDED",
        "candidate_succeeded": True,
        "artifact_ref": "attempt-context.json",
        "notes": [],
    }
    merged = merge_run_summary(evidence, attempt)
    assert merged["status"] == "FAILED"
    assert merged["candidate_succeeded"] is False
    assert merged["failure_reason"] == "no measurable gain"
    assert merged["production_verdict"] == PRODUCTION_VERDICT_MISMATCH
    assert merged["artifact_ref"] == "attempt-context.json"


def test_a_missing_or_escaping_summary_leaves_the_evidence_alone(
    tmp_path: Path,
) -> None:
    evidence: dict[str, Any] = {"status": "SUCCEEDED", "notes": []}
    assert merge_run_summary(dict(evidence), tmp_path / "absent") == evidence

    attempt = tmp_path / "attempt-01"
    attempt.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "adapter.bin").write_bytes(b"weights")
    (attempt / RUN_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "status": "succeeded",
                "training": {
                    "succeeded": True,
                    "artifact_path": "../outside",
                },
            }
        ),
        encoding="utf-8",
    )
    merged = merge_run_summary(dict(evidence), attempt)
    assert merged.get("artifact_ref") is None
    assert any("no trained artifact" in note for note in merged["notes"])
