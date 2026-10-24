"""A MoE campaign declaration reaches the worker and the evaluator.

Four settings decide how an MoE generation trains and is measured:

* ``training_backend.config.expert_placement`` -- where the routed experts live,
* ``training_backend.config.target_preset``     -- which LoRA coverage is attached,
* ``training_backend.config.tools_field``       -- the per-row tool-schema column,
* ``evaluation_execution.placement``            -- how the candidate is loaded.

Each is proven to *flow*: prepare writes it into the project template, the
training worker's own admission (``TransformersPeftRunSpec.from_resolved_config``)
reads it back, and the declared evaluation placement reaches both the in-run
template and the candidate evaluator. An undeclared campaign is proven to keep
every historical value, so no existing declaration moves. The two settings that
constrain each other are proven to refuse, at prepare and at readiness, with a
message that names them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from chowder.backends.transformers_peft import TransformersPeftRunSpec
from chowder.growth.campaign import (
    CampaignManifest,
    CampaignManifestError,
    EvaluationExecution,
)
from chowder.growth.campaign_prepare import (
    PREPARE_SCHEMA,
    CampaignPrepareRefusal,
    prepare_campaign,
)
from chowder.growth.campaign_runner import (
    READINESS_PROJECT_TEMPLATE,
    build_evaluator,
    check_campaign_readiness,
)

from test_growth_campaign_prepare import (  # the prepare path's own fixtures
    TARGET_ID,
    _load_manifest,
    _manifest_document,
    _parent_evidence,
    _probe,
    _slice_source,
)

MOE_TEMPLATE = Path(__file__).resolve().parents[1] / "docs/moe/MOE_GEN1_CAMPAIGN_TEMPLATE.json"


def _prepare(
    tmp_path: Path,
    *,
    training_backend: Mapping[str, Any] | None = None,
    evaluation_execution: Mapping[str, Any] | None = None,
    instrument: str | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Prepare one declaration and return its document, prepared paths, template."""
    document = _manifest_document(tmp_path)
    if training_backend is not None:
        document["training_backend"] = dict(training_backend)
    if evaluation_execution is not None:
        document["evaluation_execution"] = dict(evaluation_execution)
    manifest = _load_manifest(tmp_path, document)
    parent = _parent_evidence(tmp_path / "gen1", instrument=instrument or TARGET_ID)
    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=parent,
        probe=_probe,
        slice_source=_slice_source(),
    )
    template = json.loads(
        Path(prepared.inputs["project_template_path"]).read_text(encoding="utf-8")
    )
    return document, prepared, template


def _declared_again(
    tmp_path: Path, document: Mapping[str, Any], prepared: Any
) -> CampaignManifest:
    """The declaration as preparation leaves it: every emitted input named."""
    declared = dict(document)
    declared.update(prepared.inputs)
    return _load_manifest(tmp_path, declared)


def _worker_spec(template: Mapping[str, Any], tmp_path: Path) -> TransformersPeftRunSpec:
    """The training worker's own admission of this template."""
    return TransformersPeftRunSpec.from_resolved_config(
        template["config"],
        work_dir=tmp_path,
        output_dir=tmp_path / "out",
        seed=int(template["seed"]),
    )


def _moe_backend(**config: Any) -> dict[str, Any]:
    return {"provider": "local", "config": dict(config)}


# --------------------------------------------------------------------------
# each setting flows from the declaration to the worker
# --------------------------------------------------------------------------


def test_expert_placement_flows_from_the_declaration_into_the_worker_spec(
    tmp_path: Path,
) -> None:
    """``cpu-transient-grad`` is a declaration, not a constant in the template."""
    _, _, template = _prepare(
        tmp_path,
        training_backend=_moe_backend(
            # The worker requires this pairing; the preset's test below proves
            # the preset alone is a declaration too.
            expert_placement="cpu-transient-grad",
            target_preset="moe_frozen_experts",
        ),
    )
    backend = template["config"]["backend"]
    assert backend["expert_placement"] == "cpu-transient-grad"
    assert backend["lora"]["target_preset"] == "moe_frozen_experts"
    # The preset owns the coverage; an explicit list would be a second owner.
    assert "target_modules" not in backend["lora"]
    assert _worker_spec(template, tmp_path).expert_placement == "cpu-transient-grad"


def test_target_preset_flows_from_the_declaration_into_the_worker_spec(
    tmp_path: Path,
) -> None:
    """A named coverage preset replaces the hard-coded attention-only modules."""
    _, _, template = _prepare(
        tmp_path, training_backend=_moe_backend(target_preset="attention_and_mlp")
    )
    backend = template["config"]["backend"]
    assert backend["lora"]["target_preset"] == "attention_and_mlp"
    assert "target_modules" not in backend["lora"]
    spec = _worker_spec(template, tmp_path)
    assert spec.target_preset == "attention_and_mlp"
    assert spec.target_modules == ()


def test_tools_field_flows_from_the_declaration_into_the_worker_spec(
    tmp_path: Path,
) -> None:
    """Agentic rows carry their tool schemas; the field is declared, not assumed."""
    _, _, template = _prepare(
        tmp_path,
        training_backend=_moe_backend(
            dataset_format="chat", messages_field="messages", tools_field="tools"
        ),
    )
    backend = template["config"]["backend"]
    assert backend["dataset_format"] == "chat"
    assert backend["messages_field"] == "messages"
    assert backend["tools_field"] == "tools"
    spec = _worker_spec(template, tmp_path)
    assert spec.dataset_format == "chat"
    assert spec.tools_field == "tools"


def test_evaluation_placement_reaches_the_template_and_the_candidate_evaluator(
    tmp_path: Path,
) -> None:
    """One declared placement, both places the candidate is measured."""
    document, prepared, template = _prepare(
        tmp_path,
        evaluation_execution={"batch_size": 16, "placement": "experts-cpu"},
    )
    assert template["config"]["evaluation"]["placement"] == "experts-cpu"
    evaluator = build_evaluator(
        _declared_again(tmp_path, document, prepared), state_root=tmp_path / "state"
    )
    assert evaluator is not None
    assert evaluator.placement == "experts-cpu"
    # The throughput travels with it, unchanged and independent.
    assert evaluator.batch_size == 16
    # A placement no evaluator implements is refused where the manifest is read.
    with pytest.raises(CampaignManifestError):
        CampaignManifest.from_mapping(
            {**document, "evaluation_execution": {"placement": "on-the-moon"}},
            source="test",
        )


# --------------------------------------------------------------------------
# an undeclared campaign moves nothing
# --------------------------------------------------------------------------


def test_an_undeclared_campaign_keeps_every_historical_value(tmp_path: Path) -> None:
    document, prepared, template = _prepare(tmp_path)
    backend = template["config"]["backend"]
    assert backend["dataset_format"] == "text"
    assert backend["text_field"] == "text"
    # Not written at all: absent means the worker's own historical default.
    assert "expert_placement" not in backend
    assert "messages_field" not in backend
    assert "tools_field" not in backend
    assert "target_preset" not in backend["lora"]
    assert backend["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert template["config"]["evaluation"]["placement"] == "offload"

    spec = _worker_spec(template, tmp_path)
    assert spec.expert_placement == "resident"
    assert spec.target_preset == "auto"
    assert spec.tools_field == ""

    evaluator = build_evaluator(
        _declared_again(tmp_path, document, prepared), state_root=tmp_path / "state"
    )
    assert evaluator is not None
    assert evaluator.placement == "resident"

    # The manifest-level default is "unset", not a value: to_dict() therefore
    # carries no placement key and a historical declaration round-trips.
    assert EvaluationExecution().placement == ""
    assert EvaluationExecution().to_dict() == {"batch_size": 1}
    assert EvaluationExecution.from_mapping({"batch_size": 16}).placement == ""
    assert EvaluationExecution.from_mapping(
        {"batch_size": 16, "placement": "experts-cpu"}
    ).placement == "experts-cpu"


# --------------------------------------------------------------------------
# the two settings that constrain each other refuse, naming both
# --------------------------------------------------------------------------


def test_prepare_refuses_cpu_transient_grad_with_an_attention_preset(
    tmp_path: Path,
) -> None:
    """Frozen experts are the placement's premise; prepare says so before compute."""
    document = _manifest_document(tmp_path)
    document["training_backend"] = _moe_backend(
        expert_placement="cpu-transient-grad", target_preset="attention_and_mlp"
    )
    with pytest.raises(CampaignPrepareRefusal) as error:
        prepare_campaign(
            _load_manifest(tmp_path, document),
            out_dir=tmp_path / "prepared",
            parent_evidence=_parent_evidence(tmp_path / "gen1"),
            probe=_probe,
            slice_source=_slice_source(),
        )
    message = str(error.value)
    assert PREPARE_SCHEMA in message
    assert "expert_placement='cpu-transient-grad'" in message
    assert "target_preset='attention_and_mlp'" in message


def test_readiness_refuses_a_template_cpu_transient_grad_cannot_admit(
    tmp_path: Path,
) -> None:
    """A template edited away from its declaration is refused before compute."""
    document, prepared, _ = _prepare(
        tmp_path,
        training_backend=_moe_backend(
            expert_placement="cpu-transient-grad", target_preset="moe_frozen_experts"
        ),
    )
    template_path = Path(prepared.inputs["project_template_path"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["config"]["backend"]["lora"]["target_preset"] = "attention_and_mlp"
    template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")

    report = check_campaign_readiness(_declared_again(tmp_path, document, prepared))
    checks = {check.check: check for check in report.checks}
    assert checks["project_template"].status == "refused"
    assert READINESS_PROJECT_TEMPLATE in report.reason_codes
    assert "expert_placement='cpu-transient-grad'" in checks["project_template"].detail
    assert "target_preset='attention_and_mlp'" in checks["project_template"].detail


# --------------------------------------------------------------------------
# the committed template is a real declaration
# --------------------------------------------------------------------------


def _concrete(document: Mapping[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Substitute every TODO a template must never be loaded with."""
    from chowder.local_model_manifest import model_content_digest

    base = tmp_path / "moe-base"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    declared = json.loads(json.dumps(document))
    declared["base_model_path"] = str(base)
    declared["base_model_digest"] = model_content_digest(base).digest
    declared["state_root"] = str(tmp_path / "state")
    declared["budget"].update(
        {
            "device_gpu_hours_ceiling_per_recipe": 2.5,
            "wall_gpu_hours_ceiling_per_recipe": 2.5,
            "device_gpu_hours_ceiling_campaign": 7.0,
            "wall_gpu_hours_ceiling_campaign": 7.0,
        }
    )
    declared["baseline_eval_report_path"] = str(tmp_path / "gen0-arm.json")
    notes = str(declared.get("notes", ""))
    with_judge = notes.split("<TODO:", 1)
    if len(with_judge) == 2:
        # The only prose TODO: the judge path. Drop the placeholder to the end.
        declared["notes"] = with_judge[0] + "the frozen judge for this campaign."
    assert "<TODO" not in json.dumps(declared)
    return declared


def test_the_committed_moe_template_parses_and_prepares_to_a_valid_spec(
    tmp_path: Path,
) -> None:
    """The template is a declaration, not prose: it loads and the worker admits it."""
    template = json.loads(MOE_TEMPLATE.read_text(encoding="utf-8"))
    assert template["target_benchmarks"] == ["tool-use@reap-moe-v1"]
    config = template["training_backend"]["config"]
    assert config == {
        "expert_placement": "cpu-transient-grad",
        "max_length": 2048,
        "target_preset": "moe_frozen_experts",
        "dataset_format": "chat",
        "messages_field": "messages",
        "tools_field": "tools",
    }
    assert template["evaluation_execution"]["placement"] == "experts-cpu"

    document = _concrete(template, tmp_path)
    manifest = _load_manifest(tmp_path, document)
    # A parent arm that measured the tool-use instrument, as production's own
    # run root would: the target's metric is the accuracy the row reports.
    parent = _parent_evidence(tmp_path / "gen0", instrument="tool-use@reap-moe-v1")
    report_path = parent / "candidate_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["diagnostics"]["metric"] = "accuracy"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    prepared = prepare_campaign(
        manifest,
        out_dir=tmp_path / "prepared",
        parent_evidence=parent,
        probe=_probe,
        slice_source=_slice_source(),
    )
    project = json.loads(
        Path(prepared.inputs["project_template_path"]).read_text(encoding="utf-8")
    )
    assert project["config"]["evaluation"]["placement"] == "experts-cpu"
    assert project["config"]["backend"]["max_length"] == 2048
    spec = _worker_spec(project, tmp_path)
    assert spec.max_length == 2048
    assert spec.expert_placement == "cpu-transient-grad"
    assert spec.target_preset == "moe_frozen_experts"
    assert spec.dataset_format == "chat"
    assert spec.tools_field == "tools"
