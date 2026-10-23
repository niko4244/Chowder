"""Materialize four *separate* executable Gen-1 MoE training comparisons.

The production campaign planner varies learning rate only. Dataset identity is
explicitly not a search axis, so a Fable/control comparison must NOT silently
pretend to be four candidates of the single frozen two-recipe campaign. This
tool instead prepares four independent production `chowder train` projects,
with isolated registries and stable source/recipe provenance. It never trains,
selects, promotes or claims the campaign's settlement gates have been met.

Prerequisites are a real `campaign_prepare` project template, the existing
`build_moe_gen1_mix.py` SHA-attested mix, and a READY *full* 2048-token
`moe_gen1_train_preflight.py` receipt. The trained datasets are the mix's
original two distinct JSONLs, not a synthetic planner corpus.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chowder.project import project_from_mapping  # noqa: E402
from chowder.backends.tokenizer_assets import tokenizer_asset_digests  # noqa: E402
from chowder.growth.campaign_prepare import (  # noqa: E402
    CampaignPrepareRefusal,
    placement_gate_refusal,
)

FROZEN_LENGTH = 2048
FROZEN_LRS = (5e-5, 1e-4)
FROZEN_GPU_HOURS_PER_ARM = 2.5
ARM_DATA = (("fable", "train-fable.jsonl"), ("control", "train-control.jsonl"))
ALLOWED_PLACEMENTS = frozenset({"cpu-transient-grad", "cpu-transient-grad-bulk", "cpu-transient-grad-int8"})


class FourArmRefusal(ValueError):
    """The inputs cannot represent the frozen four independent comparisons."""


def _require(condition: bool, why: str) -> None:
    if not condition:
        raise FourArmRefusal(why)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FourArmRefusal(f"cannot read {path}: {exc}") from exc
    _require(isinstance(raw, dict), f"{path} must be a JSON object")
    return raw


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(row: Any) -> str:
    _require(isinstance(row, dict), "each training sample must be a JSON object")
    _require(isinstance(row.get("messages"), list), "training rows must carry chat messages")
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_arm(path: Path, expected_rows: int) -> list[str]:
    rows: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(_row_hash(json.loads(line)))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FourArmRefusal(f"invalid training rows at {path}: {exc}") from exc
    _require(rows and len(rows) == expected_rows, f"{path.name} rows differ from mix manifest")
    return rows


def _check_sources(mix: Path, receipt: Path) -> tuple[dict[str, Any], dict[str, tuple[str, int]]]:
    manifest_path = mix / "manifest.json"
    manifest = _read_json(manifest_path)
    checked = _read_json(receipt)
    _require(manifest.get("builder") == "chowder_batch/build_moe_gen1_mix.py", "wrong mix builder")
    _require(manifest.get("template_validated") is True, "mix chat templates were not validated")
    _require(manifest.get("tool_use_instrument") == "present", "protected tool-use instrument absent from mix")
    tokenizer = manifest.get("tokenizer")
    _require(isinstance(tokenizer, dict) and tokenizer.get("loaded") is True
             and isinstance(tokenizer.get("path"), str) and bool(tokenizer["path"]),
             "mix was not rendered with an identified real tokenizer")
    tokenizer_path = Path(tokenizer["path"]).resolve()
    _require(checked.get("tokenizer_manifest_path") == tokenizer["path"],
             "trainer screening used a different tokenizer than the mix builder")
    tokenizer_json = tokenizer_path / "tokenizer.json"
    _require(tokenizer_json.is_file() and checked.get("tokenizer_json_sha256") == _sha(tokenizer_json),
             "trainer screening tokenizer bytes are missing or have changed")
    # The training worker loads AutoTokenizer from backend.base_model. A READY
    # receipt for a different tokenizer can hide 2048-token answer truncation.
    try:
        screened_assets = tokenizer_asset_digests(tokenizer_path)
    except (OSError, ValueError) as exc:
        raise FourArmRefusal(f"screened tokenizer assets unavailable: {exc}") from exc
    _require(checked.get("tokenizer_assets_sha256") == screened_assets,
             "trainer receipt does not bind current tokenizer files; regenerate 2048 preflight")
    _require(checked.get("tokenizer_config_sha256") == screened_assets.get("tokenizer_config.json"),
             "trainer preflight chat template configuration differs from tokenizer")
    _require(checked.get("ready") is True and type(checked.get("max_length")) is int
             and checked["max_length"] == FROZEN_LENGTH,
             "full 2048-token trainer preflight is not READY")
    _require(checked.get("manifest_sha256") == _sha(manifest_path),
             "trainer preflight was computed for a different mix manifest")
    _require(isinstance(checked.get("arms"), dict), "trainer preflight lacks per-arm reports")
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "mix manifest missing output identities")
    sources: dict[str, tuple[str, int]] = {}
    hashes: dict[str, list[str]] = {}
    for arm, filename in ARM_DATA:
        info = outputs.get(filename)
        report = checked["arms"].get(filename)
        _require(isinstance(info, dict) and isinstance(report, dict),
                 f"{arm}: manifest or preflight evidence absent")
        _require(info.get("fable_declared") is (arm == "fable"),
                 f"{arm}: Fable declaration disagrees with arm identity")
        size = info.get("rows")
        _require(type(size) is int and size > 0 and report.get("rows") == size,
                 f"{arm}: row census differs between source and admission")
        _require(all(report.get(key) == 0 for key in
                     ("over_length", "invalid_chat", "answer_span_truncated", "answer_span_completely_removed")),
                 f"{arm}: training would truncate or reject admitted rows")
        path = mix / filename
        expected_sha = info.get("sha256")
        _require(isinstance(expected_sha, str) and len(expected_sha) == 64
                 and path.is_file() and _sha(path) == expected_sha,
                 f"{arm}: training JSONL differs from mix manifest SHA")
        _require(isinstance(info.get("sources"), list) and bool(info["sources"]),
                 f"{arm}: source provenance absent")
        if arm == "control":
            _require(not any("fable" in str(source).lower() for source in info["sources"]),
                     "control contains a Fable source")
        hashes[arm] = _load_arm(path, size)
        sources[arm] = (expected_sha, size)
    _require(sources["fable"][0] != sources["control"][0],
             "Fable and control corpora must have distinct SHA identities")
    control_cursor = 0
    for item in hashes["fable"]:
        if control_cursor < len(hashes["control"]) and item == hashes["control"][control_cursor]:
            control_cursor += 1
    _require(control_cursor == len(hashes["control"]) and len(hashes["control"]) < len(hashes["fable"]),
             "control is not a strict order-preserving subset of Fable treatment")
    _require(checked.get("fable_only_rows") == len(hashes["fable"]) - len(hashes["control"]),
             "Fable-only row count differs from screened evidence")
    return {"manifest_sha256": _sha(manifest_path), "receipt_sha256": _sha(receipt),
            "tokenizer_assets_sha256": screened_assets,
            "tokenizer_json_sha256": _sha(tokenizer_json)}, sources


def plan_four_arms(*, project_template: Path, mix: Path, train_preflight: Path,
                   out_dir: Path) -> dict[str, Any]:
    """CPU-only deterministic plan; validate real production specs, write nothing."""
    project_template = Path(project_template).resolve()
    mix = Path(mix).resolve()
    train_preflight = Path(train_preflight).resolve()
    out_dir = Path(out_dir).resolve()
    _require(project_template.is_file() and mix.is_dir() and train_preflight.is_file(),
             "prepared project, both corpora and trainer receipt are required")
    _require(not out_dir.exists(), "output exists; comparisons are never overwritten")
    _require(not any(out_dir == protected or out_dir.is_relative_to(protected)
                     or protected.is_relative_to(out_dir)
                     for protected in (project_template.parent, mix))
             and not train_preflight.is_relative_to(out_dir),
             "output would overlap a protected input directory")
    provenance, sources = _check_sources(mix, train_preflight)
    prepared = _read_json(project_template)
    backend = (prepared.get("config") or {}).get("backend")
    _require(isinstance(backend, dict) and backend.get("type") == "transformers-peft",
             "prepared project must use production transformers-peft backend")
    _require(backend.get("dataset") == "{corpus}",
             "expected actual campaign_prepare template; do not replace an already bound dataset")
    _require(backend.get("dataset_format") == "chat" and backend.get("messages_field") == "messages"
             and backend.get("tools_field") == "tools", "Gen1 must retain native chat/tool schemas")
    _require(backend.get("max_length") == FROZEN_LENGTH,
             "prepared template does not retain the frozen 2048-token training horizon")
    _require(backend.get("expert_placement") in ALLOWED_PLACEMENTS,
             "prepared project does not declare a supported frozen-expert placement")
    _require((backend.get("lora") or {}).get("target_preset") == "moe_frozen_experts",
             "prepared project does not declare MoE frozen-expert LoRA targets")
    _require(backend.get("base_model") and "<TODO" not in str(backend["base_model"]),
             "prepared project does not name the actual REAP base model")
    try:
        training_assets = tokenizer_asset_digests(Path(backend["base_model"]).resolve())
    except (OSError, ValueError) as exc:
        raise FourArmRefusal(f"training model tokenizer unavailable: {exc}") from exc
    _require(training_assets == provenance["tokenizer_assets_sha256"],
             "training model tokenizer differs from 2048-token screened tokenizer")
    try:
        placement_gate_refusal(
            expert_placement=str(backend["expert_placement"]),
            where="Gen1 four-arm project preparation",
            base_model=str(backend["base_model"]),
        )
    except CampaignPrepareRefusal as exc:
        raise FourArmRefusal(f"frozen hardware placement gate refused: {exc}") from exc
    _require(prepared.get("baseline", {}).get("mode") == "fixed",
             "a trusted measured parent baseline must remain fixed across arms")
    goal = prepared.get("goal")
    _require(isinstance(goal, dict), "prepared project has no valid goal")
    per_arm_budget = goal.get("gpu_hour_budget")
    _require(type(per_arm_budget) in (int, float) and math.isfinite(per_arm_budget)
             and 0 < per_arm_budget <= FROZEN_GPU_HOURS_PER_ARM,
             "goal.gpu_hour_budget must be finite and positive and <=2.5 GPU-hours per arm")
    experiment = prepared.get("experiment")
    estimate = experiment.get("estimated_gpu_hours") if isinstance(experiment, dict) else None
    _require(type(estimate) in (int, float) and math.isfinite(estimate)
             and 0 < estimate <= per_arm_budget,
             "experiment.estimated_gpu_hours must fit the frozen per-arm GPU budget")
    configuration = prepared.get("config")
    lifecycle = configuration.get("goal_lifecycle") if isinstance(configuration, dict) else None
    # The canonical runner uses `legacy_unbounded` only for direction-only
    # target *assessment*. It does not waive the separate measured GPU budget.
    # Honor that explicitly frozen protocol, but never invent that opt-in:
    # missing/unknown modes may mean silently changing an objective's semantics.
    _require(isinstance(lifecycle, dict)
             and lifecycle in ({"mode": "bounded"}, {"mode": "legacy_unbounded"}),
             "config.goal_lifecycle must explicitly declare bounded or "
             "legacy_unbounded; missing and unknown modes are refused")
    metrics = goal.get("metrics")
    _require(isinstance(metrics, list) and bool(metrics), "bounded goal requires metric targets")
    for metric in metrics:
        _require(isinstance(metric, dict), "bounded goal contains a non-object metric")
        targets = [metric.get(bound) for bound in ("minimum", "maximum")
                   if metric.get(bound) is not None]
        _require(all(type(target) in (int, float) and math.isfinite(target)
                     for target in targets), "numeric goal targets must be finite")
        if lifecycle["mode"] == "bounded":
            _require(bool(targets),
                     "mode='bounded' needs an explicit target on every metric; "
                     "a direction-only goal could terminate without improvement")
        else:
            _require(not targets,
                     "mode='legacy_unbounded' is only effective for direction-only "
                     "metrics; numerical goal bounds would change its semantics")
    arms: list[dict[str, Any]] = []
    for data_arm, filename in ARM_DATA:
        for lr, label in zip(FROZEN_LRS, ("lr5e-05", "lr0.0001")):
            identity = f"gen1-{data_arm}-{label}"
            project_dir = out_dir / identity
            project = copy.deepcopy(prepared)
            project["name"] = identity
            project["work_dir"] = str(project_dir)
            project["registry_path"] = str(project_dir / "runs.db")
            project["experiment"]["experiment_id"] = identity
            project["config"]["backend"]["dataset"] = str(mix / filename)
            project["config"]["backend"]["dataset_sha256"] = sources[data_arm][0]
            project["config"]["backend"]["training"]["learning_rate"] = lr
            # Keep the source's explicitly declared objective semantics as-is;
            # never manufacture the runner's legacy_unbounded opt-in.
            # Exercise the same strict spec parsing the production project loader
            # runs, on each concrete arm. This never materializes a model.
            project_from_mapping(project, source_dir=project_dir)
            arms.append({
                "id": identity, "dataset_arm": data_arm, "fable_declared": data_arm == "fable",
                "learning_rate": lr, "max_length": FROZEN_LENGTH,
                "data_sha256": sources[data_arm][0], "rows": sources[data_arm][1],
                "dataset_file": str(mix / filename),
                "sources": _read_json(mix / "manifest.json")["outputs"][filename]["sources"],
                "project_path": str(project_dir / "project.json"),
                "project": project,
                "validate_command": [sys.executable, "-m", "chowder.cli", "project-validate", str(project_dir / "project.json")],
                "train_command": [sys.executable, "-m", "chowder.cli", "train", str(project_dir / "project.json")],
            })
    return {"schema": "chowder.moe-gen1-four-independent-trials.v1",
            "status": "PREPARED_ONLY_NOT_EXECUTED", "input_project_sha256": _sha(project_template),
            "mix": str(mix), "admission": provenance, "arms": arms,
            "limitation": "Four independent executable training trials, NOT four frozen campaign candidates; "
                          "candidate adjudication, cumulative cost settlement, and N5 promotion approval "
                          "require separately preregistered evidence."}


def prepare_four_arms(*, project_template: Path, mix: Path, train_preflight: Path,
                      out_dir: Path) -> dict[str, Any]:
    """Create four production project files and an auditable cross-arm index."""
    plan = plan_four_arms(project_template=project_template, mix=mix,
                         train_preflight=train_preflight, out_dir=out_dir)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    index = {key: value for key, value in plan.items() if key != "arms"}
    index["arms"] = []
    for arm in plan["arms"]:
        project = arm["project"]
        path = Path(arm["project_path"])
        path.parent.mkdir(exist_ok=False)
        path.write_text(json.dumps(project, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        ref = {key: value for key, value in arm.items() if key != "project"}
        ref["project_sha256"] = _sha(path)
        index["arms"].append(ref)
    (out / "four-arm-plan.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-template", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--train-preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_four_arms(project_template=args.project_template, mix=args.mix,
                                   train_preflight=args.train_preflight, out_dir=args.out)
    except (FourArmRefusal, TypeError, KeyError, ValueError, OSError) as exc:
        print(f"MoE Gen1 four-arm preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
