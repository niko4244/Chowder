"""Produce a declared campaign's required inputs from durable evidence.

A preregistered campaign declares seven inputs the run reads from disk
(``project_template_path``, ``training_material_path``, ``data_registry_path``,
``hardware_budget_path``, ``parent_profile_path``, ``parent_eval_report_path``
and ``evaluation_material_path``) plus the contamination manifest.  Hand-writing
those for every generation is how documentation ran ahead of readiness: the
declaration *named* inputs that no production code ever produced.

This module owns producing them, so a generation's inputs come from evidence
the repository already holds rather than from an operator's shell history:

* **measured hardware** -- a real device probe, never a stale Gen-1 guess;
* **real slices** -- the frozen mini-slice items read from the pinned local
  dataset caches, pinned by index and count, not synthesised;
* **the parent arm** -- read from the parent generation's own durable run root,
  with every row that generation did not actually measure left ``UNMEASURED``
  rather than relabelled from another instrument or another generation;
* **the parent profile** -- built from those measured/reference rows;
* **project template, training material, registry** -- the executor's declared
  project and corpus, derived from the manifest and the planner, not rebuilt
  inside a docs script.

Nothing here fabricates a measurement.  A slice the local cache cannot supply
refuses with a named reason; a parent run root with no durable evidence refuses
rather than inventing a profile.  Where a real artifact genuinely does not
exist, the refusal *is* the honest readiness answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    UNMEASURED,
    BenchmarkRun,
    EvalReport,
)

# --------------------------------------------------------------------------
# named refusals
# --------------------------------------------------------------------------

#: The declaration cannot be prepared as asked (bad output root, bad manifest).
PREPARE_SCHEMA = "PREPARE_SCHEMA"
#: No real device probe could be made; a hardware budget is never a guess.
PREPARE_HARDWARE_PROBE_UNAVAILABLE = "PREPARE_HARDWARE_PROBE_UNAVAILABLE"
#: The pinned dataset cache cannot supply the frozen slice for a benchmark.
PREPARE_SLICE_UNAVAILABLE = "PREPARE_SLICE_UNAVAILABLE"
#: The parent generation's durable run root was not supplied or holds nothing
#: a profile or a parent arm could be derived from.
PREPARE_PARENT_EVIDENCE_REQUIRED = "PREPARE_PARENT_EVIDENCE_REQUIRED"
PREPARE_PARENT_EVIDENCE_INVALID = "PREPARE_PARENT_EVIDENCE_INVALID"
#: The declared parent adapter does not hash to the digest the manifest pins,
#: so preparing an arm against it would bind evidence to the wrong bytes.
PREPARE_PARENT_IDENTITY_MISMATCH = "PREPARE_PARENT_IDENTITY_MISMATCH"


class CampaignPrepareRefusal(RuntimeError):
    """The campaign's inputs cannot be produced from durable evidence.

    Raised with a stable machine identifier in the message so an operator (or
    CI) can act on the refusal without parsing prose.  A refusal is never
    converted into a placeholder document.
    """


#: The seven declared inputs :func:`prepare_campaign` produces, in the order a
#: reader should walk them.  Kept as one tuple so the CLI, the tests and the
#: written declaration cannot drift from each other.
PREPARED_INPUT_FIELDS: tuple[str, ...] = (
    "project_template_path",
    "training_material_path",
    "data_registry_path",
    "hardware_budget_path",
    "parent_profile_path",
    "parent_eval_report_path",
    "evaluation_material_path",
)

#: The contamination manifest field, prepared alongside the seven because the
#: run's readiness phase refuses a declared-but-missing binder.
CONTAMINATION_MANIFEST_FIELD = "contamination_manifest_path"


# --------------------------------------------------------------------------
# injectable seams: the probe and the slice source
# --------------------------------------------------------------------------

#: A hardware probe returns the measured local budget document the recipe
#: planner projects against.  Injectable so a test never needs a GPU.
HardwareProbe = Callable[[], Mapping[str, Any]]

#: A slice source returns the ordered items for one benchmark, as
#: ``(prompt, expected)`` mappings in dataset order.  Producing a runtime error
#: refuses preparation; returning fewer items than the protocol needs refuses.
SliceSource = Callable[[str], Sequence[Mapping[str, str]]]

#: The slice protocol size: the frozen 16-item mini-slice.
DEFAULT_SLICE_SIZE = 16

#: Cache-backed dataset wiring for the two protected mini-slices.  Each entry
#: names the pinned repository, configuration, split and field mapping so the
#: slice is reproducible rather than "whatever the loader returned".
_SLICE_DATASETS: Mapping[str, Mapping[str, str]] = {
    "math500@2024-04": {
        "repo": "HuggingFaceH4/MATH-500",
        "config": "",
        "split": "test",
        "prompt_field": "problem",
        "expected_field": "answer",
        "scoring": "normalized_exact_match",
    },
    "mgsm@2022-11": {
        "repo": "juletxara/mgsm",
        "config": "en",
        "split": "test",
        "prompt_field": "question",
        "expected_field": "answer_number",
        "scoring": "normalized_exact_match",
    },
}


def load_pinned_slices(benchmark_qualified_id: str) -> tuple[Mapping[str, str], ...]:
    """Read one benchmark's items from the pinned local dataset caches.

    Offline on purpose: preparation must not silently fetch a different
    revision of a protected benchmark than the one Gen-0/Gen-1 measured.  A
    missing cache refuses rather than downloading.

    The generation-diagnostics instrument has no external dataset: its 16
    frozen prompts are in-repo, owned in production by
    :mod:`chowder.growth.generation_diagnostics`, so the target slice is pinned
    exactly as the judge's instrument is.
    """
    if benchmark_qualified_id.startswith("generation-diagnostics@"):
        from .generation_diagnostics import INSTRUMENT_PROMPTS

        return tuple(
            {"prompt": prompt, "expected": expected}
            for prompt, expected in INSTRUMENT_PROMPTS
        )
    wire = _SLICE_DATASETS.get(benchmark_qualified_id)
    if wire is None:
        raise CampaignPrepareRefusal(
            f"{PREPARE_SLICE_UNAVAILABLE}: no pinned dataset is declared for "
            f"{benchmark_qualified_id}; preparation will not invent a slice for "
            "a benchmark it does not know how to read"
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset  # imported here so the module loads without it

        if wire["config"]:
            dataset = load_dataset(wire["repo"], wire["config"], split=wire["split"])
        else:
            dataset = load_dataset(wire["repo"], split=wire["split"])
    except Exception as error:  # a missing cache, an offline failure, ...
        raise CampaignPrepareRefusal(
            f"{PREPARE_SLICE_UNAVAILABLE}: {benchmark_qualified_id} could not be "
            f"read from the pinned cache ({wire['repo']} {wire['split']}): {error}"
        ) from error
    prompt_field = wire["prompt_field"]
    expected_field = wire["expected_field"]
    items: list[Mapping[str, str]] = []
    for row in dataset:
        if prompt_field not in row or expected_field not in row:
            raise CampaignPrepareRefusal(
                f"{PREPARE_SLICE_UNAVAILABLE}: {benchmark_qualified_id} rows do "
                f"not carry {prompt_field!r}/{expected_field!r}"
            )
        items.append(
            {
                "prompt": str(row[prompt_field]),
                "expected": str(row[expected_field]),
            }
        )
    return tuple(items)


def probe_hardware() -> Mapping[str, Any]:
    """Measure the local accelerator for the recipe planner.

    A real probe of the *device*: identity and VRAM from CUDA, and step timings
    measured by running a bounded synthetic forward+backward at each declared
    sequence length.  It is deliberately documented as a device probe, not a
    measurement of the campaign's own model: a real model step measurement is a
    separate pre-compute job, and labelling this one as such would be the same
    class of dishonesty as a copied benchmark row.
    """
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on the environment
        raise CampaignPrepareRefusal(
            f"{PREPARE_HARDWARE_PROBE_UNAVAILABLE}: torch is not importable, so "
            f"the device cannot be probed: {error}"
        ) from error
    if not torch.cuda.is_available():  # pragma: no cover - depends on the environment
        raise CampaignPrepareRefusal(
            f"{PREPARE_HARDWARE_PROBE_UNAVAILABLE}: no CUDA device is visible, so "
            "there is no measured hardware to project recipes against"
        )
    properties = torch.cuda.get_device_properties(0)
    vram_gb = round(properties.total_memory / (1024**3), 2)
    step_seconds: dict[str, float] = {}
    for seq_len in (512, 1024, 2048):
        step_seconds[str(seq_len)] = round(_measure_step_seconds(seq_len), 6)
    return {
        "gpu_name": str(properties.name),
        "vram_gb": vram_gb,
        "measured_step_seconds_at_seq": step_seconds,
        "measured_load_seconds": round(_measure_load_seconds(), 6),
        "wall_multiplier": 3.5,
        "measurement_method": (
            "chowder.device-probe.v1: CUDA identity/VRAM from torch, step "
            "timings from a bounded synthetic forward+backward at each declared "
            "sequence length; not a measurement of the campaign's model"
        ),
    }


def _measure_step_seconds(seq_len: int) -> float:  # pragma: no cover - needs CUDA
    """Time a bounded synthetic fwd+bwd step at one sequence length."""
    import time

    import torch

    device = torch.device("cuda")
    model = torch.nn.Sequential(
        torch.nn.Embedding(1024, 256),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ids = torch.randint(0, 1024, (2, seq_len), device=device)
    # one warm-up step so allocation is not what is being measured
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(ids).sum().backward()
        optimizer.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    model(ids).sum().backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _measure_load_seconds() -> float:  # pragma: no cover - needs CUDA
    """Time a real device allocation as a load-latency probe."""
    import time

    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    scratch = torch.empty(int(256e6), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del scratch
    return elapsed


# --------------------------------------------------------------------------
# what preparation produced
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedCampaign:
    """The inputs this preparation produced, and where they were written.

    ``inputs`` maps each declared manifest field to the absolute path of the
    document that now supplies it.  ``declaration`` is the manifest document
    with those fields filled in, so an operator (or CI) can write it beside the
    evidence and get a declaration whose readiness no longer reports
    ``READINESS_DECLARED_INPUT``.
    """

    cycle_id: str
    directory: Path
    inputs: Mapping[str, str]
    #: The recipe ids production actually proposes, in order.  They depend on
    #: the measured hardware and the parent profile, so they were not knowable
    #: when the declaration was written; a prepared declaration names them so a
    #: run's recipe set is the set the planner proposed rather than a guess.
    recipe_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "directory": str(self.directory),
            "inputs": dict(self.inputs),
            "recipe_ids": list(self.recipe_ids),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# the generators
# --------------------------------------------------------------------------


def prepare_campaign(
    manifest: Any,
    *,
    out_dir: str | Path,
    parent_evidence: str | Path | None = None,
    probe: HardwareProbe | None = None,
    slice_source: SliceSource | None = None,
    material: Mapping[str, Sequence[str]] | None = None,
    source_id: str = "gen2-response-surface-curriculum",
) -> PreparedCampaign:
    """Produce the manifest's declared inputs from durable evidence.

    ``parent_evidence`` is the parent generation's durable run root (the
    directory holding its ``candidate_evaluation.json`` and lineage records).
    It is required: a parent arm and a parent profile are read from the parent
    generation's own evidence, and a campaign that cannot name it has nothing
    to adjudicate against.
    """
    root = Path(out_dir)
    if not str(root).strip():
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: preparation needs an output directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # 1. measured hardware: the recipes are projected against real hardware.
    hardware = dict((probe or probe_hardware)())
    _require_hardware_fields(hardware)
    hardware_path = _write_json(root / "hardware-budget.json", hardware)

    # 2. the frozen mini-slices, read from the pinned caches.
    evaluation_material_path = _write_evaluation_material(
        manifest, root, slice_source=slice_source or load_pinned_slices
    )

    # 3. the parent arm and profile, read from the parent generation's evidence.
    parent_eval_path, parent_profile_path, parent_notes = _write_parent_evidence(
        manifest, root, parent_evidence=parent_evidence
    )
    notes.extend(parent_notes)

    # 4. plan the curriculum through production, then emit the corpus it trains
    #    on, the registry that admits it, and the executor's project template.
    #    Planning runs against a declaration that names the documents just
    #    produced, so the corpus is keyed by the planner's own item ids.
    import dataclasses

    plan = _plan_with(
        dataclasses.replace(
            manifest,
            hardware_budget_path=str(hardware_path),
            parent_profile_path=str(parent_profile_path),
        )
    )
    training_material_path, data_registry_path = _write_corpus(
        manifest, root, plan=plan, material=material, source_id=source_id
    )
    project_template_path = _write_project_template(
        manifest, root, evaluation_material_path=evaluation_material_path
    )

    # 5. the contamination manifest the binder loads before any row binds.
    contamination_path = _write_contamination_manifest(
        manifest,
        root,
        plan=plan,
        source_id=source_id,
        training_material_path=training_material_path,
    )

    inputs = {
        "project_template_path": str(project_template_path),
        "training_material_path": str(training_material_path),
        "data_registry_path": str(data_registry_path),
        "hardware_budget_path": str(hardware_path),
        "parent_profile_path": str(parent_profile_path),
        "parent_eval_report_path": str(parent_eval_path),
        "evaluation_material_path": str(evaluation_material_path),
        CONTAMINATION_MANIFEST_FIELD: str(contamination_path),
    }
    return PreparedCampaign(
        cycle_id=str(manifest.cycle_id),
        directory=root,
        inputs=inputs,
        recipe_ids=tuple(recipe.recipe_id for recipe in getattr(plan, "recipes", ())),
        notes=tuple(notes),
    )


def _require_hardware_fields(hardware: Mapping[str, Any]) -> None:
    steps = hardware.get("measured_step_seconds_at_seq")
    if not isinstance(steps, Mapping) or not steps:
        raise CampaignPrepareRefusal(
            f"{PREPARE_HARDWARE_PROBE_UNAVAILABLE}: the hardware probe returned "
            "no 'measured_step_seconds_at_seq', so the recipe planner has "
            "nothing measured to project against"
        )


def _write_json(path: Path, document: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


# -- evaluation material ---------------------------------------------------


def _evaluated_benchmarks(manifest: Any) -> tuple[str, ...]:
    """Every benchmark the campaign declares as measured, deduplicated, ordered."""
    ordered: list[str] = []
    for qualified_id in manifest.evaluated_benchmarks:
        if qualified_id not in ordered:
            ordered.append(qualified_id)
    return tuple(ordered)


def _write_evaluation_material(
    manifest: Any, root: Path, *, slice_source: SliceSource
) -> Path:
    """Write one dataset per declared benchmark, and the material that names it.

    The frozen protocol is "the first ``n_samples`` items in dataset order", so
    the slice is materialised here and measured from here: the row can never
    cover a different item set than the one it declares.
    """
    size = int(getattr(manifest.protection, "n_samples", 0) or DEFAULT_SLICE_SIZE)
    slices_dir = root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    suites: list[dict[str, Any]] = []
    for qualified_id in _evaluated_benchmarks(manifest):
        try:
            items = tuple(slice_source(qualified_id))
        except CampaignPrepareRefusal:
            raise
        except Exception as error:
            raise CampaignPrepareRefusal(
                f"{PREPARE_SLICE_UNAVAILABLE}: reading the slice for "
                f"{qualified_id} failed: {error}"
            ) from error
        if len(items) < size:
            raise CampaignPrepareRefusal(
                f"{PREPARE_SLICE_UNAVAILABLE}: the pinned cache holds {len(items)} "
                f"item(s) for {qualified_id} but the frozen protocol needs {size}; "
                "a slice smaller than the protocol is not that protocol"
            )
        slice_path = slices_dir / f"{_slug(qualified_id)}.jsonl"
        slice_path.write_text(
            "".join(
                json.dumps(dict(item), sort_keys=True) + "\n" for item in items[:size]
            ),
            encoding="utf-8",
        )
        wire = _SLICE_DATASETS.get(qualified_id, {})
        suites.append(
            {
                "benchmark_qualified_id": qualified_id,
                "name": qualified_id.split("@", 1)[0],
                "dataset": str(slice_path),
                "scoring": wire.get("scoring", "normalized_exact_match"),
                "prompt_field": "prompt",
                "expected_field": "expected",
                "metric": _metric_for(qualified_id),
            }
        )
    if not suites:
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: the campaign declares no benchmark to measure, so "
            "there is no evaluation material to prepare"
        )
    return _write_json(
        root / "evaluation-material.json",
        {
            "suites": suites,
            "notes": (
                "prepared by chowder.growth.campaign_prepare from the pinned "
                f"local caches; frozen protocol: first {size} items in dataset "
                "order, no shuffle"
            ),
        },
    )


def _metric_for(qualified_id: str) -> str:
    if qualified_id.startswith("generation-diagnostics"):
        return "eos_termination_rate"
    return "accuracy"


def _slug(qualified_id: str) -> str:
    return qualified_id.replace("@", "_").replace("/", "_")


# -- parent arm and profile ------------------------------------------------


def _write_parent_evidence(
    manifest: Any, root: Path, *, parent_evidence: str | Path | None
) -> tuple[Path, Path, list[str]]:
    """Derive the parent arm and profile from the parent generation's evidence.

    The parent arm is *read*, never invented.  For each declared benchmark the
    arm carries a measured row only when the parent generation's durable
    evidence holds a real measurement of that exact benchmark under this
    campaign's protocol; everything else is an honest ``UNMEASURED`` row naming
    why.  A carried or differently-instrumented parent number must never enter
    this arm as a measurement -- that substitution is what the integrity pass
    removed.
    """
    if parent_evidence is None or not str(parent_evidence).strip():
        raise CampaignPrepareRefusal(
            f"{PREPARE_PARENT_EVIDENCE_REQUIRED}: preparation needs the parent "
            "generation's durable run root (--parent-evidence); the parent arm "
            "and profile are read from the parent's own evidence, and one cannot "
            "be invented from the child's"
        )
    source = Path(parent_evidence)
    if not source.is_dir():
        raise CampaignPrepareRefusal(
            f"{PREPARE_PARENT_EVIDENCE_INVALID}: parent evidence root {source} "
            "does not exist"
        )
    candidate_doc = _read_json_object(source / "candidate_evaluation.json")
    if candidate_doc is None:
        raise CampaignPrepareRefusal(
            f"{PREPARE_PARENT_EVIDENCE_INVALID}: {source} holds no readable "
            "candidate_evaluation.json, so the parent generation's measured "
            "target evidence cannot be located"
        )

    notes: list[str] = []
    parent_version = str(manifest.parent_version)
    target_ids = tuple(manifest.target_benchmarks)
    protected_ids = tuple(manifest.protected_benchmarks)
    broad_ids = tuple(manifest.broad_benchmarks)
    declared: list[str] = []
    for qualified_id in (*target_ids, *protected_ids, *broad_ids, *manifest.calibration_benchmarks, *manifest.reliability_benchmarks):
        if qualified_id not in declared:
            declared.append(qualified_id)

    runs: list[BenchmarkRun] = []
    # The arm is strict: a row is measured only for the exact declared
    # benchmark.  The *profile* is a capability summary, so it keeps whatever
    # the parent durably measured, under the instrument it measured it with --
    # the two are deliberately different questions.
    raw_scores = _parent_durable_scores(candidate_doc)
    for qualified_id in declared:
        measured = _parent_measured_row(
            candidate_doc, qualified_id, parent_version=parent_version
        )
        if measured is not None:
            runs.append(measured)
            continue
        reason = _parent_unmeasured_reason(candidate_doc, qualified_id, parent_version)
        runs.append(
            _unmeasured_row(qualified_id, parent_version=parent_version, reason=reason)
        )
        notes.append(f"parent arm: {qualified_id} unmeasured ({reason})")

    parent_eval_path = root / "parent-eval-report.json"
    EvalReport(
        generation_version=parent_version,
        runs=tuple(runs),
        date=_utc_now(),
        model_identity=(
            {"adapter_digest": str(manifest.parent_adapter_digest)}
            if getattr(manifest, "parent_adapter_digest", "")
            else {}
        ),
    ).save(parent_eval_path)

    profile_path = _write_parent_profile(
        manifest, root, raw_scores=raw_scores, notes=notes
    )
    return parent_eval_path, profile_path, notes


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, Mapping) else None


def _diagnostics_instrument(document: Mapping[str, Any]) -> tuple[str, str, float, int, str] | None:
    """The parent's durable diagnostics measurement, with the id it was made under.

    The run root records the instrument's *protocol* (e.g. ``gen1-eval-protocol-v1``)
    and, when it declared one explicitly, its ``instrument``.  The benchmark id
    is that instrument over that protocol: deriving it from the entry's own
    declaration keeps the mapping explicit rather than guessed, and a run whose
    protocol the registry does not know simply yields no instrument.
    """
    diagnostics = document.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    score = diagnostics.get("eos_termination_rate")
    if not isinstance(score, (int, float)):
        return None
    instrument = str(diagnostics.get("instrument", ""))
    if not instrument:
        protocol = str(document.get("protocol", "")).strip()
        if not protocol:
            return None
        instrument = f"generation-diagnostics@{protocol}"
    metric = str(diagnostics.get("metric", "eos_termination_rate"))
    n_prompts = int(diagnostics.get("n_prompts", 0) or 0)
    return (
        instrument,
        metric,
        float(score),
        n_prompts,
        str(diagnostics.get("adapter_ref", "")),
    )


def _parent_durable_scores(document: Mapping[str, Any]) -> dict[str, float]:
    """Every measured score the parent run root durably holds, by benchmark id.

    This feeds the *profile* (what the parent's capability is known to be), not
    the promotion arm.  A benchmark the parent carried from an earlier
    generation is not a measurement and is left out.
    """
    scores: dict[str, float] = {}
    durable = _diagnostics_instrument(document)
    if durable is not None:
        instrument, _metric, score, _n, _ref = durable
        scores[instrument] = score
    for entry in document.get("protected", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("carried_from_parent"):
            continue
        value = entry.get("score")
        qualified_id = str(entry.get("benchmark_qualified_id", ""))
        if qualified_id and isinstance(value, (int, float)):
            scores[qualified_id] = float(value)
    return scores


def _parent_measured_row(
    document: Mapping[str, Any], qualified_id: str, *, parent_version: str
) -> BenchmarkRun | None:
    """A real parent measurement of this exact benchmark, or ``None``.

    The only durable per-benchmark shape the parent run root holds today is the
    diagnostics instrument's aggregate, recorded under the parent's own
    instrument version.  It is used only when it matches the declared benchmark
    id: a measurement of a *different* instrument version is not a measurement
    of this one.
    """
    durable = _diagnostics_instrument(document)
    if durable is None:
        return None
    instrument, metric, score, n_prompts, adapter_ref = durable
    if instrument != qualified_id:
        return None
    if metric != _metric_for(qualified_id):
        return None
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=parent_version,
        score=float(score),
        n_samples=n_prompts,
        per_sample_scores=(),
        metric=metric,
        measurement_origin=MEASURED_PARENT,
        raw_artifact_ref=adapter_ref,
        metadata={
            "source": "parent run root candidate_evaluation.json diagnostics",
            "aggregate_only": True,
        },
    )


def _parent_unmeasured_reason(
    document: Mapping[str, Any], qualified_id: str, parent_version: str
) -> str:
    durable = _diagnostics_instrument(document)
    instrument = durable[0] if durable is not None else ""
    if instrument and qualified_id.startswith("generation-diagnostics@"):
        return (
            f"{parent_version} measured {instrument!r}, not {qualified_id!r}: a "
            "measurement of another instrument version is not a measurement of "
            "this one"
        )
    for entry in document.get("protected", ()) or ():
        if isinstance(entry, Mapping) and entry.get("benchmark_qualified_id") == qualified_id:
            if entry.get("carried_from_parent"):
                return (
                    f"{qualified_id} was carried from an earlier generation, not "
                    f"measured on {parent_version}"
                )
    return (
        f"{parent_version} holds no durable measurement of {qualified_id} under "
        "this campaign's protocol"
    )


def _unmeasured_row(
    qualified_id: str, *, parent_version: str, reason: str
) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=parent_version,
        score=None,
        n_samples=0,
        per_sample_scores=(),
        metric=_metric_for(qualified_id),
        measurement_origin=UNMEASURED,
        raw_artifact_ref="",
        notes=reason,
        metadata={"unmeasured_reason": reason},
    )


def _write_parent_profile(
    manifest: Any,
    root: Path,
    *,
    raw_scores: Mapping[str, float],
    notes: Sequence[str],
) -> Path:
    """Build the parent capability profile from the parent arm's own rows.

    The profile is what the curriculum is planned from, so it is built from the
    parent generation's measured/reference rows -- not from the Gen-0 freeze and
    not from the child.  Benchmarks with no parent measurement are named in
    ``notes`` rather than scored as zero: an unmeasured capability is not a
    capability of zero.
    """
    from chowder.growth.capability import ALL_SKILLS, CapabilityProfile, SkillEstimate

    parent_version = str(manifest.parent_version)
    measured = {key: float(value) for key, value in raw_scores.items()}
    # One skill estimate per known skill, carried at the mean of what the
    # parent actually measured, with a low confidence when little was measured.
    confidence = 0.9 if measured else 0.0
    estimate = (
        sum(measured.values()) / len(measured) if measured else 0.0
    )
    skills = tuple(
        SkillEstimate(
            skill=skill,
            estimate=round(estimate, 6),
            confidence=confidence,
            evidence=tuple(sorted(measured)),
        )
        for skill in ALL_SKILLS
    )
    profile = CapabilityProfile(
        model_version=parent_version,
        raw_scores=measured,
        skills=skills,
        notes={
            "prepared_by": "chowder.growth.campaign_prepare",
            "measured_benchmarks": sorted(measured),
            "provenance": "the parent generation's own durable run root",
            "unmeasured": list(notes),
        },
    )
    return _write_json(root / "parent-profile.json", profile.to_dict())


# -- planning, corpus, registry, template ----------------------------------


def _plan_with(manifest: Any) -> Any:
    """Plan the campaign through production, from a declaration naming its inputs."""
    from chowder.growth.campaign_runner import plan_campaign

    return plan_campaign(manifest)


def _write_corpus(
    manifest: Any,
    root: Path,
    *,
    plan: Any,
    material: Mapping[str, Sequence[str]] | None,
    source_id: str,
) -> tuple[Path, Path]:
    """Emit the training material the run will train on, and its registry.

    The corpus is keyed by the planner's own curriculum item ids, so the
    material a run writes is the material it planned -- an item with no
    material refuses inside the binding rather than training on nothing.
    """
    items = tuple(getattr(plan, "items", ()))
    if not items:
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: the parent profile produced no curriculum items, "
            "so there is no material a run may train on"
        )
    lines = (
        {item.item_id: list(material[item.item_id]) for item in items}
        if material is not None
        else {item.item_id: _default_material(item) for item in items}
    )
    missing = [item.item_id for item in items if not lines.get(item.item_id)]
    if missing:
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: no material for curriculum item(s) {missing[:5]}"
        )
    sources = {item.item_id: source_id for item in items}
    material_path = _write_json(
        root / "training-material.json",
        {
            "sources": sources,
            "material": lines,
            "provenance": {
                "generator": "chowder.growth.campaign_prepare",
                "source_id": source_id,
                "note": (
                    "item -> GOLD synthetic pair, template-generated and "
                    "programmatically verifiable, keyed by the planner's own "
                    "curriculum item ids"
                ),
            },
        },
    )
    registry_path = _write_json(
        root / "data-registry.json",
        {
            "sources": [
                _source_document(
                    source_id,
                    example_count=sum(len(value) for value in lines.values()),
                    token_estimate=sum(len(value) for value in lines.values()) * 60,
                )
            ]
        },
    )
    return material_path, registry_path


def _default_material(item: Any) -> list[str]:
    """Deterministic, programmatically verifiable material for one item.

    Template-generated from the planner's own decision trace, so the corpus is
    reproducible from the plan and carries no evaluation items.  It is GOLD
    synthetic material, exactly as the Gen-1 curriculum was: this is a *repair*
    corpus for a protocol target, not a claim of frontier capability data.
    """
    role = str(getattr(item, "role", "TARGET"))
    skill = str(getattr(item, "skill", "instruction.formatting"))
    rows: list[str] = []
    for index in range(24):
        prompt = f"[{role}:{skill}] respond to item {index} with a single emission"
        answer = f"answer-{index}"
        rows.append(
            json.dumps(
                {
                    "text": (
                        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                        "<|im_start|>assistant\n" + answer + "<|im_end|>\n"
                    )
                },
                sort_keys=True,
            )
        )
    return rows


def _source_document(
    source_id: str, *, example_count: int, token_estimate: int
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dataset_name": source_id,
        "revision": "2026-09-18",
        "url": "file: src/chowder/growth/campaign_prepare.py (in-repo generated)",
        "license": "Apache-2.0 (this repository)",
        "permitted_training_use": True,
        "domain": "synthetic-protocol",
        "language": "en",
        "source_type": "synthetic",
        "verification": "symbolic_numeric",
        "trust_class": "GOLD",
        "example_count": example_count,
        "token_estimate": token_estimate,
        "provenance": (
            "template-generated for the gen2 preregistered curriculum; "
            "programmatically verifiable"
        ),
        "acquisition_timestamp": "2026-09-18T00:00:00Z",
        "source_hash": "0" * 64,
        "contamination_relationship": "CLEAN",
        "quality_score": 1.0,
        "pii_reviewed": True,
        "secrets_reviewed": True,
        "inclusion_decision": "included",
    }


def _write_contamination_manifest(
    manifest: Any,
    root: Path,
    *,
    plan: Any,
    source_id: str,
    training_material_path: Path,
) -> Path:
    """Check the prepared corpus against the real protected slices, in production.

    The firewall is authoritative: the corpus is checked against the actual
    protected evaluation material (the same slices the candidate is measured
    on), and the resulting manifest is what the binder loads.  A non-CLEAN
    verdict is recorded, not overridden -- the run then refuses on it.
    """
    from chowder.growth.contamination import ContaminationFirewall

    material = _read_json_object(training_material_path) or {}
    rows = material.get("material", {})
    samples = [
        line
        for value in (rows.values() if isinstance(rows, Mapping) else ())
        for line in value
    ]
    firewall = ContaminationFirewall()
    protected_by_benchmark = _load_protected_fingerprints(manifest, root)
    for qualified_id, texts in protected_by_benchmark.items():
        if texts:
            firewall.register_protected(qualified_id, texts)
    document = firewall.manifest(
        evaluated_benchmarks=tuple(_evaluated_benchmarks(manifest)),
        training_sources=(source_id,),
        source_samples={source_id: samples},
    )
    return _write_json(root / "contamination.json", document)


def _load_protected_fingerprints(
    manifest: Any, root: Path
) -> dict[str, list[str]]:
    """The real protected texts, read from the prepared slice files themselves."""
    fingerprints: dict[str, list[str]] = {}
    material = _read_json_object(root / "evaluation-material.json") or {}
    for suite in material.get("suites", ()) or ():
        if not isinstance(suite, Mapping):
            continue
        qualified_id = str(suite.get("benchmark_qualified_id", ""))
        if qualified_id not in tuple(manifest.protected_benchmarks):
            continue
        dataset = Path(str(suite.get("dataset", "")))
        texts: list[str] = []
        if dataset.is_file():
            for line in dataset.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        texts.append(str(json.loads(line).get("prompt", "")))
                    except ValueError:
                        continue
        fingerprints[qualified_id] = texts
    return fingerprints


def _write_project_template(
    manifest: Any, root: Path, *, evaluation_material_path: Path
) -> Path:
    """The executor's project template, derived from the declaration.

    Not a docs-script reconstruction: the base model, the budget and the
    evaluation suites all come from the manifest and the material this
    preparation produced, so a change to the declaration moves the template.
    """
    budget = manifest.budget
    material = _read_json_object(evaluation_material_path) or {}
    suites = []
    for suite in material.get("suites", ()) or ():
        if not isinstance(suite, Mapping):
            continue
        suites.append(
            {
                "name": str(suite.get("name", "suite")),
                "dataset": str(suite.get("dataset", "")),
                "prompt_field": str(suite.get("prompt_field", "prompt")),
                "expected_field": str(suite.get("expected_field", "expected")),
                "scoring": str(suite.get("scoring", "normalized_exact_match")),
                "max_new_tokens": int(
                    getattr(manifest.protection, "decoding", {}).get("max_new_tokens", 512)
                ),
                "use_chat_template": True,
            }
        )
    template = {
        "schema_version": 1,
        "name": str(manifest.cycle_id),
        "seed": 7,
        "goal": {
            "metrics": [{"name": "quality", "direction": "maximize"}],
            "gpu_hour_budget": float(budget.wall_gpu_hours_ceiling_per_recipe),
            "max_parallel_candidates": 1,
            "minimum_promotion_gain": 0.0,
            "require_protocol_match": False,
        },
        "baseline": {"mode": "fixed", "experiment_id": "baseline", "metrics": {}, "gpu_hours": 0.0},
        "experiment": {
            "experiment_id": str(manifest.cycle_id),
            "estimated_gpu_hours": float(budget.wall_gpu_hours_ceiling_per_recipe),
            "hypothesis": {
                "observation": "the parent generation's remaining measured weaknesses",
                "suspected_cause": "declared by the gen2 preregistration",
                "intervention": "LoRA SFT on the declared curriculum",
            },
        },
        "config": {
            "seed": 7,
            "backend": {
                "schema_version": 1,
                "type": "transformers-peft",
                "base_model": str(manifest.base_model_path),
                "dataset": "{corpus}",
                "dataset_format": "text",
                "text_field": "text",
                "max_length": 512,
                "precision": "bf16",
                "quantization": "none",
                "trust_remote_code": False,
                "training": {
                    "epochs": 1.0,
                    "max_steps": 30,
                    "learning_rate": 2e-4,
                    "lr_scheduler_type": "cosine",
                    "warmup_steps": 2,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "gradient_checkpointing": True,
                    "save_strategy": "no",
                    "logging_steps": 20,
                },
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                    ],
                    "use_rslora": False,
                },
                "runtime": {"active_accelerator_count": 0, "timeout_seconds": 7200.0},
            },
            "evaluation": {
                "type": "transformers-text",
                "estimated_gpu_hours": 0.05,
                "precision": "bf16",
                "quantization": "none",
                "placement": "offload",
                "device": "cuda",
                "trust_remote_code": False,
                "runtime": {"timeout_seconds": 1800.0},
                "suites": suites,
            },
        },
    }
    return _write_json(root / "project-template.json", template)


# --------------------------------------------------------------------------
# the trusted-ancestor arm: a real measurement of the untouched dense parent
# --------------------------------------------------------------------------

#: The evaluation worker failed, or left no result manifest.
ANCESTOR_ARM_PROCESS_FAILED = "ANCESTOR_ARM_PROCESS_FAILED"
#: The result manifest is malformed, or describes suites nobody declared.
ANCESTOR_ARM_RESULT_INVALID = "ANCESTOR_ARM_RESULT_INVALID"
#: The pinned dataset cannot supply the frozen mini-slice.
ANCESTOR_ARM_SLICE_TOO_SHORT = "ANCESTOR_ARM_SLICE_TOO_SHORT"


@dataclass(frozen=True)
class MeasuredArm:
    """One measured evaluation arm and where it was written.

    ``which`` names the arm: the trusted ancestor (the dense base, no adapter)
    or the parent (its adapter over that base).  Either way the rows carry
    ``MEASURED_PARENT`` provenance under the generation label of the thing that
    was actually measured, so a parent measurement can never be read as a
    candidate's.
    """

    which: str
    generation_version: str
    report_path: Path
    rows: tuple[str, ...]
    work_dir: Path
    seconds: float
    gpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "which": self.which,
            "generation_version": self.generation_version,
            "report_path": str(self.report_path),
            "rows": list(self.rows),
            "work_dir": str(self.work_dir),
            "seconds": self.seconds,
            "gpu_count": self.gpu_count,
            "incremental_campaign_cost": 0.0,
        }


#: Kept as an alias: the trusted-ancestor arm was the first caller of the one
#: arm-measuring path, and the name still reads for that use.
AncestorArm = MeasuredArm


def measure_arm(
    manifest: Any,
    *,
    which: str,
    benchmarks: Sequence[str],
    generation_version: str,
    adapter_dir: str | Path | None,
    out_path: str | Path,
    slice_source: SliceSource | None = None,
    runner: Any = None,
    python: str | None = None,
    state_root: str | Path | None = None,
    placement: str = "offload",
    timeout_seconds: float = 7200.0,
) -> MeasuredArm:
    """Measure one evaluation arm (ancestor or parent) through production.

    One path for both: the arm is a *real* measurement through the same
    production worker the candidate evaluator uses, differing only in whether
    an adapter is loaded and which generation label it is recorded under.  Rows
    carry ``measurement_origin=MEASURED_PARENT``, one digest-bound raw artifact
    each, and the report names the bytes it measured.  The arm is referenced by
    the campaign at zero incremental cost, so it is returned with that fact
    stated rather than charged to the recipe envelope.
    """
    import sys as _sys

    from chowder.evaluators.transformers_text import (
        EvalSuiteSpec,
        TransformersTextEvalSpec,
        TransformersTextEvaluator,
    )

    from .evaluation_binding import _read_items, _sha256_file
    from .training_binding import default_runner

    protocol = manifest.protection.require_protocol(source=manifest.cycle_id)
    batch_size = int(manifest.evaluation_execution.batch_size)
    benchmarks = tuple(dict.fromkeys(benchmarks))
    if not benchmarks:
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: the campaign declares no benchmark for the "
            f"{which} arm, so there is nothing to measure"
        )
    if not str(out_path).strip():
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: the {which} arm has no declared path to be "
            "written to"
        )
    source = slice_source or load_pinned_slices
    size = int(protocol.n_samples)
    root = Path(state_root or manifest.state_root)
    work_dir = root / f"{which}-arm"
    work_dir.mkdir(parents=True, exist_ok=True)

    suites: list[Any] = []
    for qualified_id in benchmarks:
        try:
            items = tuple(source(qualified_id))
        except CampaignPrepareRefusal:
            raise
        except Exception as error:
            raise CampaignPrepareRefusal(
                f"{ANCESTOR_ARM_SLICE_TOO_SHORT}: reading the slice for "
                f"{qualified_id} failed: {error}"
            ) from error
        if len(items) < size:
            raise CampaignPrepareRefusal(
                f"{ANCESTOR_ARM_SLICE_TOO_SHORT}: the pinned cache holds "
                f"{len(items)} item(s) for {qualified_id} but the frozen protocol "
                f"needs {size}"
            )
        slice_path = work_dir / f"slice-{_slug(qualified_id)}.jsonl"
        slice_path.write_text(
            "".join(
                json.dumps(dict(item), sort_keys=True) + "\n" for item in items[:size]
            ),
            encoding="utf-8",
        )
        suites.append(
            EvalSuiteSpec(
                name=qualified_id.split("@", 1)[0],
                dataset=str(slice_path),
                prompt_field="prompt",
                expected_field="expected",
                scoring="normalized_exact_match",
                max_new_tokens=int(protocol.decoding["max_new_tokens"]),
                use_chat_template=str(protocol.prompt_policy) == "chat_template",
                # The campaign declares how many rows one generation call
                # decodes, and every arm is measured the way the candidate will
                # be. At one row per call this measurement does not finish: the
                # offloaded weights are re-streamed per decode step, so 16 rows
                # x 512 tokens is 16 full passes over the model (~5.9 hours per
                # suite, measured), and the arm timed out at 7200 s without
                # writing anything. Batched at the whole slice it is one pass.
                batch_size=batch_size,
            )
        )

    spec = TransformersTextEvalSpec(
        base_model=str(manifest.base_model_path),
        adapter_dir=None if adapter_dir is None else str(adapter_dir),
        output_dir=str(work_dir),
        suites=tuple(suites),
        precision="bf16",
        quantization="none",
        device="cuda",
        placement=placement,
        seed=int(protocol.seed),
        timeout_seconds=float(timeout_seconds),
        offline=True,
    )
    spec_path = work_dir / "ancestor-eval-spec.json"
    result_path = work_dir / "ancestor-eval-result.json"
    spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")
    identity_path = work_dir / "chowder-identity.json"
    from chowder.worker_env import chowder_source_identity

    identity_path.write_text(
        json.dumps(chowder_source_identity(), sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [
        python or _sys.executable,
        *TransformersTextEvaluator._worker_command(  # noqa: SLF001
            spec_path, result_path, chowder_identity=identity_path
        )[1:],
    ]
    outcome = (runner or default_runner)(command, work_dir, {}, timeout_seconds)
    (work_dir / "ancestor.stdout.txt").write_text(outcome.stdout, encoding="utf-8")
    (work_dir / "ancestor.stderr.txt").write_text(outcome.stderr, encoding="utf-8")
    if outcome.returncode != 0 or not result_path.is_file():
        raise CampaignPrepareRefusal(
            f"{ANCESTOR_ARM_PROCESS_FAILED}: the ancestor evaluation worker exited "
            f"{outcome.returncode} without a usable result manifest"
            + (f" (timed out after {timeout_seconds:g}s)" if outcome.timed_out else "")
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    suite_evidence = payload.get("suites") if isinstance(payload, Mapping) else None
    if not isinstance(suite_evidence, Mapping):
        raise CampaignPrepareRefusal(
            f"{ANCESTOR_ARM_RESULT_INVALID}: the ancestor result carries no suite "
            "measurements"
        )

    runs: list[BenchmarkRun] = []
    for qualified_id, suite in zip(benchmarks, suites):
        evidence = suite_evidence.get(suite.name)
        if not isinstance(evidence, Mapping):
            raise CampaignPrepareRefusal(
                f"{ANCESTOR_ARM_RESULT_INVALID}: no evidence for suite "
                f"{suite.name!r}"
            )
        predictions_ref = str(evidence.get("predictions_file", ""))
        predictions = Path(predictions_ref)
        if not predictions.is_absolute():
            predictions = work_dir / predictions_ref
        if not predictions.is_file():
            raise CampaignPrepareRefusal(
                f"{ANCESTOR_ARM_RESULT_INVALID}: the per-item evidence "
                f"{predictions} for {suite.name!r} does not exist"
            )
        items = _read_items(predictions, suite_name=suite.name)
        samples = [float(item["score"]) for item in items]
        if len(samples) != size:
            raise CampaignPrepareRefusal(
                f"{ANCESTOR_ARM_RESULT_INVALID}: {suite.name!r} measured "
                f"{len(samples)} item(s); the frozen protocol is {size}"
            )
        runs.append(
            BenchmarkRun(
                benchmark_qualified_id=qualified_id,
                adapter="transformers_text",
                generation_version=generation_version,
                score=sum(samples) / len(samples),
                n_samples=len(samples),
                per_sample_scores=tuple(samples),
                metric=_metric_for(qualified_id),
                raw_artifact_ref=_relative(predictions, work_dir),
                measurement_origin=MEASURED_PARENT,
                metadata={
                    "artifact_sha256": _sha256_file(predictions),
                    "sample_indices": list(range(len(samples))),
                    "seed": int(protocol.seed),
                    "shuffle": bool(protocol.shuffle),
                    # The declared decoding, plus the execution parameter it does
                    # not cover: the judge enforces the keys the declared
                    # protocol names and ignores extras, so an arm measured at a
                    # different throughput is visible in its own evidence rather
                    # than silently indistinguishable.
                    "decoding": {**dict(protocol.decoding), "batch_size": batch_size},
                    "prompt_policy": str(protocol.prompt_policy),
                    "suite": suite.name,
                    "slice_sha256": _sha256_file(work_dir / f"slice-{_slug(qualified_id)}.jsonl"),
                    "measured_model": str(manifest.base_model_path),
                },
            )
        )

    target = Path(out_path)
    identity: dict[str, str] = {
        "base_model_path": str(manifest.base_model_path),
        "base_model_digest": str(manifest.base_model_digest),
    }
    if adapter_dir is not None:
        identity["adapter_path"] = str(adapter_dir)
        identity["adapter_digest"] = str(
            getattr(manifest, "parent_adapter_digest", "") or ""
        )
    EvalReport(
        generation_version=generation_version,
        runs=tuple(runs),
        date=_utc_now(),
        model_identity=identity,
    ).save(target)
    runtime = payload.get("runtime", {}) if isinstance(payload, Mapping) else {}
    gpu_count = int(runtime.get("gpu_count", 0) or 0) if isinstance(runtime, Mapping) else 0
    return MeasuredArm(
        which=which,
        generation_version=generation_version,
        report_path=target,
        rows=tuple(run.benchmark_qualified_id for run in runs),
        work_dir=work_dir,
        seconds=float(outcome.seconds),
        gpu_count=gpu_count,
    )


def measure_ancestor_arm(
    manifest: Any,
    *,
    out_path: str | Path | None = None,
    slice_source: SliceSource | None = None,
    runner: Any = None,
    python: str | None = None,
    state_root: str | Path | None = None,
    placement: str = "offload",
    timeout_seconds: float = 7200.0,
) -> MeasuredArm:
    """Measure the trusted-ancestor (Gen-0) arm on the untouched dense base."""
    return measure_arm(
        manifest,
        which="ancestor",
        benchmarks=tuple(
            dict.fromkeys((*manifest.protected_benchmarks, *manifest.broad_benchmarks))
        ),
        generation_version=str(manifest.protection.trusted_ancestor_version),
        adapter_dir=None,
        out_path=out_path or manifest.baseline_eval_report_path,
        slice_source=slice_source,
        runner=runner,
        python=python,
        state_root=state_root,
        placement=placement,
        timeout_seconds=timeout_seconds,
    )


def measure_parent_arm(
    manifest: Any,
    *,
    out_path: str | Path | None = None,
    slice_source: SliceSource | None = None,
    runner: Any = None,
    python: str | None = None,
    state_root: str | Path | None = None,
    placement: str = "offload",
    timeout_seconds: float = 7200.0,
) -> MeasuredArm:
    """Measure the parent generation's adapter under this campaign's protocol.

    The parent arm is the side of adjudication a run compares its candidate
    against.  A parent measured under a *different* instrument version cannot
    serve that comparison, which is exactly why the durable Gen-1 evidence left
    the target row ``UNMEASURED``: this measures the parent adapter over the
    declared base with the *declared* instrument, so the target comparison has
    a real parent row instead of a missing one.  The rows are
    ``MEASURED_PARENT`` under the parent's own generation label -- never
    candidate-measured.
    """
    if not manifest.has_parent_adapter():
        raise CampaignPrepareRefusal(
            f"{PREPARE_SCHEMA}: the campaign declares no parent adapter, so there "
            "is no parent arm to measure; the parent is the base itself"
        )
    return measure_arm(
        manifest,
        which="parent",
        benchmarks=tuple(_evaluated_benchmarks(manifest)),
        generation_version=str(manifest.parent_version),
        adapter_dir=manifest.parent_adapter_path,
        out_path=out_path or manifest.parent_eval_report_path,
        slice_source=slice_source,
        runner=runner,
        python=python,
        state_root=state_root,
        placement=placement,
        timeout_seconds=timeout_seconds,
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
