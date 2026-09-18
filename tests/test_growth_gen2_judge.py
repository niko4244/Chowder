"""The frozen gen2 judge, tested before the run it will judge.

A judge that always refuses is as useless as one that always certifies, so
every gate below is pinned in both directions with synthetic fixtures, plus
the epistemics rule inherited from the re-adjudication: undeclared provenance,
missing protocol metadata and unaligned prompt identities are UNKNOWN, never a
silent pass.

The judge must also not drift from the production engine it claims to read:
the settlement verdict is the production :func:`settle_campaign` answer, the
artifact digest is the production :func:`directory_digest`, and the frozen
instrument prompt list is cross-checked against the gen1 driver's source so
the two cannot diverge.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Mapping

import pytest

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.campaign import CampaignManifest, settle_campaign
from chowder.growth.compute_cost import ComputeCost
from chowder.growth.statistics import compare
from chowder.growth.training_binding import directory_digest

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
REPO = GEN2.parent.parent
_spec = importlib.util.spec_from_file_location("judge_gen2", GEN2 / "judge_gen2.py")
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

INSTRUMENT = judge_gen2.INSTRUMENT_ID
#: The frozen declaration itself, kept aside: fixtures re-pin its contamination
#: evidence per run root, and one test points the module global at a missing file.
FROZEN_CAMPAIGN_MANIFEST = judge_gen2.CAMPAIGN_MANIFEST
MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
DECODING = dict(judge_gen2.PROTECTED_DECODING)


# --------------------------------------------------------------------------
# instrument fixtures
# --------------------------------------------------------------------------


def _completion(*, duplicate: bool = False, echo: bool = False, answer: str = "ping") -> str:
    head = " with exactly: ping\nassistant\n" if echo else ""
    think_body = f"I should answer directly.\n{answer}\n" if duplicate else "I should answer directly.\n"
    return f"{head}<think>\n{think_body}</think>\n\n{answer}"


def _prompt_entries(*, dup: int, echo: int, correct: bool = True) -> list[dict]:
    entries = []
    for index, (prompt, expected) in enumerate(judge_gen2.INSTRUMENT_PROMPTS):
        entries.append(
            {
                "prompt_id": f"gen1-diag-{index:02d}",
                "prompt": prompt,
                "expected": expected,
                "completion": _completion(
                    duplicate=index < dup,
                    echo=index < echo,
                    answer=expected if correct else "definitely-wrong",
                ),
            }
        )
    return entries


def _instrument_run(
    version: str,
    origin: str,
    *,
    dup: int = 0,
    echo: int = 0,
    correct: bool = True,
    metadata_override: dict | None = None,
) -> BenchmarkRun:
    entries = _prompt_entries(dup=dup, echo=echo, correct=correct)
    metadata = {
        "per_prompt": entries,
        "eos_termination_rate": 1.0,
        "max_token_cap_rate": 0.0,
        "unclosed_think_rate": 0.0,
        "obvious_loop_count": 0,
        "distinct_trigram_ratio_mean": 0.97,
    }
    if metadata_override:
        metadata.update(metadata_override)
    return BenchmarkRun(
        benchmark_qualified_id=INSTRUMENT,
        adapter="chowder_custom",
        generation_version=version,
        score=0.0,
        n_samples=len(entries),
        metric="response_surface_compliance",
        measurement_origin=origin,
        raw_artifact_ref=f"raw/{version}-instrument.json",
        metadata=metadata,
    )


#: Sentinel: compute the artifact digest over the canonical payload; ``None``
#: omits the field entirely, and a string declares whatever it says.
_AUTO = object()


def _slice_ref(version: str, qualified_id: str) -> str:
    """The canonical relative artifact path for one arm's slice measurement."""
    return f"raw/{version}-{qualified_id.replace('@', '-')}-slice.json"


def _instrument_ref(version: str) -> str:
    return f"raw/{version}-instrument.json"


def _canonical_ref(run: BenchmarkRun) -> str:
    """What ``run.raw_artifact_ref`` would be if nothing had overridden it."""
    if run.benchmark_qualified_id == INSTRUMENT:
        return _instrument_ref(run.generation_version)
    return _slice_ref(run.generation_version, run.benchmark_qualified_id)


def _payload(
    qualified_id: str, version: str, score: float, n_samples: int, samples: tuple[float, ...]
) -> str:
    return json.dumps(
        {
            "benchmark_qualified_id": qualified_id,
            "generation_version": version,
            "score": score,
            "n_samples": n_samples,
            "per_sample_scores": list(samples),
        },
        sort_keys=True,
    )


def _artifact_payload(run: BenchmarkRun) -> str:
    return _payload(
        run.benchmark_qualified_id,
        run.generation_version,
        run.score,
        run.n_samples,
        tuple(run.per_sample_scores),
    )


def _slice_run(
    qualified_id: str,
    version: str,
    origin: str,
    score: float,
    *,
    n_samples: int = 16,
    indices: list[int] | None = None,
    seed: int = 1234,
    shuffle: bool = False,
    decoding: dict | None = None,
    prompt_policy: str = "chat_template",
    artifact_ref: str | None = None,
    artifact_sha256: object = _AUTO,
    per_sample_scores: tuple[float, ...] | None = None,
) -> BenchmarkRun:
    """A protocol-exact slice row, bound to an artifact the fixture materialises.

    ``artifact_ref=None`` uses the canonical relative path (and ``_write_arm``
    writes the payload the digest is computed over); a caller-supplied ref is the
    caller's business, which is how the "names an artifact that is not there"
    case is built. ``artifact_sha256=None`` omits the declared digest.
    """
    samples = tuple([score] * n_samples) if per_sample_scores is None else per_sample_scores
    metadata: dict = {
        "sample_indices": list(range(16)) if indices is None else indices,
        "seed": seed,
        "shuffle": shuffle,
        "decoding": DECODING if decoding is None else decoding,
        "prompt_policy": prompt_policy,
    }
    if artifact_sha256 is _AUTO:
        digest = hashlib.sha256(
            _payload(qualified_id, version, score, n_samples, samples).encode("utf-8")
        ).hexdigest()
        metadata["artifact_sha256"] = digest
    elif artifact_sha256 is not None:
        metadata["artifact_sha256"] = artifact_sha256
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="lm_eval",
        generation_version=version,
        score=score,
        n_samples=n_samples,
        per_sample_scores=samples,
        metric="exact_match",
        measurement_origin=origin,
        raw_artifact_ref=_slice_ref(version, qualified_id) if artifact_ref is None else artifact_ref,
        metadata=metadata,
    )


def _model_identity(role: str, **overrides: str) -> dict:
    """What an arm declares it measured, matching the frozen declarations.

    The judge binds each arm to the bytes its role names: the candidate to the
    selected artifact, the parent to the declared parent adapter, the ancestor to
    the declared dense base (the frozen manifest's own digests).
    """
    frozen = json.loads(FROZEN_CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))
    identity = {
        "candidate": {"adapter_digest": "c" * 64},
        "parent": {"adapter_digest": frozen["parent_adapter_digest"]},
        "ancestor": {"base_model_digest": frozen["base_model_digest"]},
    }[role]
    identity.update(overrides)
    return identity


def _write_arm(
    path: Path,
    version: str,
    origin: str,
    *,
    dup: int = 0,
    echo: int = 0,
    correct: bool = True,
    slices: tuple[BenchmarkRun, ...] = (),
    metadata_override: dict | None = None,
    with_instrument: bool = True,
    identity: Mapping[str, str] | None = None,
) -> Path:
    runs = []
    if with_instrument:
        runs.append(
            _instrument_run(
                version, origin, dup=dup, echo=echo, correct=correct,
                metadata_override=metadata_override,
            )
        )
    runs.extend(slices)
    EvalReport(
        generation_version=version,
        runs=tuple(runs),
        model_identity=dict(identity or {}),
    ).save(path)
    # The measurements those rows name, next to the report: the run root (or the
    # runner, which copies declared inputs in) then holds the bytes each declared
    # digest is computed over. A row whose ref a test deliberately pointed
    # elsewhere is left alone, so "names an artifact that is not there" stays
    # expressible.
    for run in runs:
        reference = str(run.raw_artifact_ref or "")
        if not reference or reference != _canonical_ref(run):
            continue
        artifact = path.parent / reference
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(_artifact_payload(run), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# run-root fixture: three provenance-bound arms + accounting + artifacts
# --------------------------------------------------------------------------


ARTIFACT_FILE = "adapter_model.safetensors"


def _build_artifact(root: Path) -> tuple[Path, str]:
    artifact = root / "candidate-adapter"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / ARTIFACT_FILE).write_text("weights-for-gen2", encoding="utf-8")
    digest, _entries = directory_digest(artifact)
    return artifact, digest


def _accounting(wall: float = 1.1, *, device: float = 0.40, device_measured: bool = False) -> dict:
    return {
        "totals": {
            "incremental": {
                "device_gpu_hours": device,
                "wall_gpu_hours": wall,
                "device_measured": device_measured,
                "source": "total",
            }
        },
        "entries": [
            {"kind": "training", "recipe_id": "gen2-recipe-a"},
            {"kind": "training", "recipe_id": "gen2-recipe-b"},
            {"kind": "evaluation", "recipe_id": "gen2-recipe-b"},
        ],
    }


def _contamination(*, benchmarks: dict | None = None, training_sources: dict | None = None) -> dict:
    return {
        "benchmarks": (
            {INSTRUMENT: {"status": "CLEAN"}, MATH: {"status": "CLEAN"}, MGSM: {"status": "CLEAN"}}
            if benchmarks is None
            else benchmarks
        ),
        "training_sources": (
            {"src-1": {"status": "CLEAN"}} if training_sources is None else training_sources
        ),
    }


def _run_root(
    tmp_path: Path,
    *,
    candidate_dup: int = 0,
    candidate_echo: int = 0,
    candidate_correct: bool = True,
    candidate_instrument_metadata: dict | None = None,
    parent_dup: int = 11,
    parent_echo: int = 7,
    candidate_slices: tuple[BenchmarkRun, ...] | None = None,
    parent_slices: tuple[BenchmarkRun, ...] | None = None,
    ancestor_slices: tuple[BenchmarkRun, ...] | None = None,
    parent_arm: bool = True,
    ancestor_arm: bool = True,
    contamination: dict | None = None,
    accounting: dict | None = None,
    artifact_digest: str | None = None,
    artifact_ref: str | None = None,
    chosen: bool = True,
    contamination_pin: Path | str | None = None,
) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)

    # Selection first: the candidate arm has to name the bytes the campaign
    # selected, so the artifact (and the digest chosen_candidate.json records)
    # exists before the arms are written.
    candidate_digest = ""
    if chosen:
        artifact, digest = _build_artifact(root)
        reference = artifact_ref if artifact_ref is not None else str(artifact)
        candidate_digest = artifact_digest if artifact_digest is not None else digest
        (root / "chosen_candidate.json").write_text(
            json.dumps(
                {
                    "recipe_id": "gen2-recipe-b",
                    "artifact_ref": reference,
                    "artifact_sha256": candidate_digest,
                }
            ),
            encoding="utf-8",
        )

    if candidate_slices is None:
        candidate_slices = (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0625),
        )
    if parent_slices is None:
        parent_slices = (
            _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0),
            _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
        )
    if ancestor_slices is None:
        ancestor_slices = (
            _slice_run(MATH, "gen0", MEASURED_PARENT, 0.0),
            _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.0),
        )

    # The candidate arm must name the artifact the campaign selected: the fixture
    # records a digest in chosen_candidate.json and the candidate arm declares the
    # same one, so the two agree unless a test deliberately breaks one of them.
    candidate_identity = {"adapter_digest": candidate_digest} if candidate_digest else {}
    _write_arm(
        root / "candidate_evaluation.json", "gen2", MEASURED_THIS_GENERATION,
        dup=candidate_dup, echo=candidate_echo, correct=candidate_correct,
        slices=candidate_slices, metadata_override=candidate_instrument_metadata,
        identity=candidate_identity,
    )
    if parent_arm:
        _write_arm(
            root / "parent_evaluation.json", "gen1", MEASURED_PARENT,
            dup=parent_dup, echo=parent_echo, slices=parent_slices,
            identity=_model_identity("parent"),
        )
    if ancestor_arm:
        _write_arm(
            root / "baseline_evaluation.json", "gen0", MEASURED_PARENT,
            slices=ancestor_slices,
            identity=_model_identity("ancestor"),
        )

    (root / "cycle_compute_accounting.json").write_text(
        json.dumps(_accounting() if accounting is None else accounting), encoding="utf-8"
    )
    pinned = root / "gen2_contamination_manifest.json"
    pinned.write_text(
        json.dumps(_contamination() if contamination is None else contamination), encoding="utf-8"
    )
    # The judged contamination evidence is the artifact the campaign pins, so the
    # fixture declares the pin it used -- as the frozen manifest does in production.
    _pin_campaign(tmp_path, root, contamination_pin or pinned)

    return root


# --------------------------------------------------------------------------
# the judged evidence set is the one the campaign pinned
# --------------------------------------------------------------------------


def _pin_campaign(tmp_path: Path, root: Path, contamination: Path | str) -> Path:
    """The frozen manifest, with its contamination pin pointed at this root.

    The judge reads the contamination artifact the *campaign* pins, so a fixture
    that wants a judged root has to declare the pin it used -- exactly as the
    frozen gen2 manifest does in production, where the pin is the state root's own
    ``gen2_contamination_manifest.json``. Every other frozen field is untouched.
    """
    document = json.loads(FROZEN_CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))
    document["contamination_manifest_path"] = str(contamination)
    path = tmp_path / "judge-campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    judge_gen2.CAMPAIGN_MANIFEST = path
    return path


def _judge_output(root: Path) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = judge_gen2.judge(root)
    return code, buffer.getvalue()


# --------------------------------------------------------------------------
# the clean run certifies; the gen1-shaped defect refuses
# --------------------------------------------------------------------------


def test_judge_certifies_a_clean_run(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    assert judge_gen2.judge(root) == 0


def test_the_frozen_policy_and_the_declared_mechanism_agree() -> None:
    """The judge's frozen values and production's declared ones are one set.

    The runner certifies a run with the manifest's declared protection; the judge
    audits with these frozen constants. If the two ever drift, one of them is
    enforcing a policy the other does not know about -- which is what gate T20
    refuses. Checked here on the shipped declaration, not on a fixture.
    """
    from chowder.growth.campaign import CampaignManifest
    from chowder.growth.certification import GEN2_PROTOCOL

    assert judge_gen2.JUDGE_PROTOCOL.to_dict() == GEN2_PROTOCOL.to_dict()

    shipped = CampaignManifest.from_file(REPO / "docs" / "gen2" / "gen2_campaign.json")
    assert shipped.protection.trusted_ancestor_version == judge_gen2.TRUSTED_ANCESTOR_VERSION
    assert shipped.protection.slice_regression_max == pytest.approx(
        judge_gen2.SLICE_REGRESSION_MAX
    )
    assert shipped.protection.require_protocol().to_dict() == judge_gen2.JUDGE_PROTOCOL.to_dict()


def test_the_run_root_must_carry_the_pinned_contamination_evidence(
    tmp_path: Path,
) -> None:
    """The audit bypass: a campaign pins KNOWN_CONTAMINATION and the run root
    happens to hold a CLEAN file with the judge's expected name.

    The pinned document is the judged evidence, so the clean file must not carry
    the gate -- and because the two disagree, the run root is refused outright
    rather than quietly read from the pin.
    """
    root = _run_root(tmp_path, contamination=_contamination())
    pinned = tmp_path / "pinned-contamination.json"
    pinned.write_text(
        json.dumps(
            _contamination(
                benchmarks={
                    INSTRUMENT: {"status": "KNOWN_CONTAMINATION"},
                    MATH: {"status": "CLEAN"},
                    MGSM: {"status": "CLEAN"},
                }
            )
        ),
        encoding="utf-8",
    )
    _pin_campaign(tmp_path, root, pinned)

    verdict, output = _judge_output(root)
    assert verdict == 1, f"a CLEAN run-root file certified a pinned KNOWN pin:\n{output}"
    assert judge_gen2.CONTAMINATION_EVIDENCE_NOT_PINNED in output
    assert "TAINTED" in output


def test_a_pin_and_a_run_root_copy_that_agree_certify(tmp_path: Path) -> None:
    """The control: pinning the run root's own CLEAN evidence still certifies."""
    root = _run_root(tmp_path)

    verdict, output = _judge_output(root)
    assert verdict == 0, output
    assert "T18" in output and "byte-identical to the pin" in output


def test_a_pin_that_is_missing_refuses_rather_than_falling_back(
    tmp_path: Path,
) -> None:
    """A pin that is not there is UNKNOWN -- never a run-root file instead."""
    root = _run_root(tmp_path)
    _pin_campaign(tmp_path, root, tmp_path / "never-written.json")

    verdict, output = _judge_output(root)
    assert verdict == 1
    assert judge_gen2.CONTAMINATION_PIN_MISSING in output
    assert "INCONCLUSIVE" in output


def test_a_campaign_that_pins_nothing_refuses(tmp_path: Path) -> None:
    """No declared pin means no judged contamination evidence: no fallback."""
    root = _run_root(tmp_path)
    _pin_campaign(tmp_path, root, "")

    verdict, output = _judge_output(root)
    assert verdict == 1
    assert judge_gen2.CONTAMINATION_PIN_ABSENT in output


def test_a_pinned_known_contamination_refuses_even_when_carried_verbatim(
    tmp_path: Path,
) -> None:
    """A pin the run copied in faithfully is still TAINTED, not certified."""
    root = _run_root(
        tmp_path,
        contamination=_contamination(
            benchmarks={
                INSTRUMENT: {"status": "CLEAN"},
                MATH: {"status": "POSSIBLE"},
                MGSM: {"status": "CLEAN"},
            }
        ),
    )

    verdict, output = _judge_output(root)
    assert verdict == 1
    assert "TAINTED" in output


# --------------------------------------------------------------------------
# every protected measurement is bound to bytes that exist
# --------------------------------------------------------------------------


def _both_slices(math_slice: BenchmarkRun) -> tuple[BenchmarkRun, ...]:
    return (math_slice, _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0))


@pytest.mark.parametrize(
    "slices, reason",
    (
        (
            _both_slices(
                _slice_run(
                    MATH, "gen2", MEASURED_THIS_GENERATION, 0.0,
                    artifact_ref="raw/never-written.json",
                )
            ),
            judge_gen2.MEASUREMENT_ARTIFACT_MISSING,
        ),
        (
            _both_slices(
                _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, artifact_sha256=None)
            ),
            judge_gen2.MEASUREMENT_DIGEST_ABSENT,
        ),
        (
            _both_slices(
                _slice_run(
                    MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, artifact_sha256="a" * 64
                )
            ),
            judge_gen2.MEASUREMENT_DIGEST_MISMATCH,
        ),
        (
            _both_slices(
                _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, per_sample_scores=())
            ),
            judge_gen2.MEASUREMENT_SAMPLES_INCONSISTENT,
        ),
        (
            _both_slices(
                _slice_run(
                    MATH, "gen2", MEASURED_THIS_GENERATION, 0.25,
                    per_sample_scores=tuple([0.0] * 16),
                )
            ),
            judge_gen2.MEASUREMENT_SAMPLES_INCONSISTENT,
        ),
        (
            _both_slices(
                _slice_run(
                    MATH, "gen2", MEASURED_THIS_GENERATION, 0.0,
                    artifact_ref="../escaped.json",
                )
            ),
            judge_gen2.MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT,
        ),
    ),
)
def test_unverifiable_measurements_refuse(
    tmp_path: Path, slices: tuple[BenchmarkRun, ...], reason: str
) -> None:
    """No fallback: an unpinned, unhashed or unsupported measurement cannot certify."""
    root = _run_root(tmp_path, candidate_slices=slices)

    verdict, output = _judge_output(root)
    assert verdict == 1, f"the judge certified an unverifiable measurement:\n{output}"
    assert reason in output


def test_a_measurement_artifact_mutated_after_recording_refuses(tmp_path: Path) -> None:
    """The digest is recomputed over the bytes, not compared to a formatted string."""
    root = _run_root(tmp_path)
    assert _judge_output(root)[0] == 0

    slice_artifact = root / "raw" / "gen2-math500-2024-04-slice.json"
    slice_artifact.write_text("{\"rewritten\": \"after the digest was recorded\"}", encoding="utf-8")

    verdict, output = _judge_output(root)
    assert verdict == 1
    assert judge_gen2.MEASUREMENT_DIGEST_MISMATCH in output


def test_a_pinned_measurement_with_real_bytes_certifies(tmp_path: Path) -> None:
    """The control: protocol-exact rows bound to real, hashed artifacts pass."""
    root = _run_root(tmp_path)
    slice_artifact = root / "raw" / "gen2-math500-2024-04-slice.json"
    assert slice_artifact.is_file()

    verdict, output = _judge_output(root)
    assert verdict == 0, output
    assert "T11" in output


def test_judge_refuses_gen1_shaped_defects(tmp_path: Path) -> None:
    """The measured gen1 defect rates (dup 0.688, echo 0.438) must refuse."""
    root = _run_root(tmp_path, candidate_dup=11, candidate_echo=7)
    assert judge_gen2.judge(root) == 1


def test_judge_refuses_when_the_candidate_arm_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# contamination coverage is exact and fail-closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benchmarks",
    (
        {},
        {MATH: {"status": "CLEAN"}},
        {MATH: {"status": "CLEAN"}, MGSM: {"status": "CLEAN"}},
        {INSTRUMENT: {"status": "CLEAN"}, MATH: {"status": "CLEAN"}},
        {INSTRUMENT: {"status": "CLEAN"}, MATH: {"reason": "not checked"}},
    ),
)
def test_contamination_coverage_must_be_exact(tmp_path: Path, benchmarks: dict) -> None:
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


def test_contamination_flag_is_a_hard_failure(tmp_path: Path) -> None:
    benchmarks = {
        INSTRUMENT: {"status": "CLEAN"},
        MATH: {"status": "POSSIBLE"},
        MGSM: {"status": "KNOWN_CONTAMINATION"},
    }
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


def test_an_empty_training_source_section_is_not_clean(tmp_path: Path) -> None:
    root = _run_root(tmp_path, contamination=_contamination(training_sources={}))
    assert judge_gen2.judge(root) == 1


def test_benchmark_coverage_alone_is_not_enough(tmp_path: Path) -> None:
    """Extra CLEAN rows never compensate for a missing required one."""
    benchmarks = {
        INSTRUMENT: {"status": "CLEAN"},
        MATH: {"status": "CLEAN"},
        "extra@v1": {"status": "CLEAN"},
    }
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# protected mini-battery: exact identity + protocol
# --------------------------------------------------------------------------


def test_both_required_slices_present_and_protocol_exact(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path)) == 0


@pytest.mark.parametrize(
    "slices",
    (
        # math only: mgsm missing
        (_slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),),
        # mgsm only: math missing
        (_slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),),
        # duplicate math rows
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0625),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong sample count
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, n_samples=8, indices=list(range(8))),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong indices (offset window)
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, indices=list(range(1, 17))),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong seed
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, seed=4321),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # shuffled
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, shuffle=True),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong decoding
        (
            _slice_run(
                MATH,
                "gen2",
                MEASURED_THIS_GENERATION,
                0.0,
                decoding={"temperature": 0.7, "do_sample": True, "max_new_tokens": 512},
            ),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong prompt policy
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, prompt_policy="raw_completion"),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # carried candidate origin
        (
            _slice_run(MATH, "gen2", "CARRIED_REFERENCE", 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # a slice that regressed past tolerance against the parent arm
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, -0.2),
        ),
    ),
)
def test_malformed_protected_evidence_refuses(tmp_path: Path, slices: tuple) -> None:
    root = _run_root(tmp_path, candidate_slices=slices)
    assert judge_gen2.judge(root) == 1


def test_an_undeclared_protected_row_cannot_substitute_for_a_required_slice(
    tmp_path: Path,
) -> None:
    slices = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run("mmlu_pro@v2", "gen2", MEASURED_THIS_GENERATION, 0.9),
    )
    assert judge_gen2.judge(_run_root(tmp_path, candidate_slices=slices)) == 1


# --------------------------------------------------------------------------
# parent evidence is its own artifact
# --------------------------------------------------------------------------


def test_the_parent_arm_must_be_independently_measured(tmp_path: Path) -> None:
    """A missing parent arm cannot be replaced by anything the candidate says."""
    assert judge_gen2.judge(_run_root(tmp_path, parent_arm=False)) == 1


def test_a_parent_row_relabelled_as_candidate_evidence_refuses(tmp_path: Path) -> None:
    """Same score, wrong provenance: the candidate arm must not borrow it."""
    slices = (
        _slice_run(MATH, "gen2", MEASURED_PARENT, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    assert judge_gen2.judge(_run_root(tmp_path, candidate_slices=slices)) == 1


def test_the_parent_arm_must_match_the_frozen_protocol_too(tmp_path: Path) -> None:
    parent_slices = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0, seed=4321),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
    )
    assert judge_gen2.judge(_run_root(tmp_path, parent_slices=parent_slices)) == 1


# --------------------------------------------------------------------------
# trusted-ancestor protection
# --------------------------------------------------------------------------


def test_gen2_cannot_promote_by_inheriting_a_gen1_regression(tmp_path: Path) -> None:
    """Case A: gen0 good, gen1 regressed badly, gen2 matches gen1."""
    ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    parent = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
    )
    candidate = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    root = _run_root(
        tmp_path, ancestor_slices=ancestor, parent_slices=parent, candidate_slices=candidate
    )
    assert judge_gen2.judge(root) == 1


def test_gen2_promotes_when_the_whole_branch_holds(tmp_path: Path) -> None:
    """Case B: gen0 good, gen1 holds, gen2 holds."""
    ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    parent = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.5),
    )
    candidate = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.5),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.5),
    )
    root = _run_root(
        tmp_path, ancestor_slices=ancestor, parent_slices=parent, candidate_slices=candidate
    )
    assert judge_gen2.judge(root) == 0


def test_an_absent_parent_arm_stays_inconclusive_unless_the_ancestor_resolves_it(
    tmp_path: Path,
) -> None:
    """Case C: gen1 unresolved, with and without independent gen0 resolution."""
    good_ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    holding = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.5),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.5),
    )
    # Gen1's protected measurement is unresolved: the arm carries the target
    # instrument (so the paired rule is applicable) but no mini-slice rows.
    # Resolved by the trusted ancestor: promotion stays possible.
    resolved = _run_root(
        tmp_path / "resolved",
        parent_slices=(),
        ancestor_slices=good_ancestor,
        candidate_slices=holding,
    )
    assert judge_gen2.judge(resolved) == 0

    # Not resolved: the candidate regressed against gen0 -> hard refusal.
    regressed = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    unresolved = _run_root(
        tmp_path / "unresolved",
        parent_slices=(),
        ancestor_slices=good_ancestor,
        candidate_slices=regressed,
    )
    assert judge_gen2.judge(unresolved) == 1


def test_a_missing_ancestor_arm_blocks_branch_protection(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path, ancestor_arm=False)) == 1


# --------------------------------------------------------------------------
# settlement is the production settlement
# --------------------------------------------------------------------------


def test_the_judge_and_production_settlement_agree(tmp_path: Path) -> None:
    campaign = CampaignManifest.from_file(judge_gen2.CAMPAIGN_MANIFEST)
    for wall in (0.2, 1.1, 1.4, 2.5):
        total = ComputeCost(
            device_gpu_hours=0.4,
            wall_gpu_hours=wall,
            source="test",
            device_measured=False,
        )
        production = settle_campaign(campaign, total=total)
        root = _run_root(tmp_path / f"wall-{wall}", accounting=_accounting(wall=wall))
        judge_ok = judge_gen2.judge(root) == 0
        assert judge_ok == production.compliant, (
            f"wall {wall}: judge certified={judge_ok}, production compliant="
            f"{production.compliant} ({production.failure_reasons})"
        )


def test_a_wall_overrun_is_a_hard_failure_not_an_unknown(tmp_path: Path) -> None:
    root = _run_root(tmp_path, accounting=_accounting(wall=2.5))
    assert judge_gen2.judge(root) == 1


def test_a_device_settlement_ceiling_cannot_be_satisfied_by_an_unmeasured_device(
    tmp_path: Path,
) -> None:
    """The policy amendment says device is admission-only *because*
    device_time_measured=false. Declaring it true restores the hard gate, and
    an unmeasured device figure must then refuse rather than pass."""
    campaign = CampaignManifest.from_file(judge_gen2.CAMPAIGN_MANIFEST)
    assert campaign.budget.device_time_measured is False
    import dataclasses

    hard = dataclasses.replace(
        campaign, budget=dataclasses.replace(campaign.budget, device_time_measured=True)
    )
    unmeasured = ComputeCost.from_wall_only(0.2, source="wall-only ledger")
    verdict = settle_campaign(hard, total=unmeasured)
    assert verdict.compliant is False
    assert any("ACTUAL_DEVICE_GPU_HOURS_UNMEASURED" in reason for reason in verdict.failure_reasons)

    measured = ComputeCost.measured(device_gpu_hours=0.1, wall_gpu_hours=0.2, source="probe")
    assert settle_campaign(hard, total=measured).compliant is True


def test_losing_recipe_accounting_is_required(tmp_path: Path) -> None:
    accounting = _accounting()
    accounting["entries"] = [{"kind": "training", "recipe_id": "gen2-recipe-a"}]
    assert judge_gen2.judge(_run_root(tmp_path, accounting=accounting)) == 1


def test_a_broken_manifest_campaign_is_inconclusive_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    # Built first (the fixture pins its own declaration), then the declaration is
    # withdrawn: an unreadable campaign is UNKNOWN, never a crash and never a pass.
    root = _run_root(tmp_path)
    monkeypatch.setattr(judge_gen2, "CAMPAIGN_MANIFEST", tmp_path / "absent.json")
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# artifact identity is recomputed
# --------------------------------------------------------------------------


def test_artifact_digest_is_recomputed(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path)) == 0


def test_a_fabricated_digest_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path, artifact_digest="a" * 64)
    assert judge_gen2.judge(root) == 1


def test_an_artifact_mutated_after_recording_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    chosen = json.loads((root / "chosen_candidate.json").read_text(encoding="utf-8"))
    Path(chosen["artifact_ref"], ARTIFACT_FILE).write_text("tampered", encoding="utf-8")
    assert judge_gen2.judge(root) == 1


def test_a_missing_artifact_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    chosen = json.loads((root / "chosen_candidate.json").read_text(encoding="utf-8"))
    import shutil

    shutil.rmtree(chosen["artifact_ref"])
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# generation identity is part of an arm's identity
# --------------------------------------------------------------------------  # noqa: E501


def test_the_ancestor_arm_must_carry_gen0_rows_not_gen1_rows(tmp_path: Path) -> None:
    """A protocol-correct gen1 row is not the gen0 arm.

    The rows here are measured, protocol-exact and parent-measured -- every
    property except the one that says *which generation produced them*. The
    ancestor arm answers "did the branch regress against the trusted ancestor",
    so a gen1 row standing in for gen0 would silently move the comparison to the
    immediate parent, which is the case branch protection exists to catch.
    """
    root = _run_root(
        tmp_path,
        ancestor_slices=(
            _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0),
            _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
        ),
    )

    code, output = _judge_output(root)

    assert code == 1, f"a gen1 row certified as the gen0 arm:\n{output}"
    assert "generation" in output


def test_an_ancestor_report_labelled_gen1_refuses(tmp_path: Path) -> None:
    """Report-level generation identity is enforced too, not only the rows."""
    root = _run_root(tmp_path)
    assert _judge_output(root)[0] == 0
    arm = root / "baseline_evaluation.json"
    document = json.loads(arm.read_text(encoding="utf-8"))
    document["generation_version"] = "gen1"
    arm.write_text(json.dumps(document), encoding="utf-8")

    code, output = _judge_output(root)

    assert code == 1, f"a mislabelled ancestor report certified:\n{output}"
    assert "generation" in output


# --------------------------------------------------------------------------
# the cycle accounts for exactly the declared recipe set
# --------------------------------------------------------------------------


def test_the_declared_recipe_set_must_be_accounted_exactly(tmp_path: Path) -> None:
    """T14 compares sets, not counts: a third recipe is as wrong as a missing one."""
    assert _judge_output(_run_root(tmp_path))[0] == 0

    fixture = _accounting()
    declared = [entry for entry in fixture["entries"]]
    assert {entry["recipe_id"] for entry in declared} == {
        "gen2-recipe-a",
        "gen2-recipe-b",
    }

    extra = json.loads(json.dumps(fixture))
    extra["entries"] = [
        *declared,
        {"kind": "training", "recipe_id": "gen2-recipe-c"},
    ]
    root = _run_root(tmp_path / "extra", accounting=extra)
    code, output = _judge_output(root)
    assert code == 1, f"an undeclared recipe certified:\n{output}"
    assert "T14" in output
    assert "gen2-recipe-c" in output

    renamed = json.loads(json.dumps(fixture))
    renamed["entries"] = [
        {"kind": "training", "recipe_id": "gen2-recipe-a"},
        {"kind": "training", "recipe_id": "something-else"},
    ]
    root = _run_root(tmp_path / "renamed", accounting=renamed)
    code, output = _judge_output(root)
    assert code == 1, f"a renamed recipe certified:\n{output}"
    assert "gen2-recipe-b" in output


# --------------------------------------------------------------------------
# the frozen paired/absolute target rule
# --------------------------------------------------------------------------


def test_paired_improvement_passes_the_frozen_rule() -> None:
    status, detail = judge_gen2._target_gate(
        [1.0] * 11 + [0.0] * 5,
        [0.0] * 16,
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.PASS
    assert "paired improvement" in detail


def test_aggregate_threshold_alone_does_not_pass_the_frozen_rule() -> None:
    """Rate at the threshold, no paired signal, too few strict improvements."""
    status, detail = judge_gen2._target_gate(
        [1.0, 1.0] + [0.0] * 14,
        [0.0] * 14 + [1.0, 1.0],
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.FAIL
    assert "absolute met" in detail
    assert "strictly better on 2/16 (need 12)" in detail


def test_a_tie_fails() -> None:
    status, _detail = judge_gen2._target_gate(
        [1.0] * 5 + [0.0] * 11,
        [1.0] * 5 + [0.0] * 11,
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.FAIL


def test_the_absolute_threshold_path_can_pass_without_a_paired_signal() -> None:
    """The prereg's second path: absolute threshold *and* the strict count.

    With tiny per-prompt deltas the paired test has no power (its effect is
    below the declared minimum), so the frozen rule's absolute branch is the
    only thing that can pass -- which is exactly why it is written down.
    """
    parent = [0.9, 0.1] * 8
    candidate = [0.8, 0.0] * 8
    assert compare(parent, candidate, min_effect=judge_gen2.TARGET_MIN_EFFECT).verdict == "flat"
    status, detail = judge_gen2._target_gate(
        parent, candidate, lower_is_better=True, absolute_threshold=0.5
    )
    assert status == judge_gen2.PASS
    assert "absolute threshold + strict count" in detail


# --------------------------------------------------------------------------
# prompt identity alignment
# --------------------------------------------------------------------------


def _reorder_and_relabel(arm_path: Path, *, ids: dict | None = None) -> None:
    report = json.loads(arm_path.read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if not per_prompt:
            continue
        if ids is not None:
            for entry in per_prompt:
                entry["prompt_id"] = ids[entry["prompt_id"]]
        run["metadata"]["per_prompt"] = list(reversed(per_prompt))
    arm_path.write_text(json.dumps(report), encoding="utf-8")


def test_reordered_but_identity_equivalent_prompts_still_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _reorder_and_relabel(root / "parent_evaluation.json")
    assert judge_gen2.judge(root) == 0


def test_a_duplicated_prompt_identity_refuses_to_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _reorder_and_relabel(
        root / "parent_evaluation.json",
        ids={f"gen1-diag-{i:02d}": "gen1-diag-00" for i in range(16)},
    )
    assert judge_gen2.judge(root) == 1


def test_a_missing_parent_prompt_refuses_to_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    report = json.loads((root / "parent_evaluation.json").read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if per_prompt:
            run["metadata"]["per_prompt"] = per_prompt[:-1]
    (root / "parent_evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert judge_gen2.judge(root) == 1


def test_a_claimed_aggregate_cannot_override_the_per_prompt_evidence(tmp_path: Path) -> None:
    """The artifact can claim anything; the judge scores the completions."""
    root = _run_root(
        tmp_path,
        candidate_dup=0,
        candidate_echo=0,
        candidate_instrument_metadata={"declared_duplication_rate": 0.0, "declared_echo_rate": 0.0},
    )
    # Now make the completions the gen1-shaped defect while the claims stay clean.
    report = json.loads((root / "candidate_evaluation.json").read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if not per_prompt:
            continue
        for entry in per_prompt:
            entry["completion"] = _completion(
                duplicate=True, echo=True, answer=str(entry.get("expected"))
            )
    (root / "candidate_evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# the frozen instrument list cannot drift from the gen1 driver
# --------------------------------------------------------------------------


def test_the_frozen_instrument_matches_the_gen1_driver_source() -> None:
    """The judge freezes its own copy of the 16 prompts; that copy must equal
    the driver's literals, or the instrument quietly changes."""
    source = (REPO / "docs" / "gen1" / "run_gen1_cycle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    literals[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    prompts = list(literals["DIAG_PROMPTS"])
    expected = list(literals["_DIAG_EXPECTED"])
    assert list(judge_gen2.INSTRUMENT_PROMPTS) == list(zip(prompts, expected))
    assert len(judge_gen2.INSTRUMENT_PROMPTS) == 16
    assert judge_gen2.CONSTRAINED_PROMPTS <= set(prompts)
