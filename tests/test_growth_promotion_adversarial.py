"""The adversarial audit of every promotion and certification path.

Each test flips exactly **one** artifact and asserts the verdict moves. A
mutation matrix is only worth having if it fails when the rule is weakened,
so the assertions that pin a fixed defect say which revert breaks them.

Three layers, in increasing cost:

1. **The rule.** ``evaluate_promotion`` over a clean, promoting input, with one
   provenance field, one contamination verdict, one cost figure or one declared
   set mutated at a time. Cheap, dense, and it covers every comparative gate.
2. **Certification.** ``certify_run_root`` over a run root whose arms carry
   real bytes, with one protocol field, digest, sample or identity mutated at a
   time.
3. **The second opinion.** A whole campaign, then the frozen Gen-2 judge over
   the run root the run wrote -- T21 auditing the recorded decision and T22
   recomputing the declared profile through production's own evaluator. The
   invariant is directional and fail-closed: the judge may never certify a
   candidate the run refused, and it may never disagree with the run about a
   declared gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from chowder.evals.result import (
    CARRIED_REFERENCE,
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    UNMEASURED,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.campaign_runner import run_campaign
from chowder.growth.certification import (
    FAIL,
    PASS,
    UNKNOWN,
    ProtectionPolicy,
    certify_run_root,
)
from chowder.growth.promotion import (
    BenchmarkResult,
    PromotionInput,
    evaluate_promotion,
)

from test_growth_campaign_runner import (
    PROTECTED_ID,
    _campaign,
    _patch_seams,
)

# ==========================================================================
# Layer 1 -- the rule, one artifact at a time
# ==========================================================================

TARGET = "target@1"
PROT = "prot@1"
BROAD = "broad@1"
CAL = "cal@1"
REL = "rel@1"
SPREAD = (0.5, 0.7, 0.6, 0.6, 0.6, 0.5, 0.7, 0.6)


def _result(
    qualified_id: str,
    score: float,
    samples: Sequence[float] = SPREAD,
    *,
    origin: str = MEASURED_THIS_GENERATION,
    contamination: str = "CLEAN",
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_qualified_id=qualified_id,
        score=score,
        samples=tuple(float(value) for value in samples),
        contamination=contamination,
        measurement_origin=origin,
    )


def _promoting_input(**overrides: Any) -> PromotionInput:
    """A complete, clean input that every mutation is applied to.

    The baseline is what matters: every declared set carries an *earned*
    measurement on both sides, so a mutation that turns a gate from "ok" to
    "satisfied anyway" is visible, and a mutation of a row nothing measured is
    distinguishable from a mutation of a row that did.
    """
    base: dict[str, Any] = {
        "candidate_version": "gen2",
        "parent_version": "gen1",
        "target_benchmarks": (TARGET,),
        "candidate_results": {
            TARGET: _result(TARGET, 0.90, (0.8, 0.9, 1.0, 0.9, 0.85, 0.95, 0.9, 0.9)),
            PROT: _result(PROT, 0.60),
            BROAD: _result(BROAD, 0.60),
            CAL: _result(CAL, 0.60),
            REL: _result(REL, 0.60),
        },
        "parent_results": {
            TARGET: _result(
                TARGET, 0.20, (0.1, 0.3, 0.2, 0.2, 0.25, 0.15, 0.2, 0.2),
                origin=MEASURED_PARENT,
            ),
            PROT: _result(PROT, 0.60, origin=MEASURED_PARENT),
            BROAD: _result(BROAD, 0.60, origin=MEASURED_PARENT),
            CAL: _result(CAL, 0.60, origin=MEASURED_PARENT),
            REL: _result(REL, 0.60, origin=MEASURED_PARENT),
        },
        "protected_benchmarks": (PROT,),
        "broad_battery_benchmarks": (BROAD,),
        "calibration_benchmarks": (CAL,),
        "reliability_benchmarks": (REL,),
        "device_gpu_hours": 0.10,
        "device_gpu_hours_ceiling": 1.0,
        "actual_wall_gpu_hours": 0.20,
        "wall_gpu_hours_ceiling": 1.0,
    }
    base.update(overrides)
    return PromotionInput(**base)


def _arm_files() -> dict[str, Path]:
    return {
        "candidate": Path("candidate_evaluation.json"),
        "parent": Path("parent_evaluation.json"),
        "ancestor": Path("baseline_evaluation.json"),
    }




def test_the_unmutated_baseline_actually_promotes() -> None:
    """The matrix is only meaningful if its clean case promotes."""
    assert evaluate_promotion(_promoting_input()).verdict == "PROMOTED"


def _poison_parent(qualified_id: str, origin: str, score: float = 0.0):
    """One parent row relabeled or repinned; everything else untouched."""

    def mutate(data: PromotionInput) -> PromotionInput:
        earned = data.parent_results[qualified_id]
        poisoned = replace(
            earned,
            measurement_origin=origin,
            score=score,
            samples=tuple([score] * len(earned.samples)),
        )
        return replace(
            data, parent_results={**data.parent_results, qualified_id: poisoned}
        )

    return mutate


def _poison_candidate(qualified_id: str, origin: str):
    def mutate(data: PromotionInput) -> PromotionInput:
        earned = data.candidate_results[qualified_id]
        poisoned = replace(earned, measurement_origin=origin)
        return replace(
            data, candidate_results={**data.candidate_results, qualified_id: poisoned}
        )

    return mutate


#: (benchmark, the comparative check the rule reads for it). Every one of
#: these is ``candidate - parent``, so every one is inflated by a low,
#: unearned baseline.
PARENT_SIDED_GATES = (
    (PROT, "protected_regression"),
    (BROAD, "broad_battery"),
    (CAL, "calibration"),
    (REL, "reliability"),
)

UNSOLD_ORIGINS = (UNMEASURED, CARRIED_REFERENCE)


@pytest.mark.parametrize("unearned_origin", UNSOLD_ORIGINS)
@pytest.mark.parametrize(("qualified_id", "check"), PARENT_SIDED_GATES)
def test_an_unearned_parent_row_never_satisfies_a_comparative_gate(
    qualified_id: str, check: str, unearned_origin: str
) -> None:
    """The audit's headline mutation, and a real defect this work fixed.

    A parent row pinned at 0.0 that nothing measured made *every* comparative
    gate read "ok" and the campaign PROMOTED, because the parent enters each
    pass condition with a plus sign. The binder's docstring called this safe --
    legacy parent rows "cannot inflate a candidate's gates" -- which is exactly
    backwards for a difference.

    Verified against the pre-fix rule: neutering ``_baseline`` makes the first
    assertion fail with ``'ok' == 'inconclusive'``.
    """
    decision = evaluate_promotion(
        _poison_parent(qualified_id, unearned_origin)(_promoting_input())
    )
    assert decision.checks[check] != "ok", (qualified_id, check, decision.checks)
    assert decision.verdict != "PROMOTED", decision.verdict


@pytest.mark.parametrize("unearned_origin", UNSOLD_ORIGINS)
def test_an_unearned_target_baseline_never_manufactures_improvement(
    unearned_origin: str,
) -> None:
    """The target gate is the one that *needs* a baseline to mean anything.

    Pinned at 0.0 against a candidate at 0.9 it is the best delta available in
    the campaign, produced by a number nothing measured.
    """
    decision = evaluate_promotion(_poison_parent(TARGET, unearned_origin)(_promoting_input()))
    assert decision.checks["target_improvement"] != "met", decision.checks
    assert decision.verdict != "PROMOTED", decision.verdict


def test_the_parent_arms_own_measurement_is_still_an_earned_baseline() -> None:
    """The other side of the same rule, so the wall is a wall and not a ban.

    ``parent_measured`` admits the parent arm's own measurement precisely
    because a campaign whose baseline was never measured could not adjudicate
    anything. What it refuses is a quotation from history and a row that
    predates provenance.
    """
    decision = evaluate_promotion(_poison_parent(TARGET, MEASURED_PARENT)(_promoting_input()))
    assert decision.checks["target_improvement"] == "met", decision.checks
    assert decision.verdict == "PROMOTED", decision.verdict


@pytest.mark.parametrize("unearned_origin", [*UNSOLD_ORIGINS, MEASURED_PARENT])
@pytest.mark.parametrize(("qualified_id", "_check"), PARENT_SIDED_GATES)
def test_an_unearned_candidate_row_never_satisfies_a_comparative_gate(
    qualified_id: str, _check: str, unearned_origin: str
) -> None:
    """The candidate wall, still intact: provenance is read on both sides now."""
    decision = evaluate_promotion(
        _poison_candidate(qualified_id, unearned_origin)(_promoting_input())
    )
    assert decision.verdict != "PROMOTED", decision.verdict


def _drop_candidate(qualified_id: str):
    def mutate(data: PromotionInput) -> PromotionInput:
        kept = {k: v for k, v in data.candidate_results.items() if k != qualified_id}
        return replace(data, candidate_results=kept)

    return mutate


@pytest.mark.parametrize("qualified_id", [PROT, BROAD, CAL, REL])
def test_a_declared_gate_cannot_be_dodged_by_never_running_the_eval(
    qualified_id: str,
) -> None:
    """Predeclared sets are closed: a declared set with no result is not a pass."""
    decision = evaluate_promotion(_drop_candidate(qualified_id)(_promoting_input()))
    assert decision.verdict != "PROMOTED", decision.verdict


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("UNKNOWN", "INCONCLUSIVE"),
        ("POSSIBLE", "TAINTED"),
        ("KNOWN_CONTAMINATION", "TAINTED"),
    ],
)
def test_contamination_is_the_firewalls_verdict_not_the_campaigns(
    verdict: str, expected: str
) -> None:
    """An unchecked row is inconclusive, not clean; a flagged one taints."""
    data = _promoting_input()
    earned = data.candidate_results[PROT]
    poisoned = replace(earned, contamination=verdict)
    decision = evaluate_promotion(
        replace(data, candidate_results={**data.candidate_results, PROT: poisoned})
    )
    assert decision.verdict == expected, (verdict, decision.verdict, decision.checks)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_gpu_hours", 99.0),
        ("actual_wall_gpu_hours", 99.0),
        ("actual_device_gpu_hours", 99.0),
    ],
)
def test_every_declared_cost_unit_is_a_hard_gate(field: str, value: float) -> None:
    """Wall, projected device and settled device each veto on their own."""
    decision = evaluate_promotion(_promoting_input(**{field: value}))
    assert decision.checks["resource_envelope"] == "violated", (field, decision.checks)
    assert decision.verdict == "REJECTED", decision.verdict


def test_the_settled_device_check_is_reachable_from_the_production_path() -> None:
    """The *reachability* half of the second finding.

    ``actual_device_gpu_hours`` was declared, checked against the device
    ceiling and accepted by no production caller: ``decide_promotion_from_runs``
    neither took the parameter nor forwarded it, so a device overrun visible
    only at settlement would have passed the promotion rule. This asserts the
    parameter survives the hop -- the arithmetic above only proves the check
    works once something can reach it.
    """
    import inspect

    from chowder.growth.cycle import GrowthCycle

    signature = inspect.signature(GrowthCycle.decide_promotion_from_runs)
    assert "actual_device_gpu_hours" in signature.parameters
    source = inspect.getsource(GrowthCycle.decide_promotion_from_runs)
    assert "actual_device_gpu_hours=actual_device_gpu_hours" in source

    from chowder.growth import campaign_runner

    assert "actual_device_gpu_hours=" in inspect.getsource(campaign_runner._adjudicate)
    # And a compliant settled device reading still promotes, so the forwarded
    # value is a gate and not a new way to refuse.
    assert evaluate_promotion(_promoting_input(actual_device_gpu_hours=0.10)).verdict == (
        "PROMOTED"
    )


def test_an_aggregate_with_no_samples_cannot_clear_a_dropped_gate() -> None:
    """The aggregate-only path is a one-sided pass, so its tolerance is real."""
    data = _promoting_input()
    flat = replace(data.candidate_results[PROT], samples=())
    assert evaluate_promotion(
        replace(data, candidate_results={**data.candidate_results, PROT: flat})
    ).verdict == "PROMOTED"
    dropped = replace(data.candidate_results[PROT], score=0.40, samples=())
    assert evaluate_promotion(
        replace(data, candidate_results={**data.candidate_results, PROT: dropped})
    ).verdict == "REJECTED"


# ==========================================================================
# Layer 2 -- certification, one artifact at a time
# ==========================================================================

CANDIDATE_VERSION = "gen2"
ANCESTOR_VERSION = "gen0"
SLICE_SAMPLES = tuple([0.0, 1.0] * 8)
PROTOCOL_METADATA: Mapping[str, Any] = {
    "sample_indices": list(range(16)),
    "seed": 1234,
    "shuffle": False,
    "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512},
    "prompt_policy": "chat_template",
}
ADAPTER = "0" * 64


def _slice_run(
    run_root: Path, arm: str, *, version: str, origin: str, samples: Sequence[float] = SLICE_SAMPLES
) -> BenchmarkRun:
    """One protocol-exact row bound to real bytes written beside its report."""
    relative = f"raw/{arm}-slice.json"
    payload = json.dumps({"arm": arm, "n": len(samples)}, sort_keys=True)
    path = run_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    metadata = {
        **PROTOCOL_METADATA,
        "artifact_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    return BenchmarkRun(
        benchmark_qualified_id=PROTECTED_ID,
        adapter="chowder_custom",
        generation_version=version,
        score=sum(samples) / len(samples),
        n_samples=len(samples),
        per_sample_scores=tuple(float(value) for value in samples),
        metric="accuracy",
        measurement_origin=origin,
        raw_artifact_ref=relative,
        metadata=metadata,
    )


def _certifiable_root(tmp_path: Path) -> tuple[Path, ProtectionPolicy]:
    """A run root whose three arms are real, protocol-exact and earned.

    The control the mutations are measured against: every requirement decides
    PASS before anything is flipped, so a FAIL after a mutation is
    attributable to the flipped artifact alone.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for arm, path in _arm_files().items():
        version = {"candidate": CANDIDATE_VERSION, "parent": "gen1", "ancestor": ANCESTOR_VERSION}[
            arm
        ]
        origin = MEASURED_THIS_GENERATION if arm == "candidate" else MEASURED_PARENT
        EvalReport(
            generation_version=version,
            runs=(_slice_run(root, arm, version=version, origin=origin),),
            model_identity={"adapter_digest": ADAPTER},
        ).save(root / path)
    policy = ProtectionPolicy(
        required_protected=(PROTECTED_ID,),
        slice_regression_max=0.0625,
        trusted_ancestor_version=ANCESTOR_VERSION,
        candidate_version=CANDIDATE_VERSION,
        parent_version="gen1",
    )
    return root, policy


def _certify(root: Path, policy: ProtectionPolicy, **digests: Any):
    arms = {role: root / path for role, path in _arm_files().items()}
    expectations = {role: ADAPTER for role in arms}
    expectations.update(digests)
    return certify_run_root(
        policy=policy,
        run_root=root,
        arms=arms,
        expected_adapter_digests=expectations,
    )


def _rewrite(path: Path, mutate: Callable[[BenchmarkRun], BenchmarkRun]) -> None:
    report = EvalReport.load(path)
    EvalReport(
        generation_version=report.generation_version,
        runs=tuple(mutate(run) for run in report.runs),
        hardware=report.hardware,
        date=report.date,
        model_identity=report.model_identity,
    ).save(path)


def test_the_unmutated_run_root_actually_certifies(tmp_path: Path) -> None:
    root, policy = _certifiable_root(tmp_path)
    assert _certify(root, policy).status == PASS


#: (name, mutate one field of the candidate's row, expected status)
CERTIFICATION_MUTATIONS: tuple[
    tuple[str, Callable[[BenchmarkRun], BenchmarkRun], str], ...
] = (
    (
        "declared digest no longer matches the bytes",
        lambda run: replace(run, metadata={**run.metadata, "artifact_sha256": "1" * 64}),
        FAIL,
    ),
    (
        "no digest at all",
        lambda run: replace(
            run, metadata={k: v for k, v in run.metadata.items() if k != "artifact_sha256"}
        ),
        FAIL,
    ),
    (
        "the digest is not a digest",
        lambda run: replace(run, metadata={**run.metadata, "artifact_sha256": "not-a-digest"}),
        FAIL,
    ),
    (
        "the named artifact is gone",
        lambda run: replace(run, raw_artifact_ref="raw/does-not-exist.json"),
        FAIL,
    ),
    (
        "the artifact ref escapes the run root",
        lambda run: replace(run, raw_artifact_ref="../../elsewhere/slice.json"),
        FAIL,
    ),
    (
        "the seed drifted",
        lambda run: replace(run, metadata={**run.metadata, "seed": 999}),
        FAIL,
    ),
    (
        "the slice was shuffled",
        lambda run: replace(run, metadata={**run.metadata, "shuffle": True}),
        FAIL,
    ),
    (
        "the decoding drifted",
        lambda run: replace(
            run, metadata={**run.metadata, "decoding": {"temperature": 0.7}}
        ),
        FAIL,
    ),
    (
        "the prompt policy drifted",
        lambda run: replace(run, metadata={**run.metadata, "prompt_policy": "raw"}),
        FAIL,
    ),
    (
        "the sample indices are not 0..N-1",
        lambda run: replace(
            run, metadata={**run.metadata, "sample_indices": list(range(1, 17))}
        ),
        FAIL,
    ),
    (
        "the score contradicts its own per-sample values",
        lambda run: replace(run, score=0.99),
        FAIL,
    ),
    (
        "the row carries fewer samples than it declares",
        lambda run: replace(run, n_samples=8, per_sample_scores=run.per_sample_scores[:8]),
        FAIL,
    ),
    (
        "the row claims the parent's provenance",
        lambda run: replace(run, measurement_origin=MEASURED_PARENT),
        FAIL,
    ),
    (
        "the row is a carried reference",
        lambda run: replace(run, measurement_origin=CARRIED_REFERENCE),
        FAIL,
    ),
    (
        "the row claims another generation",
        lambda run: replace(run, generation_version="gen7"),
        FAIL,
    ),
)


@pytest.mark.parametrize(
    ("name", "mutate", "expected"),
    CERTIFICATION_MUTATIONS,
    ids=[case[0] for case in CERTIFICATION_MUTATIONS],
)
def test_one_artifact_flipped_stops_the_candidate_arm_certifying(
    tmp_path: Path,
    name: str,
    mutate: Callable[[BenchmarkRun], BenchmarkRun],
    expected: str,
) -> None:
    root, policy = _certifiable_root(tmp_path)
    _rewrite(root / "candidate_evaluation.json", mutate)
    certification = _certify(root, policy)
    assert certification.status == expected, (name, certification.to_dict())


def test_a_duplicated_required_slice_cannot_certify(tmp_path: Path) -> None:
    """Two rows for one benchmark: which one certified?"""
    root, policy = _certifiable_root(tmp_path)
    path = root / "candidate_evaluation.json"
    report = EvalReport.load(path)
    _rewrite(path, lambda run: run)
    doubled = EvalReport(
        generation_version=report.generation_version,
        runs=(
            *report.runs,
            _slice_run(
                root, "candidate-copy",
                version=CANDIDATE_VERSION, origin=MEASURED_THIS_GENERATION,
            ),
        ),
        model_identity=report.model_identity,
    )
    doubled.save(path)
    assert _certify(root, policy).status == FAIL


def test_an_arm_bound_to_other_bytes_cannot_certify(tmp_path: Path) -> None:
    """Identity: a protocol-perfect row that measured a different artifact."""
    root, policy = _certifiable_root(tmp_path)
    certification = _certify(root, policy, candidate="9" * 64)
    assert certification.status == FAIL, certification.to_dict()
    assert any("MISMATCH" in reason for reason in certification.reasons)


def test_a_candidate_matching_an_already_regressed_parent_still_fails(
    tmp_path: Path,
) -> None:
    """Branch protection is judged against the trusted ancestor.

    gen2 holds exactly gen1's regressed score: the immediate-parent comparison
    sees no change and would pass, so only the ancestor arm can catch it.
    """
    root, policy = _certifiable_root(tmp_path)
    regressed = tuple([0.0] * 16)
    for arm in ("parent", "candidate"):
        _rewrite(
            root / _arm_files()[arm],
            lambda run, s=regressed: replace(run, score=0.0, per_sample_scores=s, n_samples=16),
        )
    _rewrite(
        root / "baseline_evaluation.json",
        lambda run: replace(
            run, score=1.0, per_sample_scores=tuple([1.0] * 16), n_samples=16
        ),
    )
    certification = _certify(root, policy)
    assert certification.status == FAIL, certification.to_dict()
    assert any("trusted-ancestor" in reason for reason in certification.reasons)


@pytest.mark.parametrize("missing", ["parent", "ancestor"])
def test_an_absent_arm_is_never_silently_the_other_arms_numbers(
    tmp_path: Path, missing: str
) -> None:
    """A missing arm is UNKNOWN evidence, and UNKNOWN refuses promotion."""
    root, policy = _certifiable_root(tmp_path)
    (root / _arm_files()[missing]).unlink()
    certification = _certify(root, policy)
    assert certification.status == UNKNOWN, certification.to_dict()


def test_no_declared_protected_set_is_never_a_pass(tmp_path: Path) -> None:
    root, _policy = _certifiable_root(tmp_path)
    policy = ProtectionPolicy(
        required_protected=(),
        slice_regression_max=0.0625,
        trusted_ancestor_version=ANCESTOR_VERSION,
        candidate_version=CANDIDATE_VERSION,
        parent_version="gen1",
    )
    assert _certify(root, policy).status == UNKNOWN


# ==========================================================================
# Layer 3 -- T21/T22 as the second opinion over a real run
# ==========================================================================

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
_spec = importlib.util.spec_from_file_location(
    "judge_gen2_adversarial", GEN2 / "judge_gen2.py"
)
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

ROW = re.compile(r"^(T\d+|INFO)\s+(.+?)\s+(PASS|UNKNOWN|FAIL|INFO)\s+(.+)$")


def _judge(
    run_root: Path, manifest_path: Path
) -> tuple[dict[str, tuple[str, str]], str, int]:
    """``{threshold: (status, detail)}``, the branch verdict, and the exit code."""
    original = judge_gen2.CAMPAIGN_MANIFEST
    judge_gen2.CAMPAIGN_MANIFEST = manifest_path
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = judge_gen2.judge(run_root)
    finally:
        judge_gen2.CAMPAIGN_MANIFEST = original
    rows: dict[str, tuple[str, str]] = {}
    final = ""
    for line in buffer.getvalue().splitlines():
        match = ROW.match(line.strip())
        if match and match.group(1).startswith("T"):
            rows[match.group(1)] = (match.group(3), match.group(4))
        if line.startswith("VERDICT: "):
            final = line.split(": ", 1)[1].strip()
    assert final, buffer.getvalue()
    return rows, final, exit_code


def _declaring_floor(value: float) -> dict[str, Any]:
    return {
        "profile_id": "gen2-protection",
        "constraints": [
            {
                "dimension": "math500",
                "kind": "absolute-floor",
                "value": value,
                "benchmark": PROTECTED_ID,
            },
        ],
    }


def test_a_clean_campaign_is_certified_by_both_authorities(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """The control: the run promotes, and the judge over the same root agrees."""
    manifest, runner, _document = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)

    run = run_campaign(manifest)
    assert run.verdict == "PROMOTED", run.verdict

    rows, _final, _exit = _judge(
        Path(manifest.state_root), tmp_path / "inputs" / "campaign.json"
    )
    assert {"T21", "T22"} <= set(rows), sorted(rows)
    assert rows["T21"][0] == "PASS", rows["T21"]
    assert rows["T22"][0] == "PASS", rows["T22"]


def test_a_run_refused_on_the_declared_gate_is_refused_here_too(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """T21 refuses and T22 *agrees*: both halves must answer the same way.

    T21 audits what the run recorded. T22 independently recomputes the same
    declared profile through production's own evaluator. A root the run refused
    must not produce a T21 that waves it through, nor a T22 that computes a
    different answer from the one in the record.
    """
    manifest, runner, _document = _campaign(
        tmp_path, with_ancestor=True, retention_profile=_declaring_floor(0.5625)
    )
    _patch_seams(monkeypatch, runner)

    run = run_campaign(manifest)
    assert run.verdict == "REJECTED", run.verdict

    rows, final, exit_code = _judge(
        Path(manifest.state_root), tmp_path / "inputs" / "campaign.json"
    )
    assert final == "REJECTED", (final, rows.get("T21"), rows.get("T22"))
    assert exit_code == 1
    assert rows["T21"][0] == "FAIL" and "DECLARED_GATE_REJECTED_RUN" in rows["T21"][1]
    assert rows["T22"][0] == "PASS" and "== recomputed" in rows["T22"][1]
    assert "RETENTION_FLOOR" in rows["T22"][1]


def _edit_record(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "campaign-run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def _tamper_raw_bytes(root: Path) -> None:
    """Edit every raw slice the run wrote, keeping the declared digests."""
    raw = root / "raw"
    assert raw.is_dir(), sorted(p.name for p in root.iterdir())
    for path in raw.rglob("*"):
        if path.is_file():
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")


#: Mutations applied to the run root *after* the run recorded its decision.
#: Each touches one artifact; the judge must not certify the result.
POST_RUN_MUTATIONS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("the candidate's raw bytes are edited after the verdict", _tamper_raw_bytes),
    ("the run record is deleted", lambda root: (root / "campaign-run.json").unlink()),
    (
        "the recorded decision is removed from the record",
        lambda root: _edit_record(root, lambda r: r.update({"promotion": None})),
    ),
    (
        "the record claims PROMOTED while naming a declared-gate breach",
        lambda root: _edit_record(
            root,
            lambda r: r.update(
                {
                    "verdict": "PROMOTED",
                    "promotion": {
                        "decision": {
                            "verdict": "PROMOTED",
                            "reasons": [
                                "RETENTION_FLOOR: candidate 0.5 is below the absolute floor 0.9"
                            ],
                        }
                    },
                }
            ),
        ),
    ),
    (
        "the record names a breach the declared profile cannot produce",
        lambda root: _edit_record(
            root,
            lambda r: r["promotion"]["decision"]["reasons"].append(
                "RETENTION_REGRESSION: regression -0.5 on 'invented' breaches the "
                "declared max-regression -0.0625"
            ),
        ),
    ),
)


@pytest.mark.parametrize(
    ("name", "mutate"), POST_RUN_MUTATIONS, ids=[case[0] for case in POST_RUN_MUTATIONS]
)
def test_the_judge_never_certifies_a_run_root_it_no_longer_agrees_with(
    tmp_path, monkeypatch, name: str, mutate: Callable[[Path], None]  # noqa: ANN001
) -> None:
    """The invariant, in its one-way and fail-closed form.

    The judge audits the same root the run wrote, so it may legitimately be
    *stricter* than the run: the run certified those bytes before they were
    edited. It may never be *more permissive*. Whatever the run decided, a
    tampered root must not come back PROMOTED, and where T22 does fire it must
    name the disagreement rather than shrug.
    """
    manifest, runner, _document = _campaign(
        tmp_path, with_ancestor=True, retention_profile=_declaring_floor(0.9)
    )
    _patch_seams(monkeypatch, runner)
    run_root = Path(manifest.state_root)
    run = run_campaign(manifest)
    assert run.verdict == "REJECTED", run.verdict

    mutate(run_root)
    rows, final, _exit = _judge(run_root, tmp_path / "inputs" / "campaign.json")

    assert final != "PROMOTED", (name, final, rows)
    if rows.get("T22", ("", ""))[0] == "FAIL":
        assert "RETENTION_RECOMPUTATION_DISAGREES" in rows["T22"][1], rows["T22"]