"""Certification: the mechanism the runner gates on and the judge audits with.

The campaign runner may not record a promoted generation until certification
says it may, so these tests pin the mechanism itself: which measurement is
usable, which arm is bound to which bytes, and what happens when a candidate
only matches an already-regressed parent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.certification import (
    ARM_ADAPTER_DIGEST_MISMATCH,
    ARM_BASE_DIGEST_MISSING,
    ARM_GENERATION_MISMATCH,
    FAIL,
    GEN2_PROTOCOL,
    PASS,
    UNKNOWN,
    MeasuredArm,
    ProtectionPolicy,
    certify_run_root,
    protocol_problems,
)

MATH = "math500@2024-04"
DECODING = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512}


def _slice(
    root: Path,
    qualified_id: str,
    version: str,
    origin: str,
    score: float,
    *,
    digest: str | None = "auto",
) -> BenchmarkRun:
    """A protocol-exact slice row, bound to bytes that exist beside the report."""
    payload = json.dumps({"benchmark": qualified_id, "version": version, "score": score})
    artifact = root / "raw" / f"{version}-{qualified_id.replace('@', '-')}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(payload, encoding="utf-8")
    metadata = {
        "sample_indices": list(range(16)),
        "seed": 1234,
        "shuffle": False,
        "decoding": dict(DECODING),
        "prompt_policy": "chat_template",
    }
    if digest is not None:
        from chowder.provenance import sha256_file

        metadata["artifact_sha256"] = digest if digest != "auto" else sha256_file(artifact)
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="lm_eval",
        generation_version=version,
        score=score,
        n_samples=16,
        per_sample_scores=tuple([score] * 16),
        metric="accuracy",
        measurement_origin=origin,
        raw_artifact_ref=f"raw/{version}-{qualified_id.replace('@', '-')}.json",
        metadata=metadata,
    )


def _arm(
    root: Path,
    name: str,
    version: str,
    origin: str,
    score: float,
    *,
    adapter_digest: str = "",
    base_digest: str = "",
    digest: str | None = "auto",
) -> Path:
    path = root / name
    identity: dict[str, str] = {}
    if adapter_digest:
        identity["adapter_digest"] = adapter_digest
    if base_digest:
        identity["base_model_digest"] = base_digest
    EvalReport(
        generation_version=version,
        runs=(_slice(root, MATH, version, origin, score, digest=digest),),
        model_identity=identity,
    ).save(path)
    return path


def _certify(
    root: Path,
    *,
    candidate: Path | None,
    parent: Path | None,
    ancestor: Path | None,
    adapter_digests: dict[str, str] | None = None,
    base_digests: dict[str, str] | None = None,
):
    return certify_run_root(
        policy=ProtectionPolicy(
            required_protected=(MATH,),
            slice_regression_max=0.0625,
            trusted_ancestor_version="gen0",
            protocol=GEN2_PROTOCOL,
            candidate_version="gen2",
            parent_version="gen1",
        ),
        run_root=root,
        arms={"candidate": candidate, "parent": parent, "ancestor": ancestor},
        expected_adapter_digests=adapter_digests,
        expected_base_digests=base_digests,
    )


# ---------------------------------------------------------------------------
# branch protection: the trusted ancestor, not just the parent
# ---------------------------------------------------------------------------


def test_a_candidate_matching_a_regressed_parent_still_fails(tmp_path: Path) -> None:
    """The case T16 exists for: gen1 regressed against gen0, gen2 matches gen1."""
    root = tmp_path / "run"
    root.mkdir()
    candidate = _arm(root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.0)
    parent = _arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.0)
    ancestor = _arm(root, "ancestor.json", "gen0", MEASURED_PARENT, 0.5)

    certification = _certify(
        root, candidate=candidate, parent=parent, ancestor=ancestor
    )

    assert certification.status == FAIL
    assert any("trusted-ancestor protection" in reason for reason in certification.reasons)
    # The immediate-parent comparison alone would have been satisfied.
    assert any(
        row.status == PASS and "candidate-vs-parent" in row.requirement
        for row in certification.rows
    )


def test_a_candidate_holding_against_both_arms_passes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    certification = _certify(
        root,
        candidate=_arm(root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.5),
        parent=_arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.5),
        ancestor=_arm(root, "ancestor.json", "gen0", MEASURED_PARENT, 0.5),
    )
    assert certification.status == PASS
    assert certification.reasons == ()


def test_an_absent_trusted_ancestor_is_undecided_not_a_pass(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    certification = _certify(
        root,
        candidate=_arm(root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.5),
        parent=_arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.5),
        ancestor=None,
    )
    assert certification.status == UNKNOWN
    assert any("trusted ancestor" in reason for reason in certification.reasons)


def test_a_protocol_correct_row_of_the_wrong_generation_is_refused(tmp_path: Path) -> None:
    """A gen1 measurement cannot stand in for the gen0 arm: the label is checked."""
    root = tmp_path / "run"
    root.mkdir()
    candidate = _arm(root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.5)
    parent = _arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.5)
    # Same provenance, same protocol, wrong generation: the row is not gen0's.
    ancestor = _arm(root, "ancestor.json", "gen1", MEASURED_PARENT, 0.5)

    certification = _certify(
        root, candidate=candidate, parent=parent, ancestor=ancestor
    )

    assert certification.status in {FAIL, UNKNOWN}
    assert any("gen0" in reason for reason in certification.reasons)


# ---------------------------------------------------------------------------
# identity: an arm is only usable when it names the bytes it measured
# ---------------------------------------------------------------------------


def test_a_candidate_arm_bound_to_another_artifact_is_a_hard_failure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    certification = _certify(
        root,
        candidate=_arm(
            root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.5,
            adapter_digest="a" * 64,
        ),
        parent=_arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.5),
        ancestor=_arm(root, "ancestor.json", "gen0", MEASURED_PARENT, 0.5),
        adapter_digests={"candidate": "b" * 64},
    )
    assert certification.status == FAIL
    assert any(ARM_ADAPTER_DIGEST_MISMATCH in reason for reason in certification.reasons)


def test_an_arm_that_names_nothing_is_undecided(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    certification = _certify(
        root,
        candidate=_arm(root, "candidate.json", "gen2", MEASURED_THIS_GENERATION, 0.5),
        parent=_arm(root, "parent.json", "gen1", MEASURED_PARENT, 0.5),
        ancestor=_arm(root, "ancestor.json", "gen0", MEASURED_PARENT, 0.5),
        base_digests={"ancestor": "c" * 64},
    )
    assert certification.status == UNKNOWN
    assert any(ARM_BASE_DIGEST_MISSING in reason for reason in certification.reasons)


def test_measuring_arms_enforces_generation_and_origin(tmp_path: Path) -> None:
    path = _arm(tmp_path, "arm.json", "gen1", MEASURED_PARENT, 0.5)
    arm = MeasuredArm.load(
        path, expected_origin=MEASURED_PARENT, expected_generation="gen0", label="ancestor"
    )
    assert arm.run_for(MATH) is None, "a gen1 row must not serve as the gen0 measurement"
    assert any(
        ARM_GENERATION_MISMATCH in problem
        for problem in arm.identity_problems(base_digest="d" * 64)
    )


# ---------------------------------------------------------------------------
# measurement evidence: declarations are not measurements
# ---------------------------------------------------------------------------


def test_a_row_whose_artifact_is_missing_is_refused(tmp_path: Path) -> None:
    run = _slice(tmp_path, MATH, "gen2", MEASURED_THIS_GENERATION, 0.5)
    object.__setattr__(run, "raw_artifact_ref", "raw/never-written.json")
    problems = protocol_problems(run, run_root=tmp_path, protocol=GEN2_PROTOCOL)
    assert any("MEASUREMENT_ARTIFACT_MISSING" in problem for problem in problems)


def test_a_row_without_a_digest_is_refused(tmp_path: Path) -> None:
    run = _slice(tmp_path, MATH, "gen2", MEASURED_THIS_GENERATION, 0.5, digest=None)
    problems = protocol_problems(run, run_root=tmp_path, protocol=GEN2_PROTOCOL)
    assert any("MEASUREMENT_DIGEST_ABSENT" in problem for problem in problems)


def test_a_row_whose_samples_contradict_its_score_is_refused(tmp_path: Path) -> None:
    run = _slice(tmp_path, MATH, "gen2", MEASURED_THIS_GENERATION, 0.5)
    object.__setattr__(run, "per_sample_scores", (0.0,) * 16)
    problems = protocol_problems(run, run_root=tmp_path, protocol=GEN2_PROTOCOL)
    assert any("MEASUREMENT_SAMPLES_INCONSISTENT" in problem for problem in problems)


def test_a_certifiable_row_has_no_problems(tmp_path: Path) -> None:
    run = _slice(tmp_path, MATH, "gen2", MEASURED_THIS_GENERATION, 0.5)
    assert protocol_problems(run, run_root=tmp_path, protocol=GEN2_PROTOCOL) == ()


# ---------------------------------------------------------------------------
# the report's identity survives its own round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", ({}, {"adapter_digest": "e" * 64}, {"base_model_digest": "f" * 64}))
def test_model_identity_round_trips(tmp_path: Path, identity: dict[str, str]) -> None:
    report = EvalReport(
        generation_version="gen2",
        runs=(_slice(tmp_path, MATH, "gen2", MEASURED_THIS_GENERATION, 0.5),),
        model_identity=identity,
    )
    path = tmp_path / "report.json"
    report.save(path)
    assert dict(EvalReport.load(path).model_identity) == identity
