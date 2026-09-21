"""The no-GPU orchestration dry-run matrix: run verdict == frozen judge verdict.

The certification boundary has one load-bearing invariant: a manifest-driven
campaign must never be able to record something the frozen judge would reject.
The coupling suite proves that for a clean run and a handful of tampered roots;
this matrix proves it *across the states the orchestration can actually reach* --
a clean promotion, an unresolved trusted-ancestor regression, an artifact that
moves under its own measurement, an evaluation that overruns the campaign
envelope, a build with no evaluator wired, an evaluation that reports no cost, a
contamination mismatch, and an identity mismatch on either the base or the
ancestor arm.

The judge is imported unmodified from ``docs/gen2`` and no threshold is restated
here. Every scenario is driven through the real CLI and then judged on the same
run root the run wrote, so what is asserted is agreement between the two
production verdicts rather than agreement between two test expectations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.growth.compute_cost import ComputeCost

import test_growth_campaign_runner as campaign_fixture
from test_growth_certification_coupling import (
    MATH,
    _campaign,
    _judge,
)

#: What the frozen judge prints when it accepts a run root. The equality asserted
#: below is exactly "the run promoted iff the judge promoted", so the string is
#: read from the judge's own output rather than restated as a policy.
PROMOTED = "VERDICT: PROMOTED"


def _judge_verdict(root: Path) -> tuple[int, str]:
    """Run the frozen judge on a run root, including one the run never wrote.

    A pre-compute refusal leaves no run root at all, and that must count as a
    judge refusal rather than an error the matrix cannot read: no root is not a
    promotion.
    """
    root.mkdir(parents=True, exist_ok=True)
    try:
        return _judge(root)
    except Exception as error:  # a root so incomplete the judge cannot read it
        return 1, f"the frozen judge could not read the run root: {error}"


def _agreement(root: Path, code: int, payload: dict) -> str:
    """Assert the run and the judge agree, and return what they agreed on.

    The invariant has two directions. A run that promoted must be certified by
    the judge on the same bytes; and a run that did *not* promote must leave no
    root the judge accepts, or the production path could write a lineage record
    the audit would reject.
    """
    verdict = str(payload.get("verdict", ""))
    judge_code, output = _judge_verdict(root)
    promoted = PROMOTED in output

    if code == 0 and verdict == "PROMOTED":
        assert judge_code == 0 and promoted, (
            "the run recorded PROMOTED but the frozen judge refused the run's "
            f"own evidence:\n{output}"
        )
        return "PROMOTED"

    assert not promoted, (
        f"the run reached {verdict!r} but the frozen judge promoted the same "
        f"run root:\n{output}"
    )
    return verdict


def _resource_failure(payload: dict) -> bool:
    """Whether the run's own record names a settlement/resource failure."""
    settlement = payload.get("settlement") or {}
    return settlement.get("budget_compliant") is False and bool(
        settlement.get("budget_failure_reasons")
    )


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


def test_matrix_clean_promotion_agrees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline: both production verdicts reach PROMOTED on the same root."""
    root, code, payload = _campaign(tmp_path, monkeypatch)

    assert code == 0, payload.get("refusal_reason") or payload
    assert _agreement(root, code, payload) == "PROMOTED"


def test_matrix_ancestor_regression_is_refused_by_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The T16 case: gen2 inherits an unresolved gen0 regression.

    The trusted ancestor measured 0.25 on the protected slice while the candidate
    sits at 0.0. The generic parent-vs-candidate rule cannot see this -- parent
    and candidate both read 0.0 -- so it is exactly the situation branch
    protection exists to catch, before any lineage record.
    """
    # MGSM is the declared *protected* slice here, so an ancestor that measured
    # 0.5 on it against a candidate at 0.0625 is an inherited regression the
    # parent-vs-candidate rule cannot see (parent and candidate both read 0.0).
    root, code, payload = _campaign(tmp_path, monkeypatch, scores={"ancestor_mgsm": 0.5})

    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"
    judge_code, output = _judge_verdict(root)
    assert judge_code == 1
    assert "T16" in output


@pytest.mark.parametrize("mutate", ("before", "after"))
def test_matrix_an_artifact_that_moves_is_refused_by_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: str
) -> None:
    """The selected bytes change under their own measurement, at either boundary.

    ``before`` is the post-training race the pre-measurement re-verification
    catches; ``after`` is the post-measurement one the pre-verdict re-verification
    catches. Either way the frozen judge would recompute the digest over bytes
    that no longer match, so the run must refuse before it binds a verdict.
    """
    root, code, payload = _campaign(tmp_path, monkeypatch, mutate=mutate)

    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"
    assert code == 1


def test_matrix_an_evaluation_overrun_vetoes_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measuring the candidate is spend, and spend can veto admission.

    The evaluation reports a real measured wall cost that pushes the campaign
    past its frozen wall ceiling. The training succeeded and the artifact is
    preserved, but a successful training process is not a budget-compliant
    experiment, so the run must not promote -- and the judge must agree.
    """
    overrun = ComputeCost(
        device_gpu_hours=0.0,
        wall_gpu_hours=0.5,
        source="dry-run matrix evaluator",
        measurement_method="wall clock",
        device_measured=False,
    )
    root, code, payload = _campaign(tmp_path, monkeypatch, cost=overrun)

    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"
    assert _resource_failure(payload), payload.get("settlement")


def test_matrix_a_missing_evaluator_refuses_before_any_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No instrument wired is knowable before training, so nothing may train.

    The build has no candidate evaluator, so a run cannot measure the artifact it
    would select. That is a pre-compute fact: the run must refuse before it
    spends, and the judge has nothing to certify.
    """
    root, code, payload = _campaign(tmp_path, monkeypatch, wired=False)

    assert code == 1
    assert payload["verdict"] == "REFUSED"
    assert payload["refused_by"] == "readiness"
    assert "evaluator" in str(payload.get("refusal_reason", "")).lower()
    # Zero compute: no attempt ran and nothing was spent, because the missing
    # instrument is knowable before the first training subprocess.
    assert payload["attempts"] == []
    assert not (root / "cycle_compute_accounting.json").exists()


def test_matrix_a_missing_evaluation_cost_is_a_refusal_not_a_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreported cost is not free compute.

    The evaluation ran and reported no cost. Charging it as zero would make a
    campaign look cheaper exactly when its accounting is least trustworthy, so
    the run refuses -- after the training spend, which is still closed out.
    """
    root, code, payload = _campaign(tmp_path, monkeypatch, cost=None)

    assert code == 1
    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"


def _contaminated(inputs: Path) -> str:
    """A contamination manifest that pins the protected slice as contaminated."""
    path = inputs / "contaminated-preparation.json"
    path.write_text(
        (
            '{"benchmarks": {"%s": {"status": "KNOWN_CONTAMINATION", '
            '"reason": "dry-run matrix"}}, "policy": {}, "training_sources": {'
            '"src-1": {"status": "KNOWN_CONTAMINATION", "reason": "dry-run matrix"}}}'
        )
        % MATH,
        encoding="utf-8",
    )
    return str(path)


def test_matrix_a_contamination_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared source or slice the firewall refuses cannot reach a verdict."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    path = _contaminated(inputs)

    root, code, payload = _campaign(
        tmp_path, monkeypatch, contamination_manifest_path=path
    )

    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"


def test_matrix_a_base_identity_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base that is not the preregistered one must be refused before compute."""
    root, code, payload = _campaign(tmp_path, monkeypatch, base_model_digest="0" * 64)

    assert code == 1
    assert payload["verdict"] == "REFUSED"
    assert _agreement(root, code, payload) != "PROMOTED"


def test_matrix_an_ancestor_identity_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ancestor arm bound to other bytes than the frozen base is not protection.

    The declared ancestor report measures an artifact whose digest is not the
    dense base the campaign pinned. Reading it as the trusted ancestor would let
    the branch-protection gate pass on bytes nobody measured, so both the run's
    own certification and the judge must refuse it.
    """
    root, code, payload = _campaign(
        tmp_path, monkeypatch, ancestor_base_digest="f" * 64
    )

    assert payload["verdict"] != "PROMOTED"
    assert _agreement(root, code, payload) != "PROMOTED"


def test_matrix_every_scenario_leaves_the_same_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant, restated across scenarios in one place.

    Each scenario gets its own ``tmp_path`` subtree so the fixtures cannot share
    state, and the assertion is the same for all of them: whatever verdict the
    campaign reached, the frozen judge agrees about the root it left behind.
    """
    scenarios = {
        "clean": {},
        "ancestor_regression": {"scores": {"ancestor_mgsm": 0.5}},
        "artifact_mutation": {"mutate": "after"},
        "evaluation_overrun": {
            "cost": ComputeCost(
                device_gpu_hours=0.0,
                wall_gpu_hours=0.5,
                source="dry-run matrix evaluator",
                measurement_method="wall clock",
            )
        },
        "missing_evaluator": {"wired": False},
        "missing_cost": {"cost": None},
        "base_identity_mismatch": {"base_model_digest": "0" * 64},
        "ancestor_identity_mismatch": {"ancestor_base_digest": "f" * 64},
    }
    for name, overrides in scenarios.items():
        case = tmp_path / name
        case.mkdir()
        root, code, payload = _campaign(case, monkeypatch, **overrides)
        # ``_campaign`` returns the run's own root; the judge reads that root.
        verdict = _agreement(root, code, payload)
        if name == "clean":
            assert verdict == "PROMOTED"
        else:
            assert verdict != "PROMOTED", f"{name} promoted when it should not have"
            assert code is not None
