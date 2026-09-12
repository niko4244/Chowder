"""The schedule a recipe asks for must be the schedule the optimiser follows.

`test_lr_scheduler_honoured.py` guards the plumbing by source. This guards the
*outcome*, from the rates a run actually logged -- the only evidence that survives
a transformers upgrade changing its cosine definition, its `num_cycles` default, or
when it reads the rate for logging.

The real 500-step trajectory of the pruned 9B is committed as a fixture
(`data/lr-trajectory-pruned9b-cosine-500.jsonl`, step + learning_rate only) because
it pins HF's actual behaviour rather than my model of it.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from chowder.schedule_audit import (
    INCONCLUSIVE,
    MAX_RESIDUAL_FRACTION,
    MIN_SEPARATION,
    SCHEDULES,
    identify_schedule,
)

PEAK = 2e-4
TOTAL = 500
FIXTURE = Path(__file__).parent / "data" / "lr-trajectory-pruned9b-cosine-500.jsonl"


def _synthetic(name: str, *, steps: range, warmup: int = 0) -> list[tuple[int, float]]:
    lam = SCHEDULES[name]
    return [(s, PEAK * lam(s, TOTAL, warmup)) for s in steps]


# ---- the real run ---------------------------------------------------------------


def _real() -> list[tuple[int, float]]:
    rows = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(r["step"], r["learning_rate"]) for r in rows]


def test_the_real_run_followed_cosine_exactly() -> None:
    """500 logged rates, zero gaps, from the run that verified today's fix.

    The fit is *numerically exact*, not bit-exact: float summation order differs
    across platforms/BLAS builds, so the residual can be ~1e-17 of peak instead
    of exactly 0.0 and the separation can be a huge finite number instead of
    inf. Demanding bit-identical math failed four CI jobs (Linux py3.10-3.12 and
    the real-CPU smoke, 2026-09-12) while Windows passed. The tolerance below is
    five orders of magnitude tighter than MAX_RESIDUAL_FRACTION (1e-2), so the
    scientific classification is untouched; separation still must clear
    MIN_SEPARATION, with infinity accepted as a valid (better) value.
    """
    observed = _real()
    assert len(observed) == 500
    verdict = identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL)
    assert verdict.matches("cosine")
    assert math.isclose(verdict.residual_fraction_of_peak, 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert verdict.separation >= MIN_SEPARATION
    assert verdict.offset == -1


def test_a_minimally_perturbed_trajectory_is_still_conclusive() -> None:
    """Regression for the CI portability defect: one logging float one ULP off
    must not flip a conclusive cosine verdict. This is the smallest perturbation
    a real run can produce, and it is what separated the passing Windows run
    from the failing Linux CI jobs."""
    observed = _real()
    step, lr = observed[249]
    observed[249] = (step, math.nextafter(lr, math.inf))
    verdict = identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL)
    assert verdict.matches("cosine")
    assert math.isclose(verdict.residual_fraction_of_peak, 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert verdict.separation >= MIN_SEPARATION


def test_the_real_run_pins_the_hf_logging_offset() -> None:
    """Trainer logs get_last_lr(), so the rate logged at step s is lambda(s-1).
    If a transformers upgrade changes that, this is where it surfaces."""
    assert identify_schedule(_real(), peak_lr=PEAK, total_steps=TOTAL).offset == -1


def test_the_real_runs_final_rate_is_the_cheapest_discriminator() -> None:
    """No fitting needed: cosine drives the rate to ~0, constant would still be at
    peak, and roughly five orders of magnitude below peak (1.97e-9 vs 2e-4) is
    only consistent with cosine."""
    final_step, final_lr = _real()[-1]
    assert final_step == 500
    assert final_lr == pytest.approx(1.973914386288467e-09)
    assert final_lr < PEAK * 1e-4


def test_the_real_run_is_not_mistaken_for_linear() -> None:
    verdict = identify_schedule(
        _real(), peak_lr=PEAK, total_steps=TOTAL, candidates=("cosine", "linear")
    )
    assert verdict.matches("cosine")


# ---- synthetic: every modelled schedule is recovered ----------------------------


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_each_modelled_schedule_is_identified(name: str) -> None:
    verdict = identify_schedule(
        _synthetic(name, steps=range(1, TOTAL + 1)), peak_lr=PEAK, total_steps=TOTAL
    )
    assert verdict.matches(name)


def test_warmup_is_modelled_rather_than_confusing_the_fit() -> None:
    warmup = 50
    observed = _synthetic("cosine", steps=range(1, TOTAL + 1), warmup=warmup)
    assert identify_schedule(
        observed, peak_lr=PEAK, total_steps=TOTAL, warmup_steps=warmup
    ).matches("cosine")


def test_a_partial_trajectory_is_still_identified() -> None:
    """A run that dies at step 120 can still be audited."""
    observed = _synthetic("cosine", steps=range(1, 121))
    assert identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL).matches("cosine")


def test_one_percent_noise_does_not_break_the_margin() -> None:
    rng = random.Random(5)
    observed = [
        (s, lr * (1 + rng.uniform(-0.01, 0.01)))
        for s, lr in _synthetic("cosine", steps=range(1, TOTAL + 1))
    ]
    verdict = identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL)
    assert verdict.matches("cosine")
    assert verdict.separation > 3.0


# ---- the honesty properties -----------------------------------------------------


def test_it_refuses_when_the_trajectory_carries_no_information() -> None:
    """Ten noisy steps: every schedule is ~peak there, so the gap between curves is
    inside the noise. The verdict must be INCONCLUSIVE, not a confident guess -- a
    run that dies at step 10 has to report "unknown"."""
    rng = random.Random(11)
    observed = [
        (s, lr * (1 + rng.uniform(-0.01, 0.01)))
        for s, lr in _synthetic("cosine", steps=range(1, 11))
    ]
    verdict = identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL)
    assert verdict.schedule == INCONCLUSIVE
    assert not verdict.conclusive
    assert not verdict.matches("cosine")
    assert not verdict.matches("linear")  # refusing is not a vote for the other one


def test_ten_exact_steps_are_still_evidence() -> None:
    """The problem is few samples AND noise, not few samples. cosine(10)=1.99803e-4
    against linear(10)=1.96e-4 differ in the third digit, so an exact fit to one of
    them is real. Pinned because the opposite intuition is the tempting one."""
    verdict = identify_schedule(
        _synthetic("cosine", steps=range(1, 11)), peak_lr=PEAK, total_steps=TOTAL
    )
    assert verdict.matches("cosine")


def test_it_refuses_when_the_true_schedule_is_not_a_candidate() -> None:
    """An inverse-sqrt run audited against linear/cosine/constant must not be
    reported as whichever of those happens to be nearest."""
    observed = [
        (s, PEAK / math.sqrt(max(1, s)))
        for s in range(1, TOTAL + 1)
    ]
    verdict = identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL)
    assert verdict.schedule == INCONCLUSIVE
    assert verdict.residual_fraction_of_peak >= MAX_RESIDUAL_FRACTION


def test_a_flat_run_at_peak_is_constant_not_a_decaying_schedule() -> None:
    """The silent-drop symptom this whole guard exists for would look like this if
    the trainer default were constant rather than linear."""
    observed = [(s, PEAK) for s in range(1, TOTAL + 1)]
    assert identify_schedule(observed, peak_lr=PEAK, total_steps=TOTAL).matches("constant")


# ---- unusable input is refused, not answered -----------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"peak_lr": 0.0}, "peak_lr must be positive"),
        ({"peak_lr": float("nan")}, "peak_lr must be positive"),
        ({"total_steps": 0}, "total_steps must be positive"),
        ({"warmup_steps": -1}, "warmup_steps cannot be negative"),
        ({"candidates": ("cosine", "vibes")}, "unmodelled schedule"),
        ({"candidates": ("cosine",)}, "at least two candidates"),
    ],
)
def test_unusable_input_raises(kwargs, message) -> None:
    base = {"peak_lr": PEAK, "total_steps": TOTAL}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        identify_schedule(_synthetic("cosine", steps=range(1, 50)), **base)


def test_no_observations_raises() -> None:
    with pytest.raises(ValueError, match="no learning-rate observations"):
        identify_schedule([], peak_lr=PEAK, total_steps=TOTAL)
