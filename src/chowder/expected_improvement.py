"""Expected-improvement estimation over historical intervention outcomes.

Roadmap Priority 6 asks for an "expected-improvement model" to drive the
GPU-hour-aware experiment policy, and immediately constrains it: *"Only
claim learned-policy improvement once validated against held-out
experiments."* This module is the deterministic/statistical first slice of
that, plus the backtest that is supposed to decide whether it earned its
place.

It is deliberately **not** a learned model. No GP, no surrogate, no
training, no RNG, no ML dependency -- ordinary sample statistics over the
real observed rewards `intervention_outcomes` already normalized out of the
registry. A fitted surrogate is a later slice, and it should have to beat
*this* on the same backtest before anyone believes it.

Not yet demonstrated superior
-----------------------------
As of this module landing, the expected-improvement selector has **not**
been shown to beat the existing UCB1 selector
(`candidate_selection.prioritize_candidates`). It is an alternative, not an
improvement, and UCB1 remains the default anywhere a default is needed.

Two reasons, both worth stating plainly rather than burying:

1. **No real data exists to validate against.** This repository contains no
   registry of real historical Chowder runs, so `backtest_selectors` has so
   far only been pointed at *constructed* outcome sets (the three regimes in
   `tests/test_expected_improvement.py`, fixed before any result was read).
   A constructed regime can be built to favour either selector, so it
   validates the harness, not the policy.

2. **On those constructed regimes the result is mixed and negative overall.**
   Mean top-1 regret over 9 rolling-origin splits each (lower is better):

       regime      expected_improvement    ucb1     input_order
       separable                0.08000   0.12000       0.12000
       noisy                    0.16556   0.11667       0.11667
       drifting                 0.22667   0.16778       0.16778

   EI wins where arms are cleanly separable and loses where within-arm
   variance is high or arm quality drifts over time -- which is what the
   method's own assumptions predict, since a stationary Gaussian per arm is
   exactly wrong under drift. Two caveats make even this weak: the
   do-nothing `input_order` control matches UCB1's regret exactly on two of
   three regimes, and EI and UCB1 make a *different first pick* in only 2-3
   of 9 splits per regime. Three differing decisions do not support a claim
   in either direction.

Reproduce with `estimate_arm_improvements` / `backtest_selectors` over the
regimes in `tests/test_expected_improvement.py`; both are deterministic.

What "expected improvement" means here
--------------------------------------
Classic EI: given an incumbent best `f*` and a posterior over an arm's next
reward, `EI = E[max(f - f*, 0)]`. Adapted to this data:

  arm       The frozenset of dotted `config_patch` key-paths an experiment
            touched -- `candidate_selection.dotted_paths`, reused via
            `intervention_outcomes.group_by_arm`, never redefined here.
            Two modules disagreeing about what an arm is would make this
            comparison meaningless.

  reward    `gate_score_vs_baseline / gpu_hours` -- byte-for-byte the
            reward UCB1 already bandits over (`decision.score /
            gpu_hours`, cf. `tournament.rank_candidates`'s efficiency).
            Deliberately identical so that a backtest difference is a
            difference in *policy*, not in what the two policies were
            shown.

  f*        The single best reward actually observed anywhere in the
            supplied history, including in arms too sparse to estimate --
            a real observation is a real incumbent regardless of which arm
            produced it.

  posterior A Gaussian centred on the arm's sample mean with the sample
            standard deviation inflated to a *predictive* standard error,
            `s * sqrt(1 + 1/n)`: the spread of one more draw from the arm,
            not the spread of the arm's mean. This is what makes a
            2-observation arm visibly less certain than a 20-observation
            one instead of merely noisier.

Honesty rules
-------------
1. **Insufficient evidence is a state, not a number.** An arm with fewer
   than `min_observations` usable observations gets
   `ArmEvidence.INSUFFICIENT` and `expected_improvement=None`. It is never
   handed a point estimate that looks as confident as a well-observed
   arm's. The floor is 2 because a single observation carries no dispersion
   information at all, and an EI computed from an assumed spread would be
   fiction. `min_observations` is a documented starting point, not a
   claimed-optimal constant.

2. **Nothing is silently dropped.** Only two `InterventionOutcome` fields
   are read, and neither is ever `None`: `gate_score_vs_baseline` and
   `gpu_hours`. So no row is discarded for absent evidence. The one
   exclusion is a *non-finite* gate score, which `gate.evaluate_candidate`
   returns (`-inf`) when a candidate's evaluation evidence was incomplete
   or its protocol did not match the baseline's. That is a statement about
   the evaluation, not about the arm, so such rows are excluded from the
   statistics -- and counted, per arm and in total, in
   `ArmEstimate.excluded_non_finite` / `ArmImprovementModel
   .excluded_non_finite`, never dropped quietly. An arm whose observations
   are *all* non-finite ends up with zero usable observations and is
   therefore treated as unexplored, not as bad.

   Fields deliberately not read, and why:

     `gate_accepted`
         Not filtered on. `None` there means "the gate verdict was never
         persisted", which `filter_outcomes` already refuses to read as
         rejection; and the reward already carries the gate's own score, so
         filtering would discard real measured outcomes twice over.

     `gate_score_vs_parent`
         `None` for any rootless row, and for the rest it is measured
         against a *different* reference point per row. Mixing per-parent
         and per-baseline deltas into one mean would compare numbers that
         are not on the same scale.

     `peak_vram_gb`, `train_runtime_seconds`, `global_step`,
     `training_gpu_hours`, `base_model`, `recipe_sha256`,
     `memory_fabric_mechanisms`, ...
         Frequently `None` (see `intervention_outcomes`' own docstring for
         exactly when). Conditioning on them would silently restrict the
         estimate to runs produced by the real transformers-peft executor.
         Segmenting history by hardware or base model is a real and useful
         idea -- do it by passing pre-filtered rows through
         `filter_outcomes`, where the restriction is visible to the caller,
         rather than hiding it in here.

3. **Deterministic.** No RNG. `group_by_arm` preserves input order,
   `sorted` is stable, and every dict here is built in first-appearance
   order, so identical inputs give an identical model and an identical
   ordering.

This module never promotes, gates, or runs anything. Like
`candidate_selection`, it only reorders a pool of not-yet-run
`Experiment`s; a low-ranked candidate can still run, just later. Where it
needs a score it replays the real `gate.evaluate_candidate`, exactly as
`candidate_selection` and `intervention_outcomes` already do.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import erf, exp, isfinite, pi, sqrt
from statistics import fmean, stdev
from typing import Callable, Mapping, Sequence

from .candidate_selection import dotted_paths, prioritize_candidates
from .gate import evaluate_candidate
from .intervention_outcomes import InterventionOutcome, group_by_arm
from .models import Experiment, ExperimentResult, Goal, Hypothesis

# Two is the hard floor, not a tuned value: `statistics.stdev` is undefined
# below it, and an EI built on an assumed spread would be fabricated.
_DEFAULT_MIN_OBSERVATIONS = 2

# Matches candidate_selection/tournament exactly so the reward the two
# selectors optimize is the same quantity.
_MIN_GPU_HOURS = 1e-9


class ArmEvidence(str, Enum):
    """Whether an arm's own history supports an expected-improvement estimate."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ArmEstimate:
    """What the history actually supports saying about one arm.

    `expected_improvement` is `None` whenever `evidence` is
    `INSUFFICIENT` -- the point of this type is that "we cannot estimate
    this yet" is representable, so it never has to be faked as a number.

    `mean_reward` is populated whenever at least one usable observation
    exists (it is then a real mean of real numbers, not an estimate), and
    `reward_stdev` whenever at least two do. Both stay `None` for an arm
    whose every observation was non-finite.
    """

    arm: frozenset[str]
    evidence: ArmEvidence
    observations: int
    excluded_non_finite: int
    mean_reward: float | None
    reward_stdev: float | None
    predictive_stdev: float | None
    expected_improvement: float | None


@dataclass(frozen=True)
class ArmImprovementModel:
    """Per-arm expected improvement over one history. Purely descriptive."""

    incumbent_reward: float | None
    min_observations: int
    estimates: Mapping[frozenset[str], ArmEstimate]
    excluded_non_finite: int

    def expected_improvement(self, arm: frozenset[str]) -> float | None:
        """This arm's EI, or `None` for an unseen or unestimable arm.

        The two `None` cases are deliberately not distinguished here: both
        mean "the history does not support an estimate", and a caller that
        needs to tell them apart should look at `estimates`.
        """
        estimate = self.estimates.get(arm)
        return estimate.expected_improvement if estimate is not None else None


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _standard_normal_pdf(z: float) -> float:
    return exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _expected_improvement(*, mean: float, predictive_stdev: float, incumbent: float) -> float:
    """Closed-form Gaussian EI: `(mu - f*)Phi(z) + sigma*phi(z)`.

    A degenerate arm (every observation identical, so `sigma == 0`) has no
    uncertainty to be paid for exploring, and its EI collapses to the
    deterministic gain `max(mu - f*, 0)` -- which is the limit of the
    formula, handled explicitly because `z` would otherwise be infinite.

    The Gaussian is an approximation: the honest predictive distribution
    for a small sample is Student-t, which has heavier tails and would
    assign *more* EI to sparse arms. Using the Gaussian is therefore the
    conservative direction for a small-sample arm, and `min_observations`
    guards the range where the difference is largest.
    """
    gain = mean - incumbent
    if predictive_stdev <= 0.0:
        return max(gain, 0.0)
    z = gain / predictive_stdev
    return gain * _standard_normal_cdf(z) + predictive_stdev * _standard_normal_pdf(z)


def _reward(score: float, gpu_hours: float) -> float:
    """The GPU-hour-normalized gate score both selectors optimize."""
    return score / max(gpu_hours, _MIN_GPU_HOURS)


def _model_from_rewards(
    rewards_by_arm: Mapping[frozenset[str], Sequence[float]],
    *,
    min_observations: int,
) -> ArmImprovementModel:
    """Build the model from already-projected per-arm rewards.

    Shared by both public entry points so the arm statistics cannot drift
    between "estimate over stored outcomes" and "select over a live
    history".
    """
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2; a single observation has no spread")

    usable: dict[frozenset[str], tuple[float, ...]] = {}
    excluded: dict[frozenset[str], int] = {}
    for arm, rewards in rewards_by_arm.items():
        finite = tuple(reward for reward in rewards if isfinite(reward))
        usable[arm] = finite
        excluded[arm] = len(rewards) - len(finite)

    observed = [reward for rewards in usable.values() for reward in rewards]
    incumbent = max(observed) if observed else None

    estimates: dict[frozenset[str], ArmEstimate] = {}
    for arm, rewards in usable.items():
        count = len(rewards)
        mean = fmean(rewards) if count >= 1 else None
        sample_stdev = stdev(rewards) if count >= 2 else None
        if incumbent is None or count < min_observations:
            estimates[arm] = ArmEstimate(
                arm=arm,
                evidence=ArmEvidence.INSUFFICIENT,
                observations=count,
                excluded_non_finite=excluded[arm],
                mean_reward=mean,
                reward_stdev=sample_stdev,
                predictive_stdev=None,
                expected_improvement=None,
            )
            continue
        # min_observations >= 2 guarantees both are real numbers here.
        assert mean is not None and sample_stdev is not None
        predictive_stdev = sample_stdev * sqrt(1.0 + 1.0 / count)
        estimates[arm] = ArmEstimate(
            arm=arm,
            evidence=ArmEvidence.SUFFICIENT,
            observations=count,
            excluded_non_finite=excluded[arm],
            mean_reward=mean,
            reward_stdev=sample_stdev,
            predictive_stdev=predictive_stdev,
            expected_improvement=_expected_improvement(
                mean=mean, predictive_stdev=predictive_stdev, incumbent=incumbent
            ),
        )

    return ArmImprovementModel(
        incumbent_reward=incumbent,
        min_observations=min_observations,
        estimates=estimates,
        excluded_non_finite=sum(excluded.values()),
    )


def estimate_arm_improvements(
    outcomes: Sequence[InterventionOutcome],
    *,
    min_observations: int = _DEFAULT_MIN_OBSERVATIONS,
) -> ArmImprovementModel:
    """Estimate per-arm expected improvement from real historical *outcomes*.

    Reads exactly two fields per row -- `gate_score_vs_baseline` and
    `gpu_hours`, neither of which is ever `None` -- so no row is dropped
    for absent evidence. See this module's docstring for what a non-finite
    score means and for why the optional fields are deliberately not read.

    Pass rows through `intervention_outcomes.filter_outcomes` first to
    condition the estimate on a base model, an intervention key-path, or a
    score threshold; doing it there keeps the restriction visible.
    """
    rewards = {
        arm: tuple(_reward(row.gate_score_vs_baseline, row.gpu_hours) for row in rows)
        for arm, rows in group_by_arm(outcomes).items()
    }
    return _model_from_rewards(rewards, min_observations=min_observations)


def prioritize_by_expected_improvement(
    candidates: Sequence[Experiment],
    *,
    history: Sequence[tuple[Experiment, ExperimentResult]],
    goal: Goal,
    baseline: ExperimentResult,
    min_observations: int = _DEFAULT_MIN_OBSERVATIONS,
) -> tuple[Experiment, ...]:
    """Reorder not-yet-run *candidates* by expected improvement over historical arms.

    Signature-compatible with `candidate_selection.prioritize_candidates`
    (one extra keyword, defaulted), takes the same history shape, replays
    the same hard gate over it, and optimizes the same reward -- so the two
    can be swapped in `backtest_selectors` and differ only in policy.

    Cold start: an arm the history cannot estimate -- never tried, tried
    too few times, or tried only in runs the gate could not score -- gets
    `+inf`, exactly as UCB1 gives an unpulled arm. `sorted` is stable, so
    such candidates keep their original relative order and an empty history
    provably returns the input order unchanged. Treating an unestimable arm
    as *unexplored* rather than as *bad* is deliberate: the alternative
    lets a single unlucky (or unscoreable) run bury an arm permanently.

    Reorders only. It never promotes, gates, mutates a candidate, or runs
    anything; a bottom-ranked candidate is still free to run.
    """
    if not candidates:
        return ()

    rewards: dict[frozenset[str], list[float]] = defaultdict(list)
    for experiment, result in history:
        decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=result)
        rewards[dotted_paths(experiment.config_patch)].append(
            _reward(decision.score, result.gpu_hours)
        )
    model = _model_from_rewards(rewards, min_observations=min_observations)

    def score(experiment: Experiment) -> float:
        improvement = model.expected_improvement(dotted_paths(experiment.config_patch))
        return float("inf") if improvement is None else improvement

    return tuple(sorted(candidates, key=lambda experiment: -score(experiment)))


# --------------------------------------------------------------------------
# Validation harness: expected improvement vs. UCB1 on held-out experiments.
# --------------------------------------------------------------------------

Selector = Callable[..., tuple[Experiment, ...]]


def _input_order(
    candidates: Sequence[Experiment],
    *,
    history: Sequence[tuple[Experiment, ExperimentResult]],
    goal: Goal,
    baseline: ExperimentResult,
) -> tuple[Experiment, ...]:
    """The do-nothing control: whatever order the candidates arrived in.

    Without it the backtest cannot tell "EI beat UCB1" from "both were
    beaten by not reordering at all", which on a small or uninformative
    history is a genuinely common outcome and the one most worth catching.
    """
    return tuple(candidates)


#: The selectors `backtest_selectors` compares, in report order. UCB1 is the
#: incumbent baseline; `input_order` is the control that keeps a win honest.
SELECTORS: Mapping[str, Selector] = {
    "expected_improvement": prioritize_by_expected_improvement,
    "ucb1": prioritize_candidates,
    "input_order": _input_order,
}


@dataclass(frozen=True)
class SelectorScore:
    """One selector's performance on one held-out set."""

    top_1_regret: float
    best_rank: int
    ordered_experiment_ids: tuple[str, ...]


@dataclass(frozen=True)
class BacktestSplit:
    """One rolling-origin split: fit on a prefix, rank the remainder."""

    fit_size: int
    held_out_size: int
    excluded_non_finite: int
    best_held_out_reward: float
    scores: Mapping[str, SelectorScore]


@dataclass(frozen=True)
class BacktestReport:
    """Every split, plus per-selector means. Report `splits` alongside any
    mean: on the handful of splits a small history supports, one split can
    move a mean a long way."""

    selector_names: tuple[str, ...]
    splits: tuple[BacktestSplit, ...]

    def mean_top_1_regret(self, selector: str) -> float | None:
        """Mean regret across splits; lower is better. `None` if no split ran."""
        if not self.splits:
            return None
        return fmean(split.scores[selector].top_1_regret for split in self.splits)

    def mean_best_rank(self, selector: str) -> float | None:
        """Mean 1-based rank of a truly-best candidate; lower is better."""
        if not self.splits:
            return None
        return fmean(float(split.scores[selector].best_rank) for split in self.splits)


def _as_candidate(row: InterventionOutcome) -> Experiment:
    """A not-yet-run `Experiment` standing in for a historical row.

    Same `config_patch`, therefore the same arm, which is the only thing
    any of these selectors looks at. `status` is left at the default
    `PLANNED`: the backtest asks what a selector would have done *before*
    this experiment ran, so it must not be handed the answer.
    """
    return Experiment(
        experiment_id=row.experiment_id,
        parent_id=row.parent_id,
        hypothesis=Hypothesis("backtest", "backtest", row.intervention or "backtest"),
        config_patch=dict(row.config_patch),
        # Only needs to be positive; no selector here reads it, and the
        # row's own measured cost is already in the reward.
        estimated_gpu_hours=max(row.gpu_hours, _MIN_GPU_HOURS),
    )


def _as_history_pair(row: InterventionOutcome) -> tuple[Experiment, ExperimentResult]:
    """A `(Experiment, ExperimentResult)` history pair rebuilt from a row.

    `InterventionOutcome` keeps the result's real `metrics` and
    `gpu_hours` but not its `evidence`, so a goal with
    `require_protocol_match=True` will score these reconstructions
    `-inf` where the original scored fine. Both selectors see the identical
    reconstruction, so the *comparison* stays fair, but the absolute
    numbers under such a goal are not the numbers the original run got --
    see `backtest_selectors`' limitations.
    """
    return _as_candidate(row), ExperimentResult(
        experiment_id=row.experiment_id,
        metrics=dict(row.metrics),
        gpu_hours=row.gpu_hours,
    )


def _score_ordering(
    ordering: Sequence[Experiment],
    *,
    true_rewards: Mapping[str, float],
    best_reward: float,
) -> SelectorScore:
    ordered_ids = tuple(experiment.experiment_id for experiment in ordering)
    best_rank = next(
        index for index, name in enumerate(ordered_ids, start=1) if true_rewards[name] == best_reward
    )
    return SelectorScore(
        top_1_regret=best_reward - true_rewards[ordered_ids[0]],
        best_rank=best_rank,
        ordered_experiment_ids=ordered_ids,
    )


def backtest_selectors(
    outcomes: Sequence[InterventionOutcome],
    *,
    goal: Goal,
    baseline: ExperimentResult,
    selectors: Mapping[str, Selector] = SELECTORS,
    min_fit_size: int = 2,
    min_held_out_size: int = 2,
) -> BacktestReport:
    """Backtest every selector in *selectors* on held-out historical experiments.

    Protocol
    --------
    Rows arrive in registry order, which is the order the experiments were
    actually recorded (`build_intervention_outcomes` guarantees it), and
    the split is **chronological**: fit on `rows[:i]`, rank `rows[i:]`. A
    random split would let a selector learn from experiments that had not
    happened yet, which is exactly the mistake this harness exists to
    avoid. Every valid `i` is used (rolling origin) rather than one
    arbitrary cut, so no single lucky split decides the result.

    Each held-out row is turned back into a not-yet-run `Experiment`
    (`_as_candidate`) and the whole held-out batch is handed to each
    selector in the same registry order, with the same fit-set history,
    the same `goal` and the same `baseline`. Nothing is run, promoted, or
    gated; only the ordering is read back.

    Metric
    ------
    A selector's job is to decide what to spend the next GPU-hours on, so
    it is scored on its own first pick:

      `top_1_regret`  the best true reward available in the held-out set
                      minus the true reward of the candidate the selector
                      put first. Zero is perfect; lower is better; never
                      negative.
      `best_rank`     the 1-based position of the first truly-best
                      candidate in the ordering -- how many runs you would
                      have paid for before reaching a best one. 1 is
                      perfect; lower is better.

    "True reward" is the row's own `gate_score_vs_baseline / gpu_hours`:
    what that experiment really scored when it really ran, through the real
    gate. Held-out rows whose true reward is non-finite are excluded (there
    is no defined truth to rank them against) and counted in
    `BacktestSplit.excluded_non_finite`.

    Limitations -- read before quoting a number
    -------------------------------------------
    * **Counterfactual-free.** Ranking historical rows measures whether a
      selector would have run the good experiments *sooner*. It cannot
      measure what would have happened had a different experiment been run
      instead, because that experiment does not exist in the record. This
      is an ordering benchmark, not a policy-value estimate.
    * **Arms, not values.** Every selector here scores an arm, so two
      held-out rows touching the same key-paths are indistinguishable to
      all of them and their relative order is decided by input order alone.
      On a history where within-arm value choice is what mattered, all
      selectors will look alike, and correctly so.
    * **Small n.** A few dozen experiments give a few dozen splits with
      heavily overlapping fit sets. Means over them are not independent
      samples; read `splits` too.
    * **Protocol-matched goals.** See `_as_history_pair`: rebuilt results
      carry no evaluation evidence, so `require_protocol_match=True`
      handicaps every selector identically and makes absolute rewards
      unrepresentative.
    * **Ties.** `best_rank` credits the first candidate attaining the best
      reward, so exact ties are resolved by the ordering itself.

    Returns a report with no splits at all when *outcomes* is too short to
    honour `min_fit_size`/`min_held_out_size` -- an empty result, not a
    fabricated one.
    """
    rows = tuple(outcomes)
    names = tuple(selectors)
    splits: list[BacktestSplit] = []

    for fit_size in range(min_fit_size, len(rows) - min_held_out_size + 1):
        fit_rows = rows[:fit_size]
        held_out_rows = rows[fit_size:]

        true_rewards: dict[str, float] = {}
        scoreable: list[InterventionOutcome] = []
        for row in held_out_rows:
            reward = _reward(row.gate_score_vs_baseline, row.gpu_hours)
            if not isfinite(reward):
                continue
            true_rewards[row.experiment_id] = reward
            scoreable.append(row)
        if len(scoreable) < min_held_out_size:
            continue

        history = tuple(_as_history_pair(row) for row in fit_rows)
        candidates = tuple(_as_candidate(row) for row in scoreable)
        best_reward = max(true_rewards.values())

        splits.append(
            BacktestSplit(
                fit_size=fit_size,
                held_out_size=len(scoreable),
                excluded_non_finite=len(held_out_rows) - len(scoreable),
                best_held_out_reward=best_reward,
                scores={
                    name: _score_ordering(
                        selectors[name](
                            candidates, history=history, goal=goal, baseline=baseline
                        ),
                        true_rewards=true_rewards,
                        best_reward=best_reward,
                    )
                    for name in names
                },
            )
        )

    return BacktestReport(selector_names=names, splits=tuple(splits))
