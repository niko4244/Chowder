# Gen-2 Preregistration — Amendment 7 (2026-09-18)

Written before any gen2 compute. It changes no threshold, no benchmark set, no
budget, no stopping rule and no verdict class. It closes three ways the run path
could still spend or record something it could not account for, and it makes two
more inputs knowable before compute.

## A. An evaluation that reports no cost is refused, not charged zero

`CandidateEvaluation.cost` was optional, a bare `EvalReport` from the seam was
accepted, and the runner charged `evaluation.cost or ComputeCost.zero(...)`. An
evaluator that reported nothing therefore settled as a **zero-cost** leg: real
candidate-evaluation compute would disappear from the campaign's accounting and
the campaign could remain inside a budget it had actually exceeded.

From this amendment:

* the seam must return a `CandidateEvaluation`; a bare `EvalReport` refuses with
  `CANDIDATE_EVALUATION_COST_UNREPORTED`, naming why;
* `CandidateEvaluation.cost is None` refuses with the same reason;
* a zero that names no measurement method refuses with
  `CANDIDATE_EVALUATION_COST_UNMEASURED` — `ComputeCost.zero(source=...)` is
  indistinguishable from an unreported cost, so an explicit, stated zero
  (`measurement_method=...`) is the only legal way to charge nothing;
* the runner charges the validated figure, never a default.

The rule this enforces is the same one the mission states for every other leg:
all compute the cycle consumed is accounted for. A candidate evaluation is
compute.

## B. Readiness runs before training, and a post-training refusal is recorded

The production evaluator was built *after* training and selection, so a campaign
could spend its training budget and only then discover that no instrument existed
to measure what it had just produced. Worse, the refusal was raised, so the
accounting that had accumulated in memory was never written: a run that had
really consumed GPU time left no durable record of it.

From this amendment the run has a **readiness phase**, and it runs before any
recipe is admitted to the executor:

| checked before compute | what a failure does |
| --- | --- |
| the declared parent and trusted-ancestor arms exist, parse, and carry measured rows | refuses |
| an evaluator can be built (declared evaluation material present, protocol renderable) | refuses |
| that evaluator covers every declared benchmark, its datasets exist, and each holds at least the declared slice size | refuses |
| the campaign projection fits the declared envelope | refuses (unchanged) |

`SubprocessEvaluationFn.admit(...)` is the evaluator's own admission seam, the
mirror of `SubprocessTrainFn.admit(...)`: the runner asks before it trains.

And a refusal that happens **after** training — a failed evaluation, an
unverifiable arm, a digest that no longer matches — no longer propagates as an
exception. The run writes its accounting (with the evaluation leg charged, when
the evaluation got as far as costing something), the compact per-attempt facts,
the selection, the reason and `campaign-run.json`, and returns a `REFUSED`
record. `CampaignRun` gains `attempts`, `refused_by` and `refusal_reason`; the
CLI exits non-zero for a refused run while still printing that record.

## C. The selected artifact's bytes are re-verified at both boundaries

The digest recorded by training was trusted until it was compared, and it was
compared only against other recorded strings. Bytes could move between training
and evaluation, or between evaluation and the verdict, and the run would certify
the recorded digest while the frozen judge recomputed a different one from disk
— the exact class of divergence the certification boundary exists to prevent.

From this amendment the run re-derives the selected artifact's digest from disk
immediately before it is measured, and again immediately before the judged
evidence set is written and a verdict is bound to it. A mismatch refuses with
`CANDIDATE_ARTIFACT_DIGEST_STALE`, records the spend, and writes no promotion.

## D. Two more things the run can know before it trains

* `parent_eval_report_path` joins the declared inputs the `run` phase requires.
  The promotion rule compares the candidate against its parent, so a campaign
  without that arm could only ever reach INCONCLUSIVE; it now says so before
  compute rather than after.
* The trusted-ancestor arm is required by the readiness phase. Without it branch
  protection is permanently undecidable, so a campaign that cannot be certified
  must not spend a training budget to find out. The gate semantics are
  unchanged: an absent or off-protocol ancestor is still `UNKNOWN` at
  certification, never a pass.

Additionally, the candidate arm's evidence must live **inside** the run root. A
row naming an absolute path is refused: a verdict that depends on a directory
somebody else can move or delete is not durable evidence.

## E. What this amendment does not claim

Nothing here is a measured gen2 result and no gen2 compute was run to write it.
The target instrument's diagnostic metadata (the judge's T1–T10) still exists
only in the historical Gen-1 driver, so T1–T10 remain `UNKNOWN`; the declaration
`docs/gen2/gen2_campaign.json` still declines to invent the eight inputs it
lacks, and both entry points still refuse before compute, naming every missing
input at once.
