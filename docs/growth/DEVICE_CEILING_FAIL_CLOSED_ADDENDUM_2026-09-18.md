# Addendum — a device ceiling could be satisfied without measuring device time (2026-09-18)

Correction to the integrity pass of 2026-09-17. That pass added actual-cost
settlement and claimed, in its own adversarial table, that "a wall figure
cannot be compared to a device ceiling". The claim was false. This note records
the counterexample, the fix at its owner, the proof, and the one place the
same hazard is deliberately left standing with a name on it.

Nothing historical is rewritten: the 2026-09-17 report and assessment keep
their text and carry dated corrections pointing here.

## The defect, reproduced

```
ComputeCost.from_wall_only(0.43)          # device 0.0, "device time not separated"
settle_cost(actual=..., device_ceiling=0.30, wall_ceiling=None)
  -> compliant=True
```

`from_wall_only` documents that device `0.0` means *not separated*, never
*device free* — and the very next function in the same file read it as device
free. Nothing in the data model recorded whether a zero was an observation or
a placeholder, so any caller could settle a device ceiling against a figure
that was never measured, and the wall figure standing in for the run did not
make the comparison "separation of units": the *units* were never the problem,
the *placeholder* was.

The real Gen-1 artifact is exactly this shape. Every entry in
`cycle_compute_accounting.json` for `gen1-protocol-compliance` reports device
`0.0` except one evaluation leg at `0.005039`; the wall total is `1.3526`.
A device ceiling declared over that ledger could not fail, at any value.

## The fix, at the owner

`src/chowder/growth/compute_cost.py`:

- `ComputeCost.device_measured: bool = False` — fail-closed default. A caller
  that never separated device time has an unmeasured zero, and must not be
  read otherwise.
- `ComputeCost.measured(device_gpu_hours=..., wall_gpu_hours=..., source=...)`
  — the explicit way to say "this device figure was observed", which includes
  an observed `0.0` (a CPU-only or idle-device run).
- `ComputeCost.from_wall_only` pins `device_measured=False`.
- `ComputeCost.plus` and the ledger's `total`/`total_for_recipe` carry the
  conjunction: a sum is a device measurement only if every contributor was.
- `to_dict`/`from_dict` round-trip the flag, and a record written before the
  field existed reads back **unmeasured** — a stored row does not become a
  device measurement by being read again.
- `settle_cost` refuses the comparison with the new machine-readable constant
  `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` when a device ceiling is declared
  against an unmeasured figure. `ACTUAL_DEVICE_GPU_HOURS_EXCEEDED` still
  reports a real measured overrun, and the two are never confused.

## Which ceiling the binding settles, and where the device ceiling lives

`SubprocessTrainingFn._settle` now decides by measurement, not by assumption:

| Case | Device ceiling | Recorded as |
| --- | --- | --- |
| Run reported no device time (today's trainer: wall only) | enforced against the recipe's projected plan at **admission**; never handed to settlement | `ceiling_enforcement.device = "admission:projected_plan"` |
| Run reported device time for every leg that contributed cost | settled post-run against the measured figure | `ceiling_enforcement.device = "settlement:measured"` |

Wall and project ceilings are always settled from the run's own measurements.
`ceiling_enforcement` exists so no reader can mistake an admission constraint
for a certified post-run number.

**The decision not taken, and why.** The alternative was to hand the declared
device ceiling to settlement unconditionally and let it refuse whenever device
time is unmeasured. That is fail-closed in the loudest possible way, and it
would make every real attempt refuse *after* burning its compute: no component
of the trainer, the registry, or the evaluation path reports device time today
(verified by searching `src/chowder/` outside `growth/`). Refusing the
configuration everywhere is not a stronger rule; it is a rule that would have
been turned off within a day. Instead the ceiling keeps its admission teeth
against the projected plan, the settlement boundary only ever judges what was
measured, and the gap between the two is written into the evidence.

## Proof

Unit (`tests/test_growth_budget_settlement.py`), all three unmeasured shapes —
wall-only, explicit zero, and a legacy record read back from disk — have
`device_measured=False`, fail a declared device ceiling, and fail it with
`ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` rather than the overrun identifier. An
observed `0.0` satisfies the same ceiling, and a measured overrun fails with
`ACTUAL_DEVICE_GPU_HOURS_EXCEEDED`.

End to end through the binding's real post-run path
(`tests/test_growth_training_binding.py`, which previously covered only the
pre-launch preflight):

1. `test_a_successful_train_that_overruns_its_wall_ceiling_settles_as_refused`
   — training succeeds, the run's wall cost exceeds the frozen ceiling, the
   attempt comes back `REFUSED` with `refused_by="budget_settlement"`, the
   machine-readable identifier in `budget_settlement.budget_failure_reasons`,
   and the artifact and evidence preserved on disk.
2. `test_a_declared_device_ceiling_is_not_settled_against_an_unmeasured_figure`
   — a wall-only run never manufactures device compliance; the ceiling is
   recorded as admission-owned.
3. `test_a_measured_device_figure_lets_the_device_ceiling_settle` — when the
   run does separate device time, the device ceiling refuses a measured
   overrun.

## Consequences

- The Gen-1 accounting artifact now settles **device-uncertified** rather than
  device-compliant: `chowder growth campaign settle` reads the artifact's own
  `device_measured` flag (absent in records written before today, so `False`)
  and reports the device ceiling as unmeasurable. The wall and project numbers
  are unchanged.
- A campaign or prereg that declares a device ceiling while running a
  wall-reporting trainer is declaring an admission constraint, not a settled
  one. Declaring it remains useful; expecting settlement from it does not.
- **Known gap, named rather than hidden:** `PromotionInput` carries
  `actual_device_gpu_hours` as a plain float and applies the same comparison a
  second time, inline. An unmeasured device figure passed there still reads as
  under-ceiling. This pass fixed the settlement owner the counterexample came
  from and did not touch the promotion surface, because the recorded Gen-1
  re-adjudication consumes it and re-deriving that verdict is a separate,
  evidence-changing decision. It is the same defect class and belongs in the
  campaign/promotion consolidation pass.
