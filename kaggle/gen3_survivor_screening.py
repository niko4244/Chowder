"""Prepare gen3-a1-reasoning's REAL screening job through the free Kaggle lane.

The survivor that the growth loop auto-consumed (dry-run proof,
``scripts/dryrun_survivor_handoff.py``) is frozen as
``gen3-a1-reasoning``: target ``reasoning`` -> ``mgsm@2022-11``, treatment
``data``, intervention "decay replay ratio 0.25 -> 0.05 after convergence",
survivor rule "transfer delta <= 0 on 2 of 3 seeds".

This runner derives the compiled ``campaign_spec`` FROM the frozen durable
artifacts (campaign.json + preregistration.json) — nothing hand-typed — and
ships it to the same operator-trainer seam acceptance run 3 proved
(``first_candidate_screening.py``), one step further: a 3-seed A/B screening
probe that actually instantiates the survivor's rule.

- control arm: constant replay ratio 0.25
- intervention arm: replay ratio 0.25 -> 0.05 at the spec's decay point
- per seed, transfer_delta = intervention_final_loss - control_final_loss
  (negative = the intervention trained better); the survivor's rule is
  satisfied on a seed iff transfer_delta <= 0, and the rule needs 2 of 3
  seeds — falsified otherwise. The runner RECOMPUTES the verdict from the
  per-seed deltas; it never trusts the trainer's own claim.

Modes (no flag = derive + print the spec, touch nothing):
  --print-spec     derive, validate and print the campaign_spec; exit
  --selftest-local execute the EXACT trainer body locally on CPU with a
                   30-step spec — proves the trainer consumes the spec and
                   evaluates the rule, without any Kaggle contact
  --launch         the real thing: quota gate -> kernel push -> poll ->
                   settle the observation into durable research memory.
                   Launching starts a real free-tier run; it is an explicit
                   operator action.

Free-lane discipline: ~6 x 300-step probes on T4x2, a fraction of one
percent of the weekly 30 device-hours (Run 3 measured 0.0071 for 300 steps).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_FROZEN_DIR = Path(__file__).resolve().parents[1] / "tmp" / "dryrun-survivor-handoff" / "gen3-a1-reasoning"
RESULT_MARKER = "CHOWDER_RESULT_JSON:"

#: The operator trainer, parameterized BY the gen3 campaign spec: the replay
#: ratios, seeds, steps, lr and batch all come from the spec, and the trainer
#: refuses a spec it cannot honor (the Run-3 lesson: prove spec_drove_run by
#: echoing every applied value back).
TRAINER_COMMAND = """python - <<'PYEOF'
import json, math, os, time
import torch
import torch.nn as nn

spec_path = os.environ["CHOWDER_CAMPAIGN_SPEC"]
with open(spec_path, encoding="utf-8") as fh:
    spec = json.load(fh)

recipe = spec.get("recipe_patch") or {}
training = recipe.get("training") or {}
lr = float(training.get("learning_rate", 0.001))
batch_size = int(training.get("batch_size", 64))
steps = int(training.get("max_steps", 0))
seeds = [int(s) for s in (spec.get("replication") or {}).get("seeds", [])]
r_start = float(training.get("replay_ratio_start", -1))
r_decay = float(training.get("replay_ratio_decayed", -1))
decay_frac = float(recipe.get("decay_point_fraction", 0.5))
gate = spec.get("screening_gate") or {}
min_seeds = int(gate.get("min_seeds_satisfied", 2))
if not (0.0 <= r_start <= 1.0) or not (0.0 <= r_decay <= 1.0):
    raise SystemExit("TRAINER_REFUSED: replay ratios must be in [0, 1]")
if steps <= 0 or lr <= 0 or not seeds or not (0 < min_seeds <= len(seeds)):
    raise SystemExit("TRAINER_REFUSED: the spec declares no usable steps/lr/seeds/gate")
decay_at = int(steps * decay_frac)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dim, hidden = 32, 64
BUFFER_CAP = 4096

def run_arm(seed: int, mode: str) -> float:
    # One real training run of the synthetic screening problem. Both arms get
    # identical data, init and fresh-batch draws (same seed); they differ ONLY
    # in the replay-ratio schedule the spec declares.
    torch.manual_seed(seed)
    torch.manual_seed(seed + 1)
    W = torch.randn(dim, 1, device=device)
    dataset = torch.randn(4096, dim, device=device)
    labels = torch.tanh(dataset @ W / math.sqrt(dim)) * 3.0
    model = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buf_x = torch.empty(0, dim, device=device)
    buf_y = torch.empty(0, 1, device=device)
    final = None
    model.train()
    for step in range(steps):
        ratio = r_start if (mode == "control" or step < decay_at) else r_decay
        idx = torch.randint(0, dataset.shape[0], (batch_size,), device=device)
        fresh_x, fresh_y = dataset[idx], labels[idx]
        x, y = fresh_x, fresh_y
        k = int(round(ratio * batch_size))
        if buf_x.shape[0] > 0 and k > 0:
            take = min(k, buf_x.shape[0])
            sel = torch.randint(0, buf_x.shape[0], (take,), device=device)
            x = torch.cat([x, buf_x[sel]], 0)
            y = torch.cat([y, buf_y[sel]], 0)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        buf_x = torch.cat([buf_x, fresh_x], 0)[-BUFFER_CAP:]
        buf_y = torch.cat([buf_y, fresh_y], 0)[-BUFFER_CAP:]
        final = float(loss.item())
    return final

(torch.cuda.synchronize() if device.type == "cuda" else None)
t0 = time.monotonic()
per_seed = []
for seed in seeds:
    control = run_arm(seed, "control")
    intervention = run_arm(seed, "intervention")
    delta = round(intervention - control, 6)
    per_seed.append({
        "seed": seed,
        "control_final_loss": round(control, 6),
        "intervention_final_loss": round(intervention, 6),
        "transfer_delta": delta,
        "rule_satisfied_on_seed": bool(delta <= 0.0),
    })
(torch.cuda.synchronize() if device.type == "cuda" else None)
wall_seconds = time.monotonic() - t0

satisfied = sum(1 for s in per_seed if s["rule_satisfied_on_seed"])
accelerators = torch.cuda.device_count() if torch.cuda.is_available() else 0
count = accelerators if accelerators > 0 else 1
result = {
    "status": "complete",
    "trainer": "gen3-survivor-ab-probe",
    "spec_honored": {
        "learning_rate": lr,
        "batch_size": batch_size,
        "steps": steps,
        "seeds": seeds,
        "replay_ratio_start": r_start,
        "replay_ratio_decayed": r_decay,
        "decay_point_fraction": decay_frac,
    },
    "falsification_rule": spec.get("falsification_rule"),
    "screening_gate": {"min_seeds_satisfied": min_seeds, "seeds_rule_satisfied": satisfied},
    "per_seed": per_seed,
    "mean_transfer_delta": round(sum(s["transfer_delta"] for s in per_seed) / len(per_seed), 6),
    "falsified": bool(satisfied < min_seeds),
    "steps_run_per_arm": steps,
    "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0) if accelerators else "none",
    "wall_seconds": round(wall_seconds, 3),
    "device_gpu_hours": round(wall_seconds / 3600.0 * count, 6),
    "accelerator_count": count,
    "metering": "measured_wall_clock_x_attached_accelerators",
    "torch": torch.__version__,
}
result_path = os.environ.get("CHOWDER_RESULT_PATH", "/kaggle/working/chowder_result.json")
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2, sort_keys=True)
print('CHOWDER_RESULT_JSON:' + json.dumps(result), flush=True)
PYEOF"""


def derive_spec(frozen_dir: Path, *, steps: int, batch_size: int,
                seeds: list[int]) -> tuple[dict, dict]:
    """Derive the campaign_spec from the frozen gen3 artifacts, fail-closed."""
    campaign_path = frozen_dir / "campaign.json"
    prereg_path = frozen_dir / "preregistration.json"
    for path in (campaign_path, prereg_path):
        if not path.exists():
            raise SystemExit(f"REFUSING: frozen artifact missing: {path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    target = prereg.get("target") or {}
    reason = str(target.get("treatment_reason") or "")
    skill = str(target.get("target_skill") or "")
    benchmarks = [str(b) for b in (target.get("target_benchmarks") or [])]
    cycle_id = str(campaign.get("cycle_id") or prereg.get("cycle_id") or "")
    if not (skill and benchmarks and cycle_id and reason):
        raise SystemExit("REFUSING: the frozen declaration lacks target fields")

    m = re.search(r"replay ratio ([0-9.]+) -> ([0-9.]+)", reason)
    if not m:
        raise SystemExit(
            f"REFUSING: no 'replay ratio X -> Y' intervention in {cycle_id}'s "
            f"treatment_reason; refusing to invent one")
    r_start, r_decayed = float(m.group(1)), float(m.group(2))
    m = re.search(r"\(proposal ([^,]+), falsification: (.+?), survivor score ([0-9.]+)\)", reason)
    if not m:
        raise SystemExit(
            f"REFUSING: no 'proposal …, falsification: …, survivor score …' "
            f"block in {cycle_id}'s treatment_reason")
    proposal_id, rule_verbatim, survivor_score = (m.group(1).strip(),
                                                  m.group(2).strip(),
                                                  float(m.group(3)))
    recipes = [str(r) for r in (campaign.get("recipes") or [])]
    lr_match = re.search(r"lr([0-9.]+)", recipes[0]) if recipes else None
    if not lr_match:
        raise SystemExit(
            f"REFUSING: no 'lr<value>' in the frozen recipe identity "
            f"{recipes!r}; refusing to guess a learning rate")
    learning_rate = float(lr_match.group(1))

    spec = {
        "recipe_patch": {
            "training": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_steps": steps,
                "replay_ratio_start": r_start,
                "replay_ratio_decayed": r_decayed,
            },
            "decay_point_fraction": 0.5,
        },
        "data": {"source_kinds": ["gen3-survivor-screening-synthetic"]},
        "evaluations": {"target_surfaces": ["loss_improvement"],
                        "transfer_surfaces": []},
        "replication": {"plan": "3-seed A/B screening pass", "seeds": seeds},
        "controls": ["constant-replay control arm at replay_ratio_start"],
        "falsification_rule": rule_verbatim,
        "screening_gate": {
            "metric": "transfer_delta = intervention_final_loss - control_final_loss",
            "seed_satisfied_iff": "transfer_delta <= 0",
            "min_seeds_satisfied": 2,
            "note": ("screening-scale operationalization of the survivor's "
                     "verbatim rule; the campaign tier re-runs it on the real "
                     "benchmark"),
        },
        "provenance": {
            "cycle_id": cycle_id,
            "parent_version": str(campaign.get("parent_version") or target.get("parent_version") or ""),
            "target_skill": skill,
            "target_benchmarks": benchmarks,
            "suggested_training_type": str(target.get("suggested_training_type") or ""),
            "proposal_id": proposal_id,
            "survivor_score": survivor_score,
            "expected_cost_gpu_hours": target.get("expected_cost_gpu_hours"),
            "frozen_recipes": recipes,
            "source": "frozen gen3 declaration from scripts/dryrun_survivor_handoff.py",
        },
    }
    provenance = spec["provenance"]
    return spec, provenance


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--commit", default=None,
                        help="chowder commit the kernel pip-installs (default: "
                             "worktree HEAD; must be on origin)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--selftest-local", action="store_true")
    parser.add_argument("--launch", action="store_true",
                        help="actually push the kernel (starts a real free-tier run)")
    args = parser.parse_args(argv)

    spec, provenance = derive_spec(Path(args.frozen_dir), steps=args.steps,
                                   batch_size=args.batch_size, seeds=args.seeds)
    if args.print_spec or (not args.launch and not args.selftest_local):
        print(json.dumps({"step": "spec-derived", "campaign_spec": spec,
                          "launch_hint": "re-run with --launch to push the kernel"},
                         indent=2, default=str))
        return 0

    if args.selftest_local:
        import os
        import tempfile
        import importlib.util
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "campaign_spec.json"
            result_path = Path(td) / "chowder_result.json"
            small = json.loads(json.dumps(spec))
            small["recipe_patch"]["training"]["max_steps"] = 30
            spec_path.write_text(json.dumps(small), encoding="utf-8")
            body = TRAINER_COMMAND.split("<<'PYEOF'\n", 1)[1].rsplit("PYEOF", 1)[0]
            env = {**os.environ,
                   "CHOWDER_CAMPAIGN_SPEC": str(spec_path),
                   "CHOWDER_RESULT_PATH": str(result_path)}
            saved = dict(os.environ)
            os.environ.update(env)
            try:
                exec(compile(body, "trainer", "exec"), {"__name__": "__main__"})
            finally:
                os.environ.clear()
                os.environ.update(saved)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        honored = result["spec_honored"]
        drove = (honored["replay_ratio_start"] == 0.25
                 and honored["replay_ratio_decayed"] == 0.05
                 and honored["steps"] == 30 and honored["seeds"] == args.seeds)
        satisfied = sum(1 for s in result["per_seed"] if s["rule_satisfied_on_seed"])
        recomputed_falsified = satisfied < 2
        print(json.dumps({
            "step": "selftest", "spec_drove_run": drove,
            "trainer_falsified": result["falsified"],
            "runner_recomputed_falsified": recomputed_falsified,
            "verdicts_agree": bool(result["falsified"] == recomputed_falsified),
            "mean_transfer_delta": result["mean_transfer_delta"],
            "per_seed": result["per_seed"],
        }, indent=2))
        return 0 if (drove and result["falsified"] == recomputed_falsified) else 1

    # --launch: the real free-tier run (same provider flow as Run 3).
    from provider_first_push import REPO_URL, head_commit, resolve_username
    commit = head_commit(args.commit)
    if len(commit) != 40:
        print(f"REFUSING: --commit must be a 40-character sha, got {commit!r}",
              file=sys.stderr)
        return 2
    username = resolve_username()
    if not username:
        print("REFUSING: no Kaggle username resolvable", file=sys.stderr)
        return 2

    import tempfile
    stamp = time.strftime("%Y%m%d-%H%M%S")
    state_root = Path(args.state_root or (Path(tempfile.gettempdir())
                                          / "chowder-gen3-screening" / stamp))
    workdir = Path(args.workdir or (state_root / "kernels"))
    experiment_id = f"gen3-survivor-screening-{stamp}"

    from chowder.scientist.compute import (
        ExperimentClass, ExperimentRequest, KaggleProvider,
    )
    provider = KaggleProvider(
        username=username,
        chowder_commit=commit,
        kernel_command=TRAINER_COMMAND,
        repo_url=REPO_URL,
        workdir=str(workdir),
        weekly_gpu_hours=20.0,
    )
    quota = provider.sync_quota_from_api()
    print(json.dumps({"step": "quota", "weekly_gpu_hours": quota.weekly_gpu_hours,
                      "used_gpu_hours": quota.used_gpu_hours,
                      "remaining_gpu_hours": quota.remaining_gpu_hours()}))

    request = ExperimentRequest(
        experiment_id=experiment_id,
        proposal_id=provenance["proposal_id"],
        hypothesis_id=f"survivor-{provenance['cycle_id']}",
        campaign_spec=spec,
        experiment_class=ExperimentClass.SCREENING,
        estimated_gpu_hours=0.05,
    )
    cost = provider.estimate_cost(request)
    if quota.remaining_gpu_hours() < cost:
        print(f"REFUSING: needs {cost} device GPU-hours, "
              f"{quota.remaining_gpu_hours()} remain", file=sys.stderr)
        return 2

    submission = provider.submit(request)
    print(json.dumps({"step": "submitted", "provider_ref": submission.provider_ref,
                      "estimated_device_gpu_hours": submission.device_gpu_hours}))

    deadline = time.time() + args.timeout_seconds
    transitions: list[dict] = []
    seen: set[str] = set()
    while time.time() < deadline:
        submission = provider.poll(submission)
        if submission.status not in seen:
            seen.add(submission.status)
            transitions.append({"status": submission.status,
                                "at_seconds": round(args.timeout_seconds
                                                    - (deadline - time.time()), 1)})
            print(json.dumps({"step": "poll", "status": submission.status}))
        if submission.status in ("complete", "failed"):
            break
        time.sleep(args.poll_seconds)
    else:
        print(json.dumps({"step": "timeout", "last_status": submission.status}),
              file=sys.stderr)
        return 3

    if submission.status != "complete":
        print(json.dumps({"step": "failed", "result": submission.result}, default=str))
        return 1

    trainer = submission.result.get("chowder_result") or {}
    honored = trainer.get("spec_honored") or {}
    spec_drove_run = (
        honored.get("learning_rate") == spec["recipe_patch"]["training"]["learning_rate"]
        and honored.get("batch_size") == spec["recipe_patch"]["training"]["batch_size"]
        and honored.get("steps") == spec["recipe_patch"]["training"]["max_steps"]
        and honored.get("seeds") == spec["replication"]["seeds"]
        and honored.get("replay_ratio_start") == spec["recipe_patch"]["training"]["replay_ratio_start"]
        and honored.get("replay_ratio_decayed") == spec["recipe_patch"]["training"]["replay_ratio_decayed"]
    )
    # Recompute the verdict from the measured per-seed deltas — never trust
    # the trainer's own claim.
    per_seed = trainer.get("per_seed") or []
    satisfied = sum(1 for s in per_seed if s.get("transfer_delta", 1.0) <= 0.0)
    recomputed_falsified = satisfied < spec["screening_gate"]["min_seeds_satisfied"]
    falsified = trainer.get("falsified")
    verdicts_agree = (falsified is None or bool(falsified) == recomputed_falsified)

    output_dir = Path(submission.result["output_dir"])
    run_id = str(output_dir)
    from chowder.scientist.observation import ExperimentObservation, Measurement
    from chowder.scientist.research_memory import ResearchMemory
    memory = ResearchMemory(
        state_root / "research",
        run_exists=lambda r: Path(r).exists(),
        run_complete=lambda r: (Path(r) / "chowder_result.json").exists(),
    )
    measurements = [
        Measurement(surface="screening:mean_transfer_delta",
                    benchmark=f"{provenance['cycle_id']}-ab-probe",
                    value=float(trainer.get("mean_transfer_delta", 0.0))),
    ]
    for s in per_seed:
        measurements.append(Measurement(
            surface="screening:transfer_delta",
            benchmark=f"{provenance['cycle_id']}-ab-probe",
            value=float(s["transfer_delta"])))
    observation = ExperimentObservation(
        observation_id=f"obs-{experiment_id}",
        run_id=run_id,
        experiment_ref=request.experiment_id,
        proposal_id=request.proposal_id,
        hypothesis_id=request.hypothesis_id,
        measurements=tuple(measurements),
        status="complete" if trainer.get("status") == "complete" else "failed",
        wall_gpu_hours=float(trainer.get("device_gpu_hours", 0.0)),
        notes=("gen3 survivor A/B screening: spec derived from the frozen "
               f"{provenance['cycle_id']} declaration (target "
               f"{provenance['target_skill']} -> {provenance['target_benchmarks']}); "
               f"rule {spec['falsification_rule']!r}: {satisfied}/{len(per_seed)} "
               f"seeds satisfied -> {'FALSIFIED' if recomputed_falsified else 'survived'}; "
               "screening-scale synthetic probe, NOT a mgsm@2022-11 measurement"),
        provider=provider.name,
        hardware_class=submission.hardware_class,
    )
    memory.record_observation(observation)

    summary = {
        "step": "complete",
        "commit": commit,
        "provider_ref": submission.provider_ref,
        "hardware_class": submission.hardware_class,
        "device_gpu_hours_measured": submission.device_gpu_hours,
        "spec_drove_run": spec_drove_run,
        "verdicts_agree": verdicts_agree,
        "falsification": {"rule": spec["falsification_rule"],
                          "seeds_satisfied": satisfied,
                          "seeds_required": spec["screening_gate"]["min_seeds_satisfied"],
                          "falsified": recomputed_falsified},
        "trainer": trainer,
        "transitions": transitions,
        "observation_id": observation.observation_id,
        "state_root": str(state_root),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
