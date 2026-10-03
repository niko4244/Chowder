"""A real candidate screening job through the Kaggle lane (acceptance run 3).

Unlike the two probe runs (acceptance runs 1–2), the kernel command here is a
TRAINING job that consumes the compiled ``campaign_spec`` the provider ships
with the kernel: the operator trainer reads the spec's ``recipe_patch`` and
uses its values as the actual training hyperparameters (learning rate, batch
size, steps, seed), trains a real model on the session's GPU for real steps,
and reports the outcome — final loss, loss delta, and the hyperparameters it
applied (echoed back so the record proves the spec was honored, not ignored).

This is the operator-trainer seam made concrete at screening scale: a real
candidate's quality signal (did training improve against the spec's
falsification rule) produced on the free lane. It is NOT yet the full growth
campaign stack (no curriculum planner, no independent evaluation tier); the
trainer is deliberately small and honest about that.

Usage:
    python kaggle/first_candidate_screening.py --commit <40-hex-sha> \
        [--learning-rate 0.003] [--steps 300] [--poll-seconds 30]

Free-lane discipline: quota is reconciled and gated before the push; the run
costs a fraction of one percent of the operator's weekly 30 device-hours.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from provider_first_push import REPO_URL, head_commit, resolve_username  # noqa: E402

RESULT_MARKER = "CHOWDER_RESULT_JSON:"

#: The operator trainer, parameterized BY the campaign spec. It refuses to run
#: against a spec it cannot parse (a trainer that silently ignored the recipe
#: would train something nobody declared) and echoes every applied value into
#: the result so the acceptance record can prove the spec drove the run.
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
batch_size = int(training.get("batch_size", 32))
steps = int(training.get("max_steps", 0) or training.get("epochs", 0) or 200)
seed = int((spec.get("replication") or {}).get("seed", 2026))
if steps <= 0:
    raise SystemExit("TRAINER_REFUSED: the spec declares no usable step count")
if lr <= 0:
    raise SystemExit("TRAINER_REFUSED: the spec declares a non-positive learning rate")

torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A real (small) supervised problem: fit a random fixed target function so
# loss must fall if the optimizer runs with the spec's hyperparameters.
dim, hidden = 32, 64
torch.manual_seed(seed + 1)
W = torch.randn(dim, 1, device=device)
dataset = torch.randn(4096, dim, device=device)
labels = torch.tanh(dataset @ W / math.sqrt(dim)) * 3.0

model = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def batch():
    idx = torch.randint(0, dataset.shape[0], (batch_size,), device=device)
    return dataset[idx], labels[idx]

(torch.cuda.synchronize() if device.type == "cuda" else None)
t0 = time.monotonic()
first_loss = None
final_loss = None
model.train()
for step in range(steps):
    x, y = batch()
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    if first_loss is None:
        first_loss = float(loss.item())
    final_loss = float(loss.item())
(torch.cuda.synchronize() if device.type == "cuda" else None)
wall_seconds = time.monotonic() - t0

accelerators = torch.cuda.device_count() if torch.cuda.is_available() else 0
count = accelerators if accelerators > 0 else 1
result = {
    "status": "complete",
    "trainer": "spec-driven-sgd-probe",
    "spec_honored": {
        "learning_rate": lr,
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
    },
    "first_loss": round(first_loss, 6),
    "final_loss": round(final_loss, 6),
    "loss_delta": round(final_loss - first_loss, 6),
    "loss_improved": bool(final_loss < first_loss),
    "steps_run": steps,
    "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0) if accelerators else "none",
    "wall_seconds": round(wall_seconds, 3),
    "wall_gpu_hours": round(wall_seconds / 3600.0, 6),
    "device_gpu_hours": round(wall_seconds / 3600.0 * count, 6),
    "accelerator_count": count,
    "metering": "measured_wall_clock_x_attached_accelerators",
    "torch": torch.__version__,
}
with open('/kaggle/working/chowder_result.json', 'w', encoding='utf-8') as fh:
    json.dump(result, fh, indent=2, sort_keys=True)
print('CHOWDER_RESULT_JSON:' + json.dumps(result), flush=True)
PYEOF"""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    args = parser.parse_args(argv)

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
                                          / "chowder-screening-acceptance" / stamp))
    workdir = Path(args.workdir or (state_root / "kernels"))
    experiment_id = f"candidate-screening-{stamp}"

    # The compiled campaign_spec, exactly as the ExperimentCompiler emits it
    # (recipe_patch in backend.* key form): the trainer consumes the SAME
    # shape the production path would.
    campaign_spec = {
        "recipe_patch": {
            "training": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_steps": args.steps,
            },
        },
        "data": {"source_kinds": ["screening-acceptance-synthetic"]},
        "evaluations": {"target_surfaces": ["loss_improvement"],
                        "transfer_surfaces": []},
        "replication": {"plan": "single screening pass", "seed": 2026},
        "controls": [],
        "falsification_rule": "loss_delta >= 0",
    }

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
        proposal_id=f"prop-{experiment_id}",
        hypothesis_id="screening-acceptance-candidate",
        campaign_spec=campaign_spec,
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
        honored.get("learning_rate") == args.learning_rate
        and honored.get("batch_size") == args.batch_size
        and honored.get("steps") == args.steps
    )
    loss_delta = trainer.get("loss_delta")
    falsified = None if loss_delta is None else bool(loss_delta >= 0.0)  # spec: loss_delta >= 0 → falsified

    # Evidence: a run-grounded observation through durable research memory.
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
        Measurement(surface="efficiency:trainer_wall_seconds",
                    benchmark="screening-acceptance",
                    value=float(trainer.get("wall_seconds", 0.0))),
    ]
    if "final_loss" in trainer:
        measurements.append(Measurement(
            surface="loss_improvement", benchmark="screening-acceptance",
            value=float(trainer["final_loss"])))
    observation = ExperimentObservation(
        observation_id=f"obs-{experiment_id}",
        run_id=run_id,
        experiment_ref=request.experiment_id,
        proposal_id=request.proposal_id,
        hypothesis_id=request.hypothesis_id,
        measurements=tuple(measurements),
        status="complete" if trainer.get("status") == "complete" else "failed",
        wall_gpu_hours=float(trainer.get("device_gpu_hours", 0.0)),
        notes=("candidate screening acceptance: compiled campaign_spec executed "
               "by the operator trainer on the Kaggle lane; spec honored: "
               f"{json.dumps(honored)}; falsification rule {campaign_spec['falsification_rule']}"
               f" → {'FALSIFIED' if falsified else 'survived'}"),
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
        "falsification": {"rule": campaign_spec["falsification_rule"],
                          "falsified": falsified},
        "trainer": trainer,
        "transitions": transitions,
        "observation_id": observation.observation_id,
        "state_root": str(state_root),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
