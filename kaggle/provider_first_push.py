"""One real end-to-end screening run through the KaggleProvider push path.

This is the provider-path acceptance run (docs/KAGGLE_PROVIDER_ACCEPTANCE.md):
pinned-commit install → BackendFingerprint → campaign-spec persistence →
operator command on the real T4×2 → chowder_result.json → kernels_output →
a run-grounded ExperimentObservation recorded into durable research memory.

What it deliberately is NOT: a full training campaign. The kernel command is
a small, real, measured GPU workload (fp16 matmul) — enough to prove the
plumbing end to end and to produce one hardware-scoped efficiency number,
without spending meaningful weekly quota.

Usage:
    python kaggle/provider_first_push.py --commit <40-hex-sha> \
        [--state-root DIR] [--workdir DIR] [--poll-seconds 30] [--timeout 3600]

The commit must be PUSHED to github.com/niko4244/Chowder (pip installs from
the remote at that sha). Credentials come from the ambient Kaggle config
(kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY); the key is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/niko4244/Chowder.git"

#: The operator command that runs INSIDE the kernel (after the template's
#: pinned-commit install + fingerprint). A real measured fp16 matmul on the
#: attached accelerator, plus the campaign-spec receipt check.
KERNEL_COMMAND = '''python -c "
import json, os, time, torch
spec = json.load(open(os.environ['CHOWDER_CAMPAIGN_SPEC']))
n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0) if n else 'none'
dev = torch.device('cuda:0' if n else 'cpu')
dt = torch.float16 if n else torch.float32
a = torch.randn(2048, 2048, device=dev, dtype=dt)
b = torch.randn(2048, 2048, device=dev, dtype=dt)
(torch.cuda.synchronize() if n else None)
t0 = time.perf_counter()
for _ in range(10):
    c = a @ b
(torch.cuda.synchronize() if n else None)
elapsed = time.perf_counter() - t0
tflops = 10 * 2 * 2048 ** 3 / elapsed / 1e12
res = {'status': 'complete', 'gpu_count': n, 'gpu_name': name,
       'matmul_fp16_tflops': round(tflops, 2), 'dtype': str(dt),
       'campaign_spec_received': bool(spec), 'torch': torch.__version__}
json.dump(res, open('/kaggle/working/chowder_result.json', 'w'), indent=2)
print(json.dumps(res))
"'''


def resolve_username() -> str:
    env_user = os.environ.get("KAGGLE_USERNAME", "")
    if env_user:
        return env_user
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR", "") or str(Path.home() / ".kaggle")
    kaggle_json = Path(config_dir) / "kaggle.json"
    if kaggle_json.exists():
        doc = json.loads(kaggle_json.read_text(encoding="utf-8"))
        return str(doc.get("username", ""))
    return ""


def head_commit(explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lower()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    return sha


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; the kaggle client's console chatter
    # (and our own JSON) can carry characters it cannot encode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None,
                        help="40-hex Chowder commit to pin (default: HEAD)")
    parser.add_argument("--state-root", default=None,
                        help="durable research-memory root (default: tempdir)")
    parser.add_argument("--workdir", default=None,
                        help="kernel-folder root (default: tempdir)")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume-ref", default=None,
                        help="resume polling an already-pushed kernel "
                             "({owner}/{slug}) instead of pushing a new one")
    args = parser.parse_args(argv)

    commit = head_commit(args.commit)
    if len(commit) != 40:
        print(f"REFUSING: --commit must be a 40-character sha, got {commit!r}",
              file=sys.stderr)
        return 2
    username = resolve_username()
    if not username:
        print("REFUSING: no Kaggle username resolvable (kaggle.json / env)",
              file=sys.stderr)
        return 2

    import tempfile
    stamp = time.strftime("%Y%m%d-%H%M%S")
    state_root = Path(args.state_root or (Path(tempfile.gettempdir())
                                          / "chowder-provider-acceptance" / stamp))
    workdir = Path(args.workdir or (state_root / "kernels"))
    experiment_id = f"provider-acceptance-{stamp}"

    from chowder.scientist.compute import (
        ExperimentClass, ExperimentRequest, KaggleProvider,
    )
    provider = KaggleProvider(
        username=username,           # key resolves from the ambient Kaggle config
        chowder_commit=commit,
        kernel_command=KERNEL_COMMAND,
        repo_url=REPO_URL,
        workdir=str(workdir),
        weekly_gpu_hours=20.0,
    )
    if not provider.configured():
        print("REFUSING: KaggleProvider reports unconfigured", file=sys.stderr)
        return 2

    # 1. reconcile the declared quota with the operator's REAL weekly budget
    quota = provider.sync_quota_from_api()
    print(json.dumps({"step": "quota", "weekly_gpu_hours": quota.weekly_gpu_hours,
                      "used_gpu_hours": quota.used_gpu_hours,
                      "remaining_gpu_hours": quota.remaining_gpu_hours()}))

    request = ExperimentRequest(
        experiment_id=experiment_id,
        proposal_id=f"prop-{experiment_id}",
        hypothesis_id="provider-acceptance",
        campaign_spec={
            "recipe_patch": {},
            "data": {"source_kinds": ["acceptance-probe"]},
            "evaluations": {"target_surfaces": ["reasoning"], "transfer_surfaces": []},
            "replication": {"plan": "single acceptance probe", "seed": 2026},
            "controls": [], "falsification_rule": "matmul_tflops <= 0",
        },
        experiment_class=ExperimentClass.SCREENING,
        estimated_gpu_hours=0.1,   # the probe is minutes, not hours
    )
    cost = provider.estimate_cost(request)

    # 2. gate on the REAL quota before creating anything
    if quota.remaining_gpu_hours() < cost:
        print(f"REFUSING: needs {cost} device GPU-hours, "
              f"{quota.remaining_gpu_hours()} remain", file=sys.stderr)
        return 2

    # 3. the real push (or resume an already-pushed kernel)
    if args.resume_ref:
        from chowder.scientist.compute import Submission
        submission = Submission(
            submission_id=f"sub-kaggle-{request.experiment_id}",
            request=request, provider_name=provider.name, status="queued",
            hardware_class=provider.hardware_class,
            device_gpu_hours=cost, experiment_class=request.experiment_class,
            provider_ref=args.resume_ref,
        )
        print(json.dumps({"step": "resumed", "provider_ref": submission.provider_ref}))
    else:
        submission = provider.submit(request)
        print(json.dumps({"step": "submitted", "provider_ref": submission.provider_ref,
                          "device_gpu_hours": submission.device_gpu_hours,
                          "hardware_class": submission.hardware_class}))

    # 4. poll the real session lifecycle
    deadline = time.time() + args.timeout_seconds
    transitions: list[dict] = []
    seen: set[str] = set()
    while time.time() < deadline:
        submission = provider.poll(submission)
        if submission.status not in seen:
            seen.add(submission.status)
            transitions.append({
                "status": submission.status,
                "at_seconds": round(time.time() - (deadline - args.timeout_seconds), 1),
            })
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

    # 5. record the run-grounded observation through durable research memory
    output_dir = Path(submission.result["output_dir"])
    run_id = str(output_dir)   # path-based grounding, per the CLI resolver convention
    from chowder.scientist.observation import ExperimentObservation, Measurement
    from chowder.scientist.research_memory import ResearchMemory
    memory = ResearchMemory(
        state_root / "research",
        run_exists=lambda r: Path(r).exists(),
        run_complete=lambda r: (Path(r) / "chowder_result.json").exists(),
    )
    probe = submission.result.get("chowder_result") or {}
    measurements = []
    if "matmul_fp16_tflops" in probe:
        measurements.append(Measurement(
            surface="efficiency:matmul_fp16_tflops", benchmark="acceptance-probe",
            value=float(probe["matmul_fp16_tflops"])))
    probe_complete = probe.get("status") == "complete" and measurements
    observation = ExperimentObservation(
        observation_id=f"obs-{experiment_id}",
        run_id=run_id,
        experiment_ref=request.experiment_id,
        proposal_id=request.proposal_id,
        hypothesis_id=request.hypothesis_id,
        measurements=tuple(measurements),
        status="complete" if probe_complete else "failed",
        wall_gpu_hours=request.estimated_gpu_hours,  # estimate; probe does not meter
        notes="provider-path acceptance probe (pinned-commit install + fingerprint "
              "+ fp16 matmul on the attached accelerator)",
        provider=provider.name,
        hardware_class=submission.hardware_class,
    )
    memory.record_observation(observation)

    summary = {
        "step": "complete",
        "commit": commit,
        "provider_ref": submission.provider_ref,
        "hardware_class": submission.hardware_class,
        "device_gpu_hours": submission.device_gpu_hours,
        "transitions": transitions,
        "output_files": submission.result.get("output_files"),
        "environment_fingerprint_present": submission.environment_fingerprint is not None,
        "probe": probe,
        "observation_id": observation.observation_id,
        "state_root": str(state_root),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
