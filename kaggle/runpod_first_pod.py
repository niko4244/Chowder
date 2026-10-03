"""One real end-to-end RunPod pod run through the RunPodProvider path.

This is the RunPod acceptance run (docs/RUNPOD_PROVIDER_ACCEPTANCE.md): REST v2
pod create → lifecycle poll → artifact confirmed from the pod's own stdout
(via GET /v2/pods/{id}/logs) → measured device-hours reconciled into the
quota model → a run-grounded ExperimentObservation in durable memory. The
pod is ALWAYS terminated in a finally block: a finished or failed probe must
not keep billing.

What it deliberately is NOT: a training campaign. The container command is a
small real GPU probe (fp16 matmul) — enough to prove create/poll/fetch/record
end to end without meaningful spend.

Usage (operator-supplied infrastructure choices — nothing is invented):
    python kaggle/runpod_first_pod.py \
        --gpu-type <id from GET /v2/catalog/gpus or the console> \
        --image <torch+cuda image, e.g. runpod/pytorch:...> \
        [--state-root DIR] [--poll-seconds 20] [--timeout-seconds 1800]

Credentials: RUNPOD_API_KEY in the environment (or --api-key). The key is
never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RESULT_MARKER = "CHOWDER_RESULT_JSON:"

#: The operator command that runs INSIDE the pod: a real measured fp16 matmul
#: on the attached accelerator, writing one JSON line (the artifact marker)
#: to stdout — the provider-side equivalent of the Kaggle kernel's
#: /kaggle/working/chowder_result.json. A heredoc keeps the payload readable
#: and free of shell-escaping pitfalls (the container runs it with its shell).
POD_COMMAND = """python - <<'PYEOF'
import json, time, torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
name = torch.cuda.get_device_name(0) if n else 'none'
dev = torch.device('cuda:0' if n else 'cpu')
dt = torch.float16 if n else torch.float32
a = torch.randn(2048, 2048, device=dev, dtype=dt)
b = torch.randn(2048, 2048, device=dev, dtype=dt)
(torch.cuda.synchronize() if n else None)
t0 = time.monotonic()
for _ in range(10):
    c = a @ b
(torch.cuda.synchronize() if n else None)
wall = time.monotonic() - t0
count = n if n else 1
res = {'status': 'complete', 'gpu_count': n, 'gpu_name': name,
       'matmul_fp16_tflops': round(10 * 2 * 2048 ** 3 / wall / 1e12, 2),
       'wall_seconds': round(wall, 3),
       'wall_gpu_hours': round(wall / 3600.0, 6),
       'device_gpu_hours': round(wall / 3600.0 * count, 6),
       'accelerator_count': count, 'dtype': str(dt),
       'torch': torch.__version__, 'metering': 'measured'}
print('CHOWDER_RESULT_JSON:' + json.dumps(res), flush=True)
PYEOF"""


def result_fetcher_from_logs(provider):
    """The artifact seam: the pod's streamed stdout, fetched through the real
    API, searched for the marker line. Returns None until the marker appears
    (the honest 'not confirmed yet')."""

    def fetch(submission):
        text = provider.logs(submission)
        for line in reversed(text.splitlines()):
            if RESULT_MARKER in line:
                payload = line.split(RESULT_MARKER, 1)[1].strip()
                return json.loads(payload)
        return None

    return fetch


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None,
                        help="RunPod API key (default: RUNPOD_API_KEY env)")
    parser.add_argument("--gpu-type", required=True,
                        help="GPU type id (GET /v2/catalog/gpus or console)")
    parser.add_argument("--image", required=True,
                        help="torch+cuda container image to run the probe")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)

    import tempfile
    stamp = time.strftime("%Y%m%d-%H%M%S")
    state_root = Path(args.state_root or (Path(tempfile.gettempdir())
                                          / "chowder-runpod-acceptance" / stamp))

    from chowder.scientist.compute import (
        ExperimentClass, ExperimentRequest, RunPodProvider,
    )
    provider = RunPodProvider(
        api_key=args.api_key or "",
        gpu_type_id=args.gpu_type,
        gpu_count=args.gpu_count,
        image=args.image,
        command=POD_COMMAND,
        weekly_gpu_hours=8.0,
    )
    missing = provider.missing_configuration()
    if missing:
        print(f"REFUSING: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    if not (args.api_key or os.environ.get("RUNPOD_API_KEY")):
        print("REFUSING: no RUNPOD_API_KEY (env or --api-key)", file=sys.stderr)
        return 2

    request = ExperimentRequest(
        experiment_id=f"runpod-acceptance-{stamp}",
        proposal_id=f"prop-{stamp}",
        hypothesis_id="runpod-acceptance",
        campaign_spec={"recipe_patch": {}, "data": {"source_kinds": ["acceptance-probe"]},
                       "evaluations": {"target_surfaces": ["reasoning"]}},
        experiment_class=ExperimentClass.SCREENING,
        estimated_gpu_hours=0.1,
    )
    cost = provider.estimate_cost(request)
    if provider.quota().remaining_gpu_hours() < cost:
        print(f"REFUSING: needs {cost} device GPU-hours, "
              f"{provider.quota().remaining_gpu_hours()} remain", file=sys.stderr)
        return 2

    provider._result_fetcher = result_fetcher_from_logs(provider)
    submission = provider.submit(request)
    print(json.dumps({"step": "submitted", "provider_ref": submission.provider_ref,
                      "hardware_class": submission.hardware_class,
                      "estimated_device_gpu_hours": submission.device_gpu_hours}))

    deadline = time.time() + args.timeout_seconds
    transitions: list[dict] = []
    seen: set[str] = set()
    try:
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
    finally:
        # billing hygiene: the probe pod never outlives the acceptance run
        try:
            terminated = provider.terminate(submission)
            print(json.dumps({"step": "terminated", "pod_deleted": terminated}))
        except Exception as error:
            print(json.dumps({"step": "terminate_failed", "detail": str(error)}),
                  file=sys.stderr)

    if submission.status != "complete":
        print(json.dumps({"step": "failed", "result": submission.result}, default=str))
        return 1

    probe = submission.result.get("chowder_result") or {}
    record_path = state_root / "chowder_result.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(probe, indent=2, sort_keys=True), encoding="utf-8")

    from chowder.scientist.observation import ExperimentObservation, Measurement
    from chowder.scientist.research_memory import ResearchMemory
    run_id = f"runpod:{submission.provider_ref}"
    measurements = []
    if "matmul_fp16_tflops" in probe:
        measurements.append(Measurement(
            surface="efficiency:matmul_fp16_tflops", benchmark="acceptance-probe",
            value=float(probe["matmul_fp16_tflops"])))
    observation = ExperimentObservation(
        observation_id=f"obs-{request.experiment_id}",
        run_id=run_id,
        experiment_ref=request.experiment_id,
        proposal_id=request.proposal_id,
        hypothesis_id=request.hypothesis_id,
        measurements=tuple(measurements),
        status="complete" if probe.get("status") == "complete" and measurements else "failed",
        wall_gpu_hours=float(probe.get("device_gpu_hours", 0.0)),
        notes="RunPod provider-path acceptance probe (REST v2 pod, measured fp16 matmul)",
        provider=provider.name,
        hardware_class=submission.hardware_class,
    )
    memory = ResearchMemory(
        state_root / "research",
        run_exists=lambda r: r == run_id,
        run_complete=lambda r: r == run_id and probe.get("status") == "complete",
    )
    memory.record_observation(observation)

    print(json.dumps({
        "step": "complete",
        "provider_ref": submission.provider_ref,
        "hardware_class": submission.hardware_class,
        "device_gpu_hours_measured": submission.device_gpu_hours,
        "quota_used_after": provider.quota().used_gpu_hours,
        "transitions": transitions,
        "probe": probe,
        "observation_id": observation.observation_id,
        "state_root": str(state_root),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
