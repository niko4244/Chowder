"""Compute providers: where a compiled experiment runs.

The ResearchDirector does not care where an experiment runs; the scheduler
decides, from explicit provider availability and quota models. The mission's
hardware-context rule lives here too: quality findings may compare across
hardware when the protocol is identical; efficiency findings never leave their
hardware context. See docs/COMPUTE_PROVIDERS.md (audit + honest status).

Proven here: the scheduling/evidence rules; Kaggle kernel-push, session
polling, output fetch and quota reconciliation against the installed Kaggle
API package (v2.2.3, kagglesdk-based: KaggleApi.kernels_push / kernels_status /
kernels_output / quota_view); the RunPod REST v2 pod path (POST /v2/pods,
GET /v2/pods/{id}) through an injectable transport. Honest limits, documented
per provider: a Kaggle push requires a pinned 40-hex commit AND an operator
supplied kernel_command (the campaign spec has no default executor), and a
RunPod pod's EXITED status is never reported `complete` unless a result
fetcher confirms an artifact — evidence over optimism, always.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SchedulerRefusal(RuntimeError):
    """Fail-closed scheduling refusal (no provider, quota, unknown pin)."""


class ExperimentClass(str, Enum):
    SCREENING = "screening"
    EXPLORATORY = "exploratory"
    SUBSTANTIAL = "substantial"
    REPLICATION = "replication"


@dataclass(frozen=True)
class ExperimentRequest:
    """What the research layer asks the scheduler to run."""

    experiment_id: str
    proposal_id: str
    hypothesis_id: str
    campaign_spec: dict[str, Any]
    experiment_class: ExperimentClass = ExperimentClass.EXPLORATORY
    estimated_gpu_hours: float = 0.0
    #: Pins the provider by name (replication-on-other-hardware does this).
    require_provider: str | None = None
    #: Claims about speed/VRAM/wall-time must run where the claim lives.
    hardware_dependent: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.proposal_id or not self.hypothesis_id:
            raise ValueError("a request names its experiment, proposal and hypothesis")
        if self.estimated_gpu_hours < 0:
            raise ValueError("estimated_gpu_hours must be non-negative")


@dataclass(frozen=True)
class ProviderQuota:
    """A provider's declared, declining budget model."""

    weekly_gpu_hours: float
    used_gpu_hours: float = 0.0
    concurrent_jobs: int = 1

    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.weekly_gpu_hours - self.used_gpu_hours)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weekly_gpu_hours": self.weekly_gpu_hours,
            "used_gpu_hours": self.used_gpu_hours,
            "remaining_gpu_hours": self.remaining_gpu_hours(),
            "concurrent_jobs": self.concurrent_jobs,
        }


@dataclass
class Submission:
    """A job handed to a provider. Executing providers carry the remote
    handle in `provider_ref` (kernel ref `owner/slug`, pod id, ...); a
    submission without one has not left the process."""

    submission_id: str
    request: ExperimentRequest
    provider_name: str
    status: str = "queued"              # queued | running | complete | failed
    hardware_class: str = ""
    device_gpu_hours: float = 0.0
    experiment_class: ExperimentClass = ExperimentClass.EXPLORATORY
    result: dict[str, Any] | None = None
    environment_fingerprint: dict[str, Any] | None = None
    provider_ref: str = ""              # remote handle, provider-namespaced


@runtime_checkable
class ComputeProvider(Protocol):
    """One external/attached compute pool. Returns typed data; never touches
    policy, thresholds, or the protected set."""

    name: str
    hardware_class: str

    def available(self) -> bool:
        """Whether this provider can accept work right now."""
        ...

    def quota(self) -> ProviderQuota:
        ...

    def estimate_cost(self, request: ExperimentRequest) -> float:
        """Device GPU-hours (wall × active accelerators), not a promise."""
        ...

    def submit(self, request: ExperimentRequest) -> Submission:
        ...

    def poll(self, submission: Submission) -> Submission:
        ...


# ---------------------------------------------------------------------------
# local provider
# ---------------------------------------------------------------------------


def _local_hardware_class() -> str:
    try:
        from ..hardware import detect_hardware
        snapshot = detect_hardware()
        if snapshot.accelerators:
            first = snapshot.accelerators[0]
            total = snapshot.topology.total_accelerator_memory_gb
            count = len(snapshot.accelerators)
            name = (getattr(first, "name", "") or "gpu").lower().replace(" ", "_")
            return f"local_{count}x_{name}_{total:.0f}gb"
    except Exception:
        pass
    return "local_cpu_only"


class LocalCudaProvider:
    """The machine Chowder runs on. Executes nothing here — the production
    training/evaluation bindings already own local execution; this provider
    exists so the scheduler treats 'local' as one choice among many, with the
    same evidence discipline."""

    name = "local_cuda"

    def __init__(self, *, accelerators: int | None = None,
                 weekly_gpu_hours: float = 1e9) -> None:
        self._hardware_class = _local_hardware_class()
        if accelerators is None:
            accelerators = self._detected_accelerators()
        if accelerators < 0:
            raise ValueError("accelerators must be non-negative")
        self._accelerators = accelerators
        self._quota = ProviderQuota(weekly_gpu_hours=weekly_gpu_hours)

    def _detected_accelerators(self) -> int:
        try:
            from ..hardware import detect_hardware
            snapshot = detect_hardware()
            return len(snapshot.accelerators)
        except Exception:
            return 0

    @property
    def hardware_class(self) -> str:
        return self._hardware_class

    def available(self) -> bool:
        return self._accelerators > 0

    def quota(self) -> ProviderQuota:
        return self._quota

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return request.estimated_gpu_hours * max(1, self._accelerators)

    def submit(self, request: ExperimentRequest) -> Submission:
        if not self.available():
            raise SchedulerRefusal("LOCAL_UNAVAILABLE: no accelerator detected")
        affordable = request.estimated_gpu_hours <= self._quota.remaining_gpu_hours()
        if not affordable:
            raise SchedulerRefusal("QUOTA_EXHAUSTED: local weekly budget exhausted")
        self._quota = replace(
            self._quota, used_gpu_hours=self._quota.used_gpu_hours + request.estimated_gpu_hours
        )
        return Submission(
            submission_id=f"sub-local-{request.experiment_id}",
            request=request,
            provider_name=self.name,
            status="queued",
            hardware_class=self.hardware_class,
            device_gpu_hours=self.estimate_cost(request),
            experiment_class=request.experiment_class,
        )

    def poll(self, submission: Submission) -> Submission:
        # Local execution happens through the production bindings; the
        # scheduler observes the resulting run, so a submission stays queued
        # until the observation is recorded. Honest: nothing pretends to run.
        return submission


# ---------------------------------------------------------------------------
# kaggle provider
# ---------------------------------------------------------------------------


class KaggleProvider:
    """Opportunistic Kaggle notebook capacity (T4×2 today, 12 h sessions,
    weekly GPU quota that varies with demand).

    Proven here (docs/COMPUTE_PROVIDERS.md §6): real kernel-push through the
    installed Kaggle API package (2.2.3, kagglesdk-based) — ``KaggleApi
    .kernels_push`` runs the pushed kernel (SAVE_AND_RUN), ``kernels_status``
    maps KernelWorkerStatus, ``kernels_output`` fetches artifacts, and
    ``quota_view`` reconciles the declared quota with the operator's real
    weekly GPU budget. A push needs three things, and refuses without any of
    them rather than pushing something useless: credentials, a pinned 40-hex
    chowder commit (the repo's "never a branch" install rule), and an
    operator-supplied ``kernel_command`` — the compiled campaign_spec is a
    declaration, and Chowder ships no default remote executor that would
    silently guess what to run.

    ``push=False`` keeps the previous spec-only behavior (submission stays
    queued, poll is a no-op) for offline scheduling tests; production
    constructs with the default ``push=True``.
    """

    name = "kaggle"
    hardware_class = "kaggle_2x_t4_16gb"
    screening = True  # the designated screening lane

    #: Two T4s per notebook (Kaggle's current GPU shape); device-hours =
    #: wall × 2, matching the growth accounting's accelerator-seconds rule.
    #: Sep 2026 accelerator id: NvidiaTeslaT4 IS the T4×2 shape (P100 retired).
    ACCELERATORS_PER_NOTEBOOK = 2
    MACHINE_SHAPE = "NvidiaTeslaT4"

    def __init__(self, *, username: str = "", api_key: str = "",
                 weekly_gpu_hours: float = 20.0, screening_lane: bool = True,
                 push: bool = True, chowder_commit: str = "",
                 kernel_command: str = "", repo_url: str = "https://github.com/niko4244/Chowder.git",
                 session_timeout_seconds: int = 43200,
                 workdir: str = "", api: Any | None = None) -> None:
        self._username = username
        self._api_key = api_key
        self._quota = ProviderQuota(
            weekly_gpu_hours=weekly_gpu_hours,
            concurrent_jobs=2,
        )
        self.screening_lane = screening_lane
        self.push = push
        self.chowder_commit = chowder_commit
        self.kernel_command = kernel_command
        self.repo_url = repo_url
        self.session_timeout_seconds = int(session_timeout_seconds)
        self._workdir = workdir  # "" → platform default under the user home
        self._api = api          # injected client (tests); None → resolve lazily
        self._client_cache: Any = None

    # -- credentials / availability ------------------------------------------

    @staticmethod
    def _env_credentials() -> tuple[str, str]:
        import os
        return (os.environ.get("KAGGLE_USERNAME", ""), os.environ.get("KAGGLE_KEY", ""))

    @staticmethod
    def _config_file_credentials() -> bool:
        """True when a kaggle.json exists where the Kaggle client looks for it.
        Checked WITHOUT importing kaggle: the package's authenticate() calls
        exit(1) when unauthenticated, so the provider resolves credentials
        itself and only imports the client once they exist."""
        import os
        from pathlib import Path
        config_dir = os.environ.get("KAGGLE_CONFIG_DIR", "")
        base = Path(config_dir) if config_dir else Path.home() / ".kaggle"
        return (base / "kaggle.json").exists()

    def resolved_username(self) -> str:
        if self._username:
            return self._username
        env_user, _ = self._env_credentials()
        return env_user

    def configured(self) -> bool:
        if self._username and self._api_key:
            return True
        env_user, env_key = self._env_credentials()
        if env_user and env_key:
            return True
        return self._config_file_credentials()

    def available(self) -> bool:
        return self.configured() and self._quota.remaining_gpu_hours() > 0

    def quota(self) -> ProviderQuota:
        return self._quota

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return request.estimated_gpu_hours * self.ACCELERATORS_PER_NOTEBOOK

    # -- the real kaggle client -----------------------------------------------

    def _client(self) -> Any:
        """The Kaggle API client: injected when given, else built from the
        installed package. Credentials are pre-resolved BEFORE any kaggle
        import — the package's authenticate() exits the process when it cannot
        authenticate, and a refusal must never kill the process."""
        if self._api is not None:
            return self._api
        if self._client_cache is not None:
            return self._client_cache
        if self._username and self._api_key:
            import os
            os.environ.setdefault("KAGGLE_USERNAME", self._username)
            os.environ.setdefault("KAGGLE_KEY", self._api_key)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as error:
            raise SchedulerRefusal(
                "KAGGLE_PACKAGE_MISSING: the kaggle package is not installed; "
                "refusing rather than pretending to push"
            ) from error
        try:
            client = KaggleApi()
            client.authenticate()
        except SystemExit as error:  # authenticate() calls exit(1) when unauthenticated
            raise SchedulerRefusal(
                "KAGGLE_NOT_CONFIGURED: kaggle authentication failed; set "
                "KAGGLE_USERNAME/KAGGLE_KEY or a kaggle.json"
            ) from error
        except Exception as error:
            raise SchedulerRefusal(
                f"KAGGLE_API_ERROR: kaggle client authentication failed: {error}"
            ) from error
        self._client_cache = client
        return client

    def _kernel_folder(self, request: ExperimentRequest) -> tuple[str, str]:
        """(slug, folder) for the request's kernel: metadata + script written
        under the provider workdir, one folder per experiment."""
        import re
        from pathlib import Path
        base = re.sub(r"[^a-z0-9-]+", "-", request.experiment_id.lower()).strip("-")
        base = base.removeprefix("sciexp-") or "exp"
        slug = f"chowder-{base}"[:60].strip("-")
        root = Path(self._workdir) if self._workdir else Path.home() / ".chowder" / "kaggle-kernels"
        folder = root / slug
        folder.mkdir(parents=True, exist_ok=True)
        return slug, str(folder)

    def _write_kernel(self, request: ExperimentRequest, slug: str, folder: str) -> None:
        """Write kernel-metadata.json + kernel.py per the Kaggle push contract
        (title ≥5 chars, slug matching the id, pinned-commit bootstrap —
        mirroring kaggle/bootstrap_environment.py's install-and-verify rule)."""
        import json
        from pathlib import Path
        ref = f"{self.resolved_username() or 'unknown'}/{slug}"
        metadata = {
            "id": ref,
            "title": slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "machine_shape": self.MACHINE_SHAPE,
        }
        (Path(folder) / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        spec_json = json.dumps(request.campaign_spec, indent=2, sort_keys=True)
        script = _KERNEL_SCRIPT_TEMPLATE
        script = script.replace("__COMMIT__", self.chowder_commit)
        script = script.replace("__REPO_URL__", self.repo_url)
        script = script.replace("__SPEC_JSON__", spec_json)
        script = script.replace("__EXPERIMENT_ID__", request.experiment_id)
        script = script.replace("__PROPOSAL_ID__", request.proposal_id)
        script = script.replace("__HYPOTHESIS_ID__", request.hypothesis_id)
        import base64
        script = script.replace(
            "__KERNEL_COMMAND_B64__",
            base64.b64encode(self.kernel_command.encode("utf-8")).decode("ascii"),
        )
        (Path(folder) / "kernel.py").write_text(script, encoding="utf-8")

    def submit(self, request: ExperimentRequest) -> Submission:
        if not self.configured():
            raise SchedulerRefusal(
                "KAGGLE_NOT_CONFIGURED: set the Kaggle username/API key to use "
                "this provider; refusing rather than pretending"
            )
        cost = self.estimate_cost(request)
        if cost > self._quota.remaining_gpu_hours():
            raise SchedulerRefusal(
                f"QUOTA_EXHAUSTED: needs {cost} device GPU-hours, "
                f"{self._quota.remaining_gpu_hours()} remain of "
                f"{self._quota.weekly_gpu_hours}"
            )
        if not self.push:
            self._quota = replace(self._quota, used_gpu_hours=self._quota.used_gpu_hours + cost)
            return Submission(
                submission_id=f"sub-kaggle-{request.experiment_id}",
                request=request,
                provider_name=self.name,
                status="queued",
                hardware_class=self.hardware_class,
                device_gpu_hours=cost,
                experiment_class=request.experiment_class,
            )
        # real push: pinned commit + an executor command, or nothing ships
        commit = (self.chowder_commit or "").strip().lower()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise SchedulerRefusal(
                "KAGGLE_KERNEL_PIN_REQUIRED: a real push installs chowder at a "
                "pinned 40-character commit (never a branch); set chowder_commit"
            )
        if not self.kernel_command.strip():
            raise SchedulerRefusal(
                "KAGGLE_EXECUTOR_NOT_CONFIGURED: the compiled campaign_spec is a "
                "declaration and Chowder ships no default remote executor; set "
                "kernel_command to the operator-approved runner"
            )
        slug, folder = self._kernel_folder(request)
        self._write_kernel(request, slug, folder)
        client = self._client()
        try:
            response = client.kernels_push(
                folder, timeout=str(self.session_timeout_seconds), acc=self.MACHINE_SHAPE)
        except SchedulerRefusal:
            raise
        except Exception as error:
            raise SchedulerRefusal(
                f"KAGGLE_API_ERROR: kernel push failed: {error}"
            ) from error
        error_field = getattr(response, "error", "") or ""
        if str(error_field).strip():
            raise SchedulerRefusal(
                f"KAGGLE_PUSH_REFUSED: kaggle rejected the kernel: {error_field}"
            )
        ref = str(getattr(response, "ref", "") or f"{self.resolved_username()}/{slug}")
        ref = self._normalize_kernel_ref(ref)
        self._quota = replace(self._quota, used_gpu_hours=self._quota.used_gpu_hours + cost)
        return Submission(
            submission_id=f"sub-kaggle-{request.experiment_id}",
            request=request,
            provider_name=self.name,
            status="queued",
            hardware_class=self.hardware_class,
            device_gpu_hours=cost,
            experiment_class=request.experiment_class,
            provider_ref=ref,
        )

    # -- session lifecycle -----------------------------------------------------

    _QUEUED_STATUS_NAMES = ("QUEUED", "CANCEL_REQUESTED", "NEW_SCRIPT")

    @staticmethod
    def _normalize_kernel_ref(ref: str) -> str:
        """The push response's `ref` is a URL path ('/code/{owner}/{slug}');
        kernels_status/kernels_output require '{owner}/{slug}'. Normalize at
        the boundary so every stored provider_ref is directly pollable."""
        parts = [p for p in str(ref).split("/") if p]
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return str(ref)

    def poll(self, submission: Submission) -> Submission:
        """Map the kernel session's real status onto the submission. A queued
        spec-only submission (push=False, no provider_ref) stays exactly where
        it is; nothing fabricates progress. On COMPLETE the output is fetched
        and chowder_result.json / environment.json become the submission's
        evidence; on ERROR the failure message travels with it."""
        if not submission.provider_ref:
            return submission
        client = self._client()
        try:
            status_response = client.kernels_status(submission.provider_ref)
        except SchedulerRefusal:
            raise
        except Exception as error:
            raise SchedulerRefusal(
                f"KAGGLE_API_ERROR: kernel status failed: {error}"
            ) from error
        raw = getattr(status_response, "status", None)
        name = getattr(raw, "name", str(raw)).split(".")[-1]
        failure = str(getattr(status_response, "failure_message", "") or "")
        if name in self._QUEUED_STATUS_NAMES:
            return replace(submission, status="queued")
        if name == "RUNNING":
            return replace(submission, status="running")
        if name == "COMPLETE":
            return self._collect_output(submission, client)
        # ERROR / CANCEL_ACKNOWLEDGED (and anything unrecognized): failed,
        # with whatever Kaggle said — never silently re-queued.
        return replace(submission, status="failed", result={
            "kernel_status": name,
            "failure_message": failure,
            "provider_ref": submission.provider_ref,
        })

    def _collect_output(self, submission: Submission, client: Any) -> Submission:
        """Fetch the completed kernel's output artifacts. chowder_result.json
        (written by the kernel script) carries the run's own claim about the
        experiment; environment.json is the BackendFingerprint the evidence
        layer expects. Missing files are recorded as missing — never invented."""
        import json
        from pathlib import Path
        slug = submission.provider_ref.split("/")[-1]
        root = Path(self._workdir) if self._workdir else Path.home() / ".chowder" / "kaggle-kernels"
        out_dir = root / slug / "output"
        import contextlib
        import io
        try:
            # the kaggle client prints per-file progress lines; on Windows its
            # console output can crash with cp1252 encode errors even with
            # quiet=True — the provider consumes the RETURNED files, not the
            # client's console chatter
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                files, _token = client.kernels_output(submission.provider_ref, str(out_dir))
        except UnicodeEncodeError:
            # the kaggle client prints a console summary AFTER downloading;
            # on a cp1252 console that print can crash even though the
            # artifacts are safely on disk. The files — not the console
            # chatter — are the evidence: fall back to what landed.
            files = (sorted(str(p) for p in out_dir.iterdir() if p.is_file())
                     if out_dir.exists() else [])
            if not files:
                raise
        except SchedulerRefusal:
            raise
        except Exception as error:
            return replace(submission, status="failed", result={
                "kernel_status": "COMPLETE",
                "failure_message": f"output fetch failed: {error}",
                "provider_ref": submission.provider_ref,
            })
        names = [Path(f).name for f in (files or [])]
        result: dict[str, Any] = {
            "kernel_status": "COMPLETE",
            "provider_ref": submission.provider_ref,
            "output_files": names,
            "output_dir": str(out_dir),
        }
        fingerprint = None
        result_path = out_dir / "chowder_result.json"
        if result_path.exists():
            try:
                result["chowder_result"] = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                result["chowder_result_error"] = str(error)
        environment_path = out_dir / "environment.json"
        if environment_path.exists():
            try:
                fingerprint = json.loads(environment_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                result["environment_fingerprint_error"] = str(error)
        probe = result.get("chowder_result")
        measured = probe.get("device_gpu_hours") if isinstance(probe, dict) else None
        if isinstance(measured, (int, float)) and measured >= 0:
            # real cost replaces the submit-time estimate: settle the
            # difference against the weekly budget honestly
            estimate = submission.device_gpu_hours
            self._quota = replace(
                self._quota,
                used_gpu_hours=max(0.0, self._quota.used_gpu_hours - estimate + float(measured)),
            )
            return replace(submission, status="complete", result=result,
                           environment_fingerprint=fingerprint,
                           device_gpu_hours=float(measured))
        return replace(submission, status="complete", result=result,
                       environment_fingerprint=fingerprint)

    def sync_quota_from_api(self) -> ProviderQuota:
        """Reconcile the declared weekly quota with the operator's REAL Kaggle
        GPU budget (quota_view → ApiAcceleratorQuota timedeltas). Reserved time
        counts against availability exactly like used time. Missing/zero fields
        keep the declared model untouched — never a fake refresh."""
        from datetime import timedelta
        client = self._client()
        try:
            view = client.quota_view()
        except Exception as error:
            raise SchedulerRefusal(
                f"KAGGLE_API_ERROR: quota view failed: {error}"
            ) from error
        gpu = getattr(view, "gpu_quota", None)
        if gpu is None:
            return self._quota
        total = getattr(gpu, "total_time_allowed", None)
        used = getattr(gpu, "time_used", None)
        reserved = getattr(gpu, "time_reserved", None)
        for field, value in (("total_time_allowed", total), ("time_used", used),
                             ("time_reserved", reserved)):
            if value is not None and not isinstance(value, timedelta):
                return self._quota
        weekly = self._quota.weekly_gpu_hours
        used_hours = self._quota.used_gpu_hours
        if total is not None and total.total_seconds() > 0:
            weekly = total.total_seconds() / 3600.0
            used_hours = 0.0
        if used is not None:
            used_hours += used.total_seconds() / 3600.0
        if reserved is not None:
            used_hours += reserved.total_seconds() / 3600.0
        self._quota = replace(self._quota, weekly_gpu_hours=weekly,
                              used_gpu_hours=used_hours)
        return self._quota


# ---------------------------------------------------------------------------
# kaggle kernel script template
# ---------------------------------------------------------------------------


#: The script written into every pushed kernel folder. Bootstrap mirrors
#: kaggle/bootstrap_environment.py: install chowder at the pinned commit, cross-
#: check the resolved commit from direct_url.json, capture the BackendFingerprint,
#: persist the compiled campaign_spec, then run the OPERATOR-SUPPLIED command —
#: Chowder ships no default remote executor, so a kernel without one refuses
#: loudly instead of training something undefined. The command writes (or the
#: template writes for it) /kaggle/working/chowder_result.json; HF tokens stay
#: in Kaggle Secrets and are the operator command's concern, as in the repo's
#: existing notebook scripts.
_KERNEL_SCRIPT_TEMPLATE = """# Chowder screening kernel (machine-generated by chowder.scientist.compute)
# experiment __EXPERIMENT_ID__ / proposal __PROPOSAL_ID__ / hypothesis __HYPOTHESIS_ID__
import json
import os
import subprocess
import sys
from pathlib import Path

import base64
import time

WORKING = Path("/kaggle/working")
RESULT_PATH = WORKING / "chowder_result.json"
COMMIT = "__COMMIT__"
REPO_URL = "__REPO_URL__"
KERNEL_COMMAND = base64.b64decode("__KERNEL_COMMAND_B64__").decode("utf-8")
CAMPAIGN_SPEC = json.loads(r'''__SPEC_JSON__''')

EXPERIMENT_ID = "__EXPERIMENT_ID__"


def _base_result(status: str, **extra) -> dict:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "batch"),
    }
    payload.update(extra)
    return payload


# Real per-run metering (wall clock of the operator command). Device-hours
# derive from the accelerators this session actually attached — the honest
# cost, replacing the request's estimate everywhere evidence is recorded.
_T0 = time.monotonic()
try:
    import torch
    _ACCELERATOR_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    _ACCELERATOR_COUNT = 0
if _ACCELERATOR_COUNT <= 0:
    _ACCELERATOR_COUNT = max(1, int(os.environ.get("CHOWDER_DEVICE_COUNT", "1")))


def main() -> int:
    WORKING.mkdir(parents=True, exist_ok=True)
    (WORKING / "campaign_spec.json").write_text(
        json.dumps(CAMPAIGN_SPEC, indent=2, sort_keys=True), encoding="utf-8")
    if not KERNEL_COMMAND.strip():
        RESULT_PATH.write_text(json.dumps(_base_result(
            "refused", reason="KAGGLE_EXECUTOR_NOT_CONFIGURED")), encoding="utf-8")
        print("REFUSED: no operator executor configured for this kernel", file=sys.stderr)
        return 3

    spec = f"chowder-ai[train,qlora] @ git+{REPO_URL}@{COMMIT}"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", spec], check=True)

    import importlib.metadata
    dist = importlib.metadata.distribution("chowder-ai")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    resolved = (direct_url.get("vcs_info") or {}).get("commit_id", "")
    if resolved and resolved != COMMIT:
        RESULT_PATH.write_text(json.dumps(_base_result(
            "refused", reason="CHOWDER_COMMIT_MISMATCH",
            requested=COMMIT, resolved=resolved)), encoding="utf-8")
        print(f"REFUSING TO CONTINUE: requested {COMMIT} but resolved {resolved}",
              file=sys.stderr)
        return 2

    try:
        from chowder.kaggle_launcher import capture_environment_fingerprint
        fingerprint = capture_environment_fingerprint(
            chowder_commit_sha=resolved or COMMIT,
            tokenizer_identity_sha256=None,
            quantization="4bit",
            dtype="float16",  # T4 cannot run bf16 (PRECISION_DIVERGENCE_REASON)
            device_map_summary='{"": 0}',
        )
        (WORKING / "environment.json").write_text(
            json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    except Exception as error:  # fingerprint is evidence, not a gate
        print(f"environment fingerprint unavailable: {error}", file=sys.stderr)

    env = dict(os.environ)
    env["CHOWDER_CAMPAIGN_SPEC"] = str(WORKING / "campaign_spec.json")
    env["CHOWDER_EXPERIMENT_ID"] = EXPERIMENT_ID
    command_started = time.monotonic()
    completed = subprocess.run(KERNEL_COMMAND, shell=True, env=env)
    wall_seconds = time.monotonic() - command_started
    wall_hours = wall_seconds / 3600.0
    device_gpu_hours = wall_hours * _ACCELERATOR_COUNT
    status = "complete" if completed.returncode == 0 else "failed"
    payload = _base_result(
        status,
        exit_code=completed.returncode,
        wall_seconds=round(wall_seconds, 3),
        wall_gpu_hours=round(wall_hours, 6),
        device_gpu_hours=round(device_gpu_hours, 6),
        accelerator_count=_ACCELERATOR_COUNT,
        metering="measured_wall_clock_x_attached_accelerators",
    )
    if RESULT_PATH.exists():
        try:
            command_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            if isinstance(command_result, dict):
                # metering fields stay the template's (measured here, not
                # claimable by the command); the command's own fields follow
                payload.update({
                    k: v for k, v in command_result.items()
                    if k not in ("status", "exit_code", "wall_seconds",
                                 "wall_gpu_hours", "device_gpu_hours",
                                 "accelerator_count", "metering", "experiment_id")
                })
        except json.JSONDecodeError as error:
            payload["command_result_error"] = str(error)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""


# ---------------------------------------------------------------------------
# runpod provider
# ---------------------------------------------------------------------------


def _runpod_http_transport(request: dict[str, Any]) -> tuple[int, str]:
    """The default transport: stdlib urllib against the RunPod REST v2 API
    (https://api.runpod.io/v2). Returns (http_status, body_text); the provider
    turns non-2xx into typed refusals. Tests inject a transport instead."""
    import json as _json
    import urllib.error
    import urllib.request
    data = _json.dumps(request["body"]).encode("utf-8") if request.get("body") is not None else None
    req = urllib.request.Request(
        request["url"], data=data, method=request["method"],
        headers=request["headers"],
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


class RunPodProvider:
    """Paid on-demand GPU capacity via the RunPod REST API v2.

    Proven here: pod create (POST /v2/pods — exactly one of gpu/cpu, container
    image + env from the body), lifecycle poll (GET /v2/pods/{id} → PodStatus
    PROVISIONING/STARTING/RUNNING/EXITED/ERROR/TERMINATED), and delete, all
    through an injectable transport (tests never touch the network; the
    default transport is stdlib urllib).

    Honest limits (docs/COMPUTE_PROVIDERS.md §7): a pod's EXITED status says
    the *container* exited — not that the experiment succeeded. Without a
    ``result_fetcher`` that confirms an artifact, EXITED maps to FAILED with
    RUNPOD_EXIT_UNVERIFIED; a complete observation still has to clear the run
    registry. Creating a pod is a billing event: the provider refuses rather
    than creating anything when credentials, GPU type, image, or executor are
    missing, and the declared quota model gates every submit.
    """

    name = "runpod"

    def __init__(self, *, api_key: str = "", gpu_type_id: str = "",
                 gpu_count: int = 1, image: str = "", command: str = "",
                 weekly_gpu_hours: float = 0.0, concurrent_jobs: int = 1,
                 cloud: str = "SECURE", disk_gb: int = 20,
                 base_url: str = "https://api.runpod.io/v2",
                 transport: Any | None = None, result_fetcher: Any | None = None,
                 screening: bool = False) -> None:
        import os
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self._gpu_type_id = gpu_type_id
        self._gpu_count = max(1, int(gpu_count))
        self._image = image
        self._command = command
        self._quota = ProviderQuota(
            weekly_gpu_hours=weekly_gpu_hours, concurrent_jobs=concurrent_jobs)
        self._cloud = cloud
        self._disk_gb = int(disk_gb)
        self._base_url = base_url.rstrip("/")
        self._transport = transport or _runpod_http_transport
        self._result_fetcher = result_fetcher
        self.screening = screening
        if gpu_type_id:
            self._hardware_class = (
                f"runpod_{self._gpu_count}x_{gpu_type_id.lower().replace(' ', '_')}"
            )
        else:
            self._hardware_class = "runpod_unconfigured"

    @property
    def hardware_class(self) -> str:
        return self._hardware_class

    def configured(self) -> bool:
        return bool(self._api_key) and bool(self._gpu_type_id)

    def missing_configuration(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self._api_key:
            missing.append("RUNPOD_API_KEY")
        if not self._gpu_type_id:
            missing.append("gpu_type_id (from GET /v2/catalog/gpus)")
        if not self._image:
            missing.append("image")
        if not self._command.strip():
            missing.append("command (operator executor)")
        return tuple(missing)

    def available(self) -> bool:
        return (
            not self.missing_configuration()
            and self._quota.remaining_gpu_hours() > 0
        )

    def quota(self) -> ProviderQuota:
        return self._quota

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return request.estimated_gpu_hours * self._gpu_count

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        status, text = self._transport({
            "method": method,
            "url": f"{self._base_url}{path}",
            "headers": headers,
            "body": body,
        })
        return int(status), text

    def submit(self, request: ExperimentRequest) -> Submission:
        missing = self.missing_configuration()
        if missing:
            raise SchedulerRefusal(
                "RUNPOD_NOT_CONFIGURED: missing " + ", ".join(missing)
                + "; refusing rather than creating an unrunnable pod"
            )
        cost = self.estimate_cost(request)
        if cost > self._quota.remaining_gpu_hours():
            raise SchedulerRefusal(
                f"QUOTA_EXHAUSTED: needs {cost} device GPU-hours, "
                f"{self._quota.remaining_gpu_hours()} remain of "
                f"{self._quota.weekly_gpu_hours}"
            )
        pod_name = f"chowder-{request.experiment_id}".lower().replace("_", "-")[:60]
        body = {
            "name": pod_name,
            "image": self._image,
            "gpu": {"id": self._gpu_type_id, "count": self._gpu_count},
            "cloud": self._cloud,
            "disk": self._disk_gb,
            "env": {
                "CHOWDER_EXPERIMENT_ID": request.experiment_id,
                "CHOWDER_PROPOSAL_ID": request.proposal_id,
                "CHOWDER_HYPOTHESIS_ID": request.hypothesis_id,
                "CHOWDER_CAMPAIGN_SPEC": json_dumps_compact(request.campaign_spec),
                "CHOWDER_KERNEL_COMMAND": self._command,
            },
        }
        if self._command:
            # Container CMD in exec form (the v2 schema: cmd is an array, not a
            # string). sh -lc gives the operator command a shell and the env
            # vars above; when the image defines an ENTRYPOINT this becomes its
            # argument list.
            body["cmd"] = ["/bin/sh", "-lc", self._command]
        try:
            status, text = self._call("POST", "/pods", body)
        except SchedulerRefusal:
            raise
        except Exception as error:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod create failed: {error}"
            ) from error
        if status < 200 or status >= 300:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod create returned HTTP {status}: {text[:300]}"
            )
        try:
            pod = json_loads(text)
        except ValueError as error:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod create returned non-JSON body: {error}"
            ) from error
        pod_id = str(pod.get("id", "") or "")
        if not pod_id:
            raise SchedulerRefusal(
                "RUNPOD_API_ERROR: pod create returned no pod id; refusing to "
                "track an unaddressable pod"
            )
        self._quota = replace(self._quota, used_gpu_hours=self._quota.used_gpu_hours + cost)
        return Submission(
            submission_id=f"sub-runpod-{request.experiment_id}",
            request=request,
            provider_name=self.name,
            status="queued",
            hardware_class=self.hardware_class,
            device_gpu_hours=cost,
            experiment_class=request.experiment_class,
            provider_ref=pod_id,
        )

    _RUNNING_STATUSES = ("PROVISIONING", "STARTING", "RUNNING")

    def poll(self, submission: Submission) -> Submission:
        if not submission.provider_ref:
            return submission
        try:
            status, text = self._call("GET", f"/pods/{submission.provider_ref}")
        except SchedulerRefusal:
            raise
        except Exception as error:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod status failed: {error}"
            ) from error
        if status < 200 or status >= 300:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod status returned HTTP {status}: {text[:300]}"
            )
        try:
            pod = json_loads(text)
        except ValueError as error:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod status returned non-JSON body: {error}"
            ) from error
        pod_status = str(pod.get("status", "")).upper()
        if pod_status in self._RUNNING_STATUSES:
            return replace(submission, status="running",
                           result={"pod_status": pod_status})
        if pod_status == "EXITED":
            return self._settle_exited(submission, pod)
        # ERROR / TERMINATED (and anything unrecognized): failed, honestly.
        return replace(submission, status="failed", result={
            "pod_status": pod_status or "UNKNOWN",
            "provider_ref": submission.provider_ref,
        })

    def _settle_exited(self, submission: Submission, pod: dict[str, Any]) -> Submission:
        """EXITED means the container stopped — success only when a result
        fetcher confirms an artifact. Without one, EXITED is failed with
        RUNPOD_EXIT_UNVERIFIED: evidence over optimism."""
        note: dict[str, Any] = {
            "pod_status": "EXITED",
            "provider_ref": submission.provider_ref,
        }
        if self._result_fetcher is None:
            note["failure_message"] = (
                "RUNPOD_EXIT_UNVERIFIED: the container exited; no result fetcher "
                "is configured to confirm an artifact, so this is not a complete"
            )
            return replace(submission, status="failed", result=note)
        try:
            artifact = self._result_fetcher(submission)
        except Exception as error:
            note["failure_message"] = f"RUNPOD_EXIT_UNVERIFIED: result fetch failed: {error}"
            return replace(submission, status="failed", result=note)
        if artifact is None:
            note["failure_message"] = (
                "RUNPOD_EXIT_UNVERIFIED: the result fetcher confirmed no artifact"
            )
            return replace(submission, status="failed", result=note)
        note["chowder_result"] = artifact
        measured = (artifact.get("device_gpu_hours")
                    if isinstance(artifact, dict) else None)
        if isinstance(measured, (int, float)) and measured >= 0:
            estimate = submission.device_gpu_hours
            self._quota = replace(
                self._quota,
                used_gpu_hours=max(0.0, self._quota.used_gpu_hours - estimate + float(measured)),
            )
            return replace(submission, status="complete", result=note,
                           device_gpu_hours=float(measured))
        return replace(submission, status="complete", result=note)

    def logs(self, submission: Submission) -> str:
        """The pod's streamed logs (GET /v2/pods/{id}/logs) — the operator's
        debugging window; never an evidence source by itself."""
        if not submission.provider_ref:
            return ""
        status, text = self._call("GET", f"/pods/{submission.provider_ref}/logs")
        if status < 200 or status >= 300:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod logs returned HTTP {status}: {text[:300]}"
            )
        return text

    def terminate(self, submission: Submission) -> bool:
        """Delete the pod (operator/loop hygiene: a refused or dead run must
        not keep billing). Returns True only on an acknowledged delete."""
        if not submission.provider_ref:
            return False
        status, text = self._call("DELETE", f"/pods/{submission.provider_ref}")
        if status < 200 or status >= 300:
            raise SchedulerRefusal(
                f"RUNPOD_API_ERROR: pod delete returned HTTP {status}: {text[:300]}"
            )
        return True


def json_dumps_compact(value: Any) -> str:
    import json
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def json_loads(text: str) -> Any:
    import json
    return json.loads(text)


# ---------------------------------------------------------------------------
# policy-driven provider construction (closed set, like every other config)
# ---------------------------------------------------------------------------


_PROVIDER_KINDS: dict[str, tuple[type, tuple[str, ...]]] = {
    "local_cuda": (LocalCudaProvider, ("accelerators", "weekly_gpu_hours")),
    "kaggle": (KaggleProvider, (
        "username", "api_key", "weekly_gpu_hours", "screening_lane", "push",
        "chowder_commit", "kernel_command", "repo_url", "session_timeout_seconds",
        "workdir",
    )),
    "runpod": (RunPodProvider, (
        "api_key", "gpu_type_id", "gpu_count", "image", "command",
        "weekly_gpu_hours", "concurrent_jobs", "cloud", "disk_gb", "base_url",
        "screening",
    )),
}


def provider_from_config(spec: dict[str, Any]) -> Any:
    """Build a provider from a policy document entry — closed set, closed keys
    (the repo's frozen-dataclass + closed-_KEYS idiom): an unknown kind or an
    unknown key refuses by name instead of being ignored."""
    if not isinstance(spec, dict):
        raise SchedulerRefusal(
            f"PROVIDER_CONFIG_INVALID: a provider entry must be a mapping, "
            f"got {type(spec).__name__}"
        )
    kind = str(spec.get("kind", ""))
    entry = _PROVIDER_KINDS.get(kind)
    if entry is None:
        raise SchedulerRefusal(
            f"UNKNOWN_PROVIDER_KIND: {kind!r}; known kinds: "
            f"{sorted(_PROVIDER_KINDS)}"
        )
    cls, allowed = entry
    unknown = sorted(set(spec) - set(allowed) - {"kind"})
    if unknown:
        raise SchedulerRefusal(
            f"UNKNOWN_PROVIDER_CONFIG_KEYS: {kind}: {unknown}"
        )
    kwargs = {k: v for k, v in spec.items() if k != "kind"}
    try:
        return cls(**kwargs)
    except SchedulerRefusal:
        raise
    except TypeError as error:
        raise SchedulerRefusal(
            f"PROVIDER_CONFIG_INVALID: {kind}: {error}"
        ) from error


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


class ExperimentScheduler:
    """Chooses where an experiment runs, fail-closed."""

    def __init__(self, providers: list[Any]) -> None:
        if not providers:
            raise SchedulerRefusal(
                "NO_PROVIDER_AVAILABLE: the scheduler was constructed with no "
                "providers; refusing rather than assuming local"
            )
        self._providers: dict[str, Any] = {}
        self._order: list[str] = []
        for provider in providers:
            if provider.name in self._providers:
                raise SchedulerRefusal(f"DUPLICATE_PROVIDER: {provider.name}")
            self._providers[provider.name] = provider
            self._order.append(provider.name)

    def providers(self) -> tuple[str, ...]:
        return tuple(self._order)

    def schedule(self, request: ExperimentRequest) -> Submission:
        """Route one request. Screening requests prefer a declared screening
        lane (before the general preference order); everything else follows
        declaration order. A pinned request runs nowhere else: an unknown pin
        refuses by name, and a pin the scheduler cannot honor right now
        refuses rather than rerouting (replication-on-different-hardware must
        not silently become replication-on-the-same-machine)."""
        if request.require_provider is not None:
            provider = self._providers.get(request.require_provider)
            if provider is None:
                raise SchedulerRefusal(
                    f"UNKNOWN_PROVIDER_PIN: {request.require_provider!r} is not a "
                    "declared provider; refusing rather than rerouting"
                )
            if not provider.available():
                raise SchedulerRefusal(
                    f"PINNED_PROVIDER_UNAVAILABLE: {request.require_provider!r} "
                    "cannot accept work now"
                )
            return provider.submit(request)
        candidates: list[str] = list(self._order)
        if request.experiment_class == ExperimentClass.SCREENING:
            screening_first = [n for n in self._order
                               if getattr(self._providers[n], "screening", False)]
            candidates = screening_first + [n for n in self._order
                                            if n not in screening_first]
        for name in candidates:
            provider = self._providers[name]
            if not provider.available():
                continue
            cost = provider.estimate_cost(request)
            if cost > provider.quota().remaining_gpu_hours():
                continue
            try:
                return provider.submit(request)
            except SchedulerRefusal:
                continue
        raise SchedulerRefusal(
            "NO_PROVIDER_AVAILABLE: every provider refused this request "
            "(unavailable or over quota)"
        )
