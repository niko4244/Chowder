"""Compute providers: scheduling policy and the hardware-context evidence
rules (docs/COMPUTE_PROVIDERS.md).

Pinned:
- screening requests prefer the declared screening lane; substantial requests
  follow declaration order;
- an exhausted quota falls back along the declared order; a pinned request
  never reroutes and an unknown pin refuses;
- an empty provider list refuses (no default provider);
- device GPU-hours are honest (Kaggle wall × 2 accelerators);
- a hardware-dependent (efficiency) claim cannot reach `replicated` citing
  runs from two hardware classes (requirement: efficiency results keep their
  hardware context);
- a quality claim may cite cross-hardware runs;
- observations carry the hardware class; the mission ledger charges the same
  regardless of provider;
- screening → survivor batching is score-driven, journaled on refusal.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from pathlib import Path

import pytest

from chowder.scientist.compute import (
    ExperimentClass,
    ExperimentRequest,
    ExperimentScheduler,
    KaggleProvider,
    LocalCudaProvider,
    ProviderQuota,
    RunPodProvider,
    SchedulerRefusal,
    Submission,
)
from chowder.scientist.observation import ExperimentObservation, Measurement
from chowder.scientist.research_memory import ResearchMemory
from chowder.scientist.research_tree import ResearchTree


def _request(**kw) -> ExperimentRequest:
    base = dict(experiment_id="e1", proposal_id="p1", hypothesis_id="h1",
                campaign_spec={}, estimated_gpu_hours=0.5)
    base.update(kw)
    return ExperimentRequest(**base)


def _providers(*, kaggle_hours: float = 10.0):
    # push=False: the routing tests exercise scheduling, not the network —
    # the real-push behavior has its own stub-client tests below.
    local = LocalCudaProvider(accelerators=1)
    kaggle = KaggleProvider(username="u", api_key="k", weekly_gpu_hours=kaggle_hours,
                            push=False)
    return local, kaggle


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_screening_prefers_the_screening_lane() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    submission = scheduler.schedule(_request(experiment_class=ExperimentClass.SCREENING))
    assert submission.provider_name == "kaggle"
    assert submission.hardware_class == "kaggle_2x_t4_16gb"
    # honest device-hours: wall 0.5 × 2 T4s
    assert submission.device_gpu_hours == pytest.approx(1.0)


def test_substantial_follows_declaration_order() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    submission = scheduler.schedule(_request(experiment_class=ExperimentClass.SUBSTANTIAL))
    assert submission.provider_name == "local_cuda"


def test_quota_exhaustion_falls_back_along_the_order() -> None:
    local, kaggle = _providers(kaggle_hours=1.0)
    scheduler = ExperimentScheduler([kaggle, local])
    # burn the kaggle quota (0.5 wall × 2 = 1.0 device-hours)
    scheduler.schedule(_request(experiment_class=ExperimentClass.SCREENING))
    submission = scheduler.schedule(_request(experiment_id="e2",
                                             experiment_class=ExperimentClass.SCREENING))
    assert submission.provider_name == "local_cuda"
    assert kaggle.available() is False


def test_pinned_request_never_reroutes_and_unknown_pin_refuses() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    pinned = scheduler.schedule(_request(require_provider="local_cuda"))
    assert pinned.provider_name == "local_cuda"
    with pytest.raises(SchedulerRefusal, match="UNKNOWN_PROVIDER_PIN"):
        scheduler.schedule(_request(require_provider="vast"))


def test_unconfigured_kaggle_is_unavailable_not_fake(tmp_path, monkeypatch) -> None:
    # isolate from the machine's real kaggle.json / KAGGLE_* env credentials
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no-kaggle-here"))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    provider = KaggleProvider()
    assert provider.configured() is False
    assert provider.available() is False
    scheduler = ExperimentScheduler([LocalCudaProvider(accelerators=0), provider])
    with pytest.raises(SchedulerRefusal, match="NO_PROVIDER_AVAILABLE"):
        scheduler.schedule(_request())


def test_env_and_config_credentials_resolve_without_importing_kaggle(
        tmp_path, monkeypatch) -> None:
    # env credentials are enough to be configured — and checking them must
    # not import kaggle (its authenticate() exits the process when unauthenticated)
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    provider = KaggleProvider(push=False)
    assert provider.configured() is True
    assert provider.resolved_username() == "env-user"
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text('{"username": "cfg-user"}', encoding="utf-8")
    assert provider.configured() is True  # kaggle.json presence is enough
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "empty"))
    assert provider.configured() is False


def test_empty_provider_list_refuses() -> None:
    with pytest.raises(SchedulerRefusal, match="NO_PROVIDER_AVAILABLE"):
        ExperimentScheduler([])


def test_local_zero_accelerators_is_unavailable() -> None:
    provider = LocalCudaProvider(accelerators=0)
    assert provider.available() is False


# ---------------------------------------------------------------------------
# the hardware-context rule
# ---------------------------------------------------------------------------


def _memory(tmp_path: Path) -> ResearchMemory:
    return ResearchMemory(tmp_path / "research",
                          run_exists=lambda r: True, run_complete=lambda r: True)


def _observation(tmp_path: Path, *, obs_id: str, run_id: str, hardware_class: str,
                 surface: str = "reasoning", value: float = 0.6):
    memory = _memory(tmp_path)
    observation = ExperimentObservation(
        observation_id=obs_id, run_id=run_id, experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface=surface, benchmark="b", value=value),),
        wall_gpu_hours=0.3, hardware_class=hardware_class,
    )
    memory.record_observation(observation)
    return memory, observation


def test_efficiency_claim_cannot_replicate_across_hardware(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-a", run_id="run-kaggle",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=42.0)
    memory, _ = _observation(tmp_path, obs_id="obs-b", run_id="run-local",
                             hardware_class="local_rtx", surface="efficiency:tokens_per_sec",
                             value=55.0)
    finding = ResearchFinding(
        finding_id="f1", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c1",
            statement="training reaches 42-55 tokens/sec",
            scope="test",
            status="replicated",
            supporting_experiments=("run-kaggle", "run-local"),
            hardware_dependent=True,
        ),),
        observation_ids=("obs-a", "obs-b"),
    )
    with pytest.raises(ValueError, match="HARDWARE_CONTEXT"):
        memory.record_finding(finding)


def test_efficiency_claim_replicates_on_one_hardware_class(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-a2", run_id="run-k1",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=42.0)
    memory, _ = _observation(tmp_path, obs_id="obs-b2", run_id="run-k2",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=43.0)
    finding = ResearchFinding(
        finding_id="f2", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c2", statement="2×T4 reaches ~42 tokens/sec", scope="test",
            status="replicated",
            supporting_experiments=("run-k1", "run-k2"),
            hardware_dependent=True,
        ),),
        observation_ids=("obs-a2", "obs-b2"),
    )
    memory.record_finding(finding)  # accepted: one hardware class
    assert memory.findings()[0].claims[0].status == "replicated"


def test_quality_claim_may_cite_cross_hardware_runs(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-q1", run_id="run-q1",
                             hardware_class="kaggle_2x_t4_16gb")
    memory, _ = _observation(tmp_path, obs_id="obs-q2", run_id="run-q2",
                             hardware_class="local_rtx")
    finding = ResearchFinding(
        finding_id="f3", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c3", statement="replay decay improves reasoning", scope="test",
            status="replicated",
            supporting_experiments=("run-q1", "run-q2"),
            hardware_dependent=False,
        ),),
        observation_ids=("obs-q1", "obs-q2"),
    )
    memory.record_finding(finding)  # accepted: quality is protocol-scoped, not hardware-scoped
    assert memory.findings()[0].claims[0].status == "replicated"


def test_observation_serialization_carries_hardware_class(tmp_path: Path) -> None:
    memory, observation = _observation(tmp_path, obs_id="obs-h", run_id="run-h",
                                       hardware_class="kaggle_2x_t4_16gb")
    stored = memory.observations()[0]
    assert stored.hardware_class == "kaggle_2x_t4_16gb"


# ---------------------------------------------------------------------------
# director-side batching
# ---------------------------------------------------------------------------


def _director_with_tree(tmp_path: Path):
    from chowder.scientist import (
        Hypothesis, MissionBudget, ResearchMission, ResearchQuestion,
    )
    from chowder.scientist.research_director import ResearchDirector
    mission = ResearchMission(
        mission_id="m-c", objective="improve reasoning",
        priorities={"reasoning": 1.0},
        budget=MissionBudget(max_gpu_hours=8.0, max_tree_nodes=10, max_parallel_branches=3),
    )
    memory = ResearchMemory(tmp_path / "research")
    tree = ResearchTree(mission_id=mission.mission_id)
    director = ResearchDirector(
        mission=mission, provider=_FakeProvider(), memory=memory, tree=tree,
    )
    return director


class _FakeProvider:
    name = "fake_deterministic"

    def available(self):
        return True

    def propose_hypotheses(self, context, *, count=3):
        return ()

    def propose_experiments(self, context, hypotheses):
        return ()

    def interpret(self, context, hypothesis, observations):
        raise NotImplementedError

    def export_state(self):
        return {}


def test_screening_and_survivor_batches_route_and_journal(tmp_path: Path) -> None:
    from chowder.scientist.lab_bridge import CompiledExperiment
    from chowder.scientist.proposal import (
        DataStrategy, ExperimentProposal, TrainingRecipeDelta,
    )
    director = _director_with_tree(tmp_path)
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([kaggle, local])

    def _pair(i: int):
        proposal = ExperimentProposal(
            proposal_id=f"p{i}", hypothesis_id=f"h{i}", experiment_type="data",
            intervention="x", variables_changed=("replay_ratio",),
            variables_held_constant=("lr",),
            training_recipe_delta=TrainingRecipeDelta(epochs=1),
            data_strategy=DataStrategy(source_kinds=("curriculum",)),
            requested_evaluations=("reasoning",), expected_outcome="up",
            falsification_rule="delta<=0", estimated_gpu_hours=0.25,
        )
        experiment = CompiledExperiment(
            experiment_id=f"sciexp-p{i}", proposal_id=f"p{i}", hypothesis_id=f"h{i}",
            campaign_spec={"recipe_patch": {}}, estimated_gpu_hours=0.25,
        )
        return proposal, experiment

    triples = director.submit_screening_batch(scheduler, (_pair(0), _pair(1), _pair(2)))
    assert len(triples) == 3
    assert all(t[2].provider_name == "kaggle" for t in triples)
    assert kaggle.quota().used_gpu_hours == pytest.approx(3 * 0.5)

    # record observations so the tree can score, then promote survivors
    for i, (proposal, experiment, submission) in enumerate(triples):
        director.record_observation(ExperimentObservation(
            observation_id=f"obs-{i}", run_id=f"run-{i}",
            experiment_ref=experiment.experiment_id,
            proposal_id=proposal.proposal_id, hypothesis_id=proposal.hypothesis_id,
            measurements=(Measurement(surface="reasoning", benchmark="b",
                                      value=(0.9 if i == 1 else 0.1),),),
            wall_gpu_hours=0.25, hardware_class=submission.hardware_class,
        ))
    survivors = director.submit_survivor_batch(scheduler, triples, keep_top=1)
    assert len(survivors) == 1
    assert survivors[0][0].proposal_id == "p1"  # the 0.9-quality branch
    assert survivors[0][2].experiment_class == ExperimentClass.SUBSTANTIAL


def test_mission_ledger_charges_identically_across_providers(tmp_path: Path) -> None:
    director = _director_with_tree(tmp_path)
    before = director.spend.spent_gpu_hours
    director.record_observation(ExperimentObservation(
        observation_id="obs-led", run_id="run-led", experiment_ref="sciexp-x",
        proposal_id="x", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.5),),
        wall_gpu_hours=0.75, hardware_class="kaggle_2x_t4_16gb",
    ))
    assert director.spend.spent_gpu_hours == pytest.approx(before + 0.75)


# ---------------------------------------------------------------------------
# kaggle: real kernel-push, session polling, quota reconciliation
# (stub client — the contract under test is the provider's behavior, not
# Kaggle's network)
# ---------------------------------------------------------------------------


class _FakeKaggleResponse:
    def __init__(self, **fields):
        self.error = ""
        self.ref = ""
        self.url = ""
        self.version_number = 0
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeStatusResponse:
    def __init__(self, status_name: str, failure_message: str = ""):
        class _S:  # KernelWorkerStatus-shaped
            name = status_name
        self.status = _S
        self.failure_message = failure_message


class _FakeKaggleApi:
    """KaggleApi-shaped stub: kernels_push / kernels_status / kernels_output /
    quota_view, serving a scripted session lifecycle."""

    def __init__(self, *, push_error: str = "", raise_on_push: Exception | None = None,
                 status_sequence: tuple[str, ...] = ("QUEUED", "RUNNING", "COMPLETE"),
                 failure_message: str = "",
                 result: dict | None = None, fingerprint: dict | None = None,
                 quota: tuple[float, float, float] | None = None,
                 push_ref: str = "u/chowder-e1") -> None:
        self.pushed: list[dict] = []
        self.status_calls: list[str] = []
        self.output_calls: list[str] = []
        self._push_error = push_error
        self._raise_on_push = raise_on_push
        self._status_sequence = list(status_sequence)
        self._failure_message = failure_message
        self._result = result
        self._fingerprint = fingerprint
        self._quota = quota
        self._push_ref = push_ref

    def kernels_push(self, folder, timeout=None, acc=None):
        self.pushed.append({"folder": folder, "timeout": timeout, "acc": acc})
        if self._raise_on_push is not None:
            raise self._raise_on_push
        if self._push_error:
            return _FakeKaggleResponse(error=self._push_error)
        return _FakeKaggleResponse(ref=self._push_ref, url="https://kaggle/u/e1",
                                   version_number=1)

    def kernels_status(self, ref):
        self.status_calls.append(ref)
        name = self._status_sequence[min(len(self.status_calls) - 1,
                                         len(self._status_sequence) - 1)]
        return _FakeStatusResponse(name, self._failure_message)

    def kernels_output(self, ref, path, file_pattern=None, force=False, **kw):
        import json as _json
        from pathlib import Path as _P
        out = _P(path)
        out.mkdir(parents=True, exist_ok=True)
        if self._result is not None:
            (out / "chowder_result.json").write_text(_json.dumps(self._result),
                                                     encoding="utf-8")
        if self._fingerprint is not None:
            (out / "environment.json").write_text(_json.dumps(self._fingerprint),
                                                  encoding="utf-8")
        files = [str(out / n) for n in ("chowder_result.json", "environment.json")]
        self.output_calls.append(ref)
        return files, ""

    def quota_view(self):
        import datetime as _dt
        total, used, reserved = self._quota or (0.0, 0.0, 0.0)

        class _Q:
            total_time_allowed = _dt.timedelta(hours=total)
            time_used = _dt.timedelta(hours=used)
            time_reserved = _dt.timedelta(hours=reserved)

        class _V:
            gpu_quota = _Q()

        return _V()


def _push_provider(**kw):
    base = dict(username="u", api_key="k", chowder_commit="a" * 40,
                kernel_command="python /kaggle/working/run_experiment.py",
                workdir=None, api=None)
    base.update(kw)
    return KaggleProvider(**base)


def _push_request(**kw):
    return _request(experiment_class=ExperimentClass.SCREENING,
                    estimated_gpu_hours=0.5, **kw)


def test_kaggle_real_push_requires_a_pinned_commit() -> None:
    stub = _FakeKaggleApi()
    provider = _push_provider(chowder_commit="", api=stub)
    with pytest.raises(SchedulerRefusal, match="KAGGLE_KERNEL_PIN_REQUIRED"):
        provider.submit(_push_request())
    provider2 = _push_provider(chowder_commit="main", api=stub)  # never a branch
    with pytest.raises(SchedulerRefusal, match="KAGGLE_KERNEL_PIN_REQUIRED"):
        provider2.submit(_push_request())
    assert stub.pushed == []  # nothing ever shipped


def test_kaggle_real_push_requires_an_executor_command() -> None:
    stub = _FakeKaggleApi()
    provider = _push_provider(kernel_command="", api=stub)
    with pytest.raises(SchedulerRefusal, match="KAGGLE_EXECUTOR_NOT_CONFIGURED"):
        provider.submit(_push_request())
    assert stub.pushed == []
    assert provider.quota().used_gpu_hours == 0  # a refusal never charges quota


def test_kaggle_real_push_writes_kernel_and_records_ref(tmp_path) -> None:
    import json
    stub = _FakeKaggleApi()
    provider = _push_provider(api=stub, workdir=str(tmp_path))
    submission = provider.submit(_push_request())
    assert submission.provider_ref == "u/chowder-e1"
    assert submission.status == "queued"
    assert submission.device_gpu_hours == pytest.approx(1.0)  # wall 0.5 × 2 T4s
    assert provider.quota().used_gpu_hours == pytest.approx(1.0)
    assert len(stub.pushed) == 1
    call = stub.pushed[0]
    assert call["acc"] == "NvidiaTeslaT4"  # the T4×2 machine shape
    folder = tmp_path / "chowder-e1"
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "u/chowder-e1"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_type"] == "script"
    script = (folder / "kernel.py").read_text(encoding="utf-8")
    assert "a" * 40 in script  # pinned commit travels with the kernel
    assert 'EXPERIMENT_ID = "e1"' in script
    assert "run_experiment.py" not in script  # command is base64, not plaintext
    assert "campaign_spec.json" in script


def test_push_response_ref_is_normalized_to_owner_slug(tmp_path) -> None:
    # the live kernels_push returns a URL-path ref ('/code/{owner}/{slug}');
    # the stored provider_ref must be pollable as {owner}/{slug}
    stub = _FakeKaggleApi(push_ref="/code/nikmarco/chowder-e1")
    provider = _push_provider(api=stub, workdir=str(tmp_path), username="nikmarco")
    submission = provider.submit(_push_request())
    assert submission.provider_ref == "nikmarco/chowder-e1"
    assert provider.poll(submission).status == "queued"  # status accepts it


def test_kaggle_push_api_errors_refuse_and_stay_honest(tmp_path) -> None:
    provider = _push_provider(api=_FakeKaggleApi(raise_on_push=RuntimeError("503")),
                              workdir=str(tmp_path))
    with pytest.raises(SchedulerRefusal, match="KAGGLE_API_ERROR"):
        provider.submit(_push_request())
    provider2 = _push_provider(api=_FakeKaggleApi(push_error="invalid metadata"),
                               workdir=str(tmp_path))
    with pytest.raises(SchedulerRefusal, match="KAGGLE_PUSH_REFUSED"):
        provider2.submit(_push_request())
    assert provider2.quota().used_gpu_hours == 0


def test_kaggle_poll_maps_the_real_status_lifecycle(tmp_path) -> None:
    result = {"experiment_id": "e1", "status": "complete", "exit_code": 0}
    fingerprint = {"python_version": "3.11.9", "gpu": ["Tesla T4", "Tesla T4"]}
    stub = _FakeKaggleApi(status_sequence=("QUEUED", "RUNNING", "COMPLETE"),
                          result=result, fingerprint=fingerprint)
    provider = _push_provider(api=stub, workdir=str(tmp_path))
    submission = provider.submit(_push_request())
    assert provider.poll(submission).status == "queued"
    running = provider.poll(submission)
    assert running.status == "running"
    complete = provider.poll(submission)
    assert complete.status == "complete"
    assert complete.result["chowder_result"] == result
    assert complete.result["output_files"] == ["chowder_result.json", "environment.json"]
    assert complete.environment_fingerprint == fingerprint
    assert stub.status_calls == ["u/chowder-e1"] * 3


def test_kaggle_poll_failure_travels_with_the_submission(tmp_path) -> None:
    stub = _FakeKaggleApi(status_sequence=("ERROR",),
                          failure_message="CUDA out of memory")
    provider = _push_provider(api=stub, workdir=str(tmp_path))
    submission = provider.submit(_push_request())
    failed = provider.poll(submission)
    assert failed.status == "failed"
    assert failed.result["kernel_status"] == "ERROR"
    assert failed.result["failure_message"] == "CUDA out of memory"


def test_output_fetch_survives_client_console_encode_errors(tmp_path) -> None:
    """The kaggle client prints a console summary AFTER downloading; on a
    cp1252 Windows console that print can raise UnicodeEncodeError even
    though the artifacts are on disk. The files are the evidence."""
    class _EncodeErrorApi(_FakeKaggleApi):
        def kernels_output(self, ref, path, **kw):
            from pathlib import Path as _P
            out = _P(path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "chowder_result.json").write_text(
                '{"status": "complete", "exit_code": 0}', encoding="utf-8")
            raise UnicodeEncodeError("charmap", "x" * 100, 55, 94, "<undefined>")

    provider = _push_provider(api=_EncodeErrorApi(status_sequence=("COMPLETE",)),
                              workdir=str(tmp_path))
    submission = provider.submit(_push_request())
    complete = provider.poll(submission)
    assert complete.status == "complete"
    assert complete.result["output_files"] == ["chowder_result.json"]
    assert complete.result["chowder_result"]["status"] == "complete"


def test_kaggle_quota_sync_uses_the_real_weekly_budget(tmp_path) -> None:
    stub = _FakeKaggleApi(quota=(30.0, 5.0, 2.0))  # 30h allowed, 5 used, 2 reserved
    provider = _push_provider(api=stub, workdir=str(tmp_path),
                              weekly_gpu_hours=20.0)
    quota = provider.sync_quota_from_api()
    assert quota.weekly_gpu_hours == pytest.approx(30.0)
    assert quota.used_gpu_hours == pytest.approx(7.0)  # used + reserved
    assert quota.remaining_gpu_hours() == pytest.approx(23.0)


def test_kaggle_pushless_poll_is_a_no_op_not_fake_progress() -> None:
    stub = _FakeKaggleApi()
    provider = _push_provider(api=stub, push=False)
    submission = provider.submit(_push_request())
    assert submission.provider_ref == ""  # never left the process
    assert provider.poll(submission) is submission  # stays queued, honestly
    assert stub.status_calls == []


# ---------------------------------------------------------------------------
# runpod: REST v2 pod lifecycle through an injectable transport
# ---------------------------------------------------------------------------


class _StubTransport:
    """Transport stub: records calls, answers from a script of (status, body)."""

    def __init__(self, responses):
        self.responses = list(responses)  # list of (status, body) or callables
        self.calls: list[dict] = []

    def __call__(self, request):
        self.calls.append(request)
        answer = self.responses.pop(0) if self.responses else (500, "{}")
        if callable(answer):
            return answer(request)
        status, body = answer
        if isinstance(body, (dict, list)):
            import json
            body = json.dumps(body)
        return status, body


def _runpod(**kw):
    base = dict(api_key="rp-key", gpu_type_id="NVIDIA A100", image="chowder/train:pin",
                command="python run_experiment.py", weekly_gpu_hours=8.0,
                result_fetcher=lambda submission: {"status": "complete",
                                                   "device_gpu_hours": 0.31})
    base.update(kw)
    return RunPodProvider(**base)


def test_runpod_unconfigured_refuses_and_names_what_is_missing() -> None:
    provider = RunPodProvider()
    assert provider.available() is False
    with pytest.raises(SchedulerRefusal, match="RUNPOD_NOT_CONFIGURED") as caught:
        provider.submit(_request())
    message = str(caught.value)
    for needed in ("RUNPOD_API_KEY", "gpu_type_id", "image", "command"):
        assert needed in message


def test_runpod_submit_creates_pod_carries_spec_and_charges_quota() -> None:
    transport = _StubTransport([(201, {"id": "pod_abc123", "status": "PROVISIONING"})])
    provider = _runpod(transport=transport, gpu_count=1)
    submission = provider.submit(_request())
    assert submission.provider_ref == "pod_abc123"
    assert submission.hardware_class == "runpod_1x_nvidia_a100"
    assert provider.quota().used_gpu_hours == pytest.approx(0.5)
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/v2/pods")
    assert call["headers"]["Authorization"] == "Bearer rp-key"
    body = call["body"]
    assert body["name"] == "chowder-e1"
    assert body["gpu"] == {"id": "NVIDIA A100", "count": 1}
    assert body["cmd"] == ["/bin/sh", "-lc", "python run_experiment.py"]
    import json
    spec = json.loads(body["env"]["CHOWDER_CAMPAIGN_SPEC"])
    assert spec == {} or isinstance(spec, dict)
    assert body["env"]["CHOWDER_EXPERIMENT_ID"] == "e1"


def test_runpod_measured_hours_replace_the_estimate() -> None:
    transport = _StubTransport([
        (201, {"id": "pod_abc", "status": "PROVISIONING"}),
        (200, {"id": "pod_abc", "status": "EXITED"}),
    ])
    provider = _runpod(transport=transport, gpu_count=2)
    submission = provider.submit(_request(estimated_gpu_hours=0.5))
    assert submission.device_gpu_hours == pytest.approx(1.0)  # 0.5 × 2 estimate
    complete = provider.poll(submission)  # fetcher returns measured device hours
    assert complete.status == "complete"
    assert complete.device_gpu_hours == pytest.approx(0.31)  # measured wins
    assert provider.quota().used_gpu_hours == pytest.approx(0.31)


def test_runpod_submit_api_error_refuses_without_charging() -> None:
    provider = _runpod(transport=_StubTransport([(503, "no capacity")]))
    with pytest.raises(SchedulerRefusal, match="RUNPOD_API_ERROR.*503"):
        provider.submit(_request())
    assert provider.quota().used_gpu_hours == 0


def test_kaggle_measured_hours_replace_the_estimate(tmp_path) -> None:
    stub = _FakeKaggleApi(
        status_sequence=("COMPLETE",),
        result={"status": "complete", "exit_code": 0,
                "wall_seconds": 112.4, "wall_gpu_hours": 0.031222,
                "device_gpu_hours": 0.062444, "accelerator_count": 2},
    )
    provider = _push_provider(api=stub, workdir=str(tmp_path))
    submission = provider.submit(_push_request())  # estimate: 0.5 × 2 = 1.0
    complete = provider.poll(submission)
    assert complete.status == "complete"
    assert complete.device_gpu_hours == pytest.approx(0.062444)  # measured wins
    assert provider.quota().used_gpu_hours == pytest.approx(0.062444)  # settled


def test_kaggle_measured_hours_survive_a_result_that_omits_them(tmp_path) -> None:
    stub = _FakeKaggleApi(status_sequence=("COMPLETE",),
                          result={"status": "complete"})  # old-style result
    provider = _push_provider(api=stub, workdir=str(tmp_path))
    submission = provider.submit(_push_request())
    complete = provider.poll(submission)
    assert complete.status == "complete"
    assert complete.device_gpu_hours == pytest.approx(1.0)  # estimate stands
    assert provider.quota().used_gpu_hours == pytest.approx(1.0)


def test_runpod_poll_lifecycle_maps_statuses() -> None:
    transport = _StubTransport([
        (201, {"id": "pod_abc", "status": "PROVISIONING"}),
        (200, {"id": "pod_abc", "status": "RUNNING"}),
        (200, {"id": "pod_abc", "status": "EXITED"}),
    ])
    provider = _runpod(transport=transport, result_fetcher=None)  # no fetcher
    submission = provider.submit(_request())
    assert provider.poll(submission).status == "running"
    exited = provider.poll(submission)
    # EXITED without a confirmed artifact is NOT a complete: fail-closed
    assert exited.status == "failed"
    assert "RUNPOD_EXIT_UNVERIFIED" in exited.result["failure_message"]


def test_runpod_exited_pod_completes_only_with_a_confirmed_artifact() -> None:
    transport = _StubTransport([
        (201, {"id": "pod_abc", "status": "PROVISIONING"}),
        (200, {"id": "pod_abc", "status": "EXITED"}),
    ])
    provider = _runpod(
        transport=transport,
        result_fetcher=lambda submission: {"experiment_id": "e1", "status": "complete"},
    )
    submission = provider.submit(_request())
    complete = provider.poll(submission)
    assert complete.status == "complete"
    assert complete.result["chowder_result"]["experiment_id"] == "e1"


def test_runpod_terminate_deletes_the_pod() -> None:
    transport = _StubTransport([
        (201, {"id": "pod_abc", "status": "PROVISIONING"}),
        (200, {"id": "pod_abc"}),
    ])
    provider = _runpod(transport=transport)
    submission = provider.submit(_request())
    assert provider.terminate(submission) is True
    assert transport.calls[-1]["method"] == "DELETE"
    assert transport.calls[-1]["url"].endswith("/pods/pod_abc")


def test_scheduler_routes_across_three_providers() -> None:
    """The seam proof: local, Kaggle, and RunPod coexist; screening prefers
    the declared lane, and a substantial run falls through exhaustion to the
    next provider in declaration order."""
    kaggle = KaggleProvider(username="u", api_key="k", weekly_gpu_hours=1.0, push=False)
    runpod = _runpod(transport=_StubTransport([(201, {"id": "pod_x", "status": "RUNNING"})]))
    local = LocalCudaProvider(accelerators=1)
    scheduler = ExperimentScheduler([kaggle, runpod, local])
    screening = scheduler.schedule(_push_request())
    assert screening.provider_name == "kaggle"  # the screening lane
    # kaggle quota is now out (1.0 device-hours); substantial falls through
    substantial = scheduler.schedule(_request(experiment_class=ExperimentClass.SUBSTANTIAL))
    assert substantial.provider_name == "runpod"
    third = scheduler.schedule(_request(experiment_id="e3",
                                        experiment_class=ExperimentClass.SUBSTANTIAL))
    assert third.provider_name == "local_cuda"
