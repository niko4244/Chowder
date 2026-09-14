"""The router-healing backend: what it refuses, and what it proves with a real model.

Two layers are tested here, deliberately separated by cost:

* The **contract** layer (spec refusals, registration, the non-computing
  ``profile``, and the parent's re-validation of a worker result) needs no model
  and runs in every job.
* The **real** layer builds a genuine tiny Qwen3 MoE once and drives actual
  subprocesses through save / resume / abrupt death. Those tests are gated on
  torch and transformers and therefore execute in the real-ML job, where a run
  that silently skipped would be visible in the skip count.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chowder.backend_selection import (
    ROUTER_HEALING_ENGINE,
    create_evaluation_executor,
    create_training_executor,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
from chowder.backends.router_healing import (
    EVAL_WORKER_RESULT_KIND,
    QUALIFIED_DEVICES,
    WORKER_RESULT_KIND,
    RouterHealingBackendError,
    RouterHealingEvalSpec,
    RouterHealingEvaluationError,
    RouterHealingEvaluator,
    RouterHealingExecutor,
    RouterHealingRunSpec,
)
from chowder.backends.router_healing_worker import train
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.lifecycle import PhaseTimer, training_lifecycle_ledger
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_file


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        vram_gb=0.0,
        ram_gb=64.0,
        nvme_gb=1000.0,
        pcie_gbps=16.0,
        ram_gbps=50.0,
        nvme_gbps=3.0,
    )


def _require_real_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")


def _spec_kwargs(**overrides) -> dict:
    base = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "corpus_path": "unused-corpus",
        "corpus_sha256": "b" * 64,
        "output_dir": "unused-out",
        "max_steps": 4,
        "learning_rate": 0.01,
        "seq_len": 16,
        "batch_size": 2,
        "seed": 0,
        "probe_window": 2,
        "max_tokens": 1024,
    }
    base.update(overrides)
    return base


# --- the contract layer ------------------------------------------------------


def test_the_only_qualified_devices_are_cpu_and_cuda():
    assert set(QUALIFIED_DEVICES) == {"cpu", "cuda"}


def test_a_spec_pins_the_device_it_has_actually_qualified():
    """An unqualified accelerator run is refused, not attempted and discovered."""
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(**_spec_kwargs(device="mps"))
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(**_spec_kwargs(device="npu"))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"max_steps": 0}, "max_steps must be positive"),
        ({"seq_len": 0}, "seq_len and batch_size must be positive"),
        ({"batch_size": -1}, "non-negative integer"),
        ({"probe_window": 0}, "probe_window must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be finite and positive"),
        ({"learning_rate": float("nan")}, "learning_rate must be finite and positive"),
        ({"max_seconds": 0.0}, "max_seconds must be finite and positive"),
        ({"scheduler": "warp"}, "unsupported router healing scheduler"),
        ({"warmup_steps": 99}, "warmup_steps cannot exceed max_steps"),
        ({"max_tokens": 8}, "smaller than a single batch"),
        ({"base_content_sha256": "short"}, "sha256 hex digest"),
    ],
)
def test_a_spec_refuses_configurations_that_could_not_train_honestly(overrides, match):
    with pytest.raises(ValueError, match=match):
        RouterHealingRunSpec(**_spec_kwargs(**overrides))


def test_the_recipe_digest_ignores_operational_paths():
    """A relocated checkout is the same experiment, not a different one."""
    one = RouterHealingRunSpec(**_spec_kwargs(base_model_dir="D:/a", output_dir="D:/b"))
    two = RouterHealingRunSpec(**_spec_kwargs(base_model_dir="E:/x", output_dir="E:/y"))
    assert one.digest() != two.digest()
    assert one.recipe_digest() == two.recipe_digest()


def test_profile_is_non_computing_and_never_touches_the_base(tmp_path):
    """A preflight that loads weights is not a preflight.

    The base path here does not exist, so a ``profile`` that resolved, hashed or
    loaded it would fail; returning an estimate proves it stayed in config.
    """
    config = {
        "backend": {
            "type": ROUTER_HEALING_ENGINE,
            "router_healing": {
                "profile": {
                    "estimated_steps": 100,
                    "seconds_per_step": 0.36,
                    "peak_vram_gb": 0.0,
                    "source": "measured",
                }
            },
        }
    }
    experiment = Experiment(
        experiment_id="exp-router-profile",
        parent_id=None,
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="i", expected_deltas={}
        ),
        config_patch={},
        estimated_gpu_hours=1.0,
    )
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    estimate = RouterHealingExecutor().profile(experiment, context)
    assert estimate.gpu_hours == pytest.approx(100 * 0.36 / 3600.0)
    assert estimate.confidence == 0.75
    assert "accelerator hours are zero" in " ".join(estimate.notes)


def test_profile_falls_back_to_the_declared_estimate_without_a_step_profile(tmp_path):
    experiment = Experiment(
        experiment_id="exp-router-profile-2",
        parent_id=None,
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="i", expected_deltas={}
        ),
        config_patch={},
        estimated_gpu_hours=2.5,
    )
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config={"backend": {"type": ROUTER_HEALING_ENGINE}}
    )
    estimate = RouterHealingExecutor().profile(experiment, context)
    assert estimate.gpu_hours == pytest.approx(2.5)
    assert estimate.confidence == 0.25


def test_the_router_engine_is_registered_and_distinct_from_peft():
    assert resolve_training_engine({"backend": {"type": "router-healing"}}) == ROUTER_HEALING_ENGINE
    executor = create_training_executor({"backend": {"type": "router-healing"}})
    assert executor.name == "transformers-router-healing"
    # The canonical PEFT spellings still resolve exactly as before.
    assert resolve_training_engine({"backend": {"type": "peft", "engine": "transformers"}}) == (
        "transformers"
    )


def test_normalizing_a_router_config_leaves_its_identity_intact():
    config = {"backend": {"type": "router-healing", "engine": "router-healing", "x": 1}}
    assert normalize_training_config_for_executor(config) == config


def test_a_router_config_cannot_select_a_different_engine():
    with pytest.raises(ValueError, match="cannot select a different engine"):
        resolve_training_engine({"backend": {"type": "router-healing", "engine": "unsloth"}})


def test_an_incomplete_spec_is_refused_before_launch(tmp_path):
    experiment = Experiment(
        experiment_id="exp-router-incomplete",
        parent_id=None,
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="i", expected_deltas={}
        ),
        config_patch={},
        estimated_gpu_hours=1.0,
    )
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config={"backend": {"type": ROUTER_HEALING_ENGINE}}
    )
    with pytest.raises(RouterHealingBackendError, match="incomplete"):
        RouterHealingExecutor().run(experiment, context)


def test_cancelling_before_launch_stops_the_run_and_records_a_reason(
    tmp_path, monkeypatch, tiny_base
):
    """Graceful stop: the controller's cancel is a terminal reason, not a hang."""
    experiment = _experiment(tiny_base)
    context = _context(tmp_path)
    executor = RouterHealingExecutor()
    # The spec is fine, so the run would launch; cancel is armed first. The run
    # id is deterministic here because `uuid4` is pinned below.
    executor._cancelled.add("exp-router-backend-000000000000")

    class FakeProcess:
        returncode = 1

        def __init__(self, *args, **kwargs):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        "chowder.backends.router_healing.subprocess.Popen", FakeProcess
    )
    monkeypatch.setattr(
        "chowder.backends.router_healing.uuid4",
        lambda: type("U", (), {"hex": "000000000000"})(),
    )
    with pytest.raises(RouterHealingBackendError, match="cancelled by the controller"):
        executor.run(experiment, context)


# --- the parent re-validates a worker result ---------------------------------


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """A real tiny Qwen3 MoE (E=4, k=2) with a real locally-trained tokenizer."""
    _require_real_model()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from chowder.base_identity import resolve_base_identity

    root = tmp_path_factory.mktemp("router-healing-backend")
    lines = [
        f"the router selects expert {index % 4} for token number {index} in this sentence"
        for index in range(400)
    ]
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(lines), encoding="utf-8")
    # A separate holdout corpus: scoring the training data would measure fit, and
    # the evaluator is built to refuse exactly that.
    holdout = root / "holdout.txt"
    holdout.write_text(
        "\n".join(
            f"held out sentence {index} asks which expert answers token {index * 3}"
            for index in range(400)
        ),
        encoding="utf-8",
    )

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        lines, trainers.WordLevelTrainer(vocab_size=64, special_tokens=["[UNK]", "[PAD]"])
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]"
    )

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model = Qwen3MoeForCausalLM(config).float()
    base_dir = root / "base"
    model.save_pretrained(base_dir)
    fast.save_pretrained(base_dir)
    identity = resolve_base_identity(base_dir)
    return {
        "base_dir": str(base_dir),
        "corpus": str(corpus),
        "holdout": str(holdout),
        "manifest_sha256": identity["manifest_sha256"],
        "content_sha256": identity["content_sha256"],
        "corpus_sha256": sha256_file(corpus),
        "holdout_sha256": sha256_file(holdout),
    }


def _experiment(tiny_base, **research_overrides) -> Experiment:
    research = {
        "base_model_dir": tiny_base["base_dir"],
        "base_manifest_sha256": tiny_base["manifest_sha256"],
        "corpus_path": tiny_base["corpus"],
        "corpus_sha256": tiny_base["corpus_sha256"],
        "holdout_corpus_path": tiny_base["holdout"],
        "holdout_corpus_sha256": tiny_base["holdout_sha256"],
        "max_steps": 4,
        "learning_rate": 0.05,
        "seq_len": 16,
        "seed": 0,
    }
    research.update(research_overrides)
    return Experiment(
        experiment_id="exp-router-backend",
        parent_id=None,
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="i", expected_deltas={}
        ),
        config_patch={"router_healing": research},
        estimated_gpu_hours=1.0,
    )


def _context(tmp_path, **knobs) -> ExecutionContext:
    settings = {"batch_size": 2, "max_tokens": 4096, "probe_window": 2}
    settings.update(knobs)
    return ExecutionContext(
        _hardware(),
        str(tmp_path),
        1,
        resolved_config={
            "backend": {"type": ROUTER_HEALING_ENGINE, "router_healing": settings}
        },
    )


def _valid_result(spec: RouterHealingRunSpec, run_dir: Path) -> dict:
    load = PhaseTimer()
    load.seconds = 0.5
    ledger = training_lifecycle_ledger(
        accelerator_count=0,
        model_load=load,
        steady_state_steps_seconds=0.25,
    )
    return {
        "kind": WORKER_RESULT_KIND,
        "spec_digest": spec.digest(),
        "global_step": spec.max_steps,
        "steps_completed": spec.max_steps,
        "loss_first": 3.0,
        "loss_last": 2.0,
        "lifecycle": ledger.to_dict(),
        "trainability": {"ok": True, "intended_components": ["g"]},
        "frozen": {"ok": True, "changed": {}},
        "coverage": {
            "ok": True,
            "require_exact": True,
            "expected_count": 2,
            "actual_count": 2,
            "missing": [],
            "extra": [],
            "unreadable": [],
            "unknown_suffixes": [],
        },
        "payload": {"payload_dir": str(run_dir / "payload")},
        "base_identity": {"content_sha256": spec.base_content_sha256},
        "resource_usage": {
            "wall_seconds": 3.0,
            "active_accelerator_count": 0,
            "visible_accelerator_count": 0,
            "peak_vram_gb_by_accelerator": {},
        },
    }


class _FakeWorkerProcess:
    """Stands in for the worker subprocess: writes a result, then reports exit 0."""

    returncode = 0

    def __init__(self, command, **process_kwargs):
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = RouterHealingRunSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        result = _valid_result(spec, result_path.parent)
        self.mutate(result, spec)
        result_path.write_text(json.dumps(result), encoding="utf-8")

    _mutate = staticmethod(lambda result, spec: None)

    @property
    def mutate(self):
        return self._mutate

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def _install_fake_worker(monkeypatch, mutate):
    def factory(command, **kwargs):
        process = _FakeWorkerProcess(command, **kwargs)
        return process

    # The mutation is carried on the class so the constructor stays a plain
    # Popen-shaped callable.
    monkeypatch.setattr(_FakeWorkerProcess, "_mutate", staticmethod(mutate))
    monkeypatch.setattr("chowder.backends.router_healing.subprocess.Popen", factory)


def test_a_clean_worker_result_becomes_an_artifact_with_zero_accelerator_hours(
    tmp_path, monkeypatch, tiny_base
):
    _install_fake_worker(monkeypatch, lambda result, spec: None)
    executor = RouterHealingExecutor()
    artifact = executor.run(_experiment(tiny_base), _context(tmp_path))
    assert artifact.gpu_hours == 0.0
    assert artifact.evidence["trainability"]["ok"] is True
    assert artifact.evidence["coverage"]["ok"] is True
    # The parent rebuilt the ledger itself rather than trusting a summary string.
    assert artifact.evidence["phase_ledger"]["phases"]["model_load"]["measured"] is True


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r, s: r.update(kind="something-else"), "expected"),
        (lambda r, s: r.update(spec_digest="c" * 64), "different run"),
        (lambda r, s: r.update(trainability={"ok": False}), "did not demonstrate trainability"),
        (lambda r, s: r.update(frozen={"ok": False, "changed": {"a": 1}}), "frozen parameters changed"),
        (
            lambda r, s: r.update(
                coverage={
                    "ok": False,
                    "require_exact": True,
                    "missing": ["model.layers.1.mlp.gate.weight"],
                    "extra": [],
                    "unreadable": [],
                    "unknown_suffixes": [],
                }
            ),
            "coverage did not qualify",
        ),
        (lambda r, s: r.update(payload=None), "published no router payload"),
        (lambda r, s: r.update(resource_usage=None), "no resource usage"),
        (lambda r, s: r.update(lifecycle=None), "model_load"),
    ],
)
def test_a_worker_result_that_fails_any_check_is_refused(
    tmp_path, monkeypatch, tiny_base, mutate, match
):
    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match=match):
        RouterHealingExecutor().run(_experiment(tiny_base), _context(tmp_path))


def test_an_unmeasured_required_phase_cannot_qualify(tmp_path, monkeypatch, tiny_base):
    """A ledger that omits the model load is unknown, not zero."""

    def mutate(result, spec):
        load = PhaseTimer()
        load.seconds = 0.5
        result["lifecycle"] = training_lifecycle_ledger(
            accelerator_count=0, model_load=load, steady_state_steps_seconds=None
        ).to_dict()

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="steady_state_steps"):
        RouterHealingExecutor().run(_experiment(tiny_base), _context(tmp_path))


def test_a_base_that_does_not_match_the_frozen_manifest_is_refused(tmp_path, tiny_base):
    experiment = _experiment(tiny_base, base_manifest_sha256="f" * 64)
    with pytest.raises(RouterHealingBackendError, match="does not match the manifest"):
        RouterHealingExecutor().run(experiment, _context(tmp_path))


def test_a_corpus_that_does_not_match_its_recorded_hash_is_refused(tmp_path, tiny_base):
    experiment = _experiment(tiny_base, corpus_sha256="e" * 64)
    with pytest.raises(RouterHealingBackendError, match="corpus hash mismatch"):
        RouterHealingExecutor().run(experiment, _context(tmp_path))


# --- the real layer ----------------------------------------------------------


def test_a_real_tiny_moe_trains_through_the_normal_worker(tmp_path, tiny_base):
    _require_real_model()
    executor = RouterHealingExecutor()
    artifact = executor.run(
        _experiment(tiny_base), _context(tmp_path, checkpoint_every=2, detailed_timing=True)
    )

    assert artifact.evidence["trainability"]["ok"] is True
    assert artifact.evidence["frozen"]["ok"] is True
    assert artifact.evidence["scope"]["router_count"] == 2
    assert artifact.evidence["freeze_summary"]["trainable_param_names"] == [
        "model.layers.0.mlp.gate.weight",
        "model.layers.1.mlp.gate.weight",
    ]
    assert artifact.evidence["limits"]["stop_reason"] == "max_steps"
    assert artifact.telemetry["steps_completed"] == 4

    # The published payload is a real, complete, verified artifact.
    payload_dir = Path(artifact.artifact_ref)
    assert (payload_dir / "router_payload.safetensors").is_file()
    assert (payload_dir / "router_payload.json").is_file()
    from chowder.router_payload import load_router_payload

    payload = load_router_payload(
        payload_dir, expected_base_content_sha256=tiny_base["content_sha256"]
    )
    assert payload["parameter_names"] == [
        "model.layers.0.mlp.gate.weight",
        "model.layers.1.mlp.gate.weight",
    ]

    # Checkpoints are complete and resumable, and the ledger measured its phases.
    checkpoints = artifact.evidence["checkpoints"]
    assert [entry["global_step"] for entry in checkpoints] == [2, 4]
    from chowder.resume_state import assert_resumable, inventory_checkpoint

    for entry in checkpoints:
        inventory = inventory_checkpoint(entry["directory"])
        assert inventory.is_complete, inventory.notes
        assert_resumable(inventory)

    ledger = artifact.evidence["phase_ledger"]
    assert ledger["phases"]["model_load"]["measured"] is True
    assert ledger["phases"]["steady_state_steps"]["measured"] is True
    assert ledger["phases"]["checkpoint_publication"]["measured"] is True
    assert ledger["phases"]["first_forward"]["measured"] is True

    # CPU: generation is the evaluator's business, and it says so rather than 0.
    assert ledger["phases"]["baseline_generation"]["measured"] is False


def test_a_real_resume_continues_from_a_published_checkpoint(tmp_path, tiny_base):
    """The fixed-horizon control: interrupting at 2 and resuming must reach 4.

    Both runs declare the same total horizon, so the resumed run executes steps
    3 and 4 with the optimizer state the checkpoint carried -- the property a
    fresh start would silently fake.
    """
    _require_real_model()
    executor = RouterHealingExecutor()
    first = executor.run(_experiment(tiny_base), _context(tmp_path / "first", checkpoint_every=2))
    checkpoint = Path(first.evidence["checkpoints"][-2]["directory"])
    assert checkpoint.name == "step-2"

    second = executor.run(
        _experiment(tiny_base),
        _context(tmp_path / "second", resume_from=str(checkpoint)),
    )
    witness = second.evidence["resume"]
    assert witness["matched"] is True
    assert witness["restored_global_step"] == 2
    assert witness["final_global_step"] == 4
    assert witness["steps_executed"] == 2
    assert witness["progress_state"] == "advanced"
    assert witness["requested_checkpoint"] == str(checkpoint)
    # Sample position is recorded, so "same data order" is inspectable rather
    # than assumed from a matching loss.
    assert first.evidence["limits"]["samples_consumed"] == 8
    assert second.evidence["limits"]["samples_consumed"] == 4

    # The equivalence half: an uninterrupted 4-step run and a run interrupted at
    # 2 and resumed to 4 must land on the same weights. Same horizon, same
    # restored optimizer state, same sample order.
    torch = pytest.importorskip("torch")
    from chowder.router_payload import load_router_payload

    straight = load_router_payload(
        first.artifact_ref, expected_base_content_sha256=tiny_base["content_sha256"]
    )
    continued = load_router_payload(
        second.artifact_ref, expected_base_content_sha256=tiny_base["content_sha256"]
    )
    assert sorted(straight["tensors"]) == sorted(continued["tensors"])
    for name in straight["tensors"]:
        assert torch.allclose(
            straight["tensors"][name], continued["tensors"][name], rtol=0.0, atol=1e-6
        ), f"continuation diverged from the fixed-horizon control at {name}"


def test_a_crashed_worker_leaves_a_terminal_reason_and_its_cost(tmp_path, monkeypatch, tiny_base):
    """A failed attempt is evidence: the reason and the wall time are durable."""

    class FailingProcess:
        returncode = 1

        def __init__(self, command, **kwargs):
            self.run_dir = Path(command[command.index("--spec") + 1]).parent

        def poll(self):
            return 1

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(
        "chowder.backends.router_healing.subprocess.Popen", FailingProcess
    )
    with pytest.raises(RouterHealingBackendError, match="exited with code 1"):
        RouterHealingExecutor().run(_experiment(tiny_base), _context(tmp_path))

    failures = list((Path(tmp_path) / ".chowder" / "runs").glob("*/run-failure.json"))
    assert len(failures) == 1, "a failed run must leave exactly one terminal record"
    record = json.loads(failures[0].read_text(encoding="utf-8"))
    assert record["kind"] == "router_healing_run_failure.v1"
    assert "exited with code 1" in record["reason"]
    assert record["wall_seconds"] is not None and record["wall_seconds"] >= 0


def test_a_resume_declared_at_its_horizon_is_refused_not_reported_as_success(
    tmp_path, tiny_base
):
    """A zero-step run cannot demonstrate trainability, so it is refused."""
    _require_real_model()
    executor = RouterHealingExecutor()
    first = executor.run(_experiment(tiny_base), _context(tmp_path / "first", checkpoint_every=2))
    final_checkpoint = Path(first.evidence["checkpoints"][-1]["directory"])
    assert final_checkpoint.name == "step-4"
    with pytest.raises(RouterHealingBackendError, match="nothing left to train|nothing to train"):
        executor.run(
            _experiment(tiny_base),
            _context(tmp_path / "second", resume_from=str(final_checkpoint)),
        )


def test_an_abruptly_killed_worker_leaves_a_complete_resumable_checkpoint(
    tmp_path, tiny_base
):
    """Abrupt death: the last published checkpoint must still be a real checkpoint.

    This launches the worker module directly, waits for checkpoint 1 to appear,
    then kills the process. Nothing the worker would have written on a graceful
    path is assumed; only what was already durable is checked.
    """
    _require_real_model()
    import subprocess
    import sys
    import time

    from chowder.resume_state import assert_resumable, inventory_checkpoint
    from chowder.worker_env import chowder_source_identity, worker_env

    run_dir = tmp_path / "killed"
    run_dir.mkdir()
    checkpoints = run_dir / "checkpoints"
    spec = RouterHealingRunSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        corpus_path=tiny_base["corpus"],
        corpus_sha256=tiny_base["corpus_sha256"],
        output_dir=str(run_dir / "output"),
        max_steps=2000,
        learning_rate=0.05,
        seq_len=16,
        batch_size=2,
        seed=0,
        probe_window=2,
        max_tokens=10_000_000,
        checkpoint_dir=str(checkpoints),
        checkpoint_every=1,
    )
    spec_path = run_dir / "run-spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    identity_path = run_dir / "chowder-identity.json"
    identity_path.write_text(json.dumps(chowder_source_identity()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "chowder.backends.router_healing_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(run_dir / "worker-result.json"),
        "--chowder-identity",
        str(identity_path),
    ]
    process = subprocess.Popen(
        command, cwd=str(run_dir), env=worker_env({"PYTHONUNBUFFERED": "1"}),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            published = sorted(checkpoints.glob("step-*")) if checkpoints.is_dir() else []
            if published:
                # Kill only once the earliest checkpoint is *complete* on
                # disk. Under heavy load the directory can be visible while
                # its files are still being written; killing inside that
                # publication window would test a partial write, not abrupt
                # death after a durable checkpoint.
                if inventory_checkpoint(published[0]).is_complete:
                    break
            if process.poll() is not None:
                raise AssertionError("the worker exited before publishing a checkpoint")
            time.sleep(0.5)
        else:  # pragma: no cover - only on a pathologically slow machine
            raise AssertionError("no checkpoint was published in time")
    finally:
        process.kill()
        process.wait(timeout=60)

    published = sorted(checkpoints.glob("step-*"))
    assert published, "a killed worker must still have left its last durable checkpoint"
    inventory = inventory_checkpoint(published[0])
    assert inventory.is_complete, inventory.notes
    assert_resumable(inventory)
    assert inventory.global_step is not None and inventory.global_step >= 1


# --- the router evaluator ----------------------------------------------------


def _eval_artifact(experiment: Experiment, payload_dir: Path) -> TrainingArtifact:
    return TrainingArtifact(
        run_id="exp-router-backend-run",
        experiment_id=experiment.experiment_id,
        artifact_ref=str(payload_dir),
        gpu_hours=0.0,
        evidence={
            "freeze_summary": {
                "trainable_param_names": [
                    "model.layers.0.mlp.gate.weight",
                    "model.layers.1.mlp.gate.weight",
                ]
            }
        },
    )


@pytest.fixture(scope="module")
def payloads(tiny_base, tmp_path_factory):
    """Three published payloads: differing, identical, and provably inert.

    The third one is not padding. A *uniform* shift of every gate entry adds the
    same multiple of the hidden state to every expert's logit, so it cannot
    change routing decisions or routing weights at all; only float rounding
    in the rest of the network can move the logits, and how much rounding
    appears depends on the host's kernels. It is the sharpest available check
    that the application control measures real behaviour rather than merely
    detecting a tensor write.
    """
    _require_real_model()
    import torch
    from transformers import AutoModelForCausalLM

    from chowder.router_payload import save_router_payload

    root = tmp_path_factory.mktemp("router-payloads")
    model = AutoModelForCausalLM.from_pretrained(tiny_base["base_dir"], dtype=torch.float32)
    names = ["model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"]
    parameters = dict(model.named_parameters())
    identical = {name: parameters[name].detach().clone() for name in names}

    # Per-expert offsets change the relative expert logits, so routing moves.
    experts = int(model.config.num_experts)
    row_offsets = (torch.arange(experts, dtype=torch.float32).unsqueeze(1) * 0.05).to(
        parameters[names[0]].dtype
    )
    different = {name: (parameters[name].detach() + row_offsets).clone() for name in names}
    uniform = {name: (parameters[name].detach() + 0.05).clone() for name in names}

    common = {
        "base_content_sha256": tiny_base["content_sha256"],
        "spec_digest": "a" * 64,
        "steps_completed": 1,
    }
    identity = save_router_payload(identical, root / "identity", **common)
    changed = save_router_payload(different, root / "changed", **common)
    inert = save_router_payload(uniform, root / "uniform", **common)
    return {
        "identity": Path(identity["payload_dir"]),
        "changed": Path(changed["payload_dir"]),
        "uniform": Path(inert["payload_dir"]),
        "names": names,
    }


def test_the_evaluator_is_dispatched_from_the_same_engine_key_as_the_trainer(tmp_path):
    evaluator = create_evaluation_executor({"backend": {"type": "router-healing"}})
    assert evaluator.name == "transformers-router-healing-evaluator"
    peft = create_evaluation_executor({"backend": {"type": "peft", "engine": "transformers"}})
    assert peft.name != evaluator.name


def test_the_evaluator_reports_zero_accelerator_hours(tmp_path):
    experiment = Experiment(
        experiment_id="exp-router-eval-profile",
        parent_id=None,
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="i", expected_deltas={}
        ),
        config_patch={},
        estimated_gpu_hours=1.0,
    )
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config={"backend": {"type": ROUTER_HEALING_ENGINE}}
    )
    estimate = RouterHealingEvaluator().profile(experiment, context)
    assert estimate.gpu_hours == 0.0
    assert "accelerator hours" in " ".join(estimate.notes)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"expected_parameter_paths": ()}, "at least one expected parameter path"),
        ({"holdout_corpus_sha256": "short"}, "sha256 digest"),
        ({"batches": 0}, "batches must be a positive integer"),
        ({"device": "npu"}, "not qualified"),
    ],
)
def test_an_eval_spec_refuses_configurations_that_cannot_score_honestly(overrides, match):
    kwargs = {
        "base_model_dir": "base",
        "base_content_sha256": "a" * 64,
        "payload_dir": "payload",
        "holdout_corpus_path": "holdout",
        "holdout_corpus_sha256": "b" * 64,
        "expected_parameter_paths": ("model.layers.0.mlp.gate.weight",),
        "output_dir": "out",
        "seq_len": 16,
        "batches": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        RouterHealingEvalSpec(**kwargs)


def test_evaluation_is_refused_without_a_holdout_corpus(tmp_path, tiny_base):
    """Scoring the training corpus is fit, not capability, so it is a refusal."""
    experiment = _experiment(tiny_base)
    experiment.config_patch["router_healing"].pop("holdout_corpus_path")
    experiment.config_patch["router_healing"].pop("holdout_corpus_sha256")
    artifact = _eval_artifact(experiment, tmp_path / "payload")
    with pytest.raises(RouterHealingEvaluationError, match="holdout_corpus_path"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def test_a_verified_payload_is_applied_and_scored_in_a_fresh_process(
    tmp_path, tiny_base, payloads
):
    """The whole point: another process, the published artifact only."""
    _require_real_model()
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["changed"])
    outcome = RouterHealingEvaluator().evaluate(
        experiment=experiment, artifact=artifact, context=_context(tmp_path)
    )

    assert outcome.gpu_hours == 0.0
    assert outcome.source_artifact_ref == str(payloads["changed"])
    assert math.isfinite(outcome.metrics["holdout_loss"])
    assert outcome.metrics["experts_per_token"] == 2.0  # the tiny model's configured top-k
    assert outcome.metrics["dead_experts"] >= 0.0

    control = outcome.evidence["application_control"]
    assert control["payload_kind"] == "replacement"
    assert control["parameters_changed"] is True
    assert control["outputs_changed"] is True
    assert control["parameters_differing"] == sorted(payloads["names"])

    # The metric that came from configuration says so; the measured one does not.
    sources = outcome.evidence["metric_sources"]
    assert "configuration read" in sources["experts_per_token"]
    assert "measured" in sources["holdout_loss"]

    # Both arms of the comparison were measured in this process, so a cost
    # breakdown exists rather than an honest "another process" placeholder.
    ledger = outcome.evidence["phase_ledger"]
    assert ledger["phases"]["baseline_generation"]["measured"] is True
    assert ledger["phases"]["candidate_generation"]["measured"] is True
    assert ledger["phases"]["model_load"]["measured"] is True
    assert ledger["phases"]["steady_state_steps"]["measured"] is False


def test_an_identity_payload_is_the_control_that_proves_the_apply_is_measured(
    tmp_path, tiny_base, payloads
):
    """A payload equal to the base must change neither parameters nor output.

    If this ever reports a changed output, the evaluation is not measuring what
    it claims, and every non-identity result becomes suspect.
    """
    _require_real_model()
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["identity"])
    outcome = RouterHealingEvaluator().evaluate(
        experiment=experiment, artifact=artifact, context=_context(tmp_path)
    )
    control = outcome.evidence["application_control"]
    assert control["identity_payload"] is True
    assert control["parameters_changed"] is False
    assert control["outputs_changed"] is False
    assert outcome.evidence["candidate_holdout_loss"] == pytest.approx(
        outcome.evidence["base_holdout_loss"]
    )


def test_a_uniform_gate_shift_is_refused_because_it_cannot_change_routing(
    tmp_path, tiny_base, payloads
):
    """Real behaviour, measured on a real model.

    Adding the same constant to every gate entry adds the same multiple of the
    hidden state to every expert's logit, so routing decisions and routing
    weights are unchanged; only float rounding can move the logits, and how
    much rounding appears depends on the host's elementwise kernels. The
    refusal therefore keys on the routing fingerprint staying inside the
    rounding tolerance, not on bit-identical logits -- demanding bit-equality
    made this control a coin flip across machines (and eventually failed CI).
    """
    _require_real_model()
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["uniform"])
    with pytest.raises(RouterHealingEvaluationError, match="changed no model output"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def test_a_payload_that_changes_parameters_but_not_output_is_refused(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """A payload wired to nothing would still produce a number; refuse it."""

    def factory(command, **kwargs):
        class FakeProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
                result_path.write_text(
                    json.dumps(
                        _valid_eval_result(spec, force_broken=True, routing="unchanged")
                    )
                )

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        return FakeProcess(command, **kwargs)

    monkeypatch.setattr(
        "chowder.backends.router_healing.subprocess.Popen", factory
    )
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["changed"])
    with pytest.raises(RouterHealingEvaluationError, match="changed no model output"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def _valid_eval_result(
    spec: RouterHealingEvalSpec,
    *,
    force_broken: bool = False,
    routing: str = "moved",
) -> dict:
    load = PhaseTimer()
    load.seconds = 0.5
    from chowder.lifecycle import (
        PHASE_BASELINE_GENERATION,
        PHASE_CANDIDATE_GENERATION,
        PHASE_STEADY_STEPS,
        training_lifecycle_ledger,
    )

    ledger = training_lifecycle_ledger(accelerator_count=0, model_load=load)
    ledger.record_unavailable(PHASE_STEADY_STEPS, "evaluation only")
    ledger.record(PHASE_BASELINE_GENERATION, 0.1, synchronized=False)
    ledger.record(PHASE_CANDIDATE_GENERATION, 0.1, synchronized=False)
    routing_control = {
        "routing_top1_equal": routing == "unchanged",
        "max_abs_routing_weight_delta": 0.0 if routing == "unchanged" else 0.2,
        "routing_unchanged_beyond_rounding": routing == "unchanged",
    }
    return {
        "kind": EVAL_WORKER_RESULT_KIND,
        "spec_digest": spec.digest(),
        "metrics": {"holdout_loss": 1.0, "experts_per_token": 2.0, "dead_experts": 0.0},
        "metric_sources": {
            "holdout_loss": "measured",
            "experts_per_token": "configuration read",
            "dead_experts": "measured",
        },
        "base_holdout_loss": 1.0,
        "candidate_holdout_loss": 1.0,
        "application_control": {
            "payload_kind": "replacement",
            "parameters_changed": True,
            "outputs_changed": not force_broken,
            **routing_control,
        },
        "lifecycle": ledger.to_dict(),
        "resource_usage": {
            "wall_seconds": 1.0,
            "active_accelerator_count": 0,
            "visible_accelerator_count": 0,
            "peak_vram_gb_by_accelerator": {},
        },
    }


# --- Uniform-shift identity control: rounding-honest, not bit-lucky ----------


def _uniform_fingerprint_spec(tiny_base, payload_dir, tmp_path):
    return RouterHealingEvalSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        payload_dir=str(payload_dir),
        holdout_corpus_path=tiny_base["holdout"],
        holdout_corpus_sha256=tiny_base["holdout_sha256"],
        expected_parameter_paths=(
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ),
        output_dir=str(tmp_path / "out"),
        seq_len=16,
        batches=1,
        device="cpu",
    )


def _loaded_tiny_model(tiny_base):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        tiny_base["base_dir"], dtype=torch.float32
    )
    model.eval()
    return model


def test_a_uniform_gate_shift_leaves_routing_and_weights_unchanged_beyond_rounding(
    tmp_path, tiny_base, payloads
):
    """Real behaviour, measured on a real model.

    A routing-invariant payload must leave *routing decisions (argmax) and
    routing weights (softmax over the gate logits)* unchanged beyond the
    float rounding that the shift mathematically forces through the rest of
    the network. Demanding bit-identical logits there turned the identity
    control into a coin flip: softmax(x+c) rounds one ulp away from
    softmax(x) whenever the elementwise kernels dispatch differently, which
    is exactly how CI failed while the same code passed locally.
    """
    _require_real_model()
    from chowder.backends.router_healing_eval_worker import (
        _routing_fingerprint,
        _routing_fingerprint_delta,
    )
    from chowder.router_payload import apply_router_payload, load_router_payload

    spec = _uniform_fingerprint_spec(tiny_base, payloads["uniform"], tmp_path)
    model = _loaded_tiny_model(tiny_base)

    payload = load_router_payload(
        payloads["uniform"], expected_base_content_sha256=tiny_base["content_sha256"]
    )
    before = _routing_fingerprint(model, spec)
    apply_router_payload(
        model, payload, expected_parameter_paths=spec.expected_parameter_paths
    )
    after = _routing_fingerprint(model, spec)

    assert before["top1_decisions"] == after["top1_decisions"]
    assert _routing_fingerprint_delta(before, after) == pytest.approx(0.0, abs=1e-6)


def test_a_routing_changing_payload_moves_the_routing_fingerprint(
    tmp_path, tiny_base, payloads
):
    """The control must still catch a payload that really changes routing."""
    _require_real_model()
    from chowder.backends.router_healing_eval_worker import (
        _routing_fingerprint,
        _routing_fingerprint_delta,
    )
    from chowder.router_payload import apply_router_payload, load_router_payload

    spec = _uniform_fingerprint_spec(tiny_base, payloads["changed"], tmp_path)
    model = _loaded_tiny_model(tiny_base)

    payload = load_router_payload(
        payloads["changed"], expected_base_content_sha256=tiny_base["content_sha256"]
    )
    before = _routing_fingerprint(model, spec)
    apply_router_payload(
        model, payload, expected_parameter_paths=spec.expected_parameter_paths
    )
    after = _routing_fingerprint(model, spec)

    assert before["top1_decisions"] != after["top1_decisions"]
    assert _routing_fingerprint_delta(before, after) > 1e-3


def test_the_uniform_gate_shift_is_refused_without_demanding_bit_identical_logits(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """End-to-end: the refusal survives float-rounding differences.

    The worker is mocked at the process boundary so ``outputs_changed`` is
    True exactly because softmax rounds one ulp differently after a uniform
    shift -- the condition that made CI flaky -- while routing decisions are
    unchanged. The refusal must now come from the routing fingerprint, not
    from demanding bit-equality that exact arithmetic promises but float
    arithmetic cannot.
    """

    def mutate(result, spec):
        control = result["application_control"]
        control["outputs_changed"] = True  # ulp-level softmax rounding
        # The real worker reports the routing truth: a uniform shift leaves
        # decisions and weights unchanged beyond rounding.
        control["routing_top1_equal"] = True
        control["max_abs_routing_weight_delta"] = 0.0
        control["routing_unchanged_beyond_rounding"] = True

    def factory(command, **kwargs):
        class FakeEvalProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(
                    **json.loads(spec_path.read_text(encoding="utf-8"))
                )
                result = _valid_eval_result(spec)
                mutate(result, spec)
                result_path.write_text(json.dumps(result), encoding="utf-8")

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        return FakeEvalProcess(command, **kwargs)

    monkeypatch.setattr(
        "chowder.backends.router_healing.subprocess.Popen", factory
    )
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["uniform"])
    with pytest.raises(RouterHealingEvaluationError, match="changed no model output"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def test_the_parent_refuses_a_payload_arm_that_reports_no_routing_fingerprint(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """A worker that stops reporting the fingerprint is refused, not believed."""

    def factory(command, **kwargs):
        class FakeEvalProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
                result = _valid_eval_result(spec)
                for field in ("routing_top1_equal", "max_abs_routing_weight_delta"):
                    result["application_control"].pop(field, None)
                result_path.write_text(json.dumps(result), encoding="utf-8")

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        return FakeEvalProcess(command, **kwargs)

    monkeypatch.setattr(
        "chowder.backends.router_healing.subprocess.Popen", factory
    )
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["changed"])
    with pytest.raises(RouterHealingEvaluationError, match="no routing fingerprint"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


# --- P11 rung 2: cuda is admitted behind a measured device preflight ---------


def test_cpu_and_cuda_are_qualified_but_mps_is_not():
    """The guard lifts for cuda alone, behind the preflight contract below."""
    assert "cpu" in QUALIFIED_DEVICES
    assert "cuda" in QUALIFIED_DEVICES
    assert "mps" not in QUALIFIED_DEVICES


def test_mps_is_still_refused():
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(**_spec_kwargs(device="mps"))


def test_the_step_cost_projection_refuses_a_run_that_cannot_fit_its_wall_budget():
    """A measured step rate that cannot fit the declared horizon is refused."""
    from chowder.backends.router_healing_worker import project_step_cost

    fitting = project_step_cost(step_seconds=1.0, max_steps=12, max_seconds=600)
    assert fitting["measured"] is True
    assert fitting["projected_wall_seconds"] == pytest.approx(12.0)
    assert fitting["would_exceed_budget"] is False

    overflowing = project_step_cost(step_seconds=1.0, max_steps=12, max_seconds=10)
    assert overflowing["projected_wall_seconds"] == pytest.approx(12.0)
    assert overflowing["would_exceed_budget"] is True


def test_the_memory_projection_refuses_a_step_peak_above_the_measured_free_memory():
    from chowder.backends.router_healing_worker import project_device_memory

    fitting = project_device_memory(free_bytes=1 << 30, peak_bytes=256 << 20)
    assert fitting["headroom_bytes"] == (1 << 30) - (256 << 20)
    assert fitting["projected_oom"] is False

    overflowing = project_device_memory(free_bytes=256 << 20, peak_bytes=(256 << 20) + 1)
    assert overflowing["projected_oom"] is True


def test_the_parent_refuses_an_accelerator_run_that_never_measured_one(
    tmp_path, monkeypatch, tiny_base
):
    """A cuda run without a device preflight is refused, not believed."""

    def mutate(result, spec):
        if spec.device != "cpu":
            result.pop("device_preflight", None)

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="device preflight"):
        RouterHealingExecutor().run(
            _experiment(tiny_base, device="cuda"), _context(tmp_path)
        )


def test_the_parent_refuses_an_accelerator_run_whose_preflight_never_measured_memory(
    tmp_path, monkeypatch, tiny_base
):
    """A preflight with a named device but no measured free memory is not one."""

    def mutate(result, spec):
        if spec.device != "cpu":
            result["device_preflight"] = {
                "device": "cuda",
                "free_memory_bytes": None,
                "step_cost_probe": {"measured": True, "step_seconds": 0.1},
            }

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="did not measure free memory"):
        RouterHealingExecutor().run(
            _experiment(tiny_base, device="cuda"), _context(tmp_path)
        )


def test_the_evaluator_refuses_an_accelerator_arm_that_never_measured_memory(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """Peak VRAM {} on a cuda arm is the unmeasured-zero lie, refused."""

    def factory(command, **kwargs):
        class FakeProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(
                    **json.loads(spec_path.read_text(encoding="utf-8"))
                )
                result = _valid_eval_result(spec)
                if spec.device != "cpu":
                    result["resource_usage"]["peak_vram_gb_by_accelerator"] = {}
                result_path.write_text(json.dumps(result))

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        return FakeProcess(command, **kwargs)

    monkeypatch.setattr("chowder.backends.router_healing.subprocess.Popen", factory)
    experiment = _experiment(tiny_base)
    artifact = _eval_artifact(experiment, payloads["changed"])
    context = _context(tmp_path)
    object.__setattr__(
        context,
        "resolved_config",
        {
            "backend": {
                "type": ROUTER_HEALING_ENGINE,
                "router_healing": {"batch_size": 2, "max_tokens": 4096, "probe_window": 2, "device": "cuda"},
            }
        },
    )
    with pytest.raises(RouterHealingEvaluationError, match="did not measure"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=context
        )


def test_a_cuda_run_reports_a_measured_preflight_and_peak_memory(tmp_path, tiny_base):
    """P11 rung 2, the real device path: the tiny router trains on cuda.

    Skips on GPU-less CI; executes where a real accelerator exists. The
    preregistered rung is qualified by a real run through the CLI; this test
    pins the worker-level contract that run depends on.
    """
    _require_real_model()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("no CUDA device on this host")
    checkpoint_dir = tmp_path / "checkpoints"
    spec = RouterHealingRunSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        corpus_path=tiny_base["corpus"],
        corpus_sha256=tiny_base["corpus_sha256"],
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        learning_rate=0.01,
        seq_len=16,
        batch_size=1,
        seed=0,
        probe_window=1,
        max_tokens=64,
        device="cuda",
        checkpoint_every=2,
        checkpoint_dir=str(checkpoint_dir),
    )
    result = train(spec)

    preflight = result["device_preflight"]
    assert preflight["device"] == "cuda"
    assert preflight["free_memory_bytes"] > 0
    assert preflight["projected_oom"] is False
    probe = preflight["step_cost_probe"]
    assert probe["measured"] is True
    assert probe["step_seconds"] > 0.0
    assert probe["peak_step_bytes"] > 0
    assert probe["projected_oom"] is False
    assert probe["would_exceed_budget"] is False

    usage = result["resource_usage"]
    assert usage["active_accelerator_count"] == 1
    assert usage["peak_vram_gb_by_accelerator"], "peak VRAM must be measured, not {}"
    assert result["trainability"]["ok"] is True
    assert result["frozen"]["ok"] is True
    assert result["limits"]["stop_reason"] == "max_steps"
