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
from pathlib import Path

import pytest

from chowder.backend_selection import (
    ROUTER_HEALING_ENGINE,
    create_training_executor,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
from chowder.backends.router_healing import (
    QUALIFIED_DEVICES,
    WORKER_RESULT_KIND,
    RouterHealingBackendError,
    RouterHealingExecutor,
    RouterHealingRunSpec,
)
from chowder.executors import ExecutionContext
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


def test_the_only_qualified_device_is_cpu():
    assert QUALIFIED_DEVICES == ("cpu",)


def test_a_spec_pins_the_device_it_has_actually_qualified():
    """An accelerator run is refused, not attempted and discovered."""
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(**_spec_kwargs(device="cuda"))
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(**_spec_kwargs(device="mps"))


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
        "manifest_sha256": identity["manifest_sha256"],
        "content_sha256": identity["content_sha256"],
        "corpus_sha256": sha256_file(corpus),
    }


def _experiment(tiny_base, **research_overrides) -> Experiment:
    research = {
        "base_model_dir": tiny_base["base_dir"],
        "base_manifest_sha256": tiny_base["manifest_sha256"],
        "corpus_path": tiny_base["corpus"],
        "corpus_sha256": tiny_base["corpus_sha256"],
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


def test_the_worker_refuses_an_accelerator_spec_before_loading_anything(tmp_path, tiny_base):
    """The device guard lives in the spec, so it fires before a subprocess exists."""
    with pytest.raises(ValueError, match="not qualified"):
        RouterHealingRunSpec(
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
        )
