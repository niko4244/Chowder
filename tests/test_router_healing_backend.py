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


def test_a_run_overrunning_its_declared_load_budget_is_refused(
    tmp_path, tiny_base
):
    """The worker must refuse a load that busts its declared ceiling.

    The exceedance must surface in the worker's own refusal -- not be carried
    silently into a successful result the parent has to catch after the fact.
    Real worker path: an impossible one-tenth-millisecond ceiling cannot hold.
    """
    spec = RouterHealingRunSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        corpus_path=tiny_base["corpus"],
        corpus_sha256=tiny_base["corpus_sha256"],
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        learning_rate=0.01,
        seq_len=16,
        batch_size=1,
        seed=0,
        probe_window=1,
        max_tokens=32,
        max_load_seconds=0.0001,
    )
    with pytest.raises(RuntimeError, match="declared budget"):
        train(spec)


def test_a_worker_that_reports_success_while_over_its_load_budget_is_refused(
    tmp_path, monkeypatch, tiny_base
):
    """The parent re-checks the budget; a lying worker result is not believed."""

    def mutate(result, spec):
        if spec.max_load_seconds is not None:
            result["load_budget"] = {
                "measured": True,
                "load_seconds": spec.max_load_seconds * 100.0,
                "max_load_seconds": spec.max_load_seconds,
                "would_exceed_load_budget": False,  # the lie
                "load_gpu_hours": 0.5,
                "max_load_gpu_hours": spec.max_load_seconds / 3600.0,
            }

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="load budget"):
        RouterHealingExecutor().run(
            _experiment(tiny_base, max_load_seconds=100.0), _context(tmp_path)
        )


def test_a_worker_that_never_measured_its_load_cost_is_refused_when_budgeted(
    tmp_path, monkeypatch, tiny_base
):
    """A declared ceiling demands a measured load; absent is unknown, not zero."""

    def mutate(result, spec):
        if spec.max_load_seconds is not None:
            result["load_budget"] = None

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="did not measure its load"):
        RouterHealingExecutor().run(
            _experiment(tiny_base, max_load_seconds=100.0), _context(tmp_path)
        )


def test_the_worker_reports_a_measured_load_budget_block(tmp_path, tiny_base):
    """The real worker path: load_budget is present and its load is measured.

    Runs in the real-ML job (it builds a real tiny base); on any device it
    proves the block exists, is measured, and converts to GPU-hours.
    """
    _require_real_model()
    spec = RouterHealingRunSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        corpus_path=tiny_base["corpus"],
        corpus_sha256=tiny_base["corpus_sha256"],
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        learning_rate=0.01,
        seq_len=16,
        batch_size=1,
        seed=0,
        probe_window=1,
        max_tokens=32,
        max_load_seconds=3600.0,
    )
    result = train(spec)
    block = result["load_budget"]
    accelerators = int(result["resource_usage"]["active_accelerator_count"])
    assert block["measured"] is True
    assert block["load_seconds"] > 0.0
    assert block["max_load_seconds"] == 3600.0
    assert block["would_exceed_load_budget"] is False
    # The conversion is device-relative: a CPU load contributes zero
    # accelerator hours, a GPU load its full measured duration.
    assert block["load_gpu_hours"] == pytest.approx(
        block["load_seconds"] * accelerators / 3600.0
    )
    # The declared ceiling converts under the same rule, so a prereg can
    # budget loads in GPU-hours and the projection checks the same number.
    assert block["max_load_gpu_hours"] == pytest.approx(3600.0 * accelerators / 3600.0)


# --- the resident-pair arm: one load, two scored arms -----------------------


def test_a_resident_pair_scores_baseline_before_any_payload_touches_the_model(
    tmp_path, tiny_base, payloads
):
    """Baseline-first ordering is what makes the in-process base score a baseline."""
    _require_real_model()
    from chowder.backends.router_healing_eval_worker import _score as _unused

    del _unused
    calls: list[str] = []
    import chowder.backends.router_healing_eval_worker as eval_worker_module

    original_score = eval_worker_module._score

    def counting_score(torch, model, batches):
        calls.append("score")
        return original_score(torch, model, batches)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_worker_module, "_score", counting_score)
    try:
        spec = RouterHealingEvalSpec(
            base_model_dir=tiny_base["base_dir"],
            base_content_sha256=tiny_base["content_sha256"],
            payload_dir=str(payloads["changed"]),
            holdout_corpus_path=tiny_base["holdout"],
            holdout_corpus_sha256=tiny_base["holdout_sha256"],
            expected_parameter_paths=(
                "model.layers.0.mlp.gate.weight",
                "model.layers.1.mlp.gate.weight",
            ),
            output_dir=str(tmp_path / "out"),
            seq_len=16,
            batches=1,
            paired_arms=True,
        )
        result = eval_worker_module.evaluate(spec)
    finally:
        monkey.undo()
    # At least two scored passes (base, then candidate) -- and the base loss
    # must be reported even though this spec names a payload.
    assert len(calls) >= 2
    assert result["arm"] == "paired"
    assert result["base_holdout_loss"] > 0.0
    assert result["candidate_holdout_loss"] > 0.0
    control = result["application_control"]
    assert control["parameters_changed"] is True
    assert control["outputs_changed"] is True


def test_a_resident_pair_that_cannot_verify_its_payload_still_reports_the_baseline(
    tmp_path, tiny_base, payloads
):
    """A base score is a completed measurement even when the candidate refuses.

    In two separate processes, a candidate-side crash leaves the baseline row
    standing. The resident pair must not make the baseline *less* durable than
    the isolated path was: a payload that fails verification after the base was
    scored reports the measured base score, `arm: base-partial`, and the
    verification error -- the parent turns that into the baseline row, not
    into a missing measurement.
    """
    _require_real_model()
    from chowder.backends import router_healing_eval_worker as eval_worker_module

    real_load = eval_worker_module.load_router_payload

    def refusing_load(payload_dir, expected_base_content_sha256):
        real_load(payload_dir, expected_base_content_sha256=expected_base_content_sha256)
        raise RuntimeError("simulated payload verification failure after load")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_worker_module, "load_router_payload", refusing_load)
    try:
        spec = RouterHealingEvalSpec(
            base_model_dir=tiny_base["base_dir"],
            base_content_sha256=tiny_base["content_sha256"],
            payload_dir=str(payloads["changed"]),
            holdout_corpus_path=tiny_base["holdout"],
            holdout_corpus_sha256=tiny_base["holdout_sha256"],
            expected_parameter_paths=(
                "model.layers.0.mlp.gate.weight",
                "model.layers.1.mlp.gate.weight",
            ),
            output_dir=str(tmp_path / "out"),
            seq_len=16,
            batches=1,
            paired_arms=True,
        )
        result = eval_worker_module.evaluate(spec)
    finally:
        monkey.undo()
    assert result["arm"] == "base-partial"
    assert result["base_holdout_loss"] > 0.0
    assert result["candidate_holdout_loss"] is None
    assert "simulated payload verification failure" in result["pair_error"]


def test_a_resident_pair_that_crashes_before_scoring_reports_no_baseline(
    tmp_path, tiny_base, payloads
):
    """A crash *before* the base score is a plain failure, not a partial arm.

    The base load itself (or the holdout protocol) failing must not be
    relabelled as a measured base: nothing was measured, so the result is
    a refused run exactly as the isolated path refuses today.
    """
    _require_real_model()
    from chowder.backends import router_healing_eval_worker as eval_worker_module

    def exploding_load(base_model_dir, load_policy, device):
        raise RuntimeError("simulated load crash")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_worker_module, "load_with_policy", exploding_load)
    try:
        spec = RouterHealingEvalSpec(
            base_model_dir=tiny_base["base_dir"],
            base_content_sha256=tiny_base["content_sha256"],
            payload_dir=str(payloads["changed"]),
            holdout_corpus_path=tiny_base["holdout"],
            holdout_corpus_sha256=tiny_base["holdout_sha256"],
            expected_parameter_paths=(
                "model.layers.0.mlp.gate.weight",
                "model.layers.1.mlp.gate.weight",
            ),
            output_dir=str(tmp_path / "out"),
            seq_len=16,
            batches=1,
            paired_arms=True,
        )
        with pytest.raises(RuntimeError, match="simulated load crash"):
            eval_worker_module.evaluate(spec)
    finally:
        monkey.undo()


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


def test_the_load_projection_budgets_the_measured_load_against_the_declared_ceiling():
    """Model loads are a first-class preflight phase with their own ceiling.

    The rung-3b CUDA run recorded an exceedance nobody preregistered for:
    three on-device model loads at 12.8 s each doubled the measured GPU-hours
    over a ceiling derived from workload-only time. The projection must make
    load cost a declared, refused-when-overrun quantity -- and convert it to
    GPU-hours, the unit a preregistration's ceiling is written in.
    """
    from chowder.backends.router_healing_worker import project_load_cost

    fitting = project_load_cost(load_seconds=12.8, max_load_seconds=30.0, accelerator_count=1)
    assert fitting["measured"] is True
    assert fitting["load_seconds"] == pytest.approx(12.8)
    assert fitting["max_load_seconds"] == pytest.approx(30.0)
    assert fitting["would_exceed_load_budget"] is False
    assert fitting["load_gpu_hours"] == pytest.approx(12.8 / 3600.0)

    overflowing = project_load_cost(load_seconds=12.8, max_load_seconds=10.0, accelerator_count=1)
    assert overflowing["would_exceed_load_budget"] is True

    # The declared ceiling itself converts, so a prereg can state its load
    # budget in GPU-hours and have the projection check the same number.
    assert fitting["max_load_gpu_hours"] == pytest.approx(30.0 / 3600.0)

    # An undeclared ceiling cannot be exceeded and must not pretend otherwise.
    undeclared = project_load_cost(load_seconds=12.8, max_load_seconds=None, accelerator_count=1)
    assert undeclared["would_exceed_load_budget"] is False
    assert undeclared["max_load_seconds"] is None


def test_the_load_projection_refuses_non_finite_loads():
    """A load time that is not a finite non-negative number is not a measurement."""
    from chowder.backends.router_healing_worker import project_load_cost

    with pytest.raises(ValueError, match="finite"):
        project_load_cost(load_seconds=float("nan"), max_load_seconds=10.0, accelerator_count=1)
    with pytest.raises(ValueError, match="non-negative"):
        project_load_cost(load_seconds=-1.0, max_load_seconds=10.0, accelerator_count=1)


def test_the_memory_projection_compares_incremental_step_demand_to_free_memory():
    """The step peak already contains the resident model; free memory does too.

    The rung-3b CUDA run caught this: with the 9B resident (~9.3 GB) the
    sampler read memory_allocated ~= 11.15 GB for one step, free-after-load
    was ~5.49 GB, and the projection refused a workload diagnostic D had
    measured fitting (11.37 GB absolute peak). The model was being demanded
    to fit twice. The projection must compare the *incremental* step demand
    (allocated minus resident-before-step) against the measured free memory.
    """
    from chowder.backends.router_healing_worker import project_device_memory

    # 5.49 GiB free with the model resident; the step sampler saw 11.15 GiB
    # allocated, of which 9.3 GiB was the resident model.
    projected = project_device_memory(
        free_bytes=5_899_288_576,
        peak_bytes=11_972_909_056,
        resident_before_step_bytes=9_300_000_000,
    )
    assert projected["resident_before_step_bytes"] == 9_300_000_000
    assert projected["incremental_step_bytes"] == 11_972_909_056 - 9_300_000_000
    assert projected["projected_oom"] is False, (
        "the model must not be demanded to fit twice"
    )
    assert projected["headroom_bytes"] == 5_899_288_576 - (
        11_972_909_056 - 9_300_000_000
    )

    # A genuinely incremental overflow still refuses.
    overflowing = project_device_memory(
        free_bytes=256 << 20,
        peak_bytes=(1 << 30) + (512 << 20),
        resident_before_step_bytes=1 << 30,
    )
    assert overflowing["incremental_step_bytes"] == 512 << 20
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


# --- P11 rung-3 amendment: the bf16-offload-transient load policy -------------


def test_the_paired_arms_mode_requires_a_payload():
    """Amortizing the load means scoring both arms in one resident process."""
    spec_kwargs = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "payload_dir": None,
        "holdout_corpus_path": "unused-holdout",
        "holdout_corpus_sha256": "c" * 64,
        "expected_parameter_paths": (),
        "output_dir": "unused-out",
        "seq_len": 16,
        "batches": 1,
        "paired_arms": True,
    }
    with pytest.raises(ValueError, match="paired_arms"):
        RouterHealingEvalSpec(**spec_kwargs)


def test_the_paired_arms_mode_requires_declared_parameter_paths():
    spec_kwargs = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "payload_dir": "some-payload",
        "holdout_corpus_path": "unused-holdout",
        "holdout_corpus_sha256": "c" * 64,
        "expected_parameter_paths": (),
        "output_dir": "unused-out",
        "seq_len": 16,
        "batches": 1,
        "paired_arms": True,
    }
    with pytest.raises(ValueError, match="at least one expected parameter path"):
        RouterHealingEvalSpec(**spec_kwargs)


def test_the_paired_arms_result_always_carries_both_scores():
    """The evaluator must surface baseline metrics from a paired candidate arm."""
    spec_kwargs = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "payload_dir": "some-payload",
        "holdout_corpus_path": "unused-holdout",
        "holdout_corpus_sha256": "c" * 64,
        "expected_parameter_paths": ("model.layers.0.mlp.gate.weight",),
        "output_dir": "unused-out",
        "seq_len": 16,
        "batches": 1,
    }
    spec = RouterHealingEvalSpec(**spec_kwargs)
    assert spec.paired_arms is False
    assert spec.payload_applied is True


def test_the_run_spec_accepts_a_declared_load_budget():
    """max_load_seconds is an optional declared ceiling on the model-load phase."""
    assert RouterHealingRunSpec(**_spec_kwargs()).max_load_seconds is None
    spec = RouterHealingRunSpec(**_spec_kwargs(max_load_seconds=120.0))
    assert spec.max_load_seconds == 120.0


def test_the_run_spec_refuses_a_non_positive_or_non_finite_load_budget():
    with pytest.raises(ValueError, match="max_load_seconds"):
        RouterHealingRunSpec(**_spec_kwargs(max_load_seconds=0.0))
    with pytest.raises(ValueError, match="max_load_seconds"):
        RouterHealingRunSpec(**_spec_kwargs(max_load_seconds=-5.0))
    with pytest.raises(ValueError, match="max_load_seconds"):
        RouterHealingRunSpec(**_spec_kwargs(max_load_seconds=float("inf")))


def test_the_load_budget_is_spec_bound_but_recipe_neutral():
    """The ceiling changes the spec a worker must honor, not the science it runs."""
    base = RouterHealingRunSpec(**_spec_kwargs())
    budgeted = RouterHealingRunSpec(**_spec_kwargs(max_load_seconds=90.0))
    assert base.digest() != budgeted.digest(), "the spec digest must bind the ceiling"
    assert base.recipe_digest() == budgeted.recipe_digest(), (
        "a scheduling ceiling is not a recipe change"
    )


def test_the_run_spec_refuses_an_unknown_load_policy():
    """A policy nobody preregistered is refused at spec time, not at load time."""
    with pytest.raises(ValueError, match="unknown load policy"):
        RouterHealingRunSpec(**_spec_kwargs(load_policy="quantized-4bit"))


def test_the_run_spec_defaults_to_fp32_resident():
    """Existing behaviour is the default: nothing changes unless declared."""
    assert RouterHealingRunSpec(**_spec_kwargs()).load_policy == "fp32-resident"


def test_the_run_spec_accepts_the_amended_offload_policy():
    spec = RouterHealingRunSpec(**_spec_kwargs(load_policy="bf16-offload-transient"))
    assert spec.load_policy == "bf16-offload-transient"


def test_the_recipe_digest_changes_with_the_load_policy():
    """A payload's recipe must say how its base was resident."""
    one = RouterHealingRunSpec(**_spec_kwargs())
    two = RouterHealingRunSpec(**_spec_kwargs(load_policy="bf16-offload-transient"))
    assert one.recipe_digest() != two.recipe_digest()


def test_the_eval_spec_carries_the_same_policy_contract():
    """The two arms must share the load contract, so both specs validate it."""
    base_fields = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "payload_dir": None,
        "holdout_corpus_path": "unused-holdout",
        "holdout_corpus_sha256": "c" * 64,
        "expected_parameter_paths": (),
        "output_dir": "unused-out",
        "seq_len": 8,
        "batches": 1,
    }
    with pytest.raises(ValueError, match="unknown load policy"):
        RouterHealingEvalSpec(**base_fields, load_policy="quantized-4bit")
    assert RouterHealingEvalSpec(**base_fields).load_policy == "fp32-resident"
    # A base arm loads the model under the policy too, so the amended policy is
    # valid exactly where the default is: with no declared parameter paths.
    amended = RouterHealingEvalSpec(**base_fields, load_policy="bf16-offload-transient")
    assert amended.load_policy == "bf16-offload-transient"


def test_an_offload_policy_on_a_base_without_experts_is_refused():
    """Nothing to offload means the request is a contradiction, not a fallback."""
    _require_real_model()
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM

    from chowder.backends.router_healing_load import resolve_load_placement

    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    model = AutoModelForCausalLM.from_config(config)
    names = [name for name, _ in model.named_parameters()]
    with pytest.raises(RuntimeError, match="has no expert parameters"):
        resolve_load_placement("unused-root", names, device="cuda")
    del model


def test_the_base_arm_scores_under_the_same_load_policy(tmp_path, monkeypatch, tiny_base):
    """Both arms must load the base under the same declared load policy.

    The rung-3b CUDA run caught the real defect: the candidate arm read
    ``load_policy`` from the research spec, but the base arm built its spec
    from project config only, so the baseline loaded fp32-resident (the exact
    WDDM-spill failure the amendment forbids) while the candidate loaded
    bf16-offload-transient. A mixed-policy comparison is refused here.
    """
    _require_real_model()
    from chowder.backends.router_healing import RouterHealingEvaluator

    evaluator = RouterHealingEvaluator()
    experiment = _experiment(tiny_base, load_policy="bf16-offload-transient")
    artifact = TrainingArtifact(
        run_id="run-base-policy",
        experiment_id=experiment.experiment_id,
        artifact_ref=str(tmp_path / "payload"),
        gpu_hours=0.0,
        telemetry={},
        evidence={
            "freeze_summary": {
                "trainable_param_names": ["model.layers.0.mlp.gate.weight"]
            },
        },
        resource_usage=None,
    )
    context = _context(tmp_path)
    config = {
        "backend": {
            "type": "router-healing",
            "router_healing": {
                "base_model_dir": tiny_base["base_dir"],
                "holdout_corpus_path": tiny_base["holdout"],
                "device": "cpu",
                # The project config carries the policy too; the research spec
                # (config_patch) also declares it. Both must agree.
                "load_policy": "bf16-offload-transient",
            },
        }
    }
    spec = evaluator._base_spec_for(context, config=config, eval_dir=tmp_path)
    assert spec.load_policy == "bf16-offload-transient", (
        "the base arm must inherit the declared load policy; a baseline under a "
        "different residency contract is not a baseline"
    )


def test_the_placement_census_refuses_a_full_resident_model(tmp_path, tiny_base):
    """Negative control: a model loaded WITHOUT offload must fail the census.

    Pins the census logic itself -- not just the happy path -- so a mutation
    that hard-codes ``verified = True`` cannot survive.
    """
    _require_real_model()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("no CUDA device on this host")
    from transformers import AutoModelForCausalLM

    from chowder.backends.router_healing_load import verify_placement_census

    model = AutoModelForCausalLM.from_pretrained(
        tiny_base["base_dir"], dtype=torch.float32, local_files_only=True
    ).to("cuda")
    census = verify_placement_census(model, device_type="cuda")
    assert census["expert_params_total"] > 0
    assert census["expert_params_on_device"] == census["expert_params_total"]
    assert census["verified"] is False, "a full-resident base must not pass the offload census"
    del model


def test_an_offload_run_on_the_tiny_moe_proves_the_census_and_trainability(
    tmp_path, tiny_base
):
    """The amended policy, end to end on a real tiny MoE, on the real device.

    Mirrors the rung-2 CUDA contract at the worker level: measured preflight,
    exact gate scope, frozen-unchanged digests -- plus the amendment's own
    placement census. Skips where no accelerator exists; the CPU suite is not
    affected because the CPU path never enters this policy's load seam.
    """
    _require_real_model()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("no CUDA device on this host")
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
        load_policy="bf16-offload-transient",
    )
    result = train(spec)

    placement = result["load_policy_report"]
    assert placement["policy"] == "bf16-offload-transient"
    assert placement["dtype"] == "torch.bfloat16"
    census = placement["placement_census"]
    assert census["expert_params_on_device"] == 0, "experts must be offloaded"
    assert census["gate_params_on_device"] == census["gate_params_total"]
    assert census["gate_params_total"] > 0
    assert census["verified"] is True
    assert result["trainability"]["ok"] is True
    assert result["frozen"]["ok"] is True
    assert result["resource_usage"]["active_accelerator_count"] == 1


def test_the_two_paths_measure_the_same_scores_and_controls(tmp_path, tiny_base, payloads):
    """Amortization may not change what either arm measures.

    The resident pair (one load, both arms) and the historical two-process
    path must produce the same base score, the same candidate score, the
    same deltas, and the same application-control verdicts on the same
    holdout blocks -- otherwise the comparison drifts when the mode flips.
    This is the equality half of the isolation question; the baseline-row
    durability is pinned separately.
    """
    _require_real_model()
    from chowder.backends import router_healing_eval_worker as eval_worker_module

    common = {
        "base_model_dir": tiny_base["base_dir"],
        "base_content_sha256": tiny_base["content_sha256"],
        "payload_dir": str(payloads["changed"]),
        "holdout_corpus_path": tiny_base["holdout"],
        "holdout_corpus_sha256": tiny_base["holdout_sha256"],
        "expected_parameter_paths": (
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ),
        "output_dir": str(tmp_path / "out"),
        "seq_len": 16,
        "batches": 1,
    }

    # Historical path: two separate evaluate() calls, base arm then candidate.
    base_result = eval_worker_module.evaluate(
        RouterHealingEvalSpec(**{**common, "payload_dir": None, "expected_parameter_paths": ()})
    )
    candidate_result = eval_worker_module.evaluate(RouterHealingEvalSpec(**common))

    # Amortized path: one load, both arms in one resident process.
    paired_result = eval_worker_module.evaluate(
        RouterHealingEvalSpec(**{**common, "paired_arms": True})
    )

    assert paired_result["base_holdout_loss"] == pytest.approx(
        base_result["base_holdout_loss"], rel=1e-12
    )
    assert paired_result["candidate_holdout_loss"] == pytest.approx(
        candidate_result["candidate_holdout_loss"], rel=1e-12
    )
    paired_control = paired_result["application_control"]
    isolated_control = candidate_result["application_control"]
    assert paired_control["outputs_changed"] == isolated_control["outputs_changed"]
    assert paired_control["routing_top1_equal"] == isolated_control["routing_top1_equal"]
    assert paired_control["max_abs_routing_weight_delta"] == pytest.approx(
        isolated_control["max_abs_routing_weight_delta"], rel=1e-9
    )
    # Dead-expert behaviour must agree too: a routing-count measured after the
    # candidate pass in the resident model would silently differ from the
    # base arm's own count if the payload leaked between arms.
    assert paired_result["metrics"]["dead_experts"] == pytest.approx(
        candidate_result["metrics"]["dead_experts"]
    )


def test_the_paired_arms_decision_reads_the_same_places_as_the_specs(tmp_path, tiny_base):
    """One resolver decides arm separation for the candidate spec.

    ``paired_arms_for`` must read exactly where ``_spec_for`` reads --
    research field first (preregistered), project knobs second -- so the
    runner's start-time decision to defer the baseline cannot drift from
    what the candidate evaluation will actually do.
    """
    _require_real_model()
    from chowder.backends.router_healing import RouterHealingExecutor, RouterHealingEvaluator

    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    artifact = TrainingArtifact(
        run_id="run-pair-decision",
        experiment_id="exp-router-backend",
        artifact_ref=str(payload_dir),
        gpu_hours=0.0,
        telemetry={},
        evidence={
            "freeze_summary": {"trainable_param_names": ["model.layers.0.mlp.gate.weight"]}
        },
        resource_usage=None,
    )
    evaluator = RouterHealingEvaluator()

    # Declared in the preregistered research spec.
    experiment = _experiment(tiny_base, paired_arms=True)
    research = RouterHealingExecutor._research_spec(experiment)
    assert evaluator._spec_for(
        experiment, artifact, _context(tmp_path), eval_dir=tmp_path / "c1"
    ).paired_arms is True
    assert RouterHealingExecutor.paired_arms_for(research, _context(tmp_path)) is True

    # Declared only as a project knob: same decision, same source order.
    knobbed = _context(tmp_path, paired_arms=True)
    plain_experiment = _experiment(tiny_base)
    assert evaluator._spec_for(
        plain_experiment, artifact, knobbed, eval_dir=tmp_path / "c2"
    ).paired_arms is True
    assert RouterHealingExecutor.paired_arms_for(
        RouterHealingExecutor._research_spec(plain_experiment), knobbed
    ) is True

    # Declared nowhere: the historical two-process path.
    assert evaluator._spec_for(
        plain_experiment, artifact, _context(tmp_path), eval_dir=tmp_path / "c3"
    ).paired_arms is False
    assert RouterHealingExecutor.paired_arms_for({}, _context(tmp_path)) is False


def test_a_paired_project_defers_the_standalone_baseline_measurement(
    tmp_path, tiny_base
):
    """The project seam: paired_arms defers the standalone baseline.

    The runner's deferred-baseline provider needs to know, before any model
    is loaded, that the automatic baseline's measurement will arrive with the
    candidate's own resident-pair evaluation -- so the run starts without the
    separate base-arm spawn and the row waits for the paired evidence.
    """
    _require_real_model()
    from chowder.backends.router_healing import RouterHealingEvaluator

    assert (
        RouterHealingEvaluator().defers_automatic_baseline(
            experiment=_experiment(tiny_base, paired_arms=True),
            context=_context(tmp_path),
        )
        is True
    )
    assert (
        RouterHealingEvaluator().defers_automatic_baseline(
            experiment=_experiment(tiny_base),
            context=_context(tmp_path),
        )
        is False
    )


def test_the_paired_arm_loads_the_model_once(tmp_path, tiny_base, payloads):
    """The measured amortization: one load where the two-process path loads twice.

    On CPU this costs wall seconds; on the 9B CUDA run it was 12.8 s of
    on-device time per load. The claim "the pair amortizes the load" is a
    measurement, not a slogan: the paired spec must report exactly one load
    in its lifecycle and no more.
    """
    _require_real_model()
    from chowder.backends import router_healing_eval_worker as eval_worker_module
    from chowder.lifecycle import PHASE_MODEL_LOAD, ledger_from_payload

    loads = 0

    def counting_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return real_load_with_policy(*args, **kwargs)

    real_load_with_policy = eval_worker_module.load_with_policy
    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_worker_module, "load_with_policy", counting_load)
    try:
        spec = RouterHealingEvalSpec(
            base_model_dir=tiny_base["base_dir"],
            base_content_sha256=tiny_base["content_sha256"],
            payload_dir=str(payloads["changed"]),
            holdout_corpus_path=tiny_base["holdout"],
            holdout_corpus_sha256=tiny_base["holdout_sha256"],
            expected_parameter_paths=(
                "model.layers.0.mlp.gate.weight",
                "model.layers.1.mlp.gate.weight",
            ),
            output_dir=str(tmp_path / "out"),
            seq_len=16,
            batches=1,
            paired_arms=True,
        )
        result = eval_worker_module.evaluate(spec)
    finally:
        monkey.undo()

    assert loads == 1, "the resident pair must load the base exactly once"
    ledger = ledger_from_payload(result["lifecycle"])
    phases = ledger.to_dict()["phases"]
    assert phases[PHASE_MODEL_LOAD]["measured"] is True
    # The historical path pays this load twice (once per arm/process); the
    # measured saving is one whole load. Assert the ledger reports the single
    # load duration so a prereg can budget it as one phase.
    assert phases[PHASE_MODEL_LOAD]["seconds"] > 0.0


def test_a_paired_result_without_a_measurable_base_score_is_refused(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """An unmeasured in-process baseline is unknown, not zero.

    The parent must refuse a paired result whose base arm never produced a
    score: the baseline completer would otherwise complete the row from a
    fabricated (or absent) number, which is exactly the 'counted as present'
    defect class P7 closed elsewhere.
    """

    def factory(command, **kwargs):
        class FakeProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
                result = _valid_eval_result(spec)
                result["arm"] = "paired"
                result["application_control"]["applied_parameters"] = list(
                    spec.expected_parameter_paths
                )
                result["base_holdout_loss"] = None
                result_path.write_text(json.dumps(result), encoding="utf-8")

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
    experiment = _experiment(tiny_base, paired_arms=True)
    artifact = _eval_artifact(experiment, payloads["changed"])
    with pytest.raises(RouterHealingEvaluationError, match="no measurable base score"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def test_a_successful_paired_evaluation_labels_itself_paired_in_evidence(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """The evidence label must say what ran: 'paired', not 'candidate'.

    The project runner's baseline completer reads ``evidence['arm']`` to decide
    whether the resident pair actually measured the base; an arm that is
    labelled 'candidate' after running as a pair makes the completer refuse a
    successful run. This pins the exact regression: the worker result says
    'paired' and the outcome evidence must repeat it, alongside the base score
    the completer completes the baseline row from.
    """

    def factory(command, **kwargs):
        class FakeProcess:
            returncode = 0

            def __init__(self, command, **process_kwargs):
                spec_path = Path(command[command.index("--spec") + 1])
                result_path = Path(command[command.index("--result") + 1])
                spec = RouterHealingEvalSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
                assert spec.paired_arms is True
                result = _valid_eval_result(spec)
                result["arm"] = "paired"
                result["application_control"]["applied_parameters"] = list(
                    spec.expected_parameter_paths
                )
                result_path.write_text(json.dumps(result), encoding="utf-8")

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
    experiment = _experiment(tiny_base, paired_arms=True)
    artifact = _eval_artifact(experiment, payloads["changed"])
    outcome = RouterHealingEvaluator().evaluate(
        experiment=experiment, artifact=artifact, context=_context(tmp_path)
    )
    assert outcome.evidence["arm"] == "paired"
    assert outcome.evidence["base_holdout_loss"] == pytest.approx(1.0)
    assert outcome.evidence["payload_applied"] is True

# --- the rung-3c aggregate ceiling: the prereg's headline enforcement --------

#: The rung-3c preregistered decomposition (frozen in
#: docs/quals/P11_RUNG3C_CUDA_PAIRED_PREREG_2026-09-15.md): measured paired
#: total 0.0161417 GPU-h x 1.5 = 0.0242, ceiling rounded to 0.025 and split
#: exactly across the three measurable phase categories.
_PREREG_CEILING = 0.025
_PREREG_SUB_BUDGETS = {"loads": 0.0107, "steps": 0.0100, "generations": 0.0043}


def test_the_run_ceiling_projection_refuses_any_budget_violation():
    """The sum ceiling and each sub-budget are enforced, on measured phases."""
    from chowder.backends.device_preflight import project_run_ceiling

    # The measured rung-3c basis: two loads (25.5 s), 12 steps at 2.0 s, and
    # 7.4 s of generations = 56.9 s = 0.0158 GPU-h, inside every line.
    fitting = project_run_ceiling(
        load_seconds=25.5,
        step_seconds=2.0,
        max_steps=12,
        eval_generation_seconds=7.4,
        accelerator_count=1,
        max_gpu_hours=_PREREG_CEILING,
        sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
    )
    assert fitting["measured"] is True
    assert fitting["would_exceed_ceiling"] is False
    assert fitting["projected_gpu_hours"] == pytest.approx(56.9 / 3600.0)
    for category in ("loads", "steps", "generations"):
        assert fitting["sub_budgets"][category]["would_exceed"] is False

    # The total fits but the loads category alone busts its sub-budget: the
    # prereg's decomposition is enforced per category, not only in sum.
    lopsided = project_run_ceiling(
        load_seconds=45.0,
        step_seconds=2.0,
        max_steps=12,
        eval_generation_seconds=7.4,
        accelerator_count=1,
        max_gpu_hours=_PREREG_CEILING,
        sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
    )
    assert lopsided["sub_budgets"]["loads"]["would_exceed"] is True
    assert lopsided["would_exceed_ceiling"] is True, (
        "a run over any sub-budget must refuse even when the sum still fits"
    )

    # The plain sum bust: 60 s of steps pushes the total to 92.9 s =
    # 0.0258 GPU-h, past the 0.025 ceiling.
    over = project_run_ceiling(
        load_seconds=25.5,
        step_seconds=5.0,
        max_steps=12,
        eval_generation_seconds=7.4,
        accelerator_count=1,
        max_gpu_hours=_PREREG_CEILING,
        sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
    )
    assert over["would_exceed_ceiling"] is True
    assert over["projected_gpu_hours"] > _PREREG_CEILING


def test_the_run_ceiling_projection_refuses_unreadable_budgets():
    """Non-finite phases, a non-positive ceiling, and a decomposition that
    does not sum to the ceiling are construction errors, not verdicts."""
    from chowder.backends.device_preflight import project_run_ceiling

    common = dict(
        step_seconds=1.0,
        max_steps=1,
        eval_generation_seconds=1.0,
        accelerator_count=1,
    )
    with pytest.raises(ValueError, match="max_gpu_hours"):
        project_run_ceiling(
            load_seconds=1.0, max_gpu_hours=0.0,
            sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS), **common,
        )
    with pytest.raises(ValueError, match="sum"):
        project_run_ceiling(
            load_seconds=1.0, max_gpu_hours=0.025,
            sub_budget_gpu_hours={"loads": 0.0107, "steps": 0.0100, "generations": 0.0040},
            **common,
        )
    with pytest.raises(ValueError, match="sub_budget"):
        project_run_ceiling(
            load_seconds=1.0, max_gpu_hours=0.025,
            sub_budget_gpu_hours={"loads": 0.0107, "steps": 0.0100},
            **common,
        )
    with pytest.raises(ValueError, match="finite"):
        project_run_ceiling(
            load_seconds=float("nan"), max_gpu_hours=0.025,
            sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS), **common,
        )


def test_the_run_spec_accepts_and_binds_the_run_ceiling():
    """The ceiling rides the spec (so workers must honor it) but stays out of
    the recipe (so it never changes what a payload is)."""
    assert RouterHealingRunSpec(**_spec_kwargs()).max_gpu_hours is None
    spec = RouterHealingRunSpec(
        **_spec_kwargs(
            max_gpu_hours=_PREREG_CEILING,
            sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
        )
    )
    assert spec.max_gpu_hours == _PREREG_CEILING
    assert spec.sub_budget_gpu_hours == _PREREG_SUB_BUDGETS
    unbudgeted = RouterHealingRunSpec(**_spec_kwargs())
    assert spec.digest() != unbudgeted.digest(), "the spec digest must bind the ceiling"
    assert spec.recipe_digest() == unbudgeted.recipe_digest(), (
        "a scheduling ceiling is not a recipe change"
    )


def test_the_run_spec_refuses_an_unreadable_run_ceiling():
    with pytest.raises(ValueError, match="max_gpu_hours"):
        RouterHealingRunSpec(
            **_spec_kwargs(max_gpu_hours=0.0, sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS))
        )
    with pytest.raises(ValueError, match="max_gpu_hours"):
        RouterHealingRunSpec(
            **_spec_kwargs(max_gpu_hours=float("inf"), sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS))
        )
    with pytest.raises(ValueError, match="sum"):
        RouterHealingRunSpec(
            **_spec_kwargs(
                max_gpu_hours=0.025,
                sub_budget_gpu_hours={"loads": 0.0107, "steps": 0.0100, "generations": 0.0040},
            )
        )
    with pytest.raises(ValueError, match="sub_budget"):
        RouterHealingRunSpec(
            **_spec_kwargs(
                max_gpu_hours=0.025,
                sub_budget_gpu_hours={"loads": 0.0107, "steps": 0.0100},
            )
        )


def _eval_spec_kwargs(**overrides) -> dict:
    base = {
        "base_model_dir": "unused-base",
        "base_content_sha256": "a" * 64,
        "payload_dir": "unused-payload",
        "holdout_corpus_path": "unused-holdout",
        "holdout_corpus_sha256": "c" * 64,
        "expected_parameter_paths": ("model.layers.0.mlp.gate.weight",),
        "output_dir": "unused-out",
        "seq_len": 16,
        "batches": 1,
    }
    base.update(overrides)
    return base


def test_the_eval_spec_accepts_and_refuses_the_run_ceiling():
    assert RouterHealingEvalSpec(**_eval_spec_kwargs()).max_gpu_hours is None
    spec = RouterHealingEvalSpec(
        **_eval_spec_kwargs(
            max_gpu_hours=_PREREG_CEILING,
            sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
        )
    )
    assert spec.max_gpu_hours == _PREREG_CEILING
    with pytest.raises(ValueError, match="max_gpu_hours"):
        RouterHealingEvalSpec(
            **_eval_spec_kwargs(max_gpu_hours=-1.0, sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS))
        )
    with pytest.raises(ValueError, match="sum"):
        RouterHealingEvalSpec(
            **_eval_spec_kwargs(
                max_gpu_hours=0.025,
                sub_budget_gpu_hours={"loads": 0.0107, "steps": 0.0100, "generations": 0.0040},
            )
        )


def test_a_training_run_overrunning_its_run_ceiling_is_refused(tmp_path, tiny_base):
    """The worker refuses an impossible ceiling before optimizer step 1 --
    the aggregate claim enforced where it can first be known."""
    _require_real_model()
    spec = RouterHealingRunSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        corpus_path=tiny_base["corpus"],
        corpus_sha256=tiny_base["corpus_sha256"],
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        learning_rate=0.01,
        seq_len=16,
        batch_size=1,
        seed=0,
        probe_window=1,
        max_tokens=32,
        max_gpu_hours=1e-9,
        sub_budget_gpu_hours={"loads": 5e-10, "steps": 4e-10, "generations": 1e-10},
    )
    with pytest.raises(RuntimeError, match="run ceiling"):
        train(spec)


def test_a_paired_evaluation_overrunning_its_run_ceiling_is_refused(
    tmp_path, tiny_base, payloads
):
    """The eval worker refuses before scoring when its measured phases cannot
    fit the declared ceiling."""
    _require_real_model()
    from chowder.backends import router_healing_eval_worker as eval_worker_module

    spec = RouterHealingEvalSpec(
        base_model_dir=tiny_base["base_dir"],
        base_content_sha256=tiny_base["content_sha256"],
        payload_dir=str(payloads["changed"]),
        holdout_corpus_path=tiny_base["holdout"],
        holdout_corpus_sha256=tiny_base["holdout_sha256"],
        expected_parameter_paths=(
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ),
        output_dir=str(tmp_path / "out"),
        seq_len=16,
        batches=1,
        max_gpu_hours=1e-9,
        sub_budget_gpu_hours={"loads": 5e-10, "steps": 4e-10, "generations": 1e-10},
    )
    with pytest.raises(RuntimeError, match="run ceiling"):
        eval_worker_module.evaluate(spec)


def test_the_parent_refuses_a_missing_run_ceiling_block_when_budgeted(
    tmp_path, monkeypatch, tiny_base
):
    """A declared ceiling demands the worker's measured run-ceiling block;
    absent is unknown, not passing."""

    def mutate(result, spec):
        result.pop("run_ceiling", None)

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="run ceiling"):
        RouterHealingExecutor().run(
            _experiment(
                tiny_base,
                max_gpu_hours=_PREREG_CEILING,
                sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
            ),
            _context(tmp_path),
        )


def test_the_parent_refuses_a_result_that_lies_about_the_run_ceiling(
    tmp_path, monkeypatch, tiny_base
):
    """The block claiming the ceiling holds while the ledger's own measured
    phases sum past it is exactly the lie the parent exists to catch."""

    def mutate(result, spec):
        block = result.get("run_ceiling")
        if isinstance(block, dict):
            block["would_exceed_ceiling"] = False
        phases = (result.get("lifecycle") or {}).get("phases") or {}
        steady = phases.get("steady_state_steps")
        if isinstance(steady, dict):
            steady["gpu_hours"] = 0.9  # 0.9 GPU-h against a 0.025 ceiling

    _install_fake_worker(monkeypatch, mutate)
    with pytest.raises(RouterHealingBackendError, match="run ceiling"):
        RouterHealingExecutor().run(
            _experiment(
                tiny_base,
                max_gpu_hours=_PREREG_CEILING,
                sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
            ),
            _context(tmp_path),
        )


def test_the_eval_parent_refuses_an_arm_that_lies_about_its_run_ceiling(
    tmp_path, monkeypatch, tiny_base, payloads
):
    """The evaluator re-derives the arm's phase costs from the ledger and the
    load block; a result whose run-ceiling block disagrees is refused."""
    from chowder.backends.router_healing import RouterHealingEvaluator

    def mutate(result, spec):
        block = result.get("run_ceiling")
        if isinstance(block, dict):
            block["would_exceed_ceiling"] = False
        phases = (result.get("lifecycle") or {}).get("phases") or {}
        load = phases.get("model_load")
        if isinstance(load, dict):
            load["gpu_hours"] = 0.9  # against a 0.025 ceiling

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
    experiment = _experiment(
        tiny_base,
        max_gpu_hours=_PREREG_CEILING,
        sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
    )
    artifact = _eval_artifact(experiment, payloads["uniform"])
    with pytest.raises(RouterHealingEvaluationError, match="run ceiling"):
        RouterHealingEvaluator().evaluate(
            experiment=experiment, artifact=artifact, context=_context(tmp_path)
        )


def test_the_specs_carry_the_preregistered_run_ceiling(tmp_path, tiny_base, payloads):
    """Both spec builders thread the ceiling from research-then-knobs, the
    same precedence as every other preregistered field."""
    from chowder.backends.router_healing import RouterHealingEvaluator

    overrides = dict(
        max_gpu_hours=_PREREG_CEILING,
        sub_budget_gpu_hours=dict(_PREREG_SUB_BUDGETS),
    )
    train_spec = RouterHealingExecutor()._spec_for(
        _experiment(tiny_base, **overrides),
        _context(tmp_path),
        run_dir=tmp_path,
    )
    assert train_spec.max_gpu_hours == _PREREG_CEILING
    assert train_spec.sub_budget_gpu_hours == _PREREG_SUB_BUDGETS

    # The candidate eval spec reads the same fields from the same research.
    experiment = _experiment(tiny_base, **overrides)
    artifact = _eval_artifact(experiment, payloads["changed"])
    eval_spec = RouterHealingEvaluator()._spec_for(
        experiment, artifact, _context(tmp_path), eval_dir=tmp_path / "eval"
    )
    assert eval_spec.max_gpu_hours == _PREREG_CEILING
    assert eval_spec.sub_budget_gpu_hours == _PREREG_SUB_BUDGETS
