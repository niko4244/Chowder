"""Resume must continue a bound experiment, including stochastic state."""

import json
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from chowder.backends.router_healing import RouterHealingExecutor
from chowder.backends.router_healing_worker import train
from chowder.base_identity import resolve_base_identity
from test_router_healing_backend import _context, _experiment, tiny_base  # noqa: F401


@pytest.fixture(scope="module")
def interrupted_run(tmp_path_factory, tiny_base):
    root = tmp_path_factory.mktemp("router-stochastic-resume")
    base = root / "dropout-base"
    shutil.copytree(tiny_base["base_dir"], base)
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    config["attention_dropout"] = 0.3
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    identity = resolve_base_identity(base)
    data = dict(tiny_base, base_dir=str(base), manifest_sha256=identity["manifest_sha256"])
    spec = RouterHealingExecutor()._spec_for(
        _experiment(data),
        _context(root, checkpoint_dir=str(root / "checkpoints"), checkpoint_every=2,
                 scheduler="cosine", warmup_steps=1),
        run_dir=root / "control",
    )
    control = train(spec)
    return spec, control, Path(control["checkpoints"][0]["directory"])


def test_dropout_resume_matches_uninterrupted_training(tmp_path, interrupted_run):
    import torch
    from safetensors.torch import load_file

    spec, control, checkpoint = interrupted_run
    resumed = train(replace(spec, resume_from=str(checkpoint),
                            output_dir=str(tmp_path / "resumed"), checkpoint_dir=None))
    assert resumed["losses"] == control["losses"][2:]
    assert resumed["step_trace"] == control["step_trace"][2:]
    assert resumed["limits"]["tokens_consumed_total"] == control["limits"]["tokens_consumed_total"]
    original = load_file(str(Path(spec.output_dir) / "payload" / "router_payload.safetensors"))
    restored = load_file(str(tmp_path / "resumed" / "payload" / "router_payload.safetensors"))
    assert original.keys() == restored.keys()
    for name in original:
        assert torch.equal(original[name], restored[name]), name


@pytest.mark.parametrize("change", [
    {"learning_rate": 0.01}, {"seed": 12}, {"max_tokens": 1024}, {"max_steps": 5},
    {"base_content_sha256": "b" * 64}, {"corpus_sha256": "c" * 64}, {"batch_size": 1},
])
def test_resume_refuses_changed_recipe(tmp_path, interrupted_run, change):
    spec, _, checkpoint = interrupted_run
    with pytest.raises(RuntimeError, match="checkpoint.*(recipe|budget)"):
        train(replace(spec, resume_from=str(checkpoint), output_dir=str(tmp_path / "out"),
                      checkpoint_dir=None, **change))


@pytest.mark.parametrize("filename", ["rng_state.pth", "scheduler.pt", "optimizer.pt",
                                      "router_state.safetensors", "trainer_state.json"])
def test_resume_refuses_missing_state(tmp_path, interrupted_run, filename):
    spec, _, checkpoint = interrupted_run
    copy = tmp_path / "checkpoint"
    shutil.copytree(checkpoint, copy)
    (copy / filename).unlink()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train(replace(spec, resume_from=str(copy), output_dir=str(tmp_path / "out"),
                      checkpoint_dir=None))


@pytest.mark.parametrize("filename", ["rng_state.pth", "scheduler.pt", "optimizer.pt",
                                      "router_state.safetensors", "trainer_state.json"])
def test_resume_refuses_corrupt_state(tmp_path, interrupted_run, filename):
    spec, _, checkpoint = interrupted_run
    copy = tmp_path / "checkpoint"
    shutil.copytree(checkpoint, copy)
    with (copy / filename).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(RuntimeError, match="checkpoint.*content hash mismatch"):
        train(replace(spec, resume_from=str(copy), output_dir=str(tmp_path / "out"),
                      checkpoint_dir=None))


def test_legacy_unbound_checkpoint_is_preserved_and_refused(tmp_path, interrupted_run):
    spec, _, checkpoint = interrupted_run
    copy = tmp_path / "legacy"
    shutil.copytree(checkpoint, copy)
    (copy / "checkpoint_manifest.json").unlink()
    before = {p.name: p.read_bytes() for p in copy.iterdir()}
    with pytest.raises(RuntimeError, match="checkpoint"):
        train(replace(spec, resume_from=str(copy), output_dir=str(tmp_path / "out"),
                      checkpoint_dir=None))
    assert {p.name: p.read_bytes() for p in copy.iterdir()} == before


@pytest.mark.parametrize("defect", ["shape", "extra", "nonfinite", "dtype"])
def test_router_restore_requires_exact_finite_tensors(tmp_path, interrupted_run, defect):
    import torch
    from safetensors.torch import load_file, save_file

    spec, _, checkpoint = interrupted_run
    copy = tmp_path / "checkpoint"
    shutil.copytree(checkpoint, copy)
    tensor_path = copy / "router_state.safetensors"
    tensors = load_file(str(tensor_path))
    name = next(iter(tensors))
    if defect == "shape":
        tensors[name] = tensors[name][:1].clone()  # copy_ alone would broadcast this.
    elif defect == "extra":
        tensors["unintended.weight"] = torch.zeros(1)
    elif defect == "nonfinite":
        tensors[name].fill_(float("nan"))
    else:
        tensors[name] = tensors[name].half()
    save_file(tensors, str(tensor_path))
    manifest_path = copy / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][tensor_path.name] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint router"):
        train(replace(spec, resume_from=str(copy), output_dir=str(tmp_path / "out"),
                      checkpoint_dir=None))


def test_resume_keeps_the_original_total_token_ceiling(tmp_path, interrupted_run):
    spec, _, _ = interrupted_run
    bounded = replace(spec, output_dir=str(tmp_path / "control"), max_tokens=96,
                      checkpoint_dir=str(tmp_path / "state"))
    control = train(bounded)
    checkpoint = control["checkpoints"][0]["directory"]
    resumed = train(replace(bounded, output_dir=str(tmp_path / "resumed"),
                            resume_from=checkpoint, checkpoint_dir=None))
    assert control["global_step"] == resumed["global_step"] == 3
    assert resumed["steps_completed"] == 1
    assert resumed["limits"]["tokens_consumed_total"] == 96
    assert resumed["limits"]["stop_reason"] == "max_tokens"
    assert resumed["losses"] == control["losses"][2:]


@pytest.mark.parametrize("existing", ["step-2", ".step-2.partial"])
def test_checkpoint_publication_preserves_existing_artifacts(tmp_path, interrupted_run, existing):
    import torch
    from chowder.backends.router_healing_worker import _publish_checkpoint

    spec, _, _ = interrupted_run
    destination = tmp_path / existing
    destination.mkdir()
    (destination / "retained.txt").write_bytes(b"retain this evidence")
    with pytest.raises(RuntimeError, match="overwrite existing state"):
        _publish_checkpoint(tmp_path, step=2, spec=spec, optimizer=None, tensors={}, torch=torch)
    assert list(destination.iterdir()) == [destination / "retained.txt"]
    assert (destination / "retained.txt").read_bytes() == b"retain this evidence"


def test_fresh_process_resume_after_forced_kill(tmp_path, interrupted_run):
    """Kill a real dropout worker immediately after atomic checkpoint publication.

    A test-only pause makes the interruption boundary deterministic. Training,
    state serialization, and the resumed CLI worker are real; no result is faked.
    """
    import subprocess
    import sys
    import time

    from chowder.worker_env import chowder_source_identity, worker_env

    spec, control, _ = interrupted_run
    first = replace(spec, output_dir=str(tmp_path / "killed"),
                    checkpoint_dir=str(tmp_path / "checkpoints"))
    spec_path = tmp_path / "spec.json"
    identity_path = tmp_path / "source-identity.json"
    result_path = tmp_path / "result.json"
    spec_path.write_text(json.dumps(first.to_dict()), encoding="utf-8")
    identity_path.write_text(json.dumps(chowder_source_identity()), encoding="utf-8")
    marker = tmp_path / "published"
    pause = (
        "import sys,time; from pathlib import Path; "
        "import chowder.backends.router_healing_worker as w\n"
        "publish = w._publish_checkpoint\n"
        "def stop_after_publish(*args, **kwargs):\n"
        "    result = publish(*args, **kwargs)\n"
        f"    Path({str(marker)!r}).write_text('ready')\n"
        "    while True: time.sleep(0.05)\n"
        "w._publish_checkpoint = stop_after_publish\n"
        "raise SystemExit(w.main())\n"
    )
    args = ["--spec", str(spec_path), "--result", str(result_path),
            "--chowder-identity", str(identity_path)]
    env = worker_env()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (tmp_path / "killed.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-c", pause, *args], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert marker.exists(), (tmp_path / "killed.log").read_text(encoding="utf-8")
            assert process.poll() is None
            process.kill()
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
    assert not result_path.exists()
    resumed_spec = replace(first, output_dir=str(tmp_path / "resumed"), checkpoint_dir=None,
                           resume_from=str(tmp_path / "checkpoints" / "step-2"))
    spec_path.write_text(json.dumps(resumed_spec.to_dict()), encoding="utf-8")
    with (tmp_path / "resumed.log").open("w", encoding="utf-8") as log:
        resumed = subprocess.run(
            [sys.executable, "-m", "chowder.backends.router_healing_worker", *args], env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=180,
        )
    assert resumed.returncode == 0, (tmp_path / "resumed.log").read_text(encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["losses"] == control["losses"][2:]
    assert result["step_trace"] == control["step_trace"][2:]
    import torch
    from safetensors.torch import load_file

    expected = load_file(control["payload"]["tensor_path"])
    actual = load_file(result["payload"]["tensor_path"])
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
