from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import hashlib
import subprocess

from chowder.kaggle_dispatch import (
    TRAIN_PULL_PATTERN,
    TRAIN_SCRIPT,
    KaggleDispatchError,
    main,
    render_job_script,
    verify_adapter,
)
from test_kaggle_dispatch import FakeKaggle

REPO = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_kernel_script_carries_the_job(tmp_path):
    job = {"commit": SHA, "data_mount": "chowder-openr1-pilot", "resume_mount": None, "save_steps": 10}
    rendered = render_job_script(TRAIN_SCRIPT, job, tmp_path)
    assert "__CHOWDER_JOB__" not in rendered.read_text(encoding="utf-8")
    assert _load(rendered, "rendered_a3").JOB == job  # importing runs no stage


def test_render_refuses_a_template_without_one_placeholder(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("JOB = {}\n")
    with pytest.raises(KaggleDispatchError):
        render_job_script(bad, {}, tmp_path / "out")


def _write_train_output(dest: Path, weights: bytes = b"lora", record: bytes = b"lora") -> None:
    adapter = dest / "cond_a3_kaggle" / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    digests = {"adapter_model.safetensors": hashlib.sha256(record).hexdigest()}
    (dest / "a3_job.json").write_text(json.dumps({"stages": {"train": {"ok": True, "result": {"adapter_sha256": digests}}}}),
                                      encoding="utf-8")


class FakeTrainKaggle(FakeKaggle):
    def __init__(self, weights: bytes = b"lora"):
        super().__init__(["complete"])
        self.weights = weights

    def __call__(self, args):
        args = list(args)
        if args[2] == "output":
            self.calls.append(args)
            _write_train_output(Path(args[args.index("-p") + 1]), weights=self.weights)
            return subprocess.CompletedProcess(args, 0, "", "")
        return super().__call__(args)


def test_verify_adapter_accepts_the_recorded_bytes_and_rejects_others(tmp_path):
    _write_train_output(tmp_path)
    assert verify_adapter(tmp_path) == {"verified_files": 1}
    _write_train_output(tmp_path / "bad", weights=b"truncated")
    with pytest.raises(KaggleDispatchError, match="does not match"):
        verify_adapter(tmp_path / "bad")


def test_train_pull_is_scoped_and_never_fetches_checkpoints():
    import re

    wanted = re.compile(TRAIN_PULL_PATTERN)
    assert wanted.search("cond_a3_kaggle/adapter/adapter_model.safetensors")
    assert wanted.search("cond_a3_kaggle/worker-result.json")
    assert not wanted.search("cond_a3_kaggle/adapter/trainer/checkpoint-4/optimizer.pt")
    assert not wanted.search("cond_a3_kaggle/adapter/trainer/checkpoint-4/adapter_model.safetensors")
    assert wanted.search("cond_a3_kaggle/adapter/tokenizer.json")  # the final adapter's own copy
    assert not wanted.search("cond_a3_kaggle/adapter/trainer/checkpoint-4/tokenizer.json")
    assert wanted.search("cond_a3_kaggle/adapter/trainer/checkpoint-4/trainer_state.json")


@pytest.mark.parametrize("devices,ok", [(None, False), ("0,1", True)])
def test_train_pilot_refuses_a_batch_the_recipe_did_not_declare(tmp_path, monkeypatch, capsys, devices, ok):
    tp = _load(REPO / "experiments" / "teacher_free_distill" / "train_pilot.py", "train_pilot_batch")
    monkeypatch.setattr(tp, "check_device_exclusivity",
                        lambda d: {"device_index": d, "device_name": "T4", "total_vram_gb": 14.6, "free_vram_gb": 14.4})
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    recipe = REPO / "experiments" / "teacher_free_distill" / "recipes" / "a3_sft_openr1_complete_kaggle_fp16.json"
    argv = ["train_pilot", "--recipe", str(recipe), "--data-dir", str(tmp_path), "--yes", "--dry-run"]
    monkeypatch.setattr("sys.argv", argv + (["--devices", devices] if devices else []))
    with pytest.raises(SystemExit):
        tp.main()  # the placeholder data always stops it at the pinned-digest check at the latest
    err = capsys.readouterr().err
    if ok:  # 1 x 16 x 2 = 32 passes the batch check and reaches the digest check
        assert "effective batch" not in err and "dataset digest mismatch" in err
    else:  # 1 x 16 x 1 = 16 on a recipe declaring 32
        assert "effective batch 16 != recipe's declared effective_batch 32" in err


def test_train_fails_when_the_pulled_adapter_does_not_verify(tmp_path):
    rc = main(["train", "--owner", "nik", "--commit", SHA, "--dataset", "nik/chowder-openr1-pilot",
               "--work-dir", str(tmp_path), "--poll-seconds", "0"], runner=FakeTrainKaggle(weights=b"cut short"))
    assert rc == 1


def test_train_push_attaches_dataset_and_resume_sources(tmp_path):
    fake = FakeTrainKaggle()
    rc = main(["train", "--owner", "nik", "--commit", SHA, "--dataset", "nik/chowder-openr1-pilot",
               "--resume-kernel", "nik/chowder-a3-train-1", "--max-steps", "3", "--work-dir", str(tmp_path), "--poll-seconds", "0"],
              runner=fake)
    assert rc == 0
    meta = json.loads(next(tmp_path.rglob("kernel-metadata.json")).read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["nik/chowder-openr1-pilot"]
    assert meta["kernel_sources"] == ["nik/chowder-a3-train-1"]
    staged = _load(next(tmp_path.rglob("kernel/run_a3_train.py")), "staged_a3")
    assert staged.JOB["resume_mount"] == "chowder-a3-train-1" and staged.JOB["commit"] == SHA
    assert staged.JOB["max_steps"] == 3
    assert staged.JOB["timeout_hours"] == 11.0  # default 690-min cap minus 30 min of margin
    pull = next(c for c in fake.calls if c[2] == "output")
    assert pull[pull.index("--file-pattern") + 1] == TRAIN_PULL_PATTERN
    record = json.loads(next(tmp_path.rglob("job_record.json")).read_text(encoding="utf-8"))
    assert record["verification"] == {"verified_files": 1}


def test_train_can_resume_from_a_dataset_mount(tmp_path):
    # An ERRORED kernel's output cannot be mounted (A4, 2026-09-28): its checkpoint
    # is re-uploaded as a dataset and attached as a second dataset source.
    fake = FakeTrainKaggle()
    rc = main(["train", "--owner", "nik", "--commit", SHA, "--dataset", "nik/chowder-openr1-a4",
               "--resume-dataset", "nik/chowder-a4-ckpt210", "--work-dir", str(tmp_path), "--poll-seconds", "0"],
              runner=fake)
    assert rc == 0
    meta = json.loads(next(tmp_path.rglob("kernel-metadata.json")).read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["nik/chowder-openr1-a4", "nik/chowder-a4-ckpt210"]
    assert meta["kernel_sources"] == []
    assert _load(next(tmp_path.rglob("kernel/run_a3_train.py")), "staged_ds").JOB["resume_mount"] == "chowder-a4-ckpt210"


def test_eval_push_attaches_adapter_kernels_and_renders_jobs(tmp_path):
    from chowder.kaggle_dispatch import EVAL_PULL_PATTERN, parse_eval_job

    assert parse_eval_job("a4@nik/chowder-a4-resume:1:75:150") == {
        "arm": "a4", "adapter_kernel": "nik/chowder-a4-resume", "adapter_mount": "chowder-a4-resume",
        "gpu": 1, "rows": [75, 150]}
    for bad in ("a4:2:0:10", "a4:0:10:10", "A4:0:0:10", "a4@nokernel:0:0:10"):
        with pytest.raises(KaggleDispatchError):
            parse_eval_job(bad)
    fake = FakeKaggle(["complete"], output_files=("eval_job.json",))
    rc = main(["eval", "--owner", "nik", "--slug", "chowder-math150-base-a3", "--commit", SHA,
               "--prompts-dataset", "nik/chowder-math150", "--job", "base:0:0:150",
               "--job", "a3@nik/chowder-a3-full:1:0:150", "--work-dir", str(tmp_path), "--poll-seconds", "0"],
              runner=fake)
    assert rc == 0
    meta = json.loads(next(tmp_path.rglob("kernel-metadata.json")).read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["nik/chowder-math150"] and meta["kernel_sources"] == ["nik/chowder-a3-full"]
    staged = _load(next(tmp_path.rglob("kernel/run_eval.py")), "staged_eval")
    assert [j["arm"] for j in staged.JOB["jobs"]] == ["base", "a3"] and staged.JOB["precision"] == "fp16"
    pull = next(c for c in fake.calls if c[2] == "output")
    assert pull[pull.index("--file-pattern") + 1] == EVAL_PULL_PATTERN


def test_train_refuses_a_short_commit(tmp_path):
    assert main(["train", "--owner", "nik", "--commit", "abc123", "--dataset", "nik/x-data",
                 "--work-dir", str(tmp_path)], runner=FakeKaggle([])) == 2


def test_kaggle_recipe_keeps_effective_batch_and_maps_fp16(tmp_path):
    tp = _load(REPO / "experiments" / "teacher_free_distill" / "train_pilot.py", "train_pilot_a3")
    recipe = json.loads((REPO / "experiments" / "teacher_free_distill" / "recipes"
                         / "a3_sft_openr1_complete_kaggle_fp16.json").read_text(encoding="utf-8"))
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    backend = tp.build_resolved_config(recipe, tmp_path)["backend"]
    assert backend["precision"] == "fp16"
    tr = backend["training"]
    assert tr["batch_size"] * tr["gradient_accumulation_steps"] * 2 == 32


def test_adapter_save_retries_only_a_windows_lock():
    from chowder.backends.transformers_worker import _save_with_lock_retry

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Error while serializing: I/O error: ... (os error 32)")

    _save_with_lock_retry(flaky, sleep=waits.append)
    assert len(calls) == 3 and waits == [2.0, 4.0]

    def broken():
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        _save_with_lock_retry(broken, sleep=waits.append)
    assert len(waits) == 2  # no retry for a different error
