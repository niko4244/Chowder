from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from chowder.kaggle_dispatch import TRAIN_SCRIPT, KaggleDispatchError, main, render_job_script
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


def test_train_push_attaches_dataset_and_resume_sources(tmp_path):
    fake = FakeKaggle(["complete"], output_files=("a3_job.json",))
    rc = main(["train", "--owner", "nik", "--commit", SHA, "--dataset", "nik/chowder-openr1-pilot",
               "--resume-kernel", "nik/chowder-a3-train-1", "--work-dir", str(tmp_path), "--poll-seconds", "0"],
              runner=fake)
    assert rc == 0
    meta = json.loads(next(tmp_path.rglob("kernel-metadata.json")).read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["nik/chowder-openr1-pilot"]
    assert meta["kernel_sources"] == ["nik/chowder-a3-train-1"]
    staged = _load(next(tmp_path.rglob("kernel/run_a3_train.py")), "staged_a3")
    assert staged.JOB["resume_mount"] == "chowder-a3-train-1" and staged.JOB["commit"] == SHA


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
