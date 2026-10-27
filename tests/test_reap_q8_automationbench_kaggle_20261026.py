"""Synthetic-only RED/GREEN checks. No Kaggle calls, GPUs or model execution."""

import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from chowder_batch import reap_q8_automationbench_kaggle_20261026 as q8


def _tar(*, evil: bool = False) -> bytes:
    required = ("pyproject.toml", "uv.lock", "automationbench/scripts/eval.py",
                "automationbench/export.py", "automationbench/task_contract.py",
                "automationbench/rubric/__init__.py", "automationbench/domains/__init__.py")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as stream:
        for name in (*required, *(('../escape.py',) if evil else ())):
            content = b"official synthetic source fixture"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _official_export(profile: str) -> dict:
    names = list(q8.PROFILES[profile]["task_names"])
    return {"meta": {"benchmark_version": "1.0.6", "toolset": "api",
                     "max_steps": 44, "total_tasks": len(names)},
            "summary": {"aborted_tasks": [], "passed_count": 0},
            "tasks": [{"name": x, "passed": False} for x in names]}


def _read_private_result(tmp_path: Path, profile: str, content: bytes, monkeypatch):
    cfg = dict(q8.PROFILES[profile])
    cfg["q4_raw_sha256"] = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(q8.PROFILES, profile, cfg)
    d = tmp_path / cfg["q4_directory"]
    d.mkdir()
    path = d / "public-automationbench.private.json"
    path.write_bytes(content)
    manifest = {
        "steering_flags": list(cfg["flags"]),
        "observed_context": {"observed_total_context": cfg["context"],
                             "observed_context_per_seq": cfg["context"],
                             "observed_sequences": 1},
        "model": {"gguf_sha256": q8.Q4_GGUF_SHA256, "gguf_bytes": 11563915264},
        "native_cuda": {"layers_cuda": "41/41", "measured_experts_offloaded": 0},
        "official_source": {"source_git_commit": q8.OFFICIAL_COMMIT},
    }
    (d / "experimental-manifest.private.json").write_text(json.dumps(manifest))
    return path


def test_frozen_profiles_have_distinct_actual_q4_arm_flags():
    six = q8.PROFILES["six"]
    two = q8.PROFILES["guard2"]
    assert six["context"] == 65536 and two["context"] == 73728
    assert list(six["flags"]) == ["--fit", "on", "-np", "1", "-c", "65536",
                                   "--n-predict", "4096", "--jinja"]
    assert list(two["flags"]) == ["--fit", "on", "-np", "1", "-c", "73728",
                                   "--repeat-penalty", "1.1", "--dry-multiplier", "0.8",
                                   "--temperature", "0.7", "--jinja"]
    assert len(six["task_names"]) == 6 and len(two["task_names"]) == 2
    assert set(six["task_names"]).isdisjoint(set(two["task_names"]))


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_prospective_q4_private_source_evidence_exact(profile):
    # This reads a file privately but exports only source hashes and names.
    actual = q8.read_q4_evidence(profile)
    assert actual["ordered_task_names"] == list(q8.PROFILES[profile]["task_names"])
    assert actual["q4_strict_pass_count"] == 0
    assert actual["q4_json_sha256"] == q8.PROFILES[profile]["q4_raw_sha256"]


def test_mutated_comparator_results_refused_before_output(tmp_path, monkeypatch):
    data = json.dumps(_official_export("six")).encode()
    path = _read_private_result(tmp_path, "six", data, monkeypatch)
    assert q8.read_q4_evidence("six", tmp_path)["task_count"] == 6
    path.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        q8.read_q4_evidence("six", tmp_path)


def test_forged_name_fails_even_when_attacker_rehashes(tmp_path, monkeypatch):
    payload = _official_export("guard2")
    payload["tasks"][0]["name"] = "sales.altered_task_name"
    data = json.dumps(payload).encode()
    _read_private_result(tmp_path, "guard2", data, monkeypatch)
    with pytest.raises(ValueError, match="task identities"):
        q8.read_q4_evidence("guard2", tmp_path)


def test_abort_launder_fails_even_when_attacker_rehashes(tmp_path, monkeypatch):
    payload = _official_export("six")
    payload["summary"]["aborted_tasks"] = [payload["tasks"][0]["name"]]
    data = json.dumps(payload).encode()
    _read_private_result(tmp_path, "six", data, monkeypatch)
    with pytest.raises(ValueError, match="task identities"):
        q8.read_q4_evidence("six", tmp_path)


def test_q4_source_profile_replay_rejected_even_with_unchanged_official_export(tmp_path, monkeypatch):
    data = json.dumps(_official_export("guard2")).encode()
    path = _read_private_result(tmp_path, "guard2", data, monkeypatch)
    witness = path.parent / "experimental-manifest.private.json"
    forged = json.loads(witness.read_text())
    forged["observed_context"]["observed_total_context"] = 65536
    witness.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="native/protocol attestation"):
        q8.read_q4_evidence("guard2", tmp_path)


@pytest.mark.parametrize("bad", ["../../foreign", "another-owner/public-archive", "nikmarco/no", "nikmarco/A_BAD"])
def test_private_official_source_dataset_slug_rejects_foreign_or_unsafe(bad):
    with pytest.raises(ValueError, match="PRIVATE Kaggle dataset"):
        q8._valid_slug(bad)


def test_no_overwrite_or_symlink_output(tmp_path):
    existing = tmp_path / "present"
    existing.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        q8.validate_private_target(existing, owned_roots=())
    symlink = tmp_path / "sym"
    try:
        symlink.symlink_to(existing, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass
    else:
        with pytest.raises(ValueError, match="exclusive"):
            q8.validate_private_target(symlink, owned_roots=())


def test_private_output_cannot_write_into_authoritative_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="may not be inside"):
        q8.validate_private_target(source / "anything", owned_roots=(source,), private_root=tmp_path)


def test_disallow_even_existing_absolute_output_outside_dedicated_private_root(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(ValueError, match="dedicated private root"):
        q8.validate_private_target(tmp_path / "other", owned_roots=(), private_root=private)


def test_official_git_source_rev_and_untracked_fail_closed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('name = "automation-bench"\nversion = "1.0.6"\n'
                                          'requires-python = ">=3.13"\n'
                                          'auto-bench = "automationbench.scripts.eval:main"')
    (source / "automationbench/rubric").mkdir(parents=True)
    (source / "automationbench/rubric/__init__.py").write_text('task_completed_correctly')
    def trusted(root, *args, **kwargs):
        if args == ("rev-parse", "HEAD"):
            return q8.OFFICIAL_COMMIT.encode()
        if args[0] == "status":
            return b""
        return _tar()
    monkeypatch.setattr(q8, "_git", trusted)
    assert q8.create_official_archive(source) == _tar()
    def shadowed(root, *args, **kwargs):
        if args[0] == "status":
            return b"?? sitecustomize.py"
        return trusted(root, *args, **kwargs)
    monkeypatch.setattr(q8, "_git", shadowed)
    with pytest.raises(ValueError, match="untracked shadow"):
        q8.create_official_archive(source)


def test_git_tar_path_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(q8, "verify_official_checkout", lambda *_: None)
    monkeypatch.setattr(q8, "_git", lambda *_args, **_kwargs: _tar(evil=True))
    with pytest.raises(ValueError, match="symlink/escape"):
        q8.create_official_archive(tmp_path)


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_kernel_render_only_two_exact_profiles_no_execution(tmp_path, monkeypatch, profile):
    config = q8.PROFILES[profile]
    monkeypatch.setattr(q8, "read_q4_evidence", lambda name, root: {
        "q4_json_sha256": config["q4_raw_sha256"],
        "q4_native_manifest_sha256": "0" * 64,
        "ordered_task_names": list(config["task_names"]), "task_count": len(config["task_names"]),
    })
    monkeypatch.setattr(q8, "create_official_archive", lambda *_: _tar())
    monkeypatch.setattr(q8, "PRIVATE_BUILD_ROOT", tmp_path)
    outfile = tmp_path / "new-private-kernel"
    built = q8.build(outfile, profile=profile,
                     official_dataset="nikmarco/pinned-private-automationbench-source",
                     source=tmp_path / "official", q4_root=tmp_path / "q4")
    assert built["status"] == "PRIVATE_Q8_KAGGLE_SOURCE_PREPARED_UNLAUNCHED"
    assert built["matched_count"] == len(config["task_names"])
    assert (outfile / q8.SOURCE_ARCHIVE_NAME).read_bytes() == _tar()
    meta = json.loads((outfile / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [q8.Q8_INPUT_DATASET,
                                       "nikmarco/pinned-private-automationbench-source"]
    generated = (outfile / meta["code_file"]).read_text()
    tree = ast.parse(generated)
    cfg = ast.literal_eval(next(x for x in tree.body if isinstance(x, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "CFG" for t in x.targets)).value.args[0])
    assert json.loads(cfg)["tasks"] == list(config["task_names"])
    assert json.loads(cfg)["native_flags"] == list(config["flags"])
    assert "--no-ensure-complete" in generated and "--max-steps" in generated
    assert "--toolset','api'" in generated
    assert "os.posix_fadvise" in generated and "'cgroup'] < 14" in generated
    assert "41/41" in generated and "n_cpu_moe" in generated
    assert "official evaluator failed" not in generated.lower() or "evaluator failed" in generated
    assert "gpu_compute_pids()" in generated
    assert "Kaggle evaluator wall-budget exceeded" in generated
    assert "RAW_Q8_OFFICIAL_PUBLIC_TASKS_REQUIRE_INDEPENDENT_ASSERTION_AUDIT" in generated
    assert "git clone" not in generated and "pip install" not in generated


def test_kernel_source_static_ast_proven_no_generated_shell_runner():
    tree = ast.parse(q8.KERNEL_TEMPLATE.replace("__LOCKED_CONFIG_LITERAL__", repr(json.dumps({"tasks": []}))))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "kill_owned" for node in tree.body)
    assert "shell=True" not in q8.KERNEL_TEMPLATE
    assert "exec(" not in q8.KERNEL_TEMPLATE
    assert "eval(" not in q8.KERNEL_TEMPLATE
    assert "enable_internet" not in q8.KERNEL_TEMPLATE  # configured by PRIVATE metadata
