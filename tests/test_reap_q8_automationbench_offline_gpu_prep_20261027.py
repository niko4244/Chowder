"""CPU-only source transformation tests: never execute a model or Kaggle job."""

import ast
import hashlib
import io
import json
import tarfile

import pytest

from chowder_batch import reap_q8_automationbench_kaggle_20261026 as original
from chowder_batch import reap_q8_automationbench_offline_gpu_prep_20261027 as prep


def source():
    config = {"profile": "six", "native_flags": list(original.PROFILES["six"]["flags"])}
    return original.KERNEL_TEMPLATE.replace("__LOCKED_CONFIG_LITERAL__", repr(json.dumps(config)))


def installer():
    raw = prep.PREFLIGHT_SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep.PREFLIGHT_SHA
    return raw.decode("utf-8")


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_patches_only_offline_bootstrap_and_preserves_original_official_protocol(profile):
    before = source()
    after = prep.patch_kernel(before, installer(), profile=profile)
    ast.parse(after)
    assert "OFFLINE_PACKAGES, offline_proof = bootstrap_offline(archive)" in after
    assert "namespace['install_and_smoke'](wheels, manifest, archive, OFFLINE_TARGET)" in after
    assert "len(list(wheels.glob('*.whl'))) != 108" in after
    assert "wheel_tar = unique('automationbench-cp313-glibc235-original-108-locked-wheels.tar.bin')" in after
    assert "wheels, manifest = stage_wheels(wheel_tar" in after
    assert "def stage_wheels(archive, destination):" in after
    assert "WHEEL_TAR_SHA = 'e7d1cc7946b17995eacc723f995702874e9a7b5fb7efe9471e5d2939bc1af6ab'" in after
    assert "WORK / ('reap-q8-wheelhouse-' + CFG['profile'])" in after
    assert "'PYTHONPATH':str(OFFLINE_PACKAGES)+os.pathsep+str(source)" in after
    assert "shutil.rmtree(owned)" in after
    assert "--no-index" in after and "--require-hashes" in after
    for unchanged in (
        "'--toolset','api','--domains','all'",
        "'--max-concurrent','1','--max-steps',str(CFG['max_steps'])",
        "'--no-ensure-complete','--export-json',str(result)",
        "R['native_argv'] = [str(executable),*CFG['native_flags']",
        "R['status'] = 'RAW_Q8_OFFICIAL_PUBLIC_TASKS_REQUIRE_INDEPENDENT_ASSERTION_AUDIT'",
        "if evaluator.returncode: raise RuntimeError('OFFICIAL evaluator failed; NO score')",
    ):
        assert unchanged in before and unchanged in after
    assert after.count("def bootstrap_offline(archive):") == 1


def test_rejects_modified_cpu_bootstrap_source():
    with pytest.raises(ValueError, match="source drift"):
        prep.patch_kernel(source(), installer() + "\n# modified", profile="six")


def test_rejects_duplicate_or_missing_original_source_anchor():
    with pytest.raises(ValueError, match="missing or ambiguous"):
        prep.patch_kernel(source().replace("def snapshot():\n", ""), installer(), profile="six")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        prep.patch_kernel(source() + "\ndef snapshot():\n    pass\n", installer(), profile="six")


def test_rejects_unregistered_profile():
    with pytest.raises(ValueError, match="unsupported arm"):
        prep.patch_kernel(source(), installer(), profile="full600")


def _wheel_tar(tmp_path, monkeypatch, *, corruption=None):
    path = tmp_path / "automationbench-cp313-glibc235-original-108-locked-wheels.tar.bin"
    entries = [("reap-q8-automationbench-offline-wheelhouse-manifest.json", b"{}")]
    entries += [(f"pkg{i}-1.0-py3-none-any.whl", b"wheel" + str(i).encode())
                for i in range(108)]
    if corruption == "missing":
        entries.pop()
    elif corruption == "duplicate":
        entries[-1] = entries[-2]
    elif corruption == "escape":
        entries[-1] = ("../escape.whl", b"evil")
    with tarfile.open(path, "w:") as archive:
        for i, (name, raw) in enumerate(entries):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            if corruption == "symlink" and i == len(entries) - 1:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                info.size = 0
            archive.addfile(info, io.BytesIO(raw) if info.isfile() else None)
    monkeypatch.setattr(prep, "WHEEL_TAR_BYTES", path.stat().st_size, raising=False)
    monkeypatch.setattr(prep, "WHEEL_TAR_SHA", hashlib.sha256(path.read_bytes()).hexdigest(),
                        raising=False)
    monkeypatch.setattr(prep, "sha_file", lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                        raising=False)
    monkeypatch.setattr(prep, "WORK", tmp_path, raising=False)
    monkeypatch.setattr(prep.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(prep.os, "posix_fadvise", lambda *_: None, raising=False)
    return path


def test_original_mounted_tar_stages_108_wheels_exclusively(tmp_path, monkeypatch):
    source = _wheel_tar(tmp_path, monkeypatch)
    destination = tmp_path / "private-wheel-scratch"
    wheels, manifest = prep.stage_wheels(source, destination)
    assert wheels == destination
    assert manifest.read_bytes() == b"{}"
    assert len(list(wheels.glob("*.whl"))) == 108
    with pytest.raises(RuntimeError, match="new exclusive"):
        prep.stage_wheels(source, destination)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "escape", "symlink"])
def test_original_mounted_tar_rejects_unsafe_members_before_writes(
    tmp_path, monkeypatch, corruption,
):
    source = _wheel_tar(tmp_path, monkeypatch, corruption=corruption)
    destination = tmp_path / "private-wheel-scratch"
    with pytest.raises(RuntimeError, match="unsafe or incomplete"):
        prep.stage_wheels(source, destination)
    assert not destination.exists()


def test_original_mounted_tar_rejects_hash_drift_before_writes(tmp_path, monkeypatch):
    source = _wheel_tar(tmp_path, monkeypatch)
    monkeypatch.setattr(prep, "WHEEL_TAR_SHA", "0" * 64, raising=False)
    destination = tmp_path / "private-wheel-scratch"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        prep.stage_wheels(source, destination)
    assert not destination.exists()
