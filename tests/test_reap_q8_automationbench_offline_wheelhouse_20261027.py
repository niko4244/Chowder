"""No-network/CPU-only adversarial proof of the original uv.lock wheelhouse."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_q8_automationbench_offline_wheelhouse_20261027 as wheels
from chowder_batch import reap_q8_automationbench_offline_cpu_preflight_20261027 as cpu


def source_manifest():
    return wheels.build_manifest(wheels.official_lock(wheels.SOURCE), glibc_minor=35)


def test_real_git_commit_lock_sha_and_original_runtime_closure():
    lock = wheels.official_lock(wheels.SOURCE)
    manifest = wheels.build_manifest(lock, glibc_minor=35)
    assert wheels.sha256_file(wheels.SOURCE / "uv.lock") == wheels.LOCK_SHA256
    assert wheels.canonical_manifest_sha(manifest) == wheels.EXPECTED_MANIFEST_SHA256_235
    assert manifest["official_source_commit"] == wheels.SOURCE_COMMIT
    assert manifest["package_count"] == 108
    assert manifest["total_locked_wheel_bytes"] == 143_537_351
    assert len({x["name"] for x in manifest["wheels"]}) == 108
    selected = {x["name"]: x["version"] for x in manifest["wheels"]}
    assert all(selected.get(name) == version for name, version in {
        "verifiers": "0.2.1", "datasets": "4.4.2", "pydantic": "2.12.5",
        "numpy": "2.4.0", "pyarrow": "22.0.0",
    }.items())
    assert "pywin32" not in {x["name"] for x in manifest["wheels"]}
    assert "win32-setctime" not in {x["name"] for x in manifest["wheels"]}
    assert "automation-bench" not in {x["name"] for x in manifest["wheels"]}
    assert "pytest" not in {x["name"] for x in manifest["wheels"]}


def test_glibc_217_fails_instead_of_selecting_incompatible_numpy_wheel():
    with pytest.raises(ValueError, match="NO_OFFLINE_LINUX_CP313_WHEEL: numpy"):
        wheels.build_manifest(wheels.official_lock(wheels.SOURCE), glibc_minor=17)


def test_unknown_or_malicious_glibc_refused():
    for value in (16, 37, -1, True, 18.5, "35"):
        with pytest.raises(ValueError, match="glibc floor"):
            wheels.target_tags(value)


def test_missing_dependency_edge_fails_closed():
    lock = {"package": [
        {"name": "automation-bench", "dependencies": [{"name": "verifiers"}]},
    ]}
    with pytest.raises(ValueError, match="transitive dependency missing"):
        wheels.runtime_closure(lock)


def test_windows_only_dep_not_included_and_requested_extras_are_traversed():
    lock = {"package": [
        {"name": "automation-bench", "dependencies": [{"name": "verifiers"},
                                                        {"name": "windows-mod", "marker": "sys_platform == 'win32'"}]},
        {"name": "verifiers", "dependencies": [{"name": "extras-mod", "extra": ["safe"]}]},
        {"name": "extras-mod", "optional-dependencies": {"safe": [{"name": "needed"}]}},
        {"name": "needed"}, {"name": "windows-mod"},
    ]}
    assert [x["name"] for x in wheels.runtime_closure(lock)] == ["extras-mod", "needed", "verifiers"]


def test_wheel_url_must_be_original_https_pypi_and_have_locked_hash():
    candidates = [{"url": "https://evil.example/pkg-1.0-py3-none-any.whl",
                   "hash": "sha256:" + "a" * 64, "size": 10}]
    with pytest.raises(ValueError, match="PyPI HTTPS"):
        wheels.select_wheel({"name": "pkg", "version": "1.0", "wheels": candidates},
                            {t: i for i, t in enumerate(wheels.target_tags(35))})


def test_developer_cannot_write_into_research_checkout(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="private output"):
        wheels.private_directory(source / "new", private_root=root)


def test_no_implicit_network_or_writes_for_default_main(monkeypatch, capsys):
    monkeypatch.setattr(wheels, "prepare", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("WRITES")))
    monkeypatch.setattr(wheels, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("NETWORK")))
    monkeypatch.setattr("sys.argv", ["wheelhouse"])
    assert wheels.main() == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "SOURCE_LOCKED_CPU_PLAN_NO_WRITES"
    assert status["package_count"] == 108


def test_exclusive_private_manifest_only_and_offline_verification_missing_wheels(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    expected = source_manifest()
    output = root / "new-private"
    summary = wheels.prepare(output, expected, private_root=root)
    assert summary["offline_complete"] is False
    assert output.is_dir()
    assert (output / wheels.MANIFEST_FILE).is_file()
    with pytest.raises(ValueError, match="wheelhouse incomplete"):
        wheels.verify(output, expected)
    with pytest.raises(ValueError, match="exclusive"):
        wheels.prepare(output, expected, private_root=root)


def test_offline_manifest_identity_tampering_cannot_pass(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    expected = source_manifest()
    output = root / "new"
    wheels.prepare(output, expected, private_root=root)
    path = output / wheels.MANIFEST_FILE
    tampered = json.loads(path.read_text())
    tampered["wheels"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="does not match source-pinned"):
        wheels.verify(output, expected)


def _synthetic_offline_bundle(tmp_path: Path, monkeypatch):
    blob = b"synthetic-wheel-not-installed"
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    entries = []
    for name in ("datasets", "pydantic", "verifiers"):
        filename = f"{name}-1.0-py3-none-any.whl"
        (wheel_dir / filename).write_bytes(blob)
        entries.append({"name": name, "version": "1.0", "filename": filename,
                        "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)})
    source_file = tmp_path / "source.tar.bin"
    source_file.write_bytes(b"synthetic-src-archive")
    manifest = {"schema": cpu.SCHEMA, "official_source_commit": cpu.OFFICIAL_COMMIT,
                "uv_lock_sha256": cpu.OFFICIAL_LOCK_SHA,
                "official_source_tar_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                "python": "CPython3.13", "platform": "linux_x86_64",
                "minimum_glibc": "2.35", "wheels": entries, "package_count": len(entries),
                "total_locked_wheel_bytes": len(blob) * len(entries)}
    path = wheel_dir / wheels.MANIFEST_FILE
    path.write_text(json.dumps(manifest))
    expected = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()
    monkeypatch.setattr(cpu, "LOCKED_WHEEL_MANIFEST_SHA", expected)
    monkeypatch.setattr(cpu, "OFFICIAL_TAR_SHA", manifest["official_source_tar_sha256"])
    monkeypatch.setattr(cpu, "glibc_minor", lambda: 35)
    monkeypatch.setattr(cpu.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpu.platform, "libc_ver", lambda: ("glibc", "2.35"))
    return wheel_dir, path, source_file


def test_cpu_preflight_inspects_only_and_does_not_install(tmp_path, monkeypatch):
    wheel_dir, manifest, tar = _synthetic_offline_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(cpu.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("EXECUTED")))
    result = cpu.inspect(wheel_dir, manifest, tar, require_native_python=False)
    assert result["status"] == "CPU_OFFLINE_SOURCES_AND_WHEELS_VERIFIED_NOT_INSTALLED"
    assert result["weights_loaded"] is False and result["package_count"] == 3


def test_cpu_preflight_refuses_missing_wheel_and_tampered_digest(tmp_path, monkeypatch):
    wheel_dir, manifest, tar = _synthetic_offline_bundle(tmp_path, monkeypatch)
    wheel = wheel_dir / "verifiers-1.0-py3-none-any.whl"
    wheel.unlink()
    with pytest.raises(ValueError, match="wheel missing"):
        cpu.inspect(wheel_dir, manifest, tar, require_native_python=False)
    wheel.write_bytes(b"synthetic-wheel-not-installeD")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cpu.inspect(wheel_dir, manifest, tar, require_native_python=False)


def test_cpu_preflight_rejects_wrong_glibc_and_rehashed_manifest(tmp_path, monkeypatch):
    wheel_dir, manifest, tar = _synthetic_offline_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(cpu, "glibc_minor", lambda: 34)
    with pytest.raises(ValueError, match="older"):
        cpu.inspect(wheel_dir, manifest, tar, require_native_python=False)
    monkeypatch.setattr(cpu, "glibc_minor", lambda: 35)
    data = json.loads(manifest.read_text())
    data["wheels"][0]["version"] = "999.0"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete source-locked"):
        cpu.inspect(wheel_dir, manifest, tar, require_native_python=False)


def test_kaggle_cpu_install_scratch_refuses_foreign_and_existing_paths(tmp_path):
    home = tmp_path / "working"
    home.mkdir()
    with pytest.raises(ValueError, match="under /kaggle/working"):
        cpu.ensure_private_target(tmp_path / "forbidden", home=home)
    existing = home / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        cpu.ensure_private_target(existing, home=home)


def test_cpu_bootstrap_contains_no_model_or_shell_workflow():
    raw = Path(cpu.__file__).read_text()
    assert "shell=True" not in raw
    assert "nvidia-smi" not in raw
    assert "uv pip install" not in raw
    assert "--no-index" in raw and "--no-deps" in raw and "--require-hashes" in raw
    assert "python3.13" in raw
    assert cpu.OFFICIAL_TAR_SHA == wheels.SOURCE_ARCHIVE_SHA256
    assert cpu.LOCKED_WHEEL_MANIFEST_SHA == wheels.EXPECTED_MANIFEST_SHA256_235
