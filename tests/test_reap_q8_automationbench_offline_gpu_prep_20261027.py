"""CPU-only source transformation tests: never execute a model or Kaggle job."""

import ast
import hashlib
import io
import json
import re
import tarfile
from types import SimpleNamespace

import pytest

from chowder_batch import reap_q8_automationbench_kaggle_20261026 as original
from chowder_batch import reap_q8_automationbench_offline_gpu_prep_20261027 as prep


def source(profile="six"):
    """Use the immutable notebook that was actually submitted, not a new render."""
    folder, filename, expected_sha = prep.gate.Q8_NOTEBOOKS[profile]
    path = prep.PRIVATE_ROOT / folder / filename
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    return raw.decode("utf-8")


def installer():
    raw = prep.PREFLIGHT_SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep.PREFLIGHT_SHA
    return raw.decode("utf-8")


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_patches_only_offline_bootstrap_and_preserves_original_official_protocol(profile):
    before = source(profile)
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
    assert "R['official_source_selected_order'] = names" in after
    assert "or set(names) != set(CFG['tasks'])" in after
    assert "or [x.get('name') for x in tasks]!=CFG['tasks']" in after
    assert "R['prelaunch_before_clean_reclaim'] = snapshot()" in after
    assert "R['prelaunch_clean_reclaim'] = reclaim_clean_offline_cache" in after
    assert "R['prelaunch'] = initial\n    check_resources(initial, prelaunch=True)" in after
    assert "if s['cgroup'] < (29.443 if prelaunch else 10)" in after
    assert "if postload['cgroup'] < 14" in after


def _replay_population_preflight(profile, selected, *, prompt_sha=None):
    """Execute only the extracted preflight with a fixed synthetic source reply."""
    patched = prep.patch_kernel(source(profile), installer(), profile=profile)
    function = next(node for node in ast.parse(patched).body
                    if isinstance(node, ast.FunctionDef) and node.name == "prove_population")
    payload = {"n": 600, "prompt_sha": prompt_sha or original.PUBLIC_PROMPTS_SHA256,
               "domains": ["sales", "marketing", "operations", "support", "finance", "hr"],
               "selected": selected}
    fake_process = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    state = {"CFG": {"tasks": list(original.PROFILES[profile]["task_names"]),
                     "public_prompt_sha256": original.PUBLIC_PROMPTS_SHA256},
             "R": {}, "subprocess": fake_process, "json": json, "re": re,
             "limited_env": lambda _: {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<synthetic-preflight>", "exec"), state)
    state["prove_population"]("python", "official-source")
    return state["R"]


def _selected(profile):
    names = list(original.PROFILES[profile]["task_names"])
    # The authentic source iterates the six tasks in a different order from
    # the original Q4 official-result export, despite identical identities.
    if profile == "six":
        names = names[2:4] + names[:2] + names[4:]
    return [{"name": name, "example_id": index + 1,
             "contract": hashlib.sha256(name.encode()).hexdigest()}
            for index, name in enumerate(names)]


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_source_order_difference_is_admitted_without_reordering_official_evaluator(profile):
    selected = _selected(profile)
    receipt = _replay_population_preflight(profile, selected)
    assert receipt["official_source_selected_order"] == [row["name"] for row in selected]
    assert [row["name"] for row in receipt["official_task_contracts"]] == list(
        original.PROFILES[profile]["task_names"])


@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "missing", "contract", "example_id"])
def test_population_preflight_refuses_identity_and_contract_forgery(mutation):
    selected = _selected("six")
    if mutation == "duplicate":
        selected[-1]["name"] = selected[0]["name"]
    elif mutation == "foreign":
        selected[-1]["name"] = "sales.forged_task"
    elif mutation == "missing":
        selected.pop()
    elif mutation == "contract":
        selected[-1]["contract"] = "0" * 63 + "X"
    else:
        selected[-1]["example_id"] = selected[0]["example_id"]
    with pytest.raises(RuntimeError, match="unique task identities"):
        _replay_population_preflight("six", selected)


def test_population_preflight_refuses_changed_full_600_prompt_hash():
    with pytest.raises(RuntimeError, match="unique task identities"):
        _replay_population_preflight("six", _selected("six"), prompt_sha="0" * 64)


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


def _reclaim_roots(tmp_path, monkeypatch, profile="six"):
    monkeypatch.setattr(prep, "WORK", tmp_path)
    monkeypatch.setattr(prep.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(prep.os, "O_NOFOLLOW", 0, raising=False)
    advised = []
    monkeypatch.setattr(prep.os, "posix_fadvise", lambda *args: advised.append(args),
                        raising=False)
    roots = tuple(tmp_path / name for name in (
        f"reap-q8-offline108-{profile}", f"reap-q8-wheelhouse-{profile}",
        "automationbench-pinned-official-source",
    ))
    for root in roots:
        root.mkdir()
    return roots, advised


def test_prelaunch_reclaim_only_advises_owned_clean_files(tmp_path, monkeypatch):
    roots, advised = _reclaim_roots(tmp_path, monkeypatch)
    (roots[0] / "installed").mkdir()
    (roots[0] / "installed" / "package.py").write_bytes(b"python")
    (roots[1] / "pinned.whl").write_bytes(b"wheel")
    (roots[2] / "pyproject.toml").write_bytes(b"source")
    before = [(root / name).read_bytes() for root, name in zip(
        roots, ("installed/package.py", "pinned.whl", "pyproject.toml"))]
    proof = prep.reclaim_clean_offline_cache(roots, "six")
    assert proof == {"advised_files": 3, "advised_bytes": 17}
    assert len(advised) == 3
    assert [(root / name).read_bytes() for root, name in zip(
        roots, ("installed/package.py", "pinned.whl", "pyproject.toml"))] == before


@pytest.mark.parametrize("attack", ["missing", "foreign", "duplicate", "wrong_profile", "no_advisory"])
def test_prelaunch_reclaim_refuses_unowned_or_unsupported_paths(tmp_path, monkeypatch, attack):
    roots, _ = _reclaim_roots(tmp_path, monkeypatch)
    profile = "six"
    if attack == "missing":
        roots[0].rmdir()
    elif attack == "foreign":
        foreign = tmp_path / "unowned"
        foreign.mkdir()
        roots = (foreign, *roots[1:])
    elif attack == "duplicate":
        roots = (roots[0], roots[0], roots[2])
    elif attack == "wrong_profile":
        profile = "guard2"
    else:
        monkeypatch.delattr(prep.os, "posix_fadvise")
    with pytest.raises(RuntimeError, match="scratch|exclusive|unavailable"):
        prep.reclaim_clean_offline_cache(roots, profile)


def test_prelaunch_reclaim_refuses_symlink_before_advising(tmp_path, monkeypatch):
    roots, advised = _reclaim_roots(tmp_path, monkeypatch)
    target = tmp_path / "outside"
    target.write_bytes(b"secret")
    try:
        (roots[0] / "bad").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("host disallows unprivileged symlink creation")
    with pytest.raises(RuntimeError, match="symlink"):
        prep.reclaim_clean_offline_cache(roots, "six")
    assert not advised
