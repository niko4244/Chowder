"""Fail-closed resumability without inventing any scientific test score."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder_batch import reap_humaneval_v38_guarded_resume_20261027 as resume


@pytest.mark.parametrize("arm", ["q4", "q8"])
def test_existing_native_prefix_verified_read_only_with_no_second_model(arm, monkeypatch):
    root = resume.AUTHORIZED_ROOTS[arm]
    if not root.is_dir():
        pytest.skip("private original evidence is unavailable")
    monkeypatch.setattr(resume.v38, "_run_research_candidate",
                        lambda *_: pytest.fail("must not execute in read-only mode"))
    if (root / (arm + "-complete.private.json")).exists():
        # Full original completion is immutable. The guarded resumer MUST
        # refuse reopening that population, even for its former prefix mode.
        # The separate independent whole-suite certificate audits completion.
        with pytest.raises(ValueError, match="already marked complete"):
            resume.continue_original(arm, root, execute=False)
        return
    report = resume.continue_original(arm, root, execute=False)
    assert report["verified_contiguous_rows"] >= (23 if arm == "q4" else 3)
    assert report["unknown_remaining_rows"] > 0
    assert report["gen0_status"] == "BLOCKED"
    assert report["status"] == "READ_ONLY_VERIFIED_ORIGINAL_PREFIX_NO_INFERENCE"


def test_refuse_foreign_or_aliased_source_roots(tmp_path):
    with pytest.raises(ValueError, match="foreign"):
        resume.original_prefix("q4", tmp_path)
    with pytest.raises(ValueError, match="foreign"):
        resume.original_prefix("q8", resume.AUTHORIZED_ROOTS["q4"])


@pytest.mark.parametrize("bad", [-1, 165, True, None, "20"])
def test_max_new_is_explicit_bounded_integer(bad):
    with pytest.raises(ValueError, match="bounded"):
        resume.continue_original("q4", resume.AUTHORIZED_ROOTS["q4"],
                                 execute=True, max_new=bad)


def test_fake_python_scoring_owner_refused(monkeypatch):
    fake = SimpleNamespace(info={"pid": 999999,
                                 "name": "python.exe",
                                 "cmdline": ["python", "-m",
                                             "chowder_batch.reap_humaneval_v38_actual_execution_20261027",
                                             "--research-execute"]})
    monkeypatch.setattr(resume.psutil, "process_iter", lambda _: [fake])
    with pytest.raises(RuntimeError, match="another original"):
        resume._assert_no_other_owners()


def test_shell_quoting_not_misidentified_as_scoring_owner(monkeypatch):
    shell = SimpleNamespace(info={"pid": 888888,
                                  "name": "powershell.exe",
                                  "cmdline": ["-Command", "reap_humaneval_v38_guarded_resume_20261027 --execute"]})
    monkeypatch.setattr(resume.psutil, "process_iter", lambda _: [shell])
    monkeypatch.setattr(resume.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=0, stdout="hermes-brainz-litellm\n"))
    resume._assert_no_other_owners()


def test_active_native_podman_scoring_capsule_refused(monkeypatch):
    monkeypatch.setattr(resume.psutil, "process_iter", lambda _: [])
    monkeypatch.setattr(resume.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=0, stdout="w10-he-v38-1234567890abcdef\n"))
    with pytest.raises(RuntimeError, match="still-active"):
        resume._assert_no_other_owners()


def test_missing_native_podman_ownership_census_refuses(monkeypatch):
    monkeypatch.setattr(resume.psutil, "process_iter", lambda _: [])
    monkeypatch.setattr(resume.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=2, stdout=""))
    with pytest.raises(RuntimeError, match="cannot inspect"):
        resume._assert_no_other_owners()


def test_source_only_private_root_cannot_be_rewritten(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "AUTHORIZED_ROOTS", {"q4": tmp_path})
    with pytest.raises(ValueError, match="root"):
        resume.original_prefix("q4", Path("F:/wrong-alias"))
