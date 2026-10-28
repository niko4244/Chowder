"""Frozen original-v2 wave tests; fixtures are NOT model evaluations."""

from pathlib import Path
import pytest

from chowder_batch import reap_v2_frozen_wave_20261027 as wave


def test_exact_registered_order_and_original_quantization():
    assert [item.arm for item in wave.STUDIES] == [
        "dense_v2_q4", "reap128_v2_q4", "dense_v2", "reap128_v2", "random128_v2",
    ]
    assert [item.limit for item in wave.STUDIES] == [200, 200, 0, 0, 0]
    assert [item.suite for item in wave.STUDIES] == ["gsm8k"] * 5
    assert all(item.model_sha256 == wave.PINNED[item.arm] for item in wave.STUDIES)
    assert [wave.NATIVE_ADMISSION_ARM[study.arm] for study in wave.STUDIES] == [
        "dense_q4", "q4", "dense_q8", "q8", "random128_q8",
    ]
    assert wave.FLAGS == ("--fit", "on", "-np", "4", "-c", "16384", "--jinja")
    assert wave.BLOCKED_HUMANEVAL_EXECUTION


def test_plan_is_exact_frozen_serve_then_original_gsm8k(tmp_path: Path):
    s = wave.STUDIES[0]
    args = wave.command(s, model=tmp_path / "m.gguf", output=tmp_path / "fresh",
                        server=tmp_path / "llama-server.exe", port=8795)
    assert args[1:3] == [str(wave.REPO / "chowder_batch" / "reap_serve.py"), "--gguf"]
    assert args.count("--gguf") == args.count("--arm") / 2 == 1
    assert args.count("--port") == 2
    assert args.count("--out") == 2
    assert args[args.index("--") + 1:] == [
        "--port", "8795", "--arm", "dense_v2_q4", "--out", str(tmp_path / "fresh"),
        "--suites", "gsm8k", "--parallel", "4", "--limit", "200",
    ]
    assert "humaneval" not in args


@pytest.mark.parametrize("bad", ["../../escape", "dense", "reap160", "humaneval", "reap128_v2_q8"])
def test_unregistered_or_ineligible_arm_refused(bad: str):
    with pytest.raises(ValueError, match="registered"):
        wave.study_for(bad)


def test_existing_model_output_refuses_before_any_launch(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture-not-real-weights")
    output = tmp_path / "owned"
    output.mkdir()
    (output / "dense_v2_q4").mkdir()
    with pytest.raises(ValueError, match="existing" ):
        wave.validate_fresh_output(output, "dense_v2_q4")


def test_output_path_is_not_created_during_audit(tmp_path: Path):
    output = tmp_path / "new"
    wave.validate_fresh_output(output, "dense_v2_q4")
    assert not output.exists()


def test_frozen_reap_serve_and_eval_sources_must_agree():
    wave.assert_frozen_code_identity()


def test_unavailable_models_are_marked_blocked_not_fake_execution(tmp_path: Path):
    report = wave.plan(lambda arm: tmp_path / (arm + ".gguf"),
                       output=tmp_path / "cohort", server=tmp_path / "none.exe", port=8795)
    assert all(row["status"] == "BLOCKED_MISSING_MODEL" for row in report["studies"])
    assert report["gen0"] == {"status": "BLOCKED", "chosen": None, "release_approved": False}
    assert report["humaneval"] == "BLOCKED_REQUIRES_ORIGINAL_APPROVED_SANDBOX"
    assert not (tmp_path / "cohort").exists()


def test_execution_is_refused_without_direct_fresh_admission(tmp_path: Path, monkeypatch):
    s = wave.STUDIES[0]
    calls = []
    monkeypatch.setattr(wave.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(wave.AdmissionRefused, match="EXPLICIT_OWNED_RUN"):
        wave.execute(s, model=tmp_path / "model.gguf", output=tmp_path / "fresh",
                     server=tmp_path / "llama-server.exe", port=8795)
    assert not calls
    assert not (tmp_path / "fresh").exists()


def test_positive_capacity_probe_is_not_permission_to_launch_model(tmp_path: Path, monkeypatch):
    from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as native

    model = tmp_path / "m.gguf"
    server = tmp_path / "llama-server.exe"
    model.write_bytes(b"synthetic-model-not-executable")
    server.write_bytes(b"synthetic-server-not-executable")
    observed = {}

    def fake_snapshot(**kwargs):
        observed.update(kwargs)
        return {"synthetic": "not a real native admission"}

    monkeypatch.setattr(native, "snapshot", fake_snapshot)
    monkeypatch.setattr(native, "assess", lambda *a, **k: {
        "status": "READY_FOR_CAPACITY_PROBE_ONLY", "reasons": [],
    })
    allowed, reasons = wave.admit_for_execution(
        study=wave.study_for("reap128_v2_q4"), model=model, server=server, port=8790)
    assert observed["arm"] == "q4"
    assert not allowed
    assert "NO_INDEPENDENT_NATIVE_LAUNCH_OWNER_ADMISSION" in reasons


def test_explicit_owned_run_calls_only_private_job_supervisor(tmp_path: Path, monkeypatch):
    from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as owner

    invocations = []
    def fake_safe_owner(**kwargs):
        invocations.append(kwargs)
        return {"status": "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY",
                "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}

    monkeypatch.setattr(owner, "run_frozen_arm", fake_safe_owner)
    expected = wave.study_for("dense_v2_q4")
    result = wave.execute(expected, model=tmp_path / "original.gguf",
                          server=tmp_path / "llama-server.exe",
                          output=tmp_path / "new-wave", receipt_dir=tmp_path / "receipts",
                          port=8790, execute_authorized=True)
    assert result["gen0"]["status"] == "BLOCKED"
    assert len(invocations) == 1
    assert invocations[0]["study"] == expected
    assert invocations[0]["execute_authorized"] is True
    assert invocations[0]["receipt_dir"] == tmp_path / "receipts"
    assert not (tmp_path / "new-wave").exists()


def test_execute_only_flag_cannot_bypass_independent_owner(tmp_path: Path):
    with pytest.raises(wave.AdmissionRefused, match="EXPLICIT_OWNED_RUN"):
        wave.execute(wave.STUDIES[0], model=tmp_path / "missing.gguf",
                     output=tmp_path / "never_created", server=tmp_path / "missing.exe",
                     port=8790, receipt_dir=tmp_path, execute_authorized=False)
    assert not (tmp_path / "never_created").exists()


def test_cli_rebinds_identical_structural_frozen_study_to_canonical_module(tmp_path: Path,
                                                                          monkeypatch):
    from dataclasses import make_dataclass
    from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as owner

    other_type = make_dataclass("OtherModuleStudy", [
        ("arm", str), ("suite", str), ("limit", int), ("model_sha256", str)
    ], frozen=True)
    original = wave.study_for("reap128_v2_q4")
    separate_class = other_type(original.arm, original.suite, original.limit,
                                original.model_sha256)
    handed = []

    def fake_owner(**kw):
        handed.append(kw["study"])
        return {"status": "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY",
                "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}

    monkeypatch.setattr(owner, "run_frozen_arm", fake_owner)
    wave.execute(separate_class, model=tmp_path / "model.gguf",
                 server=tmp_path / "server.exe", output=tmp_path / "new",
                 receipt_dir=tmp_path, port=8790, execute_authorized=True)
    assert len(handed) == 1
    assert type(handed[0]) is wave.Study
    assert handed[0] == original


def test_cli_cannot_normalize_foreign_model_as_original_study(tmp_path: Path):
    original = wave.study_for("reap128_v2_q4")
    from dataclasses import replace
    fake = replace(original, model_sha256="0" * 64)
    with pytest.raises(wave.AdmissionRefused, match="STUDY_IDENTITY_CHANGED"):
        wave.execute(fake, model=tmp_path / "model.gguf",
                     server=tmp_path / "server.exe", output=tmp_path / "new",
                     receipt_dir=tmp_path, port=8790, execute_authorized=True)
