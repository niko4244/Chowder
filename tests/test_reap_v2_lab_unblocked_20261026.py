"""Source-only RED/GREEN for separate, non-release REAP128_v2 lab testing."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_v2_lab as lab


def test_experimental_branch_unblocked_without_falsifying_gen0():
    data = lab.testing_policy()
    assert data["status"] == "UNBLOCKED_FOR_TESTING"
    assert data["experimental_testing_authorized"] is True
    assert data["research_variants_unblocked"] == ["BF16", "Q8_0", "Q4_K_M"]
    assert data["production_gen0_status"] == "BLOCKED"
    assert data["production_chosen"] is None
    assert data["release_approved"] is False
    assert data["allowed_suites"] == ["gsm8k"]


def test_status_is_read_only_and_does_not_touch_resources(monkeypatch, capsys):
    monkeypatch.setattr(lab, "run_experiment", lambda *a, **k: pytest.fail("ran model"))
    assert lab.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == lab.EXPERIMENTAL_STATUS


def test_execute_requires_explicit_new_output_path():
    with pytest.raises(SystemExit):
        lab.main(["--execute"])


@pytest.mark.parametrize("limit,port", [
    (0, 8891), (17, 8891), (-1, 8891), (True, 8891),
    (4, 8790), (4, 8767), (4, 11434), (4, 1000), (4, 65536),
])
def test_cannot_bypass_smoke_limits_or_protected_ports(tmp_path, monkeypatch, limit, port):
    monkeypatch.setattr(lab.sys, "platform", "win32")
    with pytest.raises(ValueError):
        lab._execution_preflight(tmp_path / "new", limit=limit, port=port)


def test_protected_outputs_and_existing_results_refused(tmp_path, monkeypatch):
    (tmp_path / "used").mkdir()
    with pytest.raises(ValueError):
        lab._exclusive_output(tmp_path / "used")
    with pytest.raises(ValueError):
        lab._exclusive_output(Path("not-absolute/new"))
    monkeypatch.setattr(lab.watchdog, "GGUF", tmp_path / "model.gguf")
    with pytest.raises(ValueError):
        lab._exclusive_output(tmp_path / "subdir")


def test_official_serving_command_is_short_gsm8k_only(tmp_path):
    command = lab._run_command(tmp_path / "run", limit=4, port=8891)
    assert command[0] == lab.sys.executable
    assert command.count("--suites") == 1
    assert command[command.index("--suites") + 1] == "gsm8k"
    assert command[command.index("--limit") + 1] == "4"
    assert command[command.index("--parallel") + 1] == "4"
    assert "humaneval" not in command
    assert "--gguf" in command and str(lab.watchdog.GGUF) in command
    assert "--server" in command and str(lab.watchdog.SERVER) in command


def test_source_bound_admission_refusal_never_spawns_server(tmp_path, monkeypatch):
    monkeypatch.setattr(lab.sys, "platform", "win32")
    (tmp_path / "models").mkdir()
    (tmp_path / "bin").mkdir()
    model = tmp_path / "models" / "q4.gguf"
    server = tmp_path / "bin" / "llama-server.exe"
    model.write_bytes(b"fixture")
    server.write_bytes(b"fixture")
    monkeypatch.setattr(lab.watchdog, "GGUF", model)
    monkeypatch.setattr(lab.watchdog, "SERVER", server)
    monkeypatch.setattr(lab, "_fresh_admission", lambda *_: (
        {"status": "REFUSED", "reasons": ["VRAM below floor"]}, {}))
    monkeypatch.setattr(lab.watchdog, "sha", lambda *_: pytest.fail("hashed inadmissible 11GB"))
    monkeypatch.setattr(lab.subprocess, "Popen", lambda *a, **k: pytest.fail("launched model"))
    with pytest.raises(ValueError, match="admission REFUSED"):
        lab.run_experiment(tmp_path / "fresh")


def sample_results(tmp_path, *, n=4):
    arm_dir = tmp_path / lab.ARM
    arm_dir.mkdir()
    serve = {"gguf_sha256": lab.watchdog.MODEL_SHA,
             "flags": lab.testing_policy()["registered_serving_flags"],
             "server_pid": 2045, "server_port": 8891,
             "eval_session_id": "a" * 32}
    inputs = {"suite": "gsm8k", "source_items_n": 1319,
              "source_items_sha256": "b" * 64,
              "model": {"gguf_sha256": lab.watchdog.MODEL_SHA,
                        "flags": lab.testing_policy()["registered_serving_flags"]}}
    rows = [{"id": f"gsm8k-{i}", "correct": bool(i % 2),
             "terminated": True, "text": "1"} for i in reversed(range(n))]
    raw_rows = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (arm_dir / "serve.json").write_text(json.dumps(serve))
    (arm_dir / "gsm8k.inputs.json").write_text(json.dumps(inputs))
    (arm_dir / "gsm8k.jsonl").write_bytes(raw_rows)
    (arm_dir / "summary.json").write_text(json.dumps({"gsm8k": {
        "n": n, "file_sha256": hashlib.sha256(raw_rows).hexdigest()}}))
    return arm_dir


def test_fixture_row_witness_is_descriptive_not_scientific(tmp_path, monkeypatch):
    sample_results(tmp_path)
    monkeypatch.setattr(lab.watchdog, "verify_native_shutdown", lambda *a: True)
    observed = lab._verify_smoke_results(tmp_path, limit=4, port=8891)
    assert observed["smoke_rows_verified"] == 4
    assert observed["correct_items_observed"] == 2


@pytest.mark.parametrize("mutate", [
    lambda p: (p / "gsm8k.jsonl").unlink(),
    lambda p: (p / "summary.json").write_text("{}"),
    lambda p: (p / "serve.json").write_text("{}"),
    lambda p: (p / "gsm8k.inputs.json").write_text("{}"),
    lambda p: (p / "gsm8k.jsonl").write_bytes(b'{"id":"gsm8k-0"}\n'),
])
def test_incomplete_or_fake_executed_results_cannot_report_smoke_success(tmp_path, mutate,
                                                                         monkeypatch):
    monkeypatch.setattr(lab.watchdog, "verify_native_shutdown", lambda *a: True)
    arm_dir = sample_results(tmp_path)
    mutate(arm_dir)
    with pytest.raises(ValueError):
        lab._verify_smoke_results(tmp_path, limit=4, port=8891)


def test_live_server_remaining_does_not_certify_completed_smoke(tmp_path, monkeypatch):
    sample_results(tmp_path)
    def still_resident(*_args):
        raise RuntimeError("private model server still resident")
    monkeypatch.setattr(lab.watchdog, "verify_native_shutdown", still_resident)
    with pytest.raises(RuntimeError, match="still resident"):
        lab._verify_smoke_results(tmp_path, limit=4, port=8891)


def test_synthetically_admitted_lab_invokes_supervised_wrapper_not_selection(
    tmp_path, monkeypatch,
):
    """Positive orchestration fixture; not a model, benchmark, or CUDA test."""
    output = tmp_path / "new-lab"
    admitted = {"status": "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"}
    command = ["fixed-fake-wrapper"]
    monkeypatch.setattr(lab, "_execution_preflight", lambda *_args, **_kw: (
        admitted, command))
    monkeypatch.setattr(lab, "_fresh_admission", lambda *_args: (admitted, {}))
    monkeypatch.setattr(lab, "_verify_smoke_results", lambda *_args, **_kw: (
        {"smoke_rows_verified": 4, "serving_session_id": "b" * 32}))
    launched = []
    class FixedCompletedChild:
        def poll(self):
            return 0
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(lab.subprocess, "Popen", lambda args, **kw: (
        launched.append((args, kw)), FixedCompletedChild())[1])
    result = lab.run_experiment(output, limit=4)
    assert launched and launched[0][0] == command
    assert launched[0][1]["shell"] is False
    assert result["status"] == "EXPERIMENTAL_SMOKE_FINISHED"
    assert result["smoke_rows_verified"] == 4
    assert result["production_gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False
    assert result["official_score_certified"] is False


def test_frozen_selector_and_humaneval_verifier_unchanged():
    from chowder_batch import reap_select, reap_humaneval_v2_attestation as verify
    assert reap_select.select({})["status"] == "BLOCKED"
    assert verify.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert verify.APPROVED_INDEPENDENT_WITNESS_SHA256 is None
