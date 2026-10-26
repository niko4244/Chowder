"""CPU-only adversarial checks for the isolated public AutomationBench lane.

These tests never load GGUF weights, open GPU ports or invoke model tools.
"""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from chowder_batch import reap_q4_automationbench as ab


def test_frozen_q4_flags_and_official_evaluator_command():
    server, command = ab._commands(Path("F:/private/out"), count=1)
    assert server[1:7] == ["--fit", "on", "-np", "4", "-c", "16384"]
    assert server[7] == "--jinja"
    assert "--base-url" in command
    assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8891/v1"
    assert command[command.index("--max-concurrent") + 1] == "1"
    assert command[command.index("--num-examples") + 1] == "1"
    assert command[command.index("--max-steps") + 1] == "50"
    assert command[1:4] == ["-I", "-m", "automationbench.scripts.eval"]
    assert "--toolset" in command and command[command.index("--toolset") + 1] == "api"


@pytest.mark.parametrize("count", [0, 2, 599, 601, -1, True, "600"])
def test_invalid_count_refuses(count):
    with pytest.raises(ValueError, match="one discovery task or all 600"):
        ab._commands(Path("F:/private/out"), count=count)


def test_no_resource_admission_never_hashes_or_starts(tmp_path):
    out = tmp_path / "new-private-out"
    with (patch.object(ab, "_read_only_status", return_value={"status": "REFUSED_RESOURCE_ADMISSION"}),
          patch.object(ab, "_source_check") as source,
          patch.object(ab.subprocess, "Popen") as spawn,
          patch.object(ab.watchdog, "sha") as hash_model):
        if ab.os.name == "nt":
            result = ab.run_private_public_benchmark(out, count=1)
            assert result["status"] == "REFUSED_RESOURCE_ADMISSION"
        else:
            with pytest.raises(ValueError, match="Windows"):
                ab.run_private_public_benchmark(out, count=1)
        source.assert_not_called()
        spawn.assert_not_called()
        hash_model.assert_not_called()
    assert not out.exists()


def test_modified_or_untracked_official_python_source_refused(tmp_path):
    # Even with exact Git HEAD, an untracked sitecustomize.py is importable.
    (tmp_path / "sitecustomize.py").write_text("# adversarial shadow file")
    revision = subprocess.CompletedProcess([], 0, stdout=ab.BENCH_COMMIT + "\n")
    dirty = subprocess.CompletedProcess([], 0, stdout="?? sitecustomize.py\n")
    with (patch.object(ab, "BENCH_ROOT", tmp_path),
          patch.object(ab.subprocess, "run", side_effect=[revision, dirty])):
        with pytest.raises(ValueError, match="untracked"):
            ab._source_check()


def test_ignored_dotenv_file_fails_closed_even_with_clean_git(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=not-for-benchmark")
    clean_revision = subprocess.CompletedProcess([], 0, stdout=ab.BENCH_COMMIT + "\n")
    clean_status = subprocess.CompletedProcess([], 0, stdout="")
    with (patch.object(ab, "BENCH_ROOT", tmp_path),
          patch.object(ab.subprocess, "run", side_effect=[clean_revision, clean_status])):
        with pytest.raises(ValueError, match="override"):
            ab._source_check()


def test_non_f_private_output_refused(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        ab._output_check(tmp_path / "new-run")


@pytest.mark.parametrize("ids", [[], [{"id": "other-model"}], [{"id": "reap128_v2_q4"}, {"id": "other"}]])
def test_wrong_native_model_inventory_refuses(ids):
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Length": "0"}
    response.read.return_value = __import__("json").dumps({"data": ids}).encode()
    response.__enter__.return_value = response
    with (patch.object(ab.reap_serve, "_require_owned_listener"),
          patch.object(ab.urllib.request, "urlopen", return_value=response)):
        with pytest.raises(ValueError):
            ab._served_model(ab.PORT, MagicMock())


def test_exact_native_q4_inventory_allowed():
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Length": "0"}
    response.read.return_value = b'{"data":[{"id":"reap128_v2_q4"}]}'
    response.__enter__.return_value = response
    with (patch.object(ab.reap_serve, "_require_owned_listener"),
          patch.object(ab.urllib.request, "urlopen", return_value=response)):
        assert ab._served_model(ab.PORT, MagicMock()) == ab.MODEL


def test_cuda_receipt_rejects_incomplete_or_cpu_offload(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("load_tensors: offloaded 40/41 layers to GPU\n")
    with pytest.raises(ValueError, match="CUDA"):
        ab._verify_native_cuda_startup(log)
    log.write_text("common_fit: model successfully fit\nload_tensors: offloaded 41/41 layers to GPU\n"
                   "llama_model_loader: n_cpu_moe = 4\n")
    with pytest.raises(ValueError, match="CUDA"):
        ab._verify_native_cuda_startup(log)


def test_cuda_receipt_requires_positive_source_geometry(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("common_fit: model successfully fit\nload_tensors: offloaded 41/41 layers to GPU\n")
    receipt = ab._verify_native_cuda_startup(log)
    assert receipt["layers_cuda"] == "41/41"
    assert receipt["measured_experts_offloaded"] == 0
    assert len(receipt["startup_sha256"]) == 64


def test_resource_monitor_fails_closed_on_bad_measurement(tmp_path):
    proc = MagicMock()
    proc.poll.return_value = None
    with (patch.object(ab.watchdog, "sample", return_value={"ram_available_gib": float("nan")}),
          patch.object(ab.watchdog, "unsafe", return_value=["invalid resource telemetry"])):
        with pytest.raises(RuntimeError, match="floor"):
            ab._resource_watch(proc, tmp_path, started=ab.time.monotonic(), timeout=10)


def test_runtime_samples_fsynced_with_real_phase_and_fields(tmp_path):
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 1234
    state = {"ram_available_gib": 20.0, "commit_available_gib": 27.0,
             "gpu_free_mib": 6500, "disk_free_gib": 31.0}
    log_path = tmp_path / "resource.private.jsonl"
    with (patch.object(ab.watchdog, "sample", return_value=state),
          patch.object(ab.watchdog, "unsafe", return_value=[])):
        with log_path.open("x", encoding="utf8") as logfile:
            ab._resource_watch(proc, tmp_path, started=ab.time.monotonic(),
                               timeout=10, log=logfile, phase="official_task_evaluation")
    observation = __import__("json").loads(log_path.read_text(encoding="utf8"))
    assert observation["phase"] == "official_task_evaluation"
    assert observation["owned_native_pid"] == 1234
    assert observation["measurement"] == state
    assert observation["failures"] == []


def test_runtime_floor_failure_writes_private_abort_receipt(tmp_path):
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 1234
    log_path = tmp_path / "resource.private.jsonl"
    with (patch.object(ab.watchdog, "sample", return_value={"ram_available_gib": 2}),
          patch.object(ab.watchdog, "unsafe", return_value=["physical RAM below original floor"])):
        with log_path.open("x", encoding="utf8") as logfile:
            with pytest.raises(RuntimeError, match="floor"):
                ab._resource_watch(proc, tmp_path, started=ab.time.monotonic(),
                                   timeout=10, log=logfile, phase="native_model_load")
    recorded = __import__("json").loads(log_path.read_text(encoding="utf8"))
    assert recorded["failures"] == ["physical RAM below original floor"]


def test_nonfinite_runtime_telemetry_cannot_appear_in_receipt(tmp_path):
    proc = MagicMock()
    proc.poll.return_value = None
    log_path = tmp_path / "resource.private.jsonl"
    with (patch.object(ab.watchdog, "sample", return_value={"ram_available_gib": float("nan")}),
          patch.object(ab.watchdog, "unsafe", return_value=["invalid measurement"])):
        with log_path.open("x", encoding="utf8") as logfile:
            with pytest.raises(ValueError, match="telemetry"):
                ab._resource_watch(proc, tmp_path, started=ab.time.monotonic(),
                                   timeout=10, log=logfile)
    assert log_path.read_bytes() == b""


def test_read_only_state_does_not_run_source_or_spawn(tmp_path):
    with (patch.object(ab, "_commands", return_value=([], [])),
          patch.object(ab, "_output_check"),
          patch.object(ab, "_fresh_admission", return_value=({"status": "REFUSED"}, {})),
          patch.object(ab, "_source_check") as source,
          patch.object(ab.subprocess, "Popen") as spawn):
        status = ab._read_only_status(tmp_path / "new-out", count=1)
        assert status["status"] == "REFUSED_RESOURCE_ADMISSION"
        assert status["gen0_status"] == "BLOCKED"
        assert status["release_approved"] is False
        source.assert_not_called()
        spawn.assert_not_called()
